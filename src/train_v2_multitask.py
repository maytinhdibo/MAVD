import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
import numpy as np
from torch.optim import AdamW
from transformers import (AutoModel, AutoTokenizer, get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)

from dann_utils import (ConditionalLanguageClassifier, LanguageClassifier,
                        dann_lambda_schedule, entropy_weight)
from logger import get_logger, setup_logging
from mavd import MulVulMultiTaskV2, MulVulSoftMoEV3, MulVulV2WithCCPP, MulVulV2WithLangAdapter, MulVulV2With3LangAdapter
from seed_utils import make_generator, set_seed, worker_init_fn
from ultis import create_data_loader, load_data

def ccpp_ramp_factor(epoch, start_epoch, ramp_epochs, mode):

    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0 or mode == 'none':
        return 1.0
    x = min(max((epoch - start_epoch + 1) / ramp_epochs, 0.0), 1.0)
    if mode == 'linear':
        return x
    if mode == 'cosine':
        return 0.5 * (1.0 - math.cos(math.pi * x))
    return 1.0

GROUP_LABEL_MAP = {
    'group-s': 0, 's': 0,
    'cwe-022': 0, 'cwe-078': 0,
    'group-w': 1, 'w': 1,
    'cwe-079': 1, 'cwe-089': 1,
}

CWE_LABEL_MAP = {
    'cwe-022': 0, 'cwe-078': 1, 'cwe-079': 2, 'cwe-089': 3,
}

def _norm_cwe(c):
    import re
    s = str(c).lower().strip()
    m = re.match(r'cwe-(\d+)', s)
    return f'cwe-{int(m.group(1)):03d}' if m else s

def add_group_labels(df, allow_unmapped=False, cwe_head=False):
    df = df.copy()
    if cwe_head:
        labs = df['original_CWE_ID'].map(lambda c: CWE_LABEL_MAP.get(_norm_cwe(c)))
        df['group_label'] = labs.fillna(-100).astype(int) if allow_unmapped else labs.astype(int)
        return df
    groups = df['CWE_ID'].astype(str).str.lower().str.strip().map(GROUP_LABEL_MAP)
    if groups.isna().any():
        if allow_unmapped:
            df['group_label'] = groups.fillna(-100).astype(int)
            return df
        bad = sorted(df.loc[groups.isna(), 'CWE_ID'].astype(str).unique().tolist())
        raise ValueError(f"Unmapped CWE_ID/group labels for V2: {bad}")
    df['group_label'] = groups.astype(int)
    return df

def supervised_contrastive_loss(features, labels, languages=None, temperature=0.2):

    valid = labels >= 0
    if valid.sum() < 2:
        return features.new_tensor(0.0)

    feats = F.normalize(features[valid], p=2, dim=1)
    labs = labels[valid]
    langs = languages[valid] if languages is not None else None
    n = feats.size(0)

    logits = torch.matmul(feats, feats.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
    same_group = labs.unsqueeze(0).eq(labs.unsqueeze(1)) & ~self_mask

    if langs is not None:
        diff_lang = langs.unsqueeze(0).ne(langs.unsqueeze(1))
        cross_lang_pos = same_group & diff_lang
        has_cross = cross_lang_pos.any(dim=1, keepdim=True)
        pos_mask = torch.where(has_cross, cross_lang_pos, same_group)
    else:
        pos_mask = same_group

    valid_anchor = pos_mask.any(dim=1)
    if not valid_anchor.any():
        return features.new_tensor(0.0)

    logits_masked = logits.masked_fill(self_mask, -1e9)
    log_den = torch.logsumexp(logits_masked, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~pos_mask, -1e9), dim=1)
    losses = -(log_num - log_den)
    return losses[valid_anchor].mean()

def _dtjs_probe(model, loader, device, args, vul_criterion, group_criterion,
                use_ccpp, use_js, n_batches):

    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    pos_w = vul_criterion.pos_weight

    def _cos(ga, gb):
        dot = sum((a * b).sum() for a, b in zip(ga, gb))
        na = torch.sqrt(sum((a * a).sum() for a in ga)).clamp_min(1e-12)
        nb = torch.sqrt(sum((b * b).sum() for b in gb)).clamp_min(1e-12)
        return (dot / (na * nb)).item()

    acc_cc, n_cc, acc_js, n_js = 0.0, 0, 0.0, 0
    seen = 0
    was_training = model.training
    model.train()
    for batch in loader:
        if seen >= n_batches:
            break
        seen += 1
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1).float()
        group_labels = batch['group_label'].to(device)
        languages = batch.get('language')
        languages = languages.to(device) if languages is not None else None
        if languages is None:
            continue
        vul_logits, group_logits, vul_ccpp_logits, vul_js_logits = model(
            input_ids=input_ids, attention_mask=attention_mask,
            languages=languages, return_ccpp=True)

        py = (languages == 0)
        if not py.any():
            continue
        L_py = F.binary_cross_entropy_with_logits(
            vul_logits[py], labels[py], pos_weight=pos_w)
        gl = (group_labels >= 0) & (languages == 0)
        if gl.any():
            L_py = L_py + group_criterion(group_logits[gl], group_labels[gl])
        g_py = torch.autograd.grad(L_py, bb, retain_graph=True, allow_unused=True)
        g_py = [x if x is not None else torch.zeros_like(p) for x, p in zip(g_py, bb)]
        if use_ccpp:
            cc = (languages == 1)
            if cc.any():
                L_cc = F.binary_cross_entropy_with_logits(
                    vul_ccpp_logits[cc], labels[cc], pos_weight=pos_w)
                g_cc = torch.autograd.grad(L_cc, bb, retain_graph=True, allow_unused=True)
                g_cc = [x if x is not None else torch.zeros_like(p) for x, p in zip(g_cc, bb)]
                acc_cc += _cos(g_cc, g_py); n_cc += 1
        if use_js:
            js = (languages == 2)
            if js.any():
                L_js = F.binary_cross_entropy_with_logits(
                    vul_js_logits[js], labels[js], pos_weight=pos_w)
                g_js = torch.autograd.grad(L_js, bb, retain_graph=True, allow_unused=True)
                g_js = [x if x is not None else torch.zeros_like(p) for x, p in zip(g_js, bb)]
                acc_js += _cos(g_js, g_py); n_js += 1
        del g_py
    if not was_training:
        model.eval()
    cos_cc = (acc_cc / n_cc) if n_cc else None
    cos_js = (acc_js / n_js) if n_js else None
    return cos_cc, cos_js

def train_v2(model, lang_classifier, device, optimizer, vul_criterion,
             group_criterion, lang_criterion, logger, args, train_loader,
             val_loader=None):
    model.to(device)
    if lang_classifier is not None:
        lang_classifier.to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "v2_multitask_best.pth")
    best_f1 = -1.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    swa_state, swa_n = None, 0

    dtjs_ema = {'cc': 1.0, 'js': 1.0}
    swa_path = os.path.join(args.output_dir, "v2_multitask_swa.pth")

    scheduler = None
    if getattr(args, 'lr_warmup_ratio', 0.0) > 0:
        total_steps = args.epochs * max(1, len(train_loader))
        n_warm = int(args.lr_warmup_ratio * total_steps)
        sched_kind = getattr(args, 'lr_schedule', 'constant')
        if sched_kind == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=n_warm, num_training_steps=total_steps)
        elif sched_kind == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=n_warm, num_training_steps=total_steps)
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=n_warm)
        logger.info(f"LR warmup: {n_warm}/{total_steps} steps then {sched_kind}")

    n_bumps = 0
    max_bumps = 2

    use_dann = args.lambda_dann > 0 and lang_classifier is not None

    use_proto = getattr(args, 'lambda_proto', 0.0) > 0
    proto = {}
    proto_m = 0.9

    logger.info(
        f"V2 MultiTask | epochs={args.epochs} | batch={args.batch_size} | "
        f"lr={args.learning_rate} | lambda_group={args.lambda_group} | "
        f"lambda_contrastive={args.lambda_contrastive} | "
        f"lambda_dann={args.lambda_dann}"
    )

    for epoch in range(args.epochs):
        model.train()
        if lang_classifier is not None:
            lang_classifier.train()
        total_loss = 0.0
        total_vul = 0.0
        total_group = 0.0
        total_contrastive = 0.0
        total_proto = 0.0
        total_safe = 0.0
        total_guard = 0.0
        total_margin = 0.0
        total_dann = 0.0
        total_dann_acc = 0.0
        n_pc_conflict = 0
        n_pc_batches = 0
        pc_pair_conflict = {}
        pc_pair_total = {}
        n_dann_batches = 0

        if use_dann:
            warm = args.dann_warmup_epochs
            if epoch < warm:
                lambda_grl = 0.0
            else:
                progress = (epoch - warm) / max(1, args.epochs - warm - 1)
                lambda_grl = dann_lambda_schedule(progress, max_lambda=1.0)
        else:
            lambda_grl = 0.0

        dtjs_mult = {'cc': 1.0, 'js': 1.0}
        if getattr(args, 'dtjs', False):
            cc_on = getattr(args, 'ccpp_head', False) and epoch >= args.ccpp_start_epoch
            js_on = getattr(args, 'js_head', False) and epoch >= args.js_start_epoch
            if cc_on or js_on:
                cos_cc, cos_js = _dtjs_probe(
                    model, train_loader, device, args, vul_criterion, group_criterion,
                    use_ccpp=cc_on, use_js=js_on, n_batches=args.dtjs_probe_batches)
                gmin, beta = args.dtjs_gamma_min, args.dtjs_ema
                if cos_cc is not None:
                    m = min(max(1.0 + cos_cc, gmin), 1.0)
                    dtjs_ema['cc'] = beta * dtjs_ema['cc'] + (1 - beta) * m
                if cos_js is not None:
                    m = min(max(1.0 + cos_js, gmin), 1.0)
                    dtjs_ema['js'] = beta * dtjs_ema['js'] + (1 - beta) * m
                dtjs_mult['cc'] = dtjs_ema['cc'] if cc_on else 1.0
                dtjs_mult['js'] = dtjs_ema['js'] if js_on else 1.0
                logger.info(
                    f"[DTJS epoch {epoch+1}] cos_cc={cos_cc if cos_cc is None else round(cos_cc,3)} "
                    f"cos_js={cos_js if cos_js is None else round(cos_js,3)} "
                    f"-> mult_cc={dtjs_mult['cc']:.3f} mult_js={dtjs_mult['js']:.3f}")

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1).float()
            group_labels = batch['group_label'].to(device)
            languages = batch.get('language')
            languages = languages.to(device) if languages is not None else None
            cwe_ids = batch.get('cwe_id')
            cwe_ids = cwe_ids.to(device) if cwe_ids is not None else None
            p_refs = batch.get('p_ref')
            p_refs = p_refs.to(device) if p_refs is not None else None

            if epoch == 0 and step == 0 and languages is not None:
                uniq, cnt = languages.unique(return_counts=True)
                logger.info(f"[audit] batch language dist: {dict(zip(uniq.tolist(), cnt.tolist()))}")

            optimizer.zero_grad()
            need_repr = args.lambda_contrastive > 0 or use_dann or use_proto
            use_ccpp = getattr(args, 'ccpp_head', False)
            use_js = getattr(args, 'js_head', False)
            vul_ccpp_logits = None
            vul_js_logits = None
            if use_ccpp:

                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            languages=languages, return_ccpp=True, return_repr=need_repr)
                if use_js:
                    if need_repr:
                        vul_logits, group_logits, vul_ccpp_logits, vul_js_logits, reprs = out
                    else:
                        vul_logits, group_logits, vul_ccpp_logits, vul_js_logits = out; reprs = None
                elif need_repr:
                    vul_logits, group_logits, vul_ccpp_logits, reprs = out
                else:
                    vul_logits, group_logits, vul_ccpp_logits = out; reprs = None
            elif need_repr:
                vul_logits, group_logits, reprs = model(
                    input_ids=input_ids, attention_mask=attention_mask, return_repr=True)
            else:
                vul_logits, group_logits = model(input_ids=input_ids,
                                                 attention_mask=attention_mask)
                reprs = None

            py_only = (use_dann or use_ccpp or use_js) and languages is not None

            sample_w = batch.get('sample_weight')
            sample_w = sample_w.to(device).unsqueeze(1) if sample_w is not None else None

            def _vul_bce(logits, targets, w):
                gamma = getattr(args, 'vul_focal_gamma', 0.0)
                if gamma > 0:

                    bce = F.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=vul_criterion.pos_weight, reduction='none')
                    p = torch.sigmoid(logits)
                    pt = p * targets + (1 - p) * (1 - targets)
                    focal = (1 - pt).clamp(min=1e-6) ** gamma * bce
                    if w is not None:
                        focal = focal * w
                    return focal.mean()
                if w is None:
                    return vul_criterion(logits, targets)
                return F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=vul_criterion.pos_weight, weight=w)

            if py_only:
                py_mask = (languages == 0)
                if py_mask.any():
                    vul_loss = _vul_bce(vul_logits[py_mask], labels[py_mask],
                                        sample_w[py_mask] if sample_w is not None else None)
                else:
                    vul_loss = vul_logits.new_tensor(0.0)
            else:
                vul_loss = _vul_bce(vul_logits, labels, sample_w)

            if py_only:
                gl_mask = (group_labels >= 0) & (languages == 0)
            else:
                gl_mask = (group_labels >= 0)
            if getattr(args, 'group_on_vul_only', False):

                gl_mask = gl_mask & (labels.squeeze(-1) == 1)
            if gl_mask.any():
                group_loss = group_criterion(group_logits[gl_mask], group_labels[gl_mask])
            else:
                group_loss = group_logits.new_tensor(0.0)

            ccpp_loss = None
            if use_ccpp and languages is not None and vul_ccpp_logits is not None:

                cc_mask = (languages == 1) if use_js else (languages != 0)
                if cc_mask.any():

                    cc_w = sample_w[cc_mask] if sample_w is not None else None
                    ccpp_loss = F.binary_cross_entropy_with_logits(
                        vul_ccpp_logits[cc_mask], labels[cc_mask],
                        pos_weight=vul_criterion.pos_weight, weight=cc_w)
                else:
                    ccpp_loss = vul_logits.new_tensor(0.0)

            js_loss = None
            if use_js and languages is not None and vul_js_logits is not None:
                js_mask = (languages == 2)
                if js_mask.any():
                    js_w = sample_w[js_mask] if sample_w is not None else None
                    js_loss = F.binary_cross_entropy_with_logits(
                        vul_js_logits[js_mask], labels[js_mask],
                        pos_weight=vul_criterion.pos_weight, weight=js_w)
                else:
                    js_loss = vul_logits.new_tensor(0.0)
            if epoch == 0 and step == 0:
                logger.info(f"[audit] group_loss samples: {gl_mask.sum().item()}/{len(gl_mask)} (Python+valid)")
                if ccpp_loss is not None:
                    cc_n = int((languages == 1).sum()) if use_js else int((languages != 0).sum())
                    logger.info(f"[audit] ccpp_loss samples: {cc_n}/{len(languages)} (C/C++)")
                if js_loss is not None:
                    logger.info(f"[audit] js_loss samples: {int((languages == 2).sum())}/{len(languages)} (JS)")

            ccpp_active = (ccpp_loss is not None) and (epoch >= args.ccpp_start_epoch)

            ramp = ccpp_ramp_factor(epoch, args.ccpp_start_epoch,
                                    args.ccpp_ramp_epochs, args.ccpp_ramp_mode)
            eff_ccpp = args.ccpp_loss_scale * ramp
            js_active = (js_loss is not None) and (epoch >= args.js_start_epoch)

            eff_js = args.js_loss_scale
            if getattr(args, 'dtjs', False):
                eff_ccpp = args.ccpp_loss_scale * dtjs_mult['cc']
                eff_js = args.js_loss_scale * dtjs_mult['js']
            if args.uncertainty:

                lv_v, lv_g = model.log_var_vul, model.log_var_group
                loss = (torch.exp(-lv_v) * vul_loss + lv_v
                        + torch.exp(-lv_g) * group_loss + lv_g).squeeze()
                if ccpp_active:
                    lv_c = model.log_var_ccpp
                    loss = loss + eff_ccpp * (torch.exp(-lv_c) * ccpp_loss + lv_c).squeeze()
                if js_active:
                    lv_j = model.log_var_js
                    loss = loss + eff_js * (torch.exp(-lv_j) * js_loss + lv_j).squeeze()
            else:
                loss = vul_loss + args.lambda_group * group_loss
                if ccpp_active:
                    loss = loss + eff_ccpp * args.lambda_ccpp * ccpp_loss
                if js_active:
                    loss = loss + eff_js * args.lambda_ccpp * js_loss
            if args.lambda_contrastive > 0:
                contrastive_loss = supervised_contrastive_loss(
                    reprs, group_labels, languages=languages,
                    temperature=args.contrastive_temperature,
                )
                loss = loss + args.lambda_contrastive * contrastive_loss
                total_contrastive += contrastive_loss.item()

            if use_proto and reprs is not None and languages is not None and group_labels is not None:
                lab = labels.squeeze(-1)
                vmask = (lab == 1) & (group_labels >= 0)
                p_loss = reprs.new_tensor(0.0); n_pair = 0
                for i in range(reprs.size(0)):
                    if not bool(vmask[i]):
                        continue
                    L = int(languages[i]); g = int(group_labels[i])
                    other = 0 if L != 0 else 1
                    key = (other, g)
                    if key in proto:
                        p_loss = p_loss + (1 - F.cosine_similarity(
                            reprs[i:i+1], proto[key].unsqueeze(0).detach(), dim=1)).squeeze()
                        n_pair += 1
                if n_pair > 0:
                    p_loss = p_loss / n_pair
                    loss = loss + args.lambda_proto * p_loss
                    total_proto += float(p_loss.item())

                with torch.no_grad():
                    for L in (0, 1):
                        for g in (0, 1):
                            sel = (languages == L) & (group_labels == g) & vmask
                            if bool(sel.any()):
                                m = reprs[sel].mean(0).detach()
                                k = (L, g)
                                proto[k] = m if k not in proto else proto_m * proto[k] + (1 - proto_m) * m

            if getattr(args, 'lambda_safe', 0.0) > 0 and cwe_ids is not None:
                lab = labels.squeeze(-1)
                rare = (cwe_ids == 0) | (cwe_ids == 2)
                py = (languages == 0) if languages is not None else torch.ones_like(lab, dtype=torch.bool)
                safe_rare = rare & (lab == 0) & py
                if bool(safe_rare.any()):
                    p_sr = torch.sigmoid(vul_logits.squeeze(-1))[safe_rare]
                    L_safe = (torch.relu(p_sr - 0.5) ** 2).mean()
                    loss = loss + args.lambda_safe * L_safe
                    total_safe += float(L_safe.item())

            if getattr(args, 'lambda_guard', 0.0) > 0 and cwe_ids is not None and p_refs is not None:
                lab = labels.squeeze(-1)
                rare = (cwe_ids == 0) | (cwe_ids == 2)
                py = (languages == 0) if languages is not None else torch.ones_like(lab, dtype=torch.bool)
                gmask = rare & (lab == 0) & py & (p_refs >= 0)
                if bool(gmask.any()):
                    p_g = torch.sigmoid(vul_logits.squeeze(-1))[gmask]
                    L_guard = (torch.relu(p_g - p_refs[gmask] - 0.05) ** 2).mean()
                    loss = loss + args.lambda_guard * L_guard
                    total_guard += float(L_guard.item())

            if getattr(args, 'boundary_margin', False):
                z = vul_logits.squeeze(-1)
                lab = labels.squeeze(-1)
                py = (languages == 0) if languages is not None else torch.ones_like(lab, dtype=torch.bool)
                if bool(py.any()):
                    zp, yp = z[py], lab[py]

                    Lm = (yp * F.softplus(args.margin_pos - zp)
                          + (1.0 - yp) * F.softplus(zp + args.margin_neg))
                    if args.rare_safe_beta > 0 and cwe_ids is not None:
                        rare = (cwe_ids == 0) | (cwe_ids == 2)
                        w = 1.0 + args.rare_safe_beta * ((yp == 0) & rare[py]).float()
                        L_margin = (w * Lm).mean()
                    else:
                        L_margin = Lm.mean()
                    loss = loss + args.lambda_margin * L_margin
                    total_margin += float(L_margin.item())

            if use_dann and languages is not None:
                ent_w = None
                if args.cdan:
                    pv = torch.sigmoid(vul_logits)
                    class_probs = torch.cat([pv, 1 - pv], dim=1)

                    ccpp_mask = (languages != 0).unsqueeze(1)
                    uniform = torch.full_like(class_probs, 0.5)
                    class_probs = torch.where(ccpp_mask, uniform, class_probs)
                    lang_logits = lang_classifier(reprs, class_probs.detach(), lambda_grl=lambda_grl)
                    if args.cdan_entropy:
                        ent_w = entropy_weight(class_probs)
                else:
                    lang_logits = lang_classifier(reprs, lambda_grl=lambda_grl)
                if ent_w is not None:

                    per = F.cross_entropy(lang_logits, languages, reduction='none')
                    dann_loss = (ent_w * per).sum() / ent_w.sum()
                else:
                    dann_loss = lang_criterion(lang_logits, languages)
                loss = loss + args.lambda_dann * dann_loss
                total_dann += dann_loss.item()
                with torch.no_grad():
                    lang_pred = lang_logits.argmax(dim=-1)
                    total_dann_acc += (lang_pred == languages).float().mean().item()
                n_dann_batches += 1
            use_pcgrad = (getattr(args, 'pcgrad', False) and (ccpp_active or js_active)
                          and not use_dann)
            if use_pcgrad:

                if args.uncertainty:
                    L_py = (torch.exp(-lv_v) * vul_loss + lv_v
                            + torch.exp(-lv_g) * group_loss + lv_g).squeeze()
                else:
                    L_py = vul_loss + args.lambda_group * group_loss
                tasks = [('py', L_py)]
                if ccpp_active:
                    L_cc = eff_ccpp * ((torch.exp(-lv_c) * ccpp_loss + lv_c).squeeze() if args.uncertainty
                                       else args.lambda_ccpp * ccpp_loss)
                    tasks.append(('cc', L_cc))
                if js_active:
                    L_js = eff_js * ((torch.exp(-lv_j) * js_loss + lv_j).squeeze() if args.uncertainty
                                     else args.lambda_ccpp * js_loss)
                    tasks.append(('js', L_js))

                bb = [p for p in model.backbone.parameters() if p.requires_grad]
                names = [n for n, _ in tasks]
                raw = {}
                for n, L in tasks:
                    g = torch.autograd.grad(L, bb, retain_graph=True, allow_unused=True)
                    raw[n] = [x if x is not None else torch.zeros_like(p) for x, p in zip(g, bb)]
                pc = {n: [g.clone() for g in raw[n]] for n in names}

                for ni in names:
                    for nj in names:
                        if nj == ni:
                            continue
                        dot = sum((a * b).sum() for a, b in zip(pc[ni], raw[nj]))
                        pair = tuple(sorted((ni, nj)))
                        pc_pair_total[pair] = pc_pair_total.get(pair, 0) + 1
                        if dot.item() < 0:
                            nj_sq = sum((b * b).sum() for b in raw[nj]).clamp_min(1e-12)
                            coef = dot / nj_sq
                            pc[ni] = [a - coef * b for a, b in zip(pc[ni], raw[nj])]
                            pc_pair_conflict[pair] = pc_pair_conflict.get(pair, 0) + 1
                fg = [sum(pc[n][k] for n in names) for k in range(len(bb))]
                if any(pc_pair_conflict.get(tuple(sorted((ni, nj))), 0) for ni in names for nj in names if ni != nj):
                    n_pc_conflict += 1
                n_pc_batches += 1
                del raw, pc

                total_L = sum(L for _, L in tasks)
                total_L.backward()
                for p, g in zip(bb, fg):
                    p.grad = g.detach()
            else:
                loss.backward()
            if args.grad_clip > 0:
                clip_params = list(model.parameters())
                if lang_classifier is not None:
                    clip_params += list(lang_classifier.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            total_vul += vul_loss.item()
            total_group += group_loss.item()

        n_batches = len(train_loader)
        msg = (
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/n_batches:.4f} "
            f"| Vul BCE: {total_vul/n_batches:.4f} | Group CE: {total_group/n_batches:.4f}"
        )
        if args.lambda_contrastive > 0:
            msg += f" | Contrastive: {total_contrastive/n_batches:.4f}"
        if use_proto:
            msg += f" | Proto: {total_proto/n_batches:.4f}"
        if getattr(args, 'lambda_safe', 0.0) > 0:
            msg += f" | SafeAnchor: {total_safe/n_batches:.4f}"
        if getattr(args, 'lambda_guard', 0.0) > 0:
            msg += f" | Guard: {total_guard/n_batches:.4f}"
        if getattr(args, 'boundary_margin', False):
            msg += f" | Margin: {total_margin/n_batches:.4f}"
        if use_dann and n_dann_batches > 0:
            msg += (
                f" | DANN CE: {total_dann/n_dann_batches:.4f}"
                f" | DANN Acc: {total_dann_acc/n_dann_batches:.4f}"
                f" | lambda_grl={lambda_grl:.3f}"
            )
        if n_pc_batches > 0:
            msg += f" | PCGrad conflict={n_pc_conflict}/{n_pc_batches} ({100*n_pc_conflict/n_pc_batches:.0f}%)"
            if len(pc_pair_total) > 1:
                pair_str = " ".join(
                    f"{a}-{b}={pc_pair_conflict.get((a, b), 0)}/{n}({100*pc_pair_conflict.get((a, b), 0)/n:.0f}%)"
                    for (a, b), n in sorted(pc_pair_total.items())
                )
                msg += f" | pairs: {pair_str}"
        if args.uncertainty:

            wv = float(torch.exp(-model.log_var_vul).item())
            wg = float(torch.exp(-model.log_var_group).item())
            uw = f" | w_vul={wv:.3f} w_group={wg:.3f}"
            if hasattr(model, 'log_var_ccpp'):
                _ramp = ccpp_ramp_factor(epoch, args.ccpp_start_epoch,
                                         args.ccpp_ramp_epochs, args.ccpp_ramp_mode)
                uw += (f" w_ccpp={float(torch.exp(-model.log_var_ccpp).item()):.3f}"
                       f" eff_ccpp={args.ccpp_loss_scale * _ramp:.3f}")
            if hasattr(model, 'log_var_js'):
                uw += f" w_js={float(torch.exp(-model.log_var_js).item()):.3f}"
            gcr = (total_group / max(total_vul, 1e-9))
            msg += uw + f" | GroupCE/VulBCE={gcr:.3f}"
        logger.info(msg)

        if val_loader is None:
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
            continue

        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        all_group_preds, all_group_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].cpu().numpy().astype(int)
                group_labels = batch['group_label'].cpu().numpy().astype(int)

                vul_logits, group_logits = model(input_ids=input_ids,
                                                 attention_mask=attention_mask)
                probs = torch.sigmoid(vul_logits).squeeze(1).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                group_preds = torch.argmax(group_logits, dim=1).cpu().numpy().astype(int)

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_group_preds.extend(group_preds.tolist())
                all_group_labels.extend(group_labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        p = precision_score(all_labels, all_preds, zero_division=0)
        r = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        sel = getattr(args, 'select_metric', 'vul')
        sel_f1 = macro if sel == 'macro' else f1
        group_valid = [i for i, y in enumerate(all_group_labels) if y >= 0]
        if group_valid:
            group_acc = accuracy_score(
                [all_group_labels[i] for i in group_valid],
                [all_group_preds[i] for i in group_valid],
            )
            group_f1 = f1_score(
                [all_group_labels[i] for i in group_valid],
                [all_group_preds[i] for i in group_valid],
                average='macro', zero_division=0,
            )
        else:
            group_acc = 0.0
            group_f1 = 0.0
        logger.info(
            f"[Val V2 - Epoch {epoch+1}] Acc={acc:.4f} P={p:.4f} R={r:.4f} "
            f"F1={f1:.4f} MacroF1={macro:.4f} (select={sel}) | "
            f"GroupAcc={group_acc:.4f} GroupMacroF1={group_f1:.4f}"
        )

        if getattr(args, 'swa', False) and (epoch + 1) >= args.swa_start_epoch:
            sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state, swa_n = sd, 1
            else:
                swa_n += 1
                for k in swa_state:
                    swa_state[k] += (sd[k] - swa_state[k]) / swa_n
            logger.info(f"[SWA] collected member {swa_n} (epoch {epoch+1})")

        if sel_f1 > best_f1:
            best_f1 = sel_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best V2! Val {sel}-F1: {best_f1:.4f} (Epoch {best_epoch})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        val_std = float(np.std(all_probs)) if all_probs else 1.0
        if (getattr(args, 'stuck_bump', False) and n_bumps < max_bumps
                and (epoch + 1) >= args.stuck_patience and val_std < args.stuck_std
                and best_epoch <= 1):
            n_bumps += 1
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_bump
            if scheduler is not None:
                scheduler.base_lrs = [b * args.lr_bump for b in scheduler.base_lrs]
            best_f1 = -1.0
            patience_counter = 0
            new_lr = optimizer.param_groups[0]['lr']
            logger.warning(f"[STUCK] epoch {epoch+1}: val std={val_std:.3f}<{args.stuck_std}, "
                           f"best_epoch={best_epoch} → bump LR ×{args.lr_bump} (now {new_lr:.1e}, "
                           f"bump {n_bumps}/{max_bumps})")

    if getattr(args, 'swa', False):
        if swa_state is not None:
            torch.save(swa_state, swa_path)
            logger.info(f"[SWA] saved average of {swa_n} member(s) -> {swa_path}"
                        + (" (WARNING: <2 members, ~single checkpoint)" if swa_n < 2 else ""))
        else:

            import shutil
            shutil.copyfile(best_model_path, swa_path)
            logger.warning(f"[SWA] no members collected (stopped before epoch "
                           f"{args.swa_start_epoch}); copied best ckpt to {swa_path}")
    logger.info(f"V2 training finished. Best epoch={best_epoch}: {best_model_path}")

def get_args():
    parser = argparse.ArgumentParser(description="Train V2 single-backbone multi-task model")
    parser.add_argument('--train_files', nargs='+', required=True)
    parser.add_argument('--val_files', nargs='+', default=[])
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--lambda_group', type=float, default=0.3)
    parser.add_argument('--lambda_contrastive', type=float, default=0.0)
    parser.add_argument('--contrastive_temperature', type=float, default=0.2)
    parser.add_argument('--lambda_dann', type=float, default=0.0,
                        help='DANN domain confusion weight; >0 enables backbone language '
                             'adversarial loss on CCPP-augmented training set.')
    parser.add_argument('--dann_hidden_dim', type=int, default=128)
    parser.add_argument('--warm_start_ckpt', type=str, default='',
                        help='Optional V2 base checkpoint to warm-start backbone + heads. '
                             'Useful with DANN: preserves lucky local optimum from V2 base '
                             'while gently adding cross-language signal at lower lr.')
    parser.add_argument('--allow_unmapped_group', action='store_true',
                        help='Set unmapped CWE_ID group labels to -100 for CCPP/OOD safe samples.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='model/v2_multitask')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pos_weight_cap', type=float, default=3.0)
    parser.add_argument('--uncertainty', action='store_true',
                        help='Kendall 2018 uncertainty weighting: learn per-task '
                             'loss weights (log-variance) instead of fixed lambda_group.')
    parser.add_argument('--cdan', action='store_true',
                        help='Conditional DANN: condition domain discriminator on '
                             'class prediction (outer product). Stronger than vanilla DANN.')
    parser.add_argument('--cdan_entropy', action='store_true',
                        help='CDAN+E: weight each sample domain loss by w(H)=1+exp(-H) '
                             '(entropy conditioning, Long 2018). Focuses alignment on '
                             'confident samples; usually helps F1 AND calibration.')
    parser.add_argument('--cdan_random_dim', type=int, default=None,
                        help='CDAN randomized multilinear: project f⊗g to this dim with '
                             'fixed random matrices (e.g. 1024) to shrink the discriminator.')
    parser.add_argument('--select_metric', choices=['vul', 'macro'], default='vul',
                        help='Checkpoint selection metric on val: vul-F1 (class 1) or macro-F1 (balanced).')
    parser.add_argument('--cwe_head', action='store_true',
                        help='V2.1: 4-class CWE head (022/078/079/089) instead of '
                             '2-class group head — finer auxiliary signal.')
    parser.add_argument('--dann_warmup_epochs', type=int, default=0,
                        help='Epochs with lambda_grl=0 before ramping (DANN stability).')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Max grad norm for clipping (0=off; 1.0 stabilizes DANN).')
    parser.add_argument('--cwe_reweight', action='store_true',
                        help='CWE-balanced per-sample loss weighting (upweight rare '
                             'CWEs 022/079) on Python samples, by original_CWE_ID.')
    parser.add_argument('--cwe_reweight_mode', choices=['inv', 'invsqrt'],
                        default='invsqrt')
    parser.add_argument('--cwe_weight_cap', type=float, default=3.0)
    parser.add_argument('--vul_focal_gamma', type=float, default=0.0,
                        help='Binary focal loss gamma for vul BCE (0=off). >0 focuses '
                             'on hard examples; combine with --cwe_reweight for rare CWEs.')
    parser.add_argument('--moe', action='store_true',
                        help='Use V3 SoftMoE (group experts + shared expert, group head as '
                             'soft router) instead of plain V2 single vul head.')
    parser.add_argument('--ccpp_head', action='store_true',
                        help='Cross-language: separate vul_head_ccpp consumes C/C++ BCE so the '
                             'backbone learns from C/C++ without adversarial distortion. '
                             'Python heads (vul/group) trained Python-only. Inference = V2.')
    parser.add_argument('--lambda_ccpp', type=float, default=0.5,
                        help='Weight for CCPP-head loss when NOT using --uncertainty.')
    parser.add_argument('--ccpp_loss_scale', type=float, default=1.0,
                        help='Scale the whole CCPP loss contribution (light auxiliary transfer, '
                             'ccpp.fix.md A). <1 lowers CCPP power so Python heads dominate.')
    parser.add_argument('--ccpp_start_epoch', type=int, default=0,
                        help='Epoch at which CCPP enters the loss; before it CCPP is dropped '
                             'entirely (let Python reach its optimum first).')
    parser.add_argument('--ccpp_ramp_epochs', type=int, default=0,
                        help='Ramp CCPP weight from onset over N epochs (fix.dl5.md). '
                             '0 = step onset (current dl5 behavior).')
    parser.add_argument('--ccpp_ramp_mode', choices=['none', 'linear', 'cosine'], default='none',
                        help='Shape of the CCPP weight ramp after ccpp_start_epoch.')
    parser.add_argument('--js_head', action='store_true',
                        help='3-adapter model: add a SECOND foreign stream (JS, lang id 2) with its '
                             'own adapter_js + vul_head_js, parallel to CCPP. Requires --lang_adapter '
                             '--ccpp_head. Both C/C++ and JS are auxiliary to Python.')
    parser.add_argument('--js_loss_scale', type=float, default=1.0,
                        help='Scale the whole JS-head loss contribution (mirrors --ccpp_loss_scale).')
    parser.add_argument('--js_start_epoch', type=int, default=0,
                        help='Epoch at which the JS aux loss enters; before it JS is dropped entirely '
                             '(so log_var_js does not drift).')

    parser.add_argument('--dtjs', action='store_true',
                        help='Enable Dynamic Task Scheduling: per-epoch gradient-alignment '
                             'weighting of CCPP and JS support losses (attacks 3-source '
                             'backbone gradient conflict).')
    parser.add_argument('--dtjs_probe_batches', type=int, default=8,
                        help='Number of training batches probed per epoch to estimate '
                             'support-vs-target gradient cosine (DTJS).')
    parser.add_argument('--dtjs_gamma_min', type=float, default=0.1,
                        help='Floor on the DTJS per-source multiplier (a fully conflicting '
                             'source is damped to this fraction of its base scale).')
    parser.add_argument('--dtjs_ema', type=float, default=0.5,
                        help='EMA smoothing for the DTJS multiplier across epochs '
                             '(0=snap to current probe, ->1=very sticky).')

    parser.add_argument('--boundary_margin', action='store_true',
                        help='Enable logit-space boundary-margin loss on Python samples '
                             '(NOT in Kendall; sharpens the p=0.5 decision boundary).')
    parser.add_argument('--margin_pos', type=float, default=0.15,
                        help='Vul samples pushed to logit z > +margin_pos.')
    parser.add_argument('--margin_neg', type=float, default=0.25,
                        help='Safe samples pushed to logit z < -margin_neg.')
    parser.add_argument('--lambda_margin', type=float, default=0.05,
                        help='Weight of the boundary-margin loss.')
    parser.add_argument('--rare_safe_beta', type=float, default=0.25,
                        help='Extra margin weight on CWE-022/079 safe samples (cwe_id 0/2).')

    parser.add_argument('--swa', action='store_true',
                        help='Average weights of epochs >= swa_start_epoch into '
                             'v2_multitask_swa.pth (constant-LR tail = SWA-valid).')
    parser.add_argument('--swa_start_epoch', type=int, default=8,
                        help='1-indexed epoch to start SWA collection (post-CCPP-onset).')
    parser.add_argument('--lang_adapter', action='store_true',
                        help='Direction E: per-language bottleneck adapters after pooler '
                             '(needs --ccpp_head). Python head reads Python-adapted repr.')
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help='Bottleneck dim for --lang_adapter (default 64).')
    parser.add_argument('--adapter_lr', type=float, default=0.0,
                        help='Dedicated lr for adapter params (0=use --learning_rate). Higher '
                             '(e.g. 1e-4) sharpens adapters fast so Python head gets confident '
                             'before the val peak (fix B for early-peak folds).')
    parser.add_argument('--lambda_proto', type=float, default=0.0,
                        help='BIG FIX #4: weight for CWE-prototype contrastive (class-conditional '
                             'cross-language alignment via EMA per-(lang,group) prototypes). 0=off.')
    parser.add_argument('--lambda_safe', type=float, default=0.0,
                        help='Simplified BIG FIX #7: safe-precision anchor — penalize vul-prob>0.5 '
                             'on Python SAFE rare-CWE {022,079} to recover CCPP safe-FP leak. 0=off.')
    parser.add_argument('--lambda_guard', type=float, default=0.0,
                        help='CEGA guard: penalize vul-prob exceeding Python reference p_ref '
                             '(v2_warm) on safe rare-CWE {022,079}. Needs p_ref field in data. 0=off.')
    parser.add_argument('--pcgrad', action='store_true',
                        help='PCGrad (Yu 2020): project out conflicting gradient components '
                             'on the shared backbone between Python and every currently-active '
                             'foreign task (CCPP and/or JS). Needs --ccpp_head and/or --js_head; '
                             'incompatible with DANN.')
    parser.add_argument('--uncertainty_lr', type=float, default=2e-5,
                        help='Dedicated lr for log_var_* scalar params (Kendall). Default 2e-5 '
                             '= original behavior (weights ~inert). Override (e.g. 1e-4..1e-3) to '
                             'let weighting adapt; 1e-2 makes w_group explode on the easy aux task.')
    parser.add_argument('--group_on_vul_only', action='store_true',
                        help='Compute group_loss only on VULNERABLE Python samples (group/CWE '
                             'label is only meaningful for vulnerable code → harder, less-trivial aux).')

    parser.add_argument('--lr_warmup_ratio', type=float, default=0.0,
                        help='(A) Linear LR warmup over this fraction of total steps then constant. '
                             '0=off. ~0.1 stabilizes BERT fine-tuning (Mosbach 2021).')
    parser.add_argument('--lr_schedule', choices=['constant', 'cosine', 'linear'],
                        default='constant',
                        help='(C1) Schedule AFTER warmup. constant=keep peak LR (default, old '
                             'behavior); cosine/linear decay to settle late training (ccpp.fix.md C).')
    parser.add_argument('--stuck_bump', action='store_true',
                        help='(B) Detect dead-start (val pred std collapses) mid-training and bump LR.')
    parser.add_argument('--stuck_patience', type=int, default=5,
                        help='(B) Epochs before stuck-detection activates (>warmup so it does '
                             'not misfire during the LR ramp).')
    parser.add_argument('--stuck_std', type=float, default=0.05,
                        help='(B) Val vul_prob std below this = collapsed predictions = stuck.')
    parser.add_argument('--lr_bump', type=float, default=2.0,
                        help='(B) Multiply LR by this when stuck (gentle ×2; ×5 blew past '
                             "CodeBERT's fine-tuning LR ceiling and diverged). Max 2 bumps + "
                             'only fires when best_epoch<=1 (true dead-start).')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs('log', exist_ok=True)
    log_file = os.path.join('log', f'train_v2_multitask_seed{args.seed}.log')
    setup_logging(log_file=log_file, reset_file=True)
    logger = get_logger(__name__)

    set_seed(args.seed)
    logger.info(f"Seed fixed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    backbone = AutoModel.from_pretrained(args.pretrained_model)
    n_groups = 4 if args.cwe_head else 2
    if getattr(args, 'moe', False):

        model = MulVulSoftMoEV3(backbone, num_groups=n_groups, dropout=args.dropout)
        logger.info(f"V3 SoftMoE enabled: {n_groups} group experts + shared expert (learnable alpha)")
    elif getattr(args, 'lang_adapter', False):

        if getattr(args, 'js_head', False):

            model = MulVulV2With3LangAdapter(backbone, num_groups=n_groups, dropout=args.dropout,
                                             adapter_dim=args.adapter_dim)
            logger.info(f"3-language adapter enabled: Python/CCPP/JS bottlenecks (dim={args.adapter_dim}) "
                        "+ separate vul_head_ccpp & vul_head_js; Python head reads Python-adapted repr")
        else:
            model = MulVulV2WithLangAdapter(backbone, num_groups=n_groups, dropout=args.dropout,
                                            adapter_dim=args.adapter_dim)
            logger.info(f"Language adapter enabled: per-lang bottleneck (dim={args.adapter_dim}) "
                        "+ separate vul_head_ccpp; Python head reads Python-adapted repr")
    elif getattr(args, 'ccpp_head', False):

        model = MulVulV2WithCCPP(backbone, num_groups=n_groups, dropout=args.dropout)
        logger.info("CCPP auxiliary head enabled: separate vul_head_ccpp for C/C++ (no adversarial)")
    else:
        model = MulVulMultiTaskV2(backbone, num_groups=n_groups, dropout=args.dropout)
    if args.uncertainty:

        model.register_parameter('log_var_vul', nn.Parameter(torch.zeros(1)))
        model.register_parameter('log_var_group', nn.Parameter(torch.zeros(1)))
        logger.info("Uncertainty weighting (Kendall) enabled: learning per-task log-var")
        if getattr(args, 'ccpp_head', False):
            model.register_parameter('log_var_ccpp', nn.Parameter(torch.zeros(1)))
            logger.info("  + log_var_ccpp registered (3-task uncertainty: vul/group/ccpp)")
        if getattr(args, 'js_head', False):
            model.register_parameter('log_var_js', nn.Parameter(torch.zeros(1)))
            logger.info("  + log_var_js registered (4-task uncertainty: vul/group/ccpp/js)")

    if args.warm_start_ckpt:
        if not os.path.exists(args.warm_start_ckpt):
            raise FileNotFoundError(f"warm_start_ckpt not found: {args.warm_start_ckpt}")
        v2_state = torch.load(args.warm_start_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(v2_state, strict=False)
        logger.info(
            f"Warm-started V2 from {args.warm_start_ckpt} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        for m in missing:
            logger.info(f"  missing: {m}")
        for u in unexpected:
            logger.info(f"  unexpected: {u}")

    lang_classifier = None
    if args.lambda_dann > 0:
        hidden = model.backbone.config.hidden_size
        if args.cdan:
            lang_classifier = ConditionalLanguageClassifier(
                hidden_size=hidden, num_classes=2, num_languages=2,
                hidden_dim=args.dann_hidden_dim, dropout=args.dropout,
                random_dim=args.cdan_random_dim,
            )
        else:
            lang_classifier = LanguageClassifier(
                hidden_size=hidden, num_languages=2,
                hidden_dim=args.dann_hidden_dim, dropout=args.dropout,
            )
        logger.info(
            f"DANN enabled: lambda_dann={args.lambda_dann}, hidden_dim={args.dann_hidden_dim}, "
            f"cdan={args.cdan} cdan_entropy={args.cdan_entropy} "
            f"cdan_random_dim={args.cdan_random_dim}"
        )

    if args.uncertainty:

        lv_names = {'log_var_vul', 'log_var_group', 'log_var_ccpp', 'log_var_js'}

        adapter_lr = getattr(args, 'adapter_lr', 0.0) or args.learning_rate
        sep_adapter = getattr(args, 'lang_adapter', False) and adapter_lr != args.learning_rate
        is_ad = lambda n: n.startswith('adapter_python') or n.startswith('adapter_ccpp') or n.startswith('adapter_js')
        lv_params = [p for n, p in model.named_parameters() if n in lv_names]
        ad_params = [p for n, p in model.named_parameters() if sep_adapter and is_ad(n)]
        other = [p for n, p in model.named_parameters()
                 if n not in lv_names and not (sep_adapter and is_ad(n))]
        if lang_classifier is not None:
            other += list(lang_classifier.parameters())
        groups = [{'params': other, 'lr': args.learning_rate},
                  {'params': lv_params, 'lr': args.uncertainty_lr}]
        if ad_params:
            groups.append({'params': ad_params, 'lr': adapter_lr})
        optimizer = AdamW(groups)
        logger.info(f"Optimizer: backbone/heads lr={args.learning_rate}, log_var lr={args.uncertainty_lr}"
                    + (f", adapter lr={adapter_lr}" if ad_params else ""))
    else:
        params = list(model.parameters())
        if lang_classifier is not None:
            params += list(lang_classifier.parameters())
        optimizer = AdamW(params, lr=args.learning_rate)

    logger.info(f"Loading training data: {args.train_files}")
    train_data = add_group_labels(load_data(args.train_files),
                                  allow_unmapped=args.allow_unmapped_group,
                                  cwe_head=args.cwe_head)
    num_neg = int((train_data['label'] == 0).sum())
    num_pos = int((train_data['label'] == 1).sum())
    raw_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    weight_ratio = min(raw_ratio, args.pos_weight_cap)
    logger.info(
        f"V2 class distribution - Safe: {num_neg} | Vul: {num_pos} | "
        f"raw pos_weight: {raw_ratio:.2f} | capped to: {weight_ratio:.2f}"
    )
    logger.info(f"V2 group distribution: {train_data['group_label'].value_counts().sort_index().to_dict()}")

    if args.cwe_reweight:
        import numpy as np
        cwe_col = 'original_CWE_ID' if 'original_CWE_ID' in train_data.columns else 'CWE_ID'
        lang_col_w = 'lang' if 'lang' in train_data.columns else (
            'language' if 'language' in train_data.columns else None)
        is_py = train_data[lang_col_w].astype(str).str.lower().str.strip().isin(
            ['python', 'py']) if lang_col_w else np.ones(len(train_data), dtype=bool)
        cwes = train_data[cwe_col].astype(str).str.lower().str.strip()
        py_counts = cwes[is_py].value_counts().to_dict()
        n_py = max(1, int(is_py.sum()))
        raw = {}
        for c, cnt in py_counts.items():
            if args.cwe_reweight_mode == 'inv':
                raw[c] = n_py / cnt
            else:
                raw[c] = (n_py / cnt) ** 0.5
        mean_raw = np.mean(list(raw.values())) if raw else 1.0
        cwe_w = {c: min(v / mean_raw, args.cwe_weight_cap) for c, v in raw.items()}
        weights = [cwe_w.get(cwes.iloc[i], 1.0) if is_py.iloc[i] else 1.0
                   for i in range(len(train_data))]
        train_data = train_data.reset_index(drop=True)
        train_data['sample_weight'] = weights
        logger.info(f"CWE reweight ({args.cwe_reweight_mode}, cap={args.cwe_weight_cap}): "
                    f"weights={ {k: round(v,2) for k,v in cwe_w.items()} }")

    pos_weight = torch.tensor([weight_ratio], device=device)
    vul_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    group_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    lang_criterion = nn.CrossEntropyLoss()
    if 'language' in train_data.columns or 'lang' in train_data.columns:
        lang_col = 'language' if 'language' in train_data.columns else 'lang'
        logger.info(f"V2 language distribution: {train_data[lang_col].value_counts().to_dict()}")

    generator = make_generator(args.seed)
    train_loader = create_data_loader(
        train_data, tokenizer, batch_size=args.batch_size, shuffle=True,
        generator=generator, worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
    )

    val_loader = None
    if args.val_files:
        logger.info(f"Loading validation data: {args.val_files}")
        val_data = add_group_labels(load_data(args.val_files),
                                    allow_unmapped=args.allow_unmapped_group,
                                    cwe_head=args.cwe_head)
        val_loader = create_data_loader(
            val_data, tokenizer, batch_size=args.batch_size, shuffle=False,
            worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
        )

    train_v2(model, lang_classifier, device, optimizer, vul_criterion,
             group_criterion, lang_criterion, logger, args, train_loader,
             val_loader)

if __name__ == "__main__":
    main()
