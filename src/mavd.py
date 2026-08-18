import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCodeBERT(nn.Module):

    def __init__(self, pretrained_model, num_labels: int = 1):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)

class MulVulMultiTaskV2(nn.Module):

    def __init__(self, pretrained_model, num_groups: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = pretrained_model
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(hidden, 1)
        self.group_head = nn.Linear(hidden, num_groups)

    def forward(self, input_ids, attention_mask, return_repr: bool = False, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        vul_logits = self.vul_head(pooled)
        group_logits = self.group_head(pooled)
        if return_repr:
            return vul_logits, group_logits, pooled
        return vul_logits, group_logits

class MulVulV2WithCCPP(nn.Module):

    def __init__(self, pretrained_model, num_groups: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = pretrained_model
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(hidden, 1)
        self.group_head = nn.Linear(hidden, num_groups)
        self.vul_head_ccpp = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, return_repr: bool = False,
                return_ccpp: bool = False, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        vul_logits = self.vul_head(pooled)
        group_logits = self.group_head(pooled)
        if return_ccpp:
            vul_ccpp_logits = self.vul_head_ccpp(pooled)
            if return_repr:
                return vul_logits, group_logits, vul_ccpp_logits, pooled
            return vul_logits, group_logits, vul_ccpp_logits
        if return_repr:
            return vul_logits, group_logits, pooled
        return vul_logits, group_logits

    def load_v2_checkpoint(self, v2_state_dict, copy_to_ccpp_head: bool = True):

        own_state = self.state_dict()
        loaded, skipped = [], []
        for name, param in v2_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)
        if copy_to_ccpp_head:
            own_state['vul_head_ccpp.weight'].copy_(own_state['vul_head.weight'])
            own_state['vul_head_ccpp.bias'].copy_(own_state['vul_head.bias'])
            loaded.append('vul_head.* -> vul_head_ccpp.* (identity init)')
        self.load_state_dict(own_state)
        return loaded, skipped

class Adapter(nn.Module):

    def __init__(self, hidden: int = 768, dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(hidden, dim)
        self.up = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(self.drop(self.act(self.down(h))))

class MulVulV2WithLangAdapter(MulVulV2WithCCPP):

    def __init__(self, pretrained_model, num_groups: int = 2, dropout: float = 0.1,
                 adapter_dim: int = 64):
        super().__init__(pretrained_model, num_groups, dropout)
        h = self.backbone.config.hidden_size
        self.adapter_python = Adapter(h, adapter_dim, dropout)
        self.adapter_ccpp = Adapter(h, adapter_dim, dropout)

    def forward(self, input_ids, attention_mask, languages=None,
                return_repr: bool = False, return_ccpp: bool = False, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        h_py = self.adapter_python(pooled)
        if languages is None:
            h_used = h_py
        else:
            mask_py = (languages == 0).unsqueeze(1)
            h_cc = self.adapter_ccpp(pooled)
            h_used = torch.where(mask_py, h_py, h_cc)
        vul_logits = self.vul_head(h_used)
        group_logits = self.group_head(h_used)
        if return_ccpp:
            vul_ccpp_logits = self.vul_head_ccpp(h_used)
            if return_repr:
                return vul_logits, group_logits, vul_ccpp_logits, h_used
            return vul_logits, group_logits, vul_ccpp_logits
        if return_repr:
            return vul_logits, group_logits, h_used
        return vul_logits, group_logits

class MulVulV2With3LangAdapter(MulVulV2WithLangAdapter):

    def __init__(self, pretrained_model, num_groups: int = 2, dropout: float = 0.1,
                 adapter_dim: int = 64):
        super().__init__(pretrained_model, num_groups, dropout, adapter_dim)
        h = self.backbone.config.hidden_size
        self.adapter_js = Adapter(h, adapter_dim, dropout)
        self.vul_head_js = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, languages=None,
                return_repr: bool = False, return_ccpp: bool = False, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        h_py = self.adapter_python(pooled)
        if languages is None:
            h_used = h_py
        else:
            lang = languages.unsqueeze(1)
            h_cc = self.adapter_ccpp(pooled)
            h_js = self.adapter_js(pooled)

            h_used = torch.where(lang == 0, h_py, torch.where(lang == 2, h_js, h_cc))
        vul_logits = self.vul_head(h_used)
        group_logits = self.group_head(h_used)
        if return_ccpp:
            vul_ccpp_logits = self.vul_head_ccpp(h_used)
            vul_js_logits = self.vul_head_js(h_used)
            if return_repr:
                return vul_logits, group_logits, vul_ccpp_logits, vul_js_logits, h_used
            return vul_logits, group_logits, vul_ccpp_logits, vul_js_logits
        if return_repr:
            return vul_logits, group_logits, h_used
        return vul_logits, group_logits

class MulVulSoftMoEV3(nn.Module):

    def __init__(self, pretrained_model, num_groups: int = 2, dropout: float = 0.1,
                 alpha_init: float = 0.0, alpha_learnable: bool = True):
        super().__init__()
        self.backbone = pretrained_model
        hidden = self.backbone.config.hidden_size
        self.num_groups = num_groups
        self.dropout = nn.Dropout(dropout)

        self.vul_head_shared = nn.Linear(hidden, 1)
        self.group_head = nn.Linear(hidden, num_groups)
        self.vul_heads_group = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in range(num_groups)
        ])

        if alpha_learnable:
            self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer('alpha_logit', torch.tensor(float(alpha_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, input_ids, attention_mask, return_all: bool = False,
                return_repr: bool = False, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        vul_shared = self.vul_head_shared(pooled)
        group_logits = self.group_head(pooled)
        router_prob = F.softmax(group_logits, dim=-1)

        vul_per_group = torch.cat(
            [h(pooled) for h in self.vul_heads_group], dim=-1
        )
        vul_mixture = (router_prob * vul_per_group).sum(dim=-1, keepdim=True)

        alpha = self.alpha
        vul_final = alpha * vul_shared + (1 - alpha) * vul_mixture

        if return_all:
            return {
                'vul_final': vul_final,
                'vul_shared': vul_shared,
                'vul_mixture': vul_mixture,
                'vul_per_group': vul_per_group,
                'router_prob': router_prob,
                'group_logits': group_logits,
                'pooled': pooled,
                'alpha': alpha,
            }
        if return_repr:
            return vul_final, group_logits, pooled
        return vul_final, group_logits

    def load_v2_checkpoint(self, v2_state_dict, copy_shared_to_groups: bool = True,
                            verbose: bool = False):

        own_state = self.state_dict()
        loaded, skipped = [], []
        for name, param in v2_state_dict.items():
            if name == 'vul_head.weight':
                target = 'vul_head_shared.weight'
            elif name == 'vul_head.bias':
                target = 'vul_head_shared.bias'
            else:
                target = name
            if target in own_state and own_state[target].shape == param.shape:
                own_state[target].copy_(param)
                loaded.append(f"{name} -> {target}")
            else:
                skipped.append(name)

        if copy_shared_to_groups:
            shared_w = own_state['vul_head_shared.weight']
            shared_b = own_state['vul_head_shared.bias']
            for g in range(self.num_groups):
                own_state[f'vul_heads_group.{g}.weight'].copy_(shared_w)
                own_state[f'vul_heads_group.{g}.bias'].copy_(shared_b)
                loaded.append(f"vul_head_shared.* -> vul_heads_group.{g}.* (identity init)")

        self.load_state_dict(own_state)
        if verbose:
            print(f"V3 warm-start: loaded={len(loaded)} skipped={len(skipped)}")
            for s in skipped:
                print(f"  skipped: {s}")
        return loaded, skipped

class MoEVDRouter(nn.Module):
    def __init__(self, pretrained_model, num_experts: int):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class TwoStageRouter(nn.Module):

    def __init__(self, pretrained_model, num_cwe: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone = pretrained_model
        h = self.backbone.config.hidden_size
        self.num_cwe = num_cwe
        self.dropout = nn.Dropout(dropout)

        self.binary_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, 1),
        )
        self.cwe_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, num_cwe),
        )

    def forward(self, input_ids, attention_mask, return_separate: bool = False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        binary_logit = self.binary_head(pooled)
        cwe_logits = self.cwe_head(pooled)

        if return_separate:
            return binary_logit, cwe_logits

        log_p_safe = -F.softplus(binary_logit)
        log_p_vul  = -F.softplus(-binary_logit)
        log_p_cwe  = F.log_softmax(cwe_logits, dim=-1)
        log_p_vul_per_cwe = log_p_vul + log_p_cwe

        return torch.cat([log_p_vul_per_cwe, log_p_safe], dim=-1)

def two_stage_router_loss(binary_logit, cwe_logits, label_5class,
                          safe_class: int = 4, cwe_weight: float = 0.5):

    is_vul = (label_5class != safe_class)
    binary_target = is_vul.float().unsqueeze(-1)
    binary_loss = F.binary_cross_entropy_with_logits(binary_logit, binary_target)

    if is_vul.any():
        cwe_loss = F.cross_entropy(cwe_logits[is_vul], label_5class[is_vul])
    else:
        cwe_loss = torch.tensor(0.0, device=binary_logit.device)

    return binary_loss + cwe_weight * cwe_loss, binary_loss, cwe_loss

class MulVulExpert(nn.Module):
    def __init__(self,
                 pretrained_model,
                 num_labels: int = 1,
                 num_langs: int = 2,
                 pool_length: int = 5,
                 lang_map: dict = None,
                 temperature: float = 0.1):
        super().__init__()

        self.backbone = pretrained_model
        self.hidden_size = self.backbone.config.hidden_size
        self.pool_length = pool_length
        self.lang_map = lang_map or {"python": 0, "c": 1}
        self.temperature = temperature

        self.parameter_pool = nn.Parameter(
            torch.randn(num_langs, pool_length, self.hidden_size) * 0.02
        )
        self.keys = nn.Parameter(
            torch.randn(num_langs, self.hidden_size) * 0.02
        )

        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, language=None):
        batch_size = input_ids.size(0)

        raw_embeds = self.backbone.embeddings(input_ids)
        cls_query = raw_embeds[:, 0, :]

        if language is not None:
            pool = self.parameter_pool[language]

            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            selected_keys = keys_norm[language]
            cosine_sim = torch.sum(query_norm * selected_keys, dim=1)
            aux_loss = (1.0 - cosine_sim).mean()
        else:
            query_norm = F.normalize(cls_query, p=2, dim=1)
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            scores = torch.matmul(query_norm, keys_norm.transpose(0, 1)) / self.temperature
            lang_ids = torch.argmax(scores, dim=1)
            pool = self.parameter_pool[lang_ids]
            aux_loss = torch.tensor(0.0, device=cls_query.device)

        keep_len = raw_embeds.size(1) - self.pool_length
        raw_embeds = raw_embeds[:, :keep_len, :]
        attention_mask_truncated = attention_mask[:, :keep_len]

        new_embeds = torch.cat([pool, raw_embeds], dim=1)

        pool_mask = torch.ones(batch_size, self.pool_length,
                               dtype=attention_mask.dtype,
                               device=attention_mask.device)
        new_mask = torch.cat([pool_mask, attention_mask_truncated], dim=1)

        outputs = self.backbone(inputs_embeds=new_embeds, attention_mask=new_mask)
        hidden_states = outputs.last_hidden_state

        pool_hidden = hidden_states[:, 0:self.pool_length, :]
        final_repr = pool_hidden.mean(dim=1)

        logits = self.classifier(final_repr)

        return logits, aux_loss

class SharedPoolModel(nn.Module):

    def __init__(self,
                 pretrained_model,
                 num_labels: int = 1,
                 num_langs: int = 2,
                 base_len: int = 10,
                 lang_len: int = 5):
        super().__init__()
        self.backbone = pretrained_model
        self.hidden_size = self.backbone.config.hidden_size
        self.base_len = base_len
        self.lang_len = lang_len
        self.total_pool_len = base_len + lang_len

        self.shared_base = nn.Parameter(
            torch.randn(base_len, self.hidden_size) * 0.02
        )
        self.lang_pool = nn.Parameter(
            torch.randn(num_langs, lang_len, self.hidden_size) * 0.02
        )
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, language):
        batch_size = input_ids.size(0)

        base = self.shared_base.unsqueeze(0).expand(batch_size, -1, -1)
        lang = self.lang_pool[language]
        full_pool = torch.cat([base, lang], dim=1)

        raw_embeds = self.backbone.embeddings(input_ids)
        keep_len = raw_embeds.size(1) - self.total_pool_len
        raw_embeds = raw_embeds[:, :keep_len, :]
        attention_mask_trunc = attention_mask[:, :keep_len]

        new_embeds = torch.cat([full_pool, raw_embeds], dim=1)
        pool_mask = torch.ones(batch_size, self.total_pool_len,
                               dtype=attention_mask.dtype,
                               device=attention_mask.device)
        new_mask = torch.cat([pool_mask, attention_mask_trunc], dim=1)

        outputs = self.backbone(inputs_embeds=new_embeds, attention_mask=new_mask)
        hidden_states = outputs.last_hidden_state

        pool_hidden = hidden_states[:, 0:self.total_pool_len, :]
        final_repr = pool_hidden.mean(dim=1)

        logits = self.classifier(final_repr)
        return logits

class MulVulExpertHierarchical(nn.Module):

    def __init__(self,
                 pretrained_model,
                 num_labels: int = 1,
                 num_langs: int = 2,
                 base_len: int = 10,
                 lang_len: int = 5,
                 cwe_len: int = 5,
                 use_confidence: bool = False):
        super().__init__()
        self.backbone = pretrained_model
        self.hidden_size = self.backbone.config.hidden_size
        self.base_len = base_len
        self.lang_len = lang_len
        self.cwe_len = cwe_len
        self.total_pool_len = base_len + lang_len + cwe_len
        self.use_confidence = use_confidence

        self.shared_base = nn.Parameter(
            torch.randn(base_len, self.hidden_size) * 0.02
        )
        self.lang_pool = nn.Parameter(
            torch.randn(num_langs, lang_len, self.hidden_size) * 0.02
        )
        self.cwe_pool = nn.Parameter(
            torch.randn(cwe_len, self.hidden_size) * 0.02
        )
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        if use_confidence:

            self.confidence_head = nn.Linear(self.hidden_size, 1)

    def load_shared_from_stage1(self, stage1_state_dict, strict_backbone=True):

        own_state = self.state_dict()
        loaded, skipped = [], []
        for name, param in stage1_state_dict.items():
            if name.startswith('classifier'):
                skipped.append(name)
                continue
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded.append(name)
            else:
                skipped.append(name)
        self.load_state_dict(own_state)
        return loaded, skipped

    def freeze_shared(self, freeze_backbone: bool = False):

        self.shared_base.requires_grad = False
        self.lang_pool.requires_grad = False
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, language=None):
        batch_size = input_ids.size(0)

        base = self.shared_base.unsqueeze(0).expand(batch_size, -1, -1)
        if language is None:

            language = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        lang = self.lang_pool[language]
        cwe = self.cwe_pool.unsqueeze(0).expand(batch_size, -1, -1)
        full_pool = torch.cat([base, lang, cwe], dim=1)

        raw_embeds = self.backbone.embeddings(input_ids)
        keep_len = raw_embeds.size(1) - self.total_pool_len
        raw_embeds = raw_embeds[:, :keep_len, :]
        attention_mask_trunc = attention_mask[:, :keep_len]

        new_embeds = torch.cat([full_pool, raw_embeds], dim=1)
        pool_mask = torch.ones(batch_size, self.total_pool_len,
                               dtype=attention_mask.dtype,
                               device=attention_mask.device)
        new_mask = torch.cat([pool_mask, attention_mask_trunc], dim=1)

        outputs = self.backbone(inputs_embeds=new_embeds, attention_mask=new_mask)
        hidden_states = outputs.last_hidden_state

        pool_hidden = hidden_states[:, 0:self.total_pool_len, :]
        final_repr = pool_hidden.mean(dim=1)

        logits = self.classifier(final_repr)
        aux_loss = torch.tensor(0.0, device=logits.device)
        if self.use_confidence:
            confidence_logits = self.confidence_head(final_repr)
            return logits, aux_loss, confidence_logits

        return logits, aux_loss
