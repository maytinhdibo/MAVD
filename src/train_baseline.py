import argparse
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer

from logger import get_logger, setup_logging
from mavd import BaselineCodeBERT
from seed_utils import make_generator, set_seed, worker_init_fn
from ultis import create_data_loader, load_data

def train_baseline(model, device, optimizer, criterion, logger, args, train_loader, val_loader=None):
    model.to(device)

    best_metric = -1.0 if val_loader else float('inf')
    best_epoch = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "baseline_best.pth")

    patience = 5
    patience_counter = 0

    logger.info(f"BaselineCodeBERT | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train BCE: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    logits = model(input_ids, attention_mask)
                    probs = torch.sigmoid(logits).squeeze(1)
                    preds = (probs >= 0.5).cpu().numpy().astype(int)
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy().astype(int))
            report = classification_report(
                all_labels, all_preds, target_names=['Safe', 'Vul'],
                output_dict=True, zero_division=0,
            )
            sel = getattr(args, 'select_metric', 'vul')
            if sel == 'macro':
                current_f1 = report['macro avg']['f1-score']
            else:
                current_f1 = report['Vul']['f1-score']
            logger.info(
                f"\n[Val Baseline - Epoch {epoch+1}] (select={sel})\n" +
                classification_report(all_labels, all_preds,
                                      target_names=['Safe', 'Vul'], zero_division=0)
            )
            if current_f1 > best_metric:
                best_metric = current_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best baseline! Val {sel}-F1: {current_f1:.4f} (Epoch {best_epoch})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            if avg_loss < best_metric:
                best_metric = avg_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best baseline! Train BCE: {avg_loss:.4f} (Epoch {best_epoch})")

    logger.info(f"Baseline finished. Best at epoch {best_epoch}: {best_model_path}")

def get_argparse():
    parser = argparse.ArgumentParser(description="Train Baseline CodeBERT classifier")
    parser.add_argument('--train_files', nargs='+', required=True)
    parser.add_argument('--val_files', nargs='+', default=[])
    parser.add_argument('--pretrained_model', type=str, default='microsoft/codebert-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='model/baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pos_weight_cap', type=float, default=3.0)
    parser.add_argument('--select_metric', choices=['vul', 'macro'], default='vul',
                        help='Checkpoint selection metric on val: vul-F1 (class 1) or macro-F1 (balanced).')
    return parser.parse_args()

def main():
    args = get_argparse()
    os.makedirs('log', exist_ok=True)
    log_file = os.path.join('log', f'train_baseline_seed{args.seed}.log')
    setup_logging(log_file=log_file, reset_file=True)
    logger = get_logger(__name__)

    set_seed(args.seed)
    logger.info(f"Seed fixed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    backbone = AutoModel.from_pretrained(args.pretrained_model)
    model = BaselineCodeBERT(pretrained_model=backbone, num_labels=1)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    logger.info(f"Loading training data: {args.train_files}")
    train_data = load_data(args.train_files)
    num_neg = len(train_data[train_data['label'] == 0])
    num_pos = len(train_data[train_data['label'] == 1])
    raw_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    weight_ratio = min(raw_ratio, args.pos_weight_cap)
    logger.info(
        f"Baseline class distribution - Safe: {num_neg} | Vul: {num_pos} | "
        f"raw pos_weight: {raw_ratio:.2f} | capped to: {weight_ratio:.2f}"
    )
    pos_weight = torch.tensor([weight_ratio]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    generator = make_generator(args.seed)
    train_loader = create_data_loader(
        train_data, tokenizer, batch_size=args.batch_size, shuffle=True,
        generator=generator, worker_init_fn=worker_init_fn,
    )

    val_loader = None
    if args.val_files:
        val_data = load_data(args.val_files)
        val_loader = create_data_loader(
            val_data, tokenizer, batch_size=args.batch_size, shuffle=False,
            worker_init_fn=worker_init_fn,
        )

    train_baseline(model, device, optimizer, criterion, logger, args, train_loader, val_loader)

if __name__ == "__main__":
    main()
