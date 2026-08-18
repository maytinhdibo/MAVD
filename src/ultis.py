import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import json

import torch
import torch.nn.functional as F

DEFAULT_LANGUAGE_MAP = {
    'python': 0, 'py': 0,
    'c': 1, 'cpp': 1, 'c++': 1, 'ccpp': 1,

    'js': 2, 'javascript': 2,
}

class CodeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len: int = 512, language_map=None):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.language_map = language_map or DEFAULT_LANGUAGE_MAP

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = row['code']
        label = row['label']

        lang_str = str(row.get('lang') or row.get('language', 'python')).lower().strip()

        if lang_str not in self.language_map:
            raise ValueError(
                f"Unknown language {lang_str!r}. Known: {sorted(self.language_map.keys())}. "
                f"Add it to DEFAULT_LANGUAGE_MAP in src/ultis.py if intentional."
            )
        lang_id = self.language_map[lang_str]

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        out = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'language': torch.tensor(lang_id, dtype=torch.long)
        }

        if 'confidence_label' in row.index:
            out['confidence_label'] = torch.tensor(int(row['confidence_label']), dtype=torch.long)

        if 'group_label' in row.index:
            out['group_label'] = torch.tensor(int(row['group_label']), dtype=torch.long)

        sw = row['sample_weight'] if 'sample_weight' in row.index else 1.0
        try:
            sw = float(sw)
            if sw != sw:
                sw = 1.0
        except (TypeError, ValueError):
            sw = 1.0
        out['sample_weight'] = torch.tensor(sw, dtype=torch.float)

        _cwe_map = {'cwe-022': 0, 'cwe-078': 1, 'cwe-079': 2, 'cwe-089': 3}
        cwe_raw = row.get('original_CWE_ID') if 'original_CWE_ID' in row.index else row.get('CWE_ID', '')
        out['cwe_id'] = torch.tensor(_cwe_map.get(str(cwe_raw).lower().strip(), -1), dtype=torch.long)

        pr = row['p_ref'] if 'p_ref' in row.index else -1.0
        try:
            pr = float(pr)
            if pr != pr:
                pr = -1.0
        except (TypeError, ValueError):
            pr = -1.0
        out['p_ref'] = torch.tensor(pr, dtype=torch.float)
        return out

class CodeDatasetForRouter(Dataset):
    def __init__(self, data, tokenizer, max_len: int = 512):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = row['code']
        label = row['CWE_ID']
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'CWE_ID': torch.tensor(label, dtype=torch.long)
        }

def load_data(input_data):

    if isinstance(input_data, pd.DataFrame):
        print(f"Received DataFrame directly ({len(input_data)} samples)")
        return input_data

    if not isinstance(input_data, (list, tuple)):
        raise TypeError("load_data expects a list of file paths or a pandas.DataFrame")

    data_list = []
    for file_path in input_data:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        try:
            if file_path.lower().endswith('.jsonl') or file_path.lower().endswith('.json'):

                records = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}")

                df = pd.DataFrame(records)
                print(f"Loaded JSONL: {file_path} ({len(df)} samples)")

            else:

                df = pd.read_csv(file_path)
                print(f"Loaded CSV: {file_path} ({len(df)} samples)")

            data_list.append(df)

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    if not data_list:
        raise ValueError("No valid data files could be loaded!")

    combined_df = pd.concat(data_list, ignore_index=True)
    print(f"Total samples after combining: {len(combined_df)}")
    return combined_df

def create_data_loader(data, tokenizer, batch_size=16, shuffle=True,
                       generator=None, worker_init_fn=None, num_workers=4,
                       balanced=False):

    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    dataset = CodeDataset(data, tokenizer)
    if balanced and shuffle:
        labels = data['label'].values
        class_count = pd.Series(labels).value_counts().to_dict()

        weights = np.array([1.0 / class_count[int(l)] for l in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator,
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

cwe_map = {
    'cwe-022': 0,
    'cwe-078': 1,
    'cwe-079': 2,
    'cwe-089': 3,
}

def focal_loss(logits, targets, gamma=2.0, reduction='mean'):

    ce_loss = F.cross_entropy(logits, targets, reduction='none')

    pt = torch.exp(-ce_loss)

    focal_loss_val = (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val
