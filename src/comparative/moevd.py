import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEVDRouter(nn.Module):

    def __init__(self, pretrained_model, num_experts: int):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_experts)

    def forward(self, input_ids, attention_mask):
        pooled = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(pooled)

class MoEVDExpert(nn.Module):

    def __init__(self, pretrained_model, num_labels: int = 1):
        super().__init__()
        self.backbone = pretrained_model
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled = self.backbone(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.classifier(pooled)

def focal_loss(logits, targets, gamma: float = 2.0, reduction: str = 'mean'):

    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    fl = (1.0 - pt) ** gamma * ce
    if reduction == 'mean':
        return fl.mean()
    if reduction == 'sum':
        return fl.sum()
    return fl
