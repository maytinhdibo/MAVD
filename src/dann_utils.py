import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = float(lambda_)
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, lambda_)

class LanguageClassifier(nn.Module):

    def __init__(self, hidden_size: int, num_languages: int = 2,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages),
        )

    def forward(self, features: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
        reversed_features = gradient_reverse(features, lambda_grl)
        return self.classifier(reversed_features)

class ConditionalLanguageClassifier(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int = 2,
                 num_languages: int = 2, hidden_dim: int = 128, dropout: float = 0.1,
                 random_dim: int = None):
        super().__init__()
        self.num_classes = num_classes
        full_dim = hidden_size * num_classes
        self.use_random = random_dim is not None and full_dim > random_dim
        if self.use_random:

            self.register_buffer('Rf', torch.randn(hidden_size, random_dim))
            self.register_buffer('Rg', torch.randn(num_classes, random_dim))
            in_dim = random_dim
        else:
            in_dim = full_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_languages),
        )

    def forward(self, features: torch.Tensor, class_probs: torch.Tensor,
                lambda_grl: float = 1.0) -> torch.Tensor:
        rev = gradient_reverse(features, lambda_grl)
        g = class_probs.detach()
        if self.use_random:
            import math
            zf = rev @ self.Rf
            zg = g @ self.Rg
            op = zf * zg / math.sqrt(self.Rf.size(1))
            return self.classifier(op)
        op = torch.bmm(rev.unsqueeze(2), g.unsqueeze(1))
        return self.classifier(op.view(op.size(0), -1))

def entropy_weight(class_probs: torch.Tensor) -> torch.Tensor:

    p = class_probs.detach().clamp(min=1e-6, max=1.0)
    H = -(p * p.log()).sum(dim=1)
    return 1.0 + torch.exp(-H)

def dann_lambda_schedule(progress: float, gamma: float = 10.0,
                          max_lambda: float = 1.0) -> float:

    import math
    progress = max(0.0, min(1.0, float(progress)))
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
