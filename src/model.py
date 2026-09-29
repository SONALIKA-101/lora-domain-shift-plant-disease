"""
Models for the study: a DINOv2 backbone with four adaptation strategies.

Backbone: DINOv2 (via HF transformers)
LoRA:     on the attention query/value projections (via peft)
DoGA:     a domain-adversarial head attached to the shared features through a
          gradient reversal layer (Ganin and Lempitsky, 2015). A domain
          classifier is trained on the features, while the gradient reversal
          pushes the (LoRA-adapted) backbone to make the domain harder to
          predict. Whether this improves generalization to an unseen domain is
          what the accompanying study evaluates.

Modes (selectable via config `baselines.mode` or the --mode flag):
    - linear_probe : frozen backbone, only a linear head trains
    - lora_plain   : LoRA adapters + classifier head, no domain-adversarial term
    - full_finetune: entire backbone unfrozen
    - doga         : LoRA + domain-adversarial head
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class GradientReversalFunction(torch.autograd.Function):
    """Standard gradient reversal layer (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambd = 0.0  # ramped up during training via set_lambda()

    def set_lambda(self, lambd: float):
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)


class DomainClassifierHead(nn.Module):
    def __init__(self, feature_dim: int, num_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x):
        return self.net(x)


class DoGAModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, num_domains: int,
                 lora_cfg: dict, mode: str = "doga"):
        super().__init__()
        assert mode in {"linear_probe", "lora_plain", "full_finetune", "doga"}
        self.mode = mode

        base_model = AutoModel.from_pretrained(backbone_name)
        feature_dim = base_model.config.hidden_size

        if mode == "linear_probe":
            for p in base_model.parameters():
                p.requires_grad = False
            self.backbone = base_model

        elif mode == "full_finetune":
            self.backbone = base_model  # all params trainable

        else:  # lora_plain or doga
            for p in base_model.parameters():
                p.requires_grad = False
            peft_cfg = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                target_modules=lora_cfg["target_modules"],
                bias="none",
            )
            self.backbone = get_peft_model(base_model, peft_cfg)

        self.classifier = nn.Linear(feature_dim, num_classes)

        self.use_domain_head = (mode == "doga")
        if self.use_domain_head:
            self.grl = GradientReversalLayer()
            self.domain_head = DomainClassifierHead(feature_dim, num_domains)

    def encode(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        # DINOv2 pooled output = CLS token representation
        return out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
            else out.last_hidden_state[:, 0]

    def forward(self, pixel_values, domain_lambda: float = 0.0):
        features = self.encode(pixel_values)
        class_logits = self.classifier(features)

        domain_logits = None
        if self.use_domain_head:
            self.grl.set_lambda(domain_lambda)
            reversed_feats = self.grl(features)
            domain_logits = self.domain_head(reversed_feats)

        return {"class_logits": class_logits, "domain_logits": domain_logits, "features": features}

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
