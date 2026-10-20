"""Multi-Task IndoBERT with PEFT/LoRA adapters, Dual MLP Heads, and Multi-Class Focal Loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss with inverse class-frequency weighting alpha_t.

    Formula:
        L_Focal(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t), gamma = 2.0
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from raw unnormalized logits and integer class targets."""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1, 1)
        pt = probs.gather(1, targets).squeeze(1)
        log_pt = log_probs.gather(1, targets).squeeze(1)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets.squeeze(1))
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DualMLPHead(nn.Module):
    """2-layer MLP classification head (768 -> 256 -> num_classes) with LayerNorm & Dropout."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pooled representation `h_CLS` into class logits."""
        return self.net(x)


class IndoBertMultiTaskLoRA(nn.Module):
    """Shared-backbone IndoBERT with PEFT/LoRA and parallel Category & Urgency heads."""

    def __init__(
        self,
        backbone_name: str = "indobenchmark/indobert-base-p1",
        num_categories: int = 4,
        num_urgencies: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.10,
        target_modules: Optional[List[str]] = None,
        category_label_smoothing: float = 0.05,
        urgency_focal_gamma: float = 2.0,
        urgency_class_weights: Optional[torch.Tensor] = None,
        use_pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_categories = num_categories
        self.num_urgencies = num_urgencies
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["query", "key", "value", "dense"]

        if use_pretrained_backbone:
            base_model = AutoModel.from_pretrained(backbone_name)
        else:
            config = AutoConfig.from_pretrained(backbone_name)
            base_model = AutoModel.from_config(config)

        # Freeze all original IndoBERT parameters before injecting LoRA matrices
        for param in base_model.parameters():
            param.requires_grad = False

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_cfg)
        hidden_size = int(base_model.config.hidden_size)

        self.category_head = DualMLPHead(
            input_dim=hidden_size, hidden_dim=256, num_classes=num_categories
        )
        self.urgency_head = DualMLPHead(
            input_dim=hidden_size, hidden_dim=256, num_classes=num_urgencies
        )

        self.category_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=category_label_smoothing
        )
        self.urgency_loss_fn = MultiClassFocalLoss(
            alpha=urgency_class_weights, gamma=urgency_focal_gamma
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_labels: Optional[torch.Tensor] = None,
        urgency_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single-pass forward computation returning category/urgency logits and joint loss."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled CLS token representation h_CLS (768-d)
        h_cls = outputs.last_hidden_state[:, 0, :]

        cat_logits = self.category_head(h_cls)
        urg_logits = self.urgency_head(h_cls)

        result: Dict[str, torch.Tensor] = {
            "category_logits": cat_logits,
            "urgency_logits": urg_logits,
        }

        if category_labels is not None and urgency_labels is not None:
            loss_cat = self.category_loss_fn(cat_logits, category_labels)
            loss_urg = self.urgency_loss_fn(urg_logits, urgency_labels)
            total_loss = loss_cat + loss_urg
            result["loss"] = total_loss
            result["category_loss"] = loss_cat
            result["urgency_loss"] = loss_urg

        return result

    def parameter_summary(self) -> Dict[str, float]:
        """Calculate trainable vs total parameters to verify ~1.08% PEFT efficiency."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0
        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "trainable_percentage": round(pct, 4),
        }

    def save_checkpoint(self, save_dir: str | Path) -> None:
        """Save LoRA adapter weights and dual classification heads to `save_dir`."""
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(target / "adapter"))
        torch.save(
            {
                "category_head": self.category_head.state_dict(),
                "urgency_head": self.urgency_head.state_dict(),
                "backbone_name": self.backbone_name,
                "num_categories": self.num_categories,
                "num_urgencies": self.num_urgencies,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
                "target_modules": self.target_modules,
            },
            target / "heads.pt",
        )

    @classmethod
    def load_checkpoint(
        cls,
        save_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> "IndoBertMultiTaskLoRA":
        """Load saved LoRA adapter and classification heads from `save_dir`."""
        target = Path(save_dir)
        heads_meta = torch.load(target / "heads.pt", map_location=device, weights_only=True)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.backbone_name = heads_meta["backbone_name"]
        instance.num_categories = heads_meta["num_categories"]
        instance.num_urgencies = heads_meta["num_urgencies"]
        instance.lora_r = heads_meta["lora_r"]
        instance.lora_alpha = heads_meta["lora_alpha"]
        instance.lora_dropout = heads_meta["lora_dropout"]
        instance.target_modules = heads_meta["target_modules"]

        base_model = AutoModel.from_pretrained(instance.backbone_name)
        for param in base_model.parameters():
            param.requires_grad = False

        instance.backbone = PeftModel.from_pretrained(
            base_model, str(target / "adapter")
        )
        hidden_size = int(base_model.config.hidden_size)
        instance.category_head = DualMLPHead(
            input_dim=hidden_size, hidden_dim=256, num_classes=instance.num_categories
        )
        instance.urgency_head = DualMLPHead(
            input_dim=hidden_size, hidden_dim=256, num_classes=instance.num_urgencies
        )
        instance.category_head.load_state_dict(heads_meta["category_head"])
        instance.urgency_head.load_state_dict(heads_meta["urgency_head"])
        instance.category_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        instance.urgency_loss_fn = MultiClassFocalLoss(gamma=2.0)

        instance.to(device)
        instance.eval()
        return instance


def compute_inverse_class_weights(labels: np.ndarray, num_classes: int = 4) -> torch.Tensor:
    """Compute normalized inverse class-frequency weights strictly from `train_tickets.csv`."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = len(labels) / (num_classes * counts)
    inv_freq = inv_freq / np.sum(inv_freq) * num_classes
    return torch.tensor(inv_freq, dtype=torch.float32)
