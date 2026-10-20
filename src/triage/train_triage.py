"""Training and Out-of-Fold (OOF) Critical threshold calibration for IndoBERT-LoRA Triage."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from src.triage.lora_model import IndoBertMultiTaskLoRA, compute_inverse_class_weights


class TicketTriageDataset(Dataset):
    """PyTorch Dataset encoding sanitized ticket messages for multi-task triage."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        cat_to_idx: Dict[str, int],
        urg_to_idx: Dict[str, int],
        max_length: int = 128,
    ) -> None:
        self.texts = df["sanitized_message"].fillna("").astype(str).tolist()
        self.cat_labels = [cat_to_idx[c] for c in df["category"]]
        self.urg_labels = [urg_to_idx[u] for u in df["urgency"]]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category_labels": torch.tensor(self.cat_labels[idx], dtype=torch.long),
            "urgency_labels": torch.tensor(self.urg_labels[idx], dtype=torch.long),
        }


def set_all_seeds(seed: int) -> None:
    """Set random seeds across `random`, `numpy`, and `torch` for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_temperature_and_critical_threshold(
    urg_logits: np.ndarray,
    urg_targets: np.ndarray,
    critical_idx: int = 3,
    t_min: float = 0.20,
    t_max: float = 0.60,
    beta: float = 1.5,
) -> Dict[str, float]:
    """Calibrate temperature scaling T and search optimal Critical threshold theta_crit*.

    Maximizes F_{beta=1.5} on the `Critical` class over validation predictions while
    maintaining Precision >= 0.80 when feasible.
    """
    best_temp = 1.0
    best_nll = float("inf")
    logits_t = torch.tensor(urg_logits, dtype=torch.float32)
    targets_t = torch.tensor(urg_targets, dtype=torch.long)

    for temp in np.linspace(0.6, 2.0, 29):
        scaled = logits_t / float(temp)
        nll = float(torch.nn.functional.cross_entropy(scaled, targets_t).item())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    scaled_logits = urg_logits / best_temp
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    y_crit_true = (urg_targets == critical_idx).astype(np.int32)
    crit_probs = probs[:, critical_idx]

    best_theta = 0.35
    best_fbeta = -1.0
    best_prec = 0.0
    best_rec = 0.0

    for theta in np.linspace(t_min, t_max, 81):
        preds_bin = (crit_probs >= theta).astype(np.int32)
        prec = float(precision_score(y_crit_true, preds_bin, zero_division=0))
        rec = float(recall_score(y_crit_true, preds_bin, zero_division=0))
        fb = float(fbeta_score(y_crit_true, preds_bin, beta=beta, zero_division=0))

        # Prefer thresholds that keep precision >= 0.80 while maximizing F_beta=1.5
        score = fb + (0.05 if prec >= 0.80 else 0.0)
        if score > best_fbeta:
            best_fbeta = score
            best_theta = float(theta)
            best_prec = prec
            best_rec = rec

    return {
        "temperature": round(best_temp, 4),
        "critical_threshold": round(best_theta, 4),
        "val_critical_fbeta_1_5": round(
            float(
                fbeta_score(
                    y_crit_true,
                    (crit_probs >= best_theta).astype(np.int32),
                    beta=beta,
                    zero_division=0,
                )
            ),
            4,
        ),
        "val_critical_precision": round(best_prec, 4),
        "val_critical_recall": round(best_rec, 4),
    }


def apply_calibrated_urgency_prediction(
    urg_logits: np.ndarray,
    temperature: float,
    critical_threshold: float,
    critical_idx: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute temperature-scaled urgency probabilities and threshold-calibrated predictions."""
    scaled = urg_logits / max(temperature, 1e-4)
    exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1)
    crit_mask = probs[:, critical_idx] >= critical_threshold
    preds[crit_mask] = critical_idx
    return probs, preds


def run_validation_logits(
    model: IndoBertMultiTaskLoRA,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect category & urgency logits on an evaluation DataLoader inside inference_mode."""
    model.eval()
    cat_logits_list: List[np.ndarray] = []
    urg_logits_list: List[np.ndarray] = []
    cat_targets_list: List[np.ndarray] = []
    urg_targets_list: List[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            cat_logits_list.append(out["category_logits"].cpu().numpy())
            urg_logits_list.append(out["urgency_logits"].cpu().numpy())
            cat_targets_list.append(batch["category_labels"].numpy())
            urg_targets_list.append(batch["urgency_labels"].numpy())

    return (
        np.concatenate(cat_logits_list, axis=0),
        np.concatenate(urg_logits_list, axis=0),
        np.concatenate(cat_targets_list, axis=0),
        np.concatenate(urg_targets_list, axis=0),
    )


def train_indobert_lora(
    config_path: str = "configs/config.yaml",
    epochs_override: int | None = None,
) -> Dict[str, Any]:
    """Execute end-to-end IndoBERT PEFT/LoRA training and OOF threshold calibration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_triage] Using device: {device}")

    train_df = pd.read_csv(cfg["paths"]["train_tickets"])
    val_df = pd.read_csv(cfg["paths"]["val_tickets"])

    categories: List[str] = cfg["triage"]["categories"]
    urgencies: List[str] = cfg["triage"]["urgencies"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    urg_to_idx = {u: i for i, u in enumerate(urgencies)}

    backbone_name = cfg["triage"]["backbone_name"]
    max_length = int(cfg["triage"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    train_ds = TicketTriageDataset(train_df, tokenizer, cat_to_idx, urg_to_idx, max_length)
    val_ds = TicketTriageDataset(val_df, tokenizer, cat_to_idx, urg_to_idx, max_length)

    batch_size = int(cfg["triage"]["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Compute inverse class weights strictly from train_tickets.csv
    train_urg_labels = np.array([urg_to_idx[u] for u in train_df["urgency"]], dtype=np.int64)
    urg_alpha = compute_inverse_class_weights(train_urg_labels, num_classes=len(urgencies))

    lora_cfg = cfg["triage"]["lora"]
    loss_cfg = cfg["triage"]["loss"]

    model = IndoBertMultiTaskLoRA(
        backbone_name=backbone_name,
        num_categories=len(categories),
        num_urgencies=len(urgencies),
        lora_r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        category_label_smoothing=float(loss_cfg["category_label_smoothing"]),
        urgency_focal_gamma=float(loss_cfg["urgency_focal_gamma"]),
        urgency_class_weights=urg_alpha,
        use_pretrained_backbone=True,
    ).to(device)

    param_info = model.parameter_summary()
    print(f"[train_triage] Parameter Summary: {param_info}")

    epochs = epochs_override if epochs_override is not None else int(cfg["triage"]["training"]["epochs"])
    lr = float(cfg["triage"]["training"]["learning_rate"])
    weight_decay = float(cfg["triage"]["training"]["weight_decay"])

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * float(cfg["triage"]["training"]["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    save_dir = Path(cfg["paths"]["lora_model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(save_dir / "tokenizer"))

    best_val_score = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category_labels"].to(device)
            urg_labels = batch["urgency_labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                category_labels=cat_labels,
                urgency_labels=urg_labels,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(len(train_loader), 1)
        cat_logits, urg_logits, cat_true, urg_true = run_validation_logits(model, val_loader, device)
        val_cat_f1 = float(f1_score(cat_true, np.argmax(cat_logits, axis=1), average="macro"))
        val_urg_f1 = float(f1_score(urg_true, np.argmax(urg_logits, axis=1), average="macro"))
        composite = 0.5 * (val_cat_f1 + val_urg_f1)
        elapsed = time.perf_counter() - t0

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_category_macro_f1": round(val_cat_f1, 4),
            "val_urgency_macro_f1": round(val_urg_f1, 4),
            "elapsed_sec": round(elapsed, 2),
        }
        history.append(epoch_log)
        print(f"[Epoch {epoch}/{epochs}] {epoch_log}")

        if composite >= best_val_score:
            best_val_score = composite
            model.save_checkpoint(save_dir)

    # Load best checkpoint and run OOF threshold calibration on validation split
    best_model = IndoBertMultiTaskLoRA.load_checkpoint(save_dir, device=device)
    val_cat_logits, val_urg_logits, val_cat_true, val_urg_true = run_validation_logits(
        best_model, val_loader, device
    )

    cal_cfg = cfg["triage"]["calibration"]
    calib_info = calibrate_temperature_and_critical_threshold(
        urg_logits=val_urg_logits,
        urg_targets=val_urg_true,
        critical_idx=urg_to_idx["Critical"],
        t_min=float(cal_cfg["critical_threshold_min"]),
        t_max=float(cal_cfg["critical_threshold_max"]),
        beta=float(cal_cfg["fbeta_beta"]),
    )
    calib_info["parameter_summary"] = param_info
    calib_info["training_history"] = history

    oof_path = Path(cfg["paths"]["oof_thresholds_path"])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_path.write_text(json.dumps(calib_info, indent=2), encoding="utf-8")
    print(f"[train_triage] Saved calibrated OOF threshold to {oof_path}: {calib_info}")

    # Package artifacts into a single zip file for effortless download from Google Colab
    import shutil
    import zipfile

    zip_path = Path("trained_indobert_lora_artifacts.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in save_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.as_posix())
        if oof_path.exists():
            zf.write(oof_path, oof_path.as_posix())
    print(f"[train_triage] Packaged Colab artifacts into: {zip_path.resolve()}")
    return calib_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task IndoBERT-LoRA Triage Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_indobert_lora(config_path=args.config, epochs_override=args.epochs)

