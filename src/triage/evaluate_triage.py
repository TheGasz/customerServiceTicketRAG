"""Comprehensive triage evaluation, OOF calibration verification, and slicing error analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml

from src.preprocessing.pii_masker import extract_technical_entities
from src.triage.baseline_tfidf import BaselineTriageModel
from src.triage.lora_model import IndoBertMultiTaskLoRA
from src.triage.train_triage import (
    TicketTriageDataset,
    apply_calibrated_urgency_prediction,
    calibrate_temperature_and_critical_threshold,
    run_validation_logits,
)


class LightweightContextualFallback:
    """Fast CPU/GPU fallback approximating contextual character+word representation.

    Used only when the full `models/indobert_lora` checkpoint has not yet been trained
    on Google Colab, ensuring end-to-end execution and unit testing never crash.
    """

    def __init__(self, categories: List[str], urgencies: List[str], seed: int = 42) -> None:
        self.categories = categories
        self.urgencies = urgencies
        self.cat_to_idx = {c: i for i, c in enumerate(categories)}
        self.urg_to_idx = {u: i for i, u in enumerate(urgencies)}

        self.cat_pipe = Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000, sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=4.0, max_iter=500, random_state=seed)),
            ]
        )
        self.urg_pipe = Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000, sublinear_tf=True)),
                        ]
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=4.0, class_weight="balanced", max_iter=500, random_state=seed
                    ),
                ),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> "LightweightContextualFallback":
        """Fit strictly on `train_tickets.csv`."""
        texts = train_df["sanitized_message"].fillna("").astype(str).tolist()
        y_cat = np.array([self.cat_to_idx[c] for c in train_df["category"]], dtype=np.int64)
        y_urg = np.array([self.urg_to_idx[u] for u in train_df["urgency"]], dtype=np.int64)
        self.cat_pipe.fit(texts, y_cat)
        self.urg_pipe.fit(texts, y_urg)
        return self

    def predict_logits(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return log-probabilities as pseudo-logits."""
        cat_probs = np.clip(self.cat_pipe.predict_proba(texts), 1e-6, 1.0 - 1e-6)
        urg_probs = np.clip(self.urg_pipe.predict_proba(texts), 1e-6, 1.0 - 1e-6)
        return np.log(cat_probs), np.log(urg_probs)


class TriageRuntimePredictor:
    """Unified runtime inference engine for FastAPI and Streamlit with automatic LoRA/Fallback loading."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.categories: List[str] = self.cfg["triage"]["categories"]
        self.urgencies: List[str] = self.cfg["triage"]["urgencies"]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.urg_to_idx = {u: i for i, u in enumerate(self.urgencies)}
        self.max_length = int(self.cfg["triage"]["max_length"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lora_dir = Path(self.cfg["paths"]["lora_model_dir"])
        oof_path = Path(self.cfg["paths"]["oof_thresholds_path"])

        self.temperature = 1.0
        self.critical_threshold = 0.35
        if oof_path.exists():
            oof_data = json.loads(oof_path.read_text(encoding="utf-8"))
            self.temperature = float(oof_data.get("temperature", 1.0))
            self.critical_threshold = float(oof_data.get("critical_threshold", 0.35))

        self.lora_model: Optional[IndoBertMultiTaskLoRA] = None
        self.tokenizer: Optional[Any] = None
        self.fallback_model: Optional[LightweightContextualFallback] = None
        self.mode = "fallback"

        if (lora_dir / "heads.pt").exists() and (lora_dir / "adapter").exists():
            tok_dir = lora_dir / "tokenizer"
            tok_source = str(tok_dir) if tok_dir.exists() else self.cfg["triage"]["backbone_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(tok_source)
            self.lora_model = IndoBertMultiTaskLoRA.load_checkpoint(lora_dir, device=self.device)
            self.lora_model.eval()
            self.mode = "indobert_lora"
        else:
            train_df = pd.read_csv(self.cfg["paths"]["train_tickets"])
            self.fallback_model = LightweightContextualFallback(
                categories=self.categories,
                urgencies=self.urgencies,
                seed=int(self.cfg["project"]["seed"]),
            ).fit(train_df)

    def predict_single(self, sanitized_text: str) -> Dict[str, Any]:
        """Run single-pass multi-task inference inside `torch.inference_mode()`."""
        if self.mode == "indobert_lora" and self.lora_model is not None and self.tokenizer is not None:
            with torch.inference_mode():
                enc = self.tokenizer(
                    [sanitized_text],
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
                cat_logits = out["category_logits"].cpu().numpy()
                urg_logits = out["urgency_logits"].cpu().numpy()
        else:
            assert self.fallback_model is not None
            cat_logits, urg_logits = self.fallback_model.predict_logits([sanitized_text])

        # Softmax for category
        cat_exp = np.exp(cat_logits - np.max(cat_logits, axis=1, keepdims=True))
        cat_probs = (cat_exp / np.sum(cat_exp, axis=1, keepdims=True))[0]
        cat_idx = int(np.argmax(cat_probs))

        # Calibrated urgency probabilities & OOF Critical threshold
        crit_idx = self.urg_to_idx["Critical"]
        urg_probs_batch, urg_preds_batch = apply_calibrated_urgency_prediction(
            urg_logits=urg_logits,
            temperature=self.temperature,
            critical_threshold=self.critical_threshold,
            critical_idx=crit_idx,
        )
        urg_probs = urg_probs_batch[0]
        urg_idx = int(urg_preds_batch[0])

        return {
            "category": self.categories[cat_idx],
            "category_confidence": round(float(cat_probs[cat_idx]), 4),
            "category_probs": {
                self.categories[i]: round(float(cat_probs[i]), 4)
                for i in range(len(self.categories))
            },
            "urgency": self.urgencies[urg_idx],
            "urgency_confidence": round(float(urg_probs[urg_idx]), 4),
            "urgency_probs": {
                self.urgencies[i]: round(float(urg_probs[i]), 4)
                for i in range(len(self.urgencies))
            },
            "model_mode": self.mode,
        }


def _compute_metrics_from_logits(
    cat_logits: np.ndarray,
    urg_logits: np.ndarray,
    cat_true: np.ndarray,
    urg_true: np.ndarray,
    temperature: float = 1.0,
    critical_threshold: Optional[float] = None,
    critical_idx: int = 3,
) -> Dict[str, Any]:
    """Compute Category Macro-F1, Urgency Macro-F1, Critical PR-AUC, Precision, and Recall."""
    cat_exp = np.exp(cat_logits - np.max(cat_logits, axis=1, keepdims=True))
    cat_probs = cat_exp / np.sum(cat_exp, axis=1, keepdims=True)
    cat_preds = np.argmax(cat_probs, axis=1)

    if critical_threshold is None:
        urg_exp = np.exp(urg_logits - np.max(urg_logits, axis=1, keepdims=True))
        urg_probs = urg_exp / np.sum(urg_exp, axis=1, keepdims=True)
        urg_preds = np.argmax(urg_probs, axis=1)
    else:
        urg_probs, urg_preds = apply_calibrated_urgency_prediction(
            urg_logits=urg_logits,
            temperature=temperature,
            critical_threshold=critical_threshold,
            critical_idx=critical_idx,
        )

    cat_macro_f1 = float(f1_score(cat_true, cat_preds, average="macro"))
    urg_macro_f1 = float(f1_score(urg_true, urg_preds, average="macro"))

    y_crit_bin = (urg_true == critical_idx).astype(np.int32)
    crit_pr_auc = float(average_precision_score(y_crit_bin, urg_probs[:, critical_idx]))

    prec, rec, f1_cls, _ = precision_recall_fscore_support(
        urg_true, urg_preds, labels=[0, 1, 2, 3], zero_division=0
    )

    return {
        "category_macro_f1": round(cat_macro_f1, 4),
        "urgency_macro_f1": round(urg_macro_f1, 4),
        "critical_pr_auc": round(crit_pr_auc, 4),
        "critical_precision": round(float(prec[critical_idx]), 4),
        "critical_recall": round(float(rec[critical_idx]), 4),
        "critical_f1": round(float(f1_cls[critical_idx]), 4),
        "cat_preds": cat_preds,
        "urg_preds": urg_preds,
        "cat_probs": cat_probs,
        "urg_probs": urg_probs,
    }


def evaluate_all_triage_models(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Run full Tahap 2 benchmark, confusion matrix plotting, and slicing analysis."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    train_df = pd.read_csv(cfg["paths"]["train_tickets"])
    val_df = pd.read_csv(cfg["paths"]["val_tickets"])
    test_df = pd.read_csv(cfg["paths"]["test_tickets"])

    categories: List[str] = cfg["triage"]["categories"]
    urgencies: List[str] = cfg["triage"]["urgencies"]
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    urg_to_idx = {u: i for i, u in enumerate(urgencies)}
    crit_idx = urg_to_idx["Critical"]

    # 1. Baseline TF-IDF + LightGBM
    baseline_path = Path(cfg["paths"]["baseline_model_path"])
    if baseline_path.exists():
        baseline_model = BaselineTriageModel.load(baseline_path)
    else:
        baseline_model = BaselineTriageModel(categories, urgencies, seed=seed).fit(train_df)
        baseline_model.save(baseline_path)

    base_test = baseline_model.evaluate(test_df)

    # 2. IndoBERT-LoRA (or Fallback if not yet trained on Colab)
    lora_dir = Path(cfg["paths"]["lora_model_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_cat_val = np.array([cat_to_idx[c] for c in val_df["category"]], dtype=np.int64)
    y_urg_val = np.array([urg_to_idx[u] for u in val_df["urgency"]], dtype=np.int64)
    y_cat_test = np.array([cat_to_idx[c] for c in test_df["category"]], dtype=np.int64)
    y_urg_test = np.array([urg_to_idx[u] for u in test_df["urgency"]], dtype=np.int64)

    if (lora_dir / "heads.pt").exists() and (lora_dir / "adapter").exists():
        tok_dir = lora_dir / "tokenizer"
        tok_src = str(tok_dir) if tok_dir.exists() else cfg["triage"]["backbone_name"]
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        lora_model = IndoBertMultiTaskLoRA.load_checkpoint(lora_dir, device=device)

        val_ds = TicketTriageDataset(val_df, tokenizer, cat_to_idx, urg_to_idx, int(cfg["triage"]["max_length"]))
        test_ds = TicketTriageDataset(test_df, tokenizer, cat_to_idx, urg_to_idx, int(cfg["triage"]["max_length"]))
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        val_cat_logits, val_urg_logits, _, _ = run_validation_logits(lora_model, val_loader, device)
        test_cat_logits, test_urg_logits, _, _ = run_validation_logits(lora_model, test_loader, device)
        model_source = "indobert_lora_trained"
    else:
        fb = LightweightContextualFallback(categories, urgencies, seed=seed).fit(train_df)
        val_cat_logits, val_urg_logits = fb.predict_logits(
            val_df["sanitized_message"].fillna("").astype(str).tolist()
        )
        test_cat_logits, test_urg_logits = fb.predict_logits(
            test_df["sanitized_message"].fillna("").astype(str).tolist()
        )
        model_source = "lightweight_contextual_fallback"

    # Calibrate OOF threshold on validation logits
    calib = calibrate_temperature_and_critical_threshold(
        urg_logits=val_urg_logits,
        urg_targets=y_urg_val,
        critical_idx=crit_idx,
        t_min=float(cfg["triage"]["calibration"]["critical_threshold_min"]),
        t_max=float(cfg["triage"]["calibration"]["critical_threshold_max"]),
        beta=float(cfg["triage"]["calibration"]["fbeta_beta"]),
    )
    oof_path = Path(cfg["paths"]["oof_thresholds_path"])
    if not oof_path.exists():
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")

    lora_argmax = _compute_metrics_from_logits(
        test_cat_logits, test_urg_logits, y_cat_test, y_urg_test, critical_threshold=None
    )
    lora_calibrated = _compute_metrics_from_logits(
        test_cat_logits,
        test_urg_logits,
        y_cat_test,
        y_urg_test,
        temperature=calib["temperature"],
        critical_threshold=calib["critical_threshold"],
        critical_idx=crit_idx,
    )

    # Entity Exact-Match F1 on Test Split
    ner_matches = 0
    for _, row in test_df.fillna("").iterrows():
        ext = extract_technical_entities(str(row["sanitized_message"]))
        if (
            (ext["product_model"] or "") == str(row["product_model"]).upper()
            and (ext["error_code"] or "") == str(row["error_code"]).upper()
            and (ext["serial_number"] or "") == str(row["serial_number"]).upper()
        ):
            ner_matches += 1
    ner_exact_f1 = round(ner_matches / len(test_df), 4)

    # Slicing Error Analysis across 4 sub-populations + Formal & Multi-Issue
    slices = {
        "Tiket Formal (>15 Kata)": (test_df["word_count"] > 15) & (~test_df["is_slang_or_typo"]),
        "Tiket Pendek (<12 Kata)": test_df["word_count"] < 12,
        "Tiket Slang/Typo": test_df["is_slang_or_typo"].astype(bool),
        "Tiket Dengan Kode Error Eksplisit": test_df["has_explicit_code"].astype(bool),
        "Tiket Tanpa Kode Error": ~test_df["has_explicit_code"].astype(bool),
        "Tiket Keluhan Multi-Masalah": test_df["is_multi_issue"].astype(bool),
    }

    base_cat_preds = np.array(base_test["cat_preds"])
    base_urg_preds = np.array(base_test["urg_preds"])
    lora_cat_preds = lora_calibrated["cat_preds"]
    lora_urg_preds = lora_calibrated["urg_preds"]

    slicing_results: Dict[str, Dict[str, float]] = {}
    for slice_name, mask in slices.items():
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        b_f1 = 0.5 * (
            f1_score(y_cat_test[idx], base_cat_preds[idx], average="macro", zero_division=0)
            + f1_score(y_urg_test[idx], base_urg_preds[idx], average="macro", zero_division=0)
        )
        l_f1 = 0.5 * (
            f1_score(y_cat_test[idx], lora_cat_preds[idx], average="macro", zero_division=0)
            + f1_score(y_urg_test[idx], lora_urg_preds[idx], average="macro", zero_division=0)
        )
        slicing_results[slice_name] = {
            "sample_count": int(len(idx)),
            "baseline_macro_f1": round(float(b_f1), 4),
            "indobert_lora_macro_f1": round(float(l_f1), 4),
        }

    formal_f1 = slicing_results.get("Tiket Formal (>15 Kata)", {}).get("indobert_lora_macro_f1", 0.0)
    slang_f1 = slicing_results.get("Tiket Slang/Typo", {}).get("indobert_lora_macro_f1", 0.0)
    slang_gap = round(abs(formal_f1 - slang_f1), 4)

    # Top-10 highest-loss / worst errors analysis
    urg_probs = lora_calibrated["urg_probs"]
    true_urg_probs = urg_probs[np.arange(len(y_urg_test)), y_urg_test]
    sample_losses = -np.log(np.clip(true_urg_probs, 1e-6, 1.0))
    worst_indices = np.argsort(sample_losses)[::-1][:10]

    top_errors: List[Dict[str, Any]] = []
    for r_idx in worst_indices:
        row = test_df.iloc[int(r_idx)]
        top_errors.append(
            {
                "ticket_id": str(row["ticket_id"]),
                "sanitized_message": str(row["sanitized_message"])[:120] + "...",
                "true_category": str(row["category"]),
                "pred_category": categories[int(lora_cat_preds[r_idx])],
                "true_urgency": str(row["urgency"]),
                "pred_urgency": urgencies[int(lora_urg_preds[r_idx])],
                "loss": round(float(sample_losses[r_idx]), 4),
            }
        )

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Plot Confusion Matrix (Category & Urgency side by side)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cm_cat = confusion_matrix(y_cat_test, lora_cat_preds, labels=[0, 1, 2, 3])
    cm_urg = confusion_matrix(y_urg_test, lora_urg_preds, labels=[0, 1, 2, 3])

    sns.heatmap(
        cm_cat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        ax=axes[0],
    )
    axes[0].set_title("Category Confusion Matrix (IndoBERT-LoRA)")
    axes[0].set_xlabel("Predicted Category")
    axes[0].set_ylabel("True Category")

    sns.heatmap(
        cm_urg,
        annot=True,
        fmt="d",
        cmap="OrRd",
        xticklabels=urgencies,
        yticklabels=urgencies,
        ax=axes[1],
    )
    axes[1].set_title("Urgency Confusion Matrix (OOF Calibrated)")
    axes[1].set_xlabel("Predicted Urgency")
    axes[1].set_ylabel("True Urgency")

    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix_triage.png"
    plt.savefig(cm_path, dpi=180)
    plt.close(fig)

    benchmark_payload = {
        "model_source": model_source,
        "entity_exact_match_f1": ner_exact_f1,
        "oof_calibration": calib,
        "models": {
            "baseline_tfidf_lightgbm": {
                "category_macro_f1": base_test["category_macro_f1"],
                "urgency_macro_f1": base_test["urgency_macro_f1"],
                "critical_pr_auc": base_test["critical_pr_auc"],
                "critical_precision": base_test["critical_precision"],
                "critical_recall": base_test["critical_recall"],
                "critical_f1": base_test["critical_f1"],
            },
            "indobert_lora_argmax": {
                "category_macro_f1": lora_argmax["category_macro_f1"],
                "urgency_macro_f1": lora_argmax["urgency_macro_f1"],
                "critical_pr_auc": lora_argmax["critical_pr_auc"],
                "critical_precision": lora_argmax["critical_precision"],
                "critical_recall": lora_argmax["critical_recall"],
                "critical_f1": lora_argmax["critical_f1"],
            },
            "indobert_lora_oof_calibrated": {
                "category_macro_f1": lora_calibrated["category_macro_f1"],
                "urgency_macro_f1": lora_calibrated["urgency_macro_f1"],
                "critical_pr_auc": lora_calibrated["critical_pr_auc"],
                "critical_precision": lora_calibrated["critical_precision"],
                "critical_recall": lora_calibrated["critical_recall"],
                "critical_f1": lora_calibrated["critical_f1"],
            },
        },
        "slicing_analysis": slicing_results,
        "formal_vs_slang_gap": slang_gap,
        "top_10_worst_predictions": top_errors,
    }

    (reports_dir / "triage_benchmark.json").write_text(
        json.dumps(benchmark_payload, indent=2), encoding="utf-8"
    )

    # Write Markdown Slicing Report
    md_lines = [
        "# Laporan Evaluasi Triase Multi-Task & Slicing Error Analysis",
        "",
        f"- **Model Mode**: `{model_source}`",
        f"- **NER Exact-Match F1**: `{ner_exact_f1:.4f}` (Target >= 0.95)",
        f"- **OOF Critical Threshold ($\\theta_{{crit}}^*$)**: `{calib['critical_threshold']}` (Temperature $T = {calib['temperature']}$)",
        f"- **Gap Macro-F1 (Formal vs Slang/Typo)**: `{slang_gap * 100:.2f}%` (Target <= 8.0%)",
        "",
        "## 1. Perbandingan Performa Model pada Held-Out Test Split (180 Tiket)",
        "",
        "| Model Konfigurasi | Category Macro-F1 | Urgency Macro-F1 | Critical PR-AUC | Critical Precision | Critical Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_label, m_key in [
        ("Baseline (TF-IDF + LightGBM)", "baseline_tfidf_lightgbm"),
        ("IndoBERT-LoRA (Default Argmax)", "indobert_lora_argmax"),
        ("IndoBERT-LoRA (OOF Threshold Calibrated)", "indobert_lora_oof_calibrated"),
    ]:
        m = benchmark_payload["models"][m_key]
        md_lines.append(
            f"| {m_label} | {m['category_macro_f1']:.4f} | {m['urgency_macro_f1']:.4f} | "
            f"{m['critical_pr_auc']:.4f} | {m['critical_precision']:.4f} | {m['critical_recall']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Slicing Error Analysis Berdasarkan Karakteristik Tiket",
            "",
            "| Segmen Sub-Populasi Tiket | Jumlah Sampel | Baseline Macro-F1 | IndoBERT-LoRA Macro-F1 | Delta Peningkatan |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for s_name, s_val in slicing_results.items():
        delta = s_val["indobert_lora_macro_f1"] - s_val["baseline_macro_f1"]
        md_lines.append(
            f"| {s_name} | {s_val['sample_count']} | {s_val['baseline_macro_f1']:.4f} | "
            f"{s_val['indobert_lora_macro_f1']:.4f} | +{delta:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Daftar 10 Sampel dengan Nilai Loss Tertinggi (Top-Loss Errors)",
            "",
            "| Ticket ID | True Category | Pred Category | True Urgency | Pred Urgency | Cross-Entropy Loss | Cuplikan Pesan |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for err in top_errors:
        md_lines.append(
            f"| `{err['ticket_id']}` | {err['true_category']} | {err['pred_category']} | "
            f"{err['true_urgency']} | {err['pred_urgency']} | {err['loss']:.4f} | {err['sanitized_message']} |"
        )

    (reports_dir / "triage_slicing_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    return benchmark_payload


if __name__ == "__main__":
    res = evaluate_all_triage_models()
    print("=== TRIAGE BENCHMARK SUMMARY ===")
    print(json.dumps(res["models"], indent=2))
    print(f"Formal vs Slang Gap: {res['formal_vs_slang_gap']}")
