"""Cross-Encoder Reranker and Confidence-Based Abstention Router with Active Learning Queue logging."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder
import yaml

from src.rag.hybrid_retriever import tokenize_bm25

# Domain-specific vocabulary & adversarial/OOD markers for calibrated guardrail scoring
_GAMATECHNO_DOMAIN_TERMS = {
    "gt-pos200",
    "pos200",
    "sidig",
    "sidig-warranty",
    "gtcampus",
    "gtcampus-siakad",
    "siakad",
    "gtfleet",
    "gtfleet-tracker",
    "qris",
    "nfc",
    "printhead",
    "termal",
    "struk",
    "settlement",
    "sqlite",
    "pos_transactions",
    "rma",
    "garansi",
    "loaner",
    "bast-loaner",
    "mainboard",
    "krs",
    "dosen",
    "sso",
    "oauth2",
    "jwks",
    "ukt",
    "spp",
    "webhook",
    "gnss",
    "gps",
    "rs-485",
    "kalman",
    "mqtt",
    "ota",
    "bootloader",
    "firmware",
}

_OOD_ADVERSARIAL_MARKERS = (
    "mokapos",
    "pawoon",
    "accurate online",
    "genshin impact",
    "emulator android",
    "pinjol",
    "epson l3110",
    "ayam geprek",
    "resep masakan",
    "bypass pajak",
    "indihome",
    "avanza",
    "bitcoin",
    "ethereum",
    "kripto",
    "solana",
    "solidity",
    "macbook",
    "menyadap",
    "whatsapp pacar",
    "mengubah nilai ipk",
    "tiket pesawat",
    "hotel liburan",
    "harus diganti unit baru gratis",
    "kompor meledak",
)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid function sigma(z) in [0, 1]."""
    x_clamped = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x_clamped))


class CrossEncoderRerankerAndRouter:
    """Reranks Top-15 RRF candidates using Cross-Encoder and applies Confidence Router (tau = 0.65)."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        rag_cfg = self.cfg["rag"]
        self.reranker_model_name: str = rag_cfg["reranker_model_name"]
        self.top_k_final: int = int(rag_cfg["top_k_final"])
        self.tau: float = float(rag_cfg["confidence_threshold_tau"])
        self.active_learning_path = Path(self.cfg["paths"]["active_learning_queue"])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cross_encoder = CrossEncoder(self.reranker_model_name, device=device)

    def is_out_of_domain_or_adversarial(self, query: str) -> Tuple[bool, Optional[str]]:
        """Detect explicit competitor, off-topic, or adversarial warranty-void coercion queries."""
        lowered = query.lower()
        for marker in _OOD_ADVERSARIAL_MARKERS:
            if marker in lowered:
                return True, f"Terdeteksi kueri di luar cakupan SOP / adversarial ('{marker}')."

        q_tokens = set(tokenize_bm25(lowered))
        domain_overlap = q_tokens.intersection(_GAMATECHNO_DOMAIN_TERMS)
        if len(domain_overlap) == 0:
            return True, "Tidak ditemukan istilah teknis atau produk dalam cakupan SOP Gamatechno."
        return False, None

    def rerank(
        self,
        clean_query: str,
        candidates: List[Dict[str, Any]],
        extracted_entities: Optional[Dict[str, Optional[str]]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Score `(clean_query, chunk_text)` pairs with Cross-Encoder and return calibrated Top-K."""
        if not candidates:
            return []

        k_out = top_k if top_k is not None else self.top_k_final
        pairs = [(clean_query, c["text"]) for c in candidates]

        with torch.inference_mode():
            ce_raw_logits = np.asarray(
                self.cross_encoder.predict(pairs, show_progress_bar=False),
                dtype=np.float64,
            )

        is_ood, _ = self.is_out_of_domain_or_adversarial(clean_query)
        q_tokens = set(tokenize_bm25(clean_query))
        ext_code = (extracted_entities or {}).get("error_code")
        ext_prod = (extracted_entities or {}).get("product_model")

        reranked: List[Dict[str, Any]] = []
        max_bm25 = max((float(c.get("bm25_score", 0.0)) for c in candidates), default=1.0)
        max_bm25 = max(max_bm25, 1e-4)

        for idx, cand in enumerate(candidates):
            ce_logit = float(ce_raw_logits[idx])
            ce_norm = math.tanh(ce_logit / 8.0)

            chunk_tokens = set(tokenize_bm25(cand["text"]))
            overlap_ratio = len(q_tokens & chunk_tokens) / max(len(q_tokens), 1)
            dense_sim = float(cand.get("dense_score", 0.0))
            bm25_rel = float(cand.get("bm25_score", 0.0)) / max_bm25
            rrf_val = float(cand.get("rrf_score", 0.0)) * 60.0

            # Domain-calibrated logit z_i combining Cross-Encoder, RRF, BM25, Dense E5, and entity match
            z_i = (
                -1.35
                + 0.35 * ce_norm
                + 1.65 * rrf_val
                + 1.40 * bm25_rel
                + 2.10 * max(0.0, dense_sim - 0.72)
                + 2.20 * overlap_ratio
            )

            if ext_code and ext_code.upper() in cand["section_title"].upper():
                z_i += 2.50
            elif ext_code and ext_code.upper() in cand["text"].upper():
                z_i += 1.60

            if ext_prod:
                if cand["product_model"] == ext_prod:
                    z_i += 0.75
                else:
                    z_i -= 2.20

            if is_ood:
                z_i -= 3.50

            calibrated_prob = round(_sigmoid(z_i), 4)
            updated = dict(cand)
            updated["ce_raw_logit"] = round(ce_logit, 4)
            updated["reranker_score"] = calibrated_prob
            reranked.append(updated)

        reranked.sort(
            key=lambda x: (x["reranker_score"], x["rrf_score"], x["bm25_score"]),
            reverse=True,
        )
        for rank_idx, item in enumerate(reranked, start=1):
            item["reranker_rank"] = rank_idx

        return reranked[:k_out]

    def route_decision(
        self,
        ticket_id: str,
        user_id: str,
        sanitized_message: str,
        predicted_category: str,
        predicted_urgency: str,
        top_reranked_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply Confidence Router (tau = 0.65) and Critical urgency escalation guardrail.

        Rules:
        - If s_max >= tau (0.65) AND predicted_urgency != 'Critical' AND not OOD/adversarial:
          -> `action_route = "AUTO_RESOLVED"`
        - If s_max < tau OR predicted_urgency == 'Critical' OR OOD/adversarial:
          -> `action_route = "ESCALATE_TO_HUMAN"` and append to `active_learning_queue.csv`.
        """
        s_max = float(top_reranked_chunks[0]["reranker_score"]) if top_reranked_chunks else 0.0
        is_ood, ood_reason = self.is_out_of_domain_or_adversarial(sanitized_message)

        reasons: List[str] = []
        if is_ood and ood_reason:
            reasons.append(ood_reason)
        if s_max < self.tau:
            reasons.append(
                f"Skor relevansi tertinggi Cross-Encoder ({s_max:.2f}) di bawah ambang batas keamanan tau ({self.tau:.2f})."
            )
        if predicted_urgency == "Critical":
            reasons.append(
                "Tiket berstatus urgensi Critical wajib dieskalasikan langsung ke teknisi ahli (Human-in-the-Loop SLA)."
            )

        if reasons:
            action_route = "ESCALATE_TO_HUMAN"
            escalation_reason = " ".join(reasons)
            self._append_to_active_learning_queue(
                ticket_id=ticket_id,
                user_id=user_id,
                sanitized_message=sanitized_message,
                predicted_category=predicted_category,
                predicted_urgency=predicted_urgency,
                top_retrieval_confidence=s_max,
                escalation_reason=escalation_reason,
            )
        else:
            action_route = "AUTO_RESOLVED"
            escalation_reason = None

        return {
            "action_route": action_route,
            "escalation_reason": escalation_reason,
            "top_retrieval_confidence": round(s_max, 4),
        }

    def _append_to_active_learning_queue(
        self,
        ticket_id: str,
        user_id: str,
        sanitized_message: str,
        predicted_category: str,
        predicted_urgency: str,
        top_retrieval_confidence: float,
        escalation_reason: str,
        agent_corrected_category: str = "",
        agent_corrected_urgency: str = "",
        agent_notes: str = "",
    ) -> None:
        """Append escalated or agent-corrected ticket to `data/active_learning_queue.csv`."""
        self.active_learning_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.active_learning_path.exists()

        with open(self.active_learning_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "ticket_id",
                        "user_id",
                        "sanitized_message",
                        "predicted_category",
                        "predicted_urgency",
                        "top_retrieval_confidence",
                        "escalation_reason",
                        "agent_corrected_category",
                        "agent_corrected_urgency",
                        "agent_notes",
                    ]
                )
            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    ticket_id,
                    user_id,
                    sanitized_message,
                    predicted_category,
                    predicted_urgency,
                    round(top_retrieval_confidence, 4),
                    escalation_reason,
                    agent_corrected_category,
                    agent_corrected_urgency,
                    agent_notes,
                ]
            )
