"""Strict Pydantic v2 data schemas for Enterprise AI Support Copilot API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    """Incoming customer support ticket request payload."""

    ticket_id: str = Field(
        ..., pattern=r"^TCK-\d{4,8}$", description="ID unik tiket masuk"
    )
    user_id: str = Field(
        ..., pattern=r"^USR-\d{4,6}$", description="ID pelanggan untuk pelacakan riwayat"
    )
    message: str = Field(
        ...,
        min_length=8,
        max_length=2000,
        description="Pesan keluhan mentah pelanggan",
    )


class SourceCitation(BaseModel):
    """Grounded SOP citation metadata and procedure snippet."""

    doc_id: str = Field(..., description="Nama berkas dokumen SOP rujukan")
    product_model: str = Field(..., description="Kode produk terkait dokumen SOP")
    section_title: str = Field(
        ..., description="Judul bab/sub-bab spesifik di dalam SOP"
    )
    reranker_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Skor relevansi terkalibrasi Cross-Encoder",
    )
    snippet: str = Field(
        ..., description="Kutipan teks prosedur dari potongan dokumen"
    )


class LatencyBreakdown(BaseModel):
    """Per-stage execution time breakdown in milliseconds."""

    pii_and_ner_ms: float
    triage_inference_ms: float
    hybrid_retrieval_ms: float
    reranker_ms: float
    generation_ms: float
    total_ms: float


class SupportResponse(BaseModel):
    """Validated end-to-end triage, entity extraction, and grounded RAG response."""

    ticket_id: str
    sanitized_message: str
    category: Literal[
        "Hardware_POS", "Billing_Payment", "Academic_Portal", "Warranty_Claim"
    ]
    category_confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: Literal["Low", "Medium", "High", "Critical"]
    urgency_confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_entities: Dict[str, Optional[str]]
    action_route: Literal["AUTO_RESOLVED", "ESCALATE_TO_HUMAN"]
    escalation_reason: Optional[str] = None
    suggested_reply: str
    internal_triage_notes: str
    citations: List[SourceCitation]
    top_retrieval_confidence: float = Field(..., ge=0.0, le=1.0)
    latency: LatencyBreakdown
