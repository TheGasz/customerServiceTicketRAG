"""FastAPI production microservice integrating the 5-stage Triage & Grounded Hybrid RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import Any, AsyncIterator, Dict, Tuple

from fastapi import FastAPI, HTTPException
import yaml

from src.api.schemas import (
    LatencyBreakdown,
    SourceCitation,
    SupportResponse,
    TicketRequest,
)
from src.preprocessing.pii_masker import extract_technical_entities, mask_pii
from src.rag.generator import GroundedRAGGenerator
from src.rag.hybrid_retriever import HybridRAGRetriever
from src.rag.reranker import CrossEncoderRerankerAndRouter
from src.triage.evaluate_triage import TriageRuntimePredictor

_PIPELINE_STATE: Dict[str, Any] = {}


def load_pipeline_components(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Initialize and cache all ML models and RAG indexes in memory once."""
    if _PIPELINE_STATE.get("initialized"):
        return _PIPELINE_STATE

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    triage_engine = TriageRuntimePredictor(config_path=config_path)
    retriever = HybridRAGRetriever(config_path=config_path)
    reranker = CrossEncoderRerankerAndRouter(config_path=config_path)
    generator = GroundedRAGGenerator(
        tau=float(cfg["rag"]["confidence_threshold_tau"])
    )

    _PIPELINE_STATE.update(
        {
            "initialized": True,
            "cfg": cfg,
            "triage_engine": triage_engine,
            "retriever": retriever,
            "reranker": reranker,
            "generator": generator,
        }
    )
    return _PIPELINE_STATE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load IndoBERT-LoRA triage model, BM25/FAISS indexes, and Cross-Encoder once at startup."""
    load_pipeline_components("configs/config.yaml")
    yield
    _PIPELINE_STATE.clear()


app = FastAPI(
    title="Enterprise AI Support Copilot API",
    description=(
        "End-to-End Multi-Task Ticket Triage (IndoBERT PEFT/LoRA + Focal Loss + OOF Calibration) "
        "& Metadata-Filtered Grounded Hybrid RAG Service."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def execute_five_stage_pipeline(
    request: TicketRequest,
    state: Dict[str, Any] | None = None,
) -> Tuple[SupportResponse, Dict[str, Any]]:
    """Run all 5 pipeline stages and return both `SupportResponse` and diagnostic metadata."""
    active_state = state if state and state.get("initialized") else load_pipeline_components()
    cfg = active_state["cfg"]
    triage_engine: TriageRuntimePredictor = active_state["triage_engine"]
    retriever: HybridRAGRetriever = active_state["retriever"]
    reranker: CrossEncoderRerankerAndRouter = active_state["reranker"]
    generator: GroundedRAGGenerator = active_state["generator"]

    t_total_start = time.perf_counter()

    # Stage 1 & 2a: PII Sanitization & Hybrid Technical Entity Extraction
    t0 = time.perf_counter()
    sanitized_message = mask_pii(request.message)
    extracted_entities = extract_technical_entities(
        sanitized_message, infer_product_from_context=True
    )
    pii_and_ner_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 2b: Multi-Task Triage Inference (IndoBERT-LoRA + OOF Calibration)
    t1 = time.perf_counter()
    triage_out = triage_engine.predict_single(sanitized_message)
    triage_inference_ms = (time.perf_counter() - t1) * 1000.0

    # Stage 3: Metadata Pre-Filtering & Hybrid RRF Retrieval (BM25 + Dense E5)
    t2 = time.perf_counter()
    rrf_candidates = retriever.retrieve(
        query=sanitized_message,
        extracted_entities=extracted_entities,
        mode="hybrid_rrf_filtered",
        top_k=int(cfg["rag"]["top_k_rerank_input"]),
        deduplicate_by_section=True,
    )
    hybrid_retrieval_ms = (time.perf_counter() - t2) * 1000.0

    # Stage 4: Cross-Encoder Reranking & Confidence-Based Abstention Router
    t3 = time.perf_counter()
    top_reranked = reranker.rerank(
        clean_query=sanitized_message,
        candidates=rrf_candidates,
        extracted_entities=extracted_entities,
        top_k=int(cfg["rag"]["top_k_final"]),
    )
    route_out = reranker.route_decision(
        ticket_id=request.ticket_id,
        user_id=request.user_id,
        sanitized_message=sanitized_message,
        predicted_category=triage_out["category"],
        predicted_urgency=triage_out["urgency"],
        top_reranked_chunks=top_reranked,
    )
    reranker_ms = (time.perf_counter() - t3) * 1000.0

    # Stage 5: Grounded Response Generation & Strict Pydantic v2 Validation
    t4 = time.perf_counter()
    gen_out = generator.generate(
        ticket_id=request.ticket_id,
        category=triage_out["category"],
        urgency=triage_out["urgency"],
        extracted_entities=extracted_entities,
        action_route=route_out["action_route"],
        escalation_reason=route_out["escalation_reason"],
        top_chunks=top_reranked,
        top_retrieval_confidence=route_out["top_retrieval_confidence"],
    )
    generation_ms = (time.perf_counter() - t4) * 1000.0
    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    citations = [
        SourceCitation(
            doc_id=c["doc_id"],
            product_model=c["product_model"],
            section_title=c["section_title"],
            reranker_score=float(c["reranker_score"]),
            snippet=c["text"][:450],
        )
        for c in top_reranked
    ]

    latency = LatencyBreakdown(
        pii_and_ner_ms=round(pii_and_ner_ms, 2),
        triage_inference_ms=round(triage_inference_ms, 2),
        hybrid_retrieval_ms=round(hybrid_retrieval_ms, 2),
        reranker_ms=round(reranker_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
    )

    response_obj = SupportResponse.model_validate(
        {
            "ticket_id": request.ticket_id,
            "sanitized_message": sanitized_message,
            "category": triage_out["category"],
            "category_confidence": triage_out["category_confidence"],
            "urgency": triage_out["urgency"],
            "urgency_confidence": triage_out["urgency_confidence"],
            "extracted_entities": extracted_entities,
            "action_route": route_out["action_route"],
            "escalation_reason": route_out["escalation_reason"],
            "suggested_reply": gen_out["suggested_reply"],
            "internal_triage_notes": gen_out["internal_triage_notes"],
            "citations": citations,
            "top_retrieval_confidence": route_out["top_retrieval_confidence"],
            "latency": latency,
        }
    )

    diagnostics = {
        "category_probs": triage_out["category_probs"],
        "urgency_probs": triage_out["urgency_probs"],
        "model_mode": triage_out["model_mode"],
        "reranked_candidates": top_reranked,
        "all_rrf_candidates": rrf_candidates,
    }
    return response_obj, diagnostics


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Return service health and loaded model status."""
    state = load_pipeline_components()
    return {
        "status": "ok",
        "triage_model_mode": state["triage_engine"].mode,
        "sop_chunks_indexed": len(state["retriever"].chunks),
    }


@app.post("/v1/tickets/triage-and-answer", response_model=SupportResponse)
async def triage_and_answer_endpoint(payload: TicketRequest) -> SupportResponse:
    """Process an incoming support ticket through the 5-stage AI Support Copilot pipeline."""
    try:
        response_obj, _ = execute_five_stage_pipeline(payload, state=_PIPELINE_STATE)
        return response_obj
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Internal pipeline processing error: {exc}"
        ) from exc
