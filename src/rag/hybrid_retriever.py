"""Metadata-filtered Hybrid Retriever combining BM25Okapi, FAISS Multilingual-E5, and Dynamic Weighted RRF."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import yaml

from src.rag.chunker import SOPChunk, load_and_chunk_kb_docs

_TECH_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{2,}")
_STANDARD_WORD_RE = re.compile(r"[A-Za-z]{4,}")


def tokenize_bm25(text: str, preserve_codes: bool = True) -> List[str]:
    """Tokenize Indonesian technical text.

    When `preserve_codes=True` (used in Hybrid RRF), preserves alphanumeric error codes
    (`E-402`, `ERR_VA_TIMEOUT`). When `preserve_codes=False` (naive lexical baseline),
    uses standard alphabetic word tokens.
    """
    pattern = _TECH_TOKEN_RE if preserve_codes else _STANDARD_WORD_RE
    return [tok.lower() for tok in pattern.findall(text)]


class HybridRAGRetriever:
    """Hybrid Lexical (BM25Okapi) + Dense Semantic (FAISS Inner Product + E5) Retriever with Weighted RRF."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        rag_cfg = self.cfg["rag"]
        self.dense_model_name: str = rag_cfg["dense_model_name"]
        self.embedding_dim: int = int(rag_cfg["embedding_dim"])
        self.top_k_bm25: int = int(rag_cfg["top_k_bm25"])
        self.top_k_dense: int = int(rag_cfg["top_k_dense"])
        self.rrf_k: int = int(rag_cfg["rrf_k"])
        self.weights_with_code: Dict[str, float] = rag_cfg["weights_with_code"]
        self.weights_semantic_only: Dict[str, float] = rag_cfg["weights_semantic_only"]

        self.chunks: List[SOPChunk] = load_and_chunk_kb_docs(self.cfg["paths"]["kb_docs_dir"])

        # Technical code-aware BM25 index for Hybrid RRF and standard word BM25 for lexical baseline
        self.tokenized_corpus_tech: List[List[str]] = [
            tokenize_bm25(c.lexical_text, preserve_codes=True) for c in self.chunks
        ]
        self.bm25_tech = BM25Okapi(self.tokenized_corpus_tech)

        self.tokenized_corpus_naive: List[List[str]] = [
            tokenize_bm25(c.subsection_title + " " + c.text.split("\n", 1)[-1], preserve_codes=False)
            for c in self.chunks
        ]
        self.bm25_naive = BM25Okapi(self.tokenized_corpus_naive)

        # Load Multilingual-E5 (384-d) and build L2-normalized FAISS IndexFlatIP
        self.embedder = SentenceTransformer(self.dense_model_name)
        passage_texts = [c.dense_passage_text for c in self.chunks]
        embeddings = self.embedder.encode(
            passage_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        self.chunk_embeddings = embeddings
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    def get_dynamic_rrf_weights(
        self, extracted_entities: Optional[Dict[str, Optional[str]]] = None
    ) -> Tuple[float, float]:
        """Return dynamic RRF weights (w_lex, w_sem) based on presence of explicit error code or serial number."""
        has_explicit = False
        if extracted_entities:
            if extracted_entities.get("error_code") or extracted_entities.get("serial_number"):
                has_explicit = True

        if has_explicit:
            return (
                float(self.weights_with_code["w_lex"]),
                float(self.weights_with_code["w_sem"]),
            )
        return (
            float(self.weights_semantic_only["w_lex"]),
            float(self.weights_semantic_only["w_sem"]),
        )

    def retrieve(
        self,
        query: str,
        extracted_entities: Optional[Dict[str, Optional[str]]] = None,
        mode: str = "hybrid_rrf_filtered",
        top_k: int = 15,
        deduplicate_by_section: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute retrieval using BM25, Dense FAISS, or Weighted RRF with optional metadata pre-filtering."""
        target_product = None
        if mode == "hybrid_rrf_filtered" and extracted_entities:
            target_product = extracted_entities.get("product_model")

        if target_product:
            eligible_indices = [
                i for i, c in enumerate(self.chunks) if c.product_model == target_product
            ]
            if not eligible_indices:
                eligible_indices = list(range(len(self.chunks)))
        else:
            eligible_indices = list(range(len(self.chunks)))

        eligible_set = set(eligible_indices)

        # 1. Lexical Path (BM25Okapi)
        bm25_ranks: Dict[int, int] = {}
        bm25_raw_scores = np.zeros(len(self.chunks), dtype=np.float64)
        if mode != "dense_only":
            if mode in ("bm25_only", "hybrid_rrf"):
                q_tok = tokenize_bm25(query, preserve_codes=False)
                bm25_raw_scores = np.asarray(self.bm25_naive.get_scores(q_tok), dtype=np.float64)
            else:
                q_tok = tokenize_bm25(query, preserve_codes=True)
                bm25_raw_scores = np.asarray(self.bm25_tech.get_scores(q_tok), dtype=np.float64)

            bm25_sorted = sorted(
                eligible_indices, key=lambda idx: float(bm25_raw_scores[idx]), reverse=True
            )[: self.top_k_bm25]
            bm25_ranks = {
                chunk_idx: rank_pos + 1 for rank_pos, chunk_idx in enumerate(bm25_sorted)
            }

        # 2. Semantic Path (FAISS Inner Product + Multilingual-E5)
        dense_ranks: Dict[int, int] = {}
        dense_sim_all = np.zeros(len(self.chunks), dtype=np.float32)
        if mode != "bm25_only":
            e5_query = f"query: {query}"
            q_emb = self.embedder.encode(
                [e5_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            dense_sim_all = np.dot(self.chunk_embeddings, q_emb[0])
            dense_sorted = sorted(
                eligible_indices, key=lambda idx: float(dense_sim_all[idx]), reverse=True
            )[: self.top_k_dense]
            dense_ranks = {
                chunk_idx: rank_pos + 1 for rank_pos, chunk_idx in enumerate(dense_sorted)
            }

        # 3. Combine Rankings via Weighted RRF
        if mode == "hybrid_rrf":
            w_lex, w_sem = 0.50, 0.50
        else:
            w_lex, w_sem = self.get_dynamic_rrf_weights(extracted_entities)
        candidate_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        if not candidate_indices:
            candidate_indices = eligible_set

        scored_candidates: List[Dict[str, Any]] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            r_bm25 = bm25_ranks.get(idx, 999)
            r_dense = dense_ranks.get(idx, 999)
            s_bm25 = float(bm25_raw_scores[idx])
            s_dense = float(dense_sim_all[idx])

            if mode == "bm25_only":
                primary_score = s_bm25
            elif mode == "dense_only":
                primary_score = s_dense
            else:
                lex_term = (w_lex / (self.rrf_k + r_bm25)) if idx in bm25_ranks else 0.0
                sem_term = (w_sem / (self.rrf_k + r_dense)) if idx in dense_ranks else 0.0
                primary_score = lex_term + sem_term

            scored_candidates.append(
                {
                    "chunk_index": idx,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "product_model": chunk.product_model,
                    "category": chunk.category,
                    "section_title": chunk.section_title,
                    "subsection_title": chunk.subsection_title,
                    "text": chunk.text,
                    "bm25_score": round(s_bm25, 4),
                    "bm25_rank": r_bm25,
                    "dense_score": round(s_dense, 4),
                    "dense_rank": r_dense,
                    "rrf_score": round(float(primary_score), 6),
                    "w_lex": w_lex,
                    "w_sem": w_sem,
                }
            )

        scored_candidates.sort(
            key=lambda x: (x["rrf_score"], x["bm25_score"], x["dense_score"]),
            reverse=True,
        )

        if not deduplicate_by_section:
            return scored_candidates[:top_k]

        deduped: List[Dict[str, Any]] = []
        seen_sections = set()
        for item in scored_candidates:
            sec_key = (item["doc_id"], item["section_title"])
            if sec_key not in seen_sections:
                seen_sections.add(sec_key)
                deduped.append(item)
                if len(deduped) >= top_k:
                    break
        return deduped
