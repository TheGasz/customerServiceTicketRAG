# Project Notes: Gamatechno AI Support Copilot

## Overview
- **Goal**: End-to-End Ticket Triage (IndoBERT PEFT/LoRA + Focal Loss + OOF Calibration) & Grounded Hybrid RAG Pipeline (BM25 + FAISS Multilingual-E5 + Weighted RRF + Cross-Encoder Reranker + Confidence Router).
- **Target**: PT Gamatechno Indonesia AI Support Copilot (v2.0 Production Blueprint).
- **Workflow Constraint**:
  1. Commit every stage/change to Git (and push to GitHub once remote is configured).
  2. When reaching the deep learning training step (`IndoBERT-LoRA` training), STOP and inform the user which script/notebook to run on Google Colab, and how to place the trained weights back.

## Architecture & Stages
1. **Tahap 1 (Fondasi & Data)**:
   - `configs/config.yaml`
   - `requirements.txt`
   - `data/generate_synthetic_data.py` -> generates 8 SOP docs in `data/kb_docs/` and 1,200 tickets (240 unique `user_id`: `USR-0001`..`USR-0240`) split via `GroupShuffleSplit`/`GroupKFold` into `train_tickets.csv` (840), `val_tickets.csv` (180), `test_tickets.csv` (180) with zero `user_id` overlap and disjoint template pools. Also generates 30 adversarial/OOD trap queries (`data/tickets/ood_trap_tickets.csv`).
   - `src/preprocessing/pii_masker.py` -> `mask_pii()` (`[MASKED_PHONE]`, `[MASKED_EMAIL]`, `[MASKED_NIK]`) and `extract_technical_entities()` (`product_model`, `error_code`, `serial_number`).
   - `src/preprocessing/splitter.py` -> group split utility verifying zero `user_id` overlap.
   - `tests/test_pii_and_entities.py` & `tests/test_no_leakage_split.py`.
2. **Tahap 2 (Pipeline Triase Multi-Task & Evaluasi)**:
   - `src/triage/baseline_tfidf.py` -> `sklearn.pipeline.Pipeline` (TF-IDF fit ONLY on `train_tickets.csv` + LightGBM/Classifier) for Category & Urgency.
   - `src/triage/lora_model.py` -> `indobenchmark/indobert-base-p1` + `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.10, target_modules=["query", "key", "value", "dense"])`, dual heads (`CategoryHead` 768->256->4 with CrossEntropy label_smoothing=0.05; `UrgencyHead` 768->256->4 with Multi-Class Focal Loss gamma=2.0), plus lightweight CPU fallback mode.
   - `src/triage/train_triage.py` + Colab notebook/script ready for user to train on Google Colab T4.
   - `src/triage/evaluate_triage.py` -> produces `reports/triage_benchmark.json`, `reports/confusion_matrix_triage.png`, and `reports/triage_slicing_report.md`.
3. **Tahap 3 (Pipeline Hybrid RAG, Reranker & Benchmark Ablasi)**:
   - `src/rag/chunker.py` -> Markdown heading chunker (`#`, `##`, `###`) with `product_model` and `doc_id` metadata.
   - `src/rag/hybrid_retriever.py` -> Metadata pre-filtering + BM25Okapi (Top-15) + FAISS Inner Product `intfloat/multilingual-e5-small` (Top-15) + dynamic Weighted RRF ($k=60$, $w_{lex}=0.55, w_{sem}=0.45$ with explicit code/SN, else $0.35, 0.65$).
   - `src/rag/reranker.py` -> Cross-Encoder reranker + calibrated probability $\sigma(z_i)$ + Confidence Router ($\tau = 0.65$ or `urgency == "Critical"` -> `ESCALATE_TO_HUMAN` + log to `data/active_learning_queue.csv`).
   - `src/rag/generator.py` -> Grounded response generator with strict citation `[doc_id: section_title]`.
   - `src/rag/evaluate_rag.py` -> produces `reports/rag_ablation_table.md` and `reports/guardrail_audit.json`.
4. **Tahap 4 (FastAPI, Streamlit Dual-View UI & Dokumentasi)**:
   - `src/api/schemas.py` -> Strict Pydantic v2 schemas (`TicketRequest`, `SourceCitation`, `LatencyBreakdown`, `SupportResponse`).
   - `src/api/main.py` -> FastAPI with `@asynccontextmanager async def lifespan(app: FastAPI)` and `POST /v1/tickets/triage-and-answer`.
   - `app_streamlit.py` -> 3-tab Streamlit UI.
   - `tests/test_api_contract.py` & `README.md`.
