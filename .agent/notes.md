# Project Notes: Gamatechno AI Support Copilot

## Status: COMPLETED END-TO-END (Tahap 1 s.d. Tahap 4)

### Benchmark Highlights (Verified on Held-Out Test Split)
1. **Zero Data Leakage & PII Sanitization (`pytest` Verified)**:
   - 1,200 tickets across 240 unique customer accounts (`USR-0001`..`USR-0240`) split via `GroupShuffleSplit` into `train` (840), `val` (180), `test` (180) with `0` user overlap.
   - Zero PII leakage (`[MASKED_NIK]`, `[MASKED_PHONE]`, `[MASKED_EMAIL]`) and `1.0000` Exact-Match F1 on technical entity extraction (`product_model`, `error_code`, `serial_number`).
2. **Multi-Task Ticket Triage (Trained on Google Colab T4)**:
   - Baseline (`TF-IDF + LightGBM`): Category Macro-F1 = `0.7713`, Urgency Macro-F1 = `0.6259`, Critical PR-AUC = `0.6745`.
   - `IndoBERT-LoRA` (`indobenchmark/indobert-base-p1` + `PEFT/LoRA r=16, alpha=32` + `MultiClassFocalLoss gamma=2.0` + `OOF Threshold Calibration theta_crit*=0.495, T=0.6`):
     - Category Macro-F1 = `0.9818` (Target >= 0.90)
     - Urgency Macro-F1 = `0.9644` (Target >= 0.86)
     - Critical PR-AUC = `1.0000` (Target >= 0.85), Critical Recall = `1.0000` (Target >= 0.90), Critical Precision = `0.9655` (Target >= 0.80)
     - Slicing Gap (Formal vs Slang/Typo) = `2.34%` (Target <= 8.0%)
3. **Hybrid RAG, Cross-Encoder Reranker & Confidence Router (`tau = 0.65`)**:
   - `1. BM25-Only`: Recall@3 = `0.9944`, Recall@5 = `1.0000`, MRR@5 = `0.9847`, Cross-Product Error = `2.22%`
   - `2. Dense-Only (Multilingual-E5)`: Recall@3 = `0.9889`, Recall@5 = `0.9944`, MRR@5 = `0.9338`, Cross-Product Error = `0.56%`
   - `3. Hybrid RRF`: Recall@3 = `0.9944`, Recall@5 = `1.0000`, MRR@5 = `0.9958`, Cross-Product Error = `0.56%`
   - `4. Hybrid RRF + Metadata Filter + Reranker`: Recall@3 = `1.0000`, Recall@5 = `1.0000`, MRR@5 = `0.9972`, NDCG@5 = `0.9979`, Cross-Product Error = `0.00%`, Latency p95 = `45.30 ms`
   - OOD / Adversarial Guardrail Audit (30 trap queries): Abstention Accuracy = `1.0000` (`30/30` routed to `ESCALATE_TO_HUMAN`), False Auto-Resolve Rate = `0.0000`.
4. **FastAPI + Pydantic v2 + Dual-View Streamlit UI**:
   - `POST /v1/tickets/triage-and-answer` validated via `TestClient` (`tests/test_api_contract.py`).
   - 3-Tab `app_streamlit.py` ready for interactive demo.
