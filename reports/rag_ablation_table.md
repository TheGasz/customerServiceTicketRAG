# Studi Ablasi Retrieval Hybrid RAG & Evaluasi Latency

| Konfigurasi Retrieval | Recall@3 | Recall@5 | MRR@5 | NDCG@5 | Cross-Product Error | Latency p50 (ms) | Latency p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BM25-Only (Lexical)** | 0.9944 | 1.0000 | 0.9847 | 0.9886 | 2.22% | 0.60 | 0.79 |
| **2. Dense-Only (Multilingual-E5)** | 0.9889 | 0.9944 | 0.9338 | 0.9493 | 0.56% | 10.02 | 16.60 |
| **3. Hybrid RRF (BM25 + Dense)** | 0.9944 | 1.0000 | 0.9958 | 0.9968 | 0.56% | 10.77 | 12.68 |
| **4. Hybrid RRF + Metadata Filter + Reranker** | 1.0000 | 1.0000 | 0.9972 | 0.9979 | 0.00% | 42.01 | 45.30 |

## Metrik Grounded Generation & Guardrail Anti-Halusinasi

- **Faithfulness Score (Top-3 Groundedness)**: `1.0000` (Target >= 0.95)
- **Strict Citation Precision (`[doc_id: section_title]`)**: `1.0000` (Target >= 0.92)
- **OOD / Adversarial Abstention Accuracy (30 Tiket Jebakan)**: `1.0000` (30/30 ditahan ke `ESCALATE_TO_HUMAN`, Target >= 0.90)
- **False Auto-Resolve Rate**: `0.0000` (Target <= 0.10)
