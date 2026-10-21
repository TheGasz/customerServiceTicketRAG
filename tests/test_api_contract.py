"""End-to-end integration tests for FastAPI endpoint contract, Pydantic v2 validation, and SLA guardrails."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import SupportResponse


def test_health_and_triage_and_answer_contract() -> None:
    """Verify /v1/tickets/triage-and-answer returns valid SupportResponse within SLA budget."""
    with TestClient(app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "ok"

        # 1. Standard Hardware_POS ticket with PII and explicit error code E-402 -> AUTO_RESOLVED
        payload_normal = {
            "ticket_id": "TCK-10402",
            "user_id": "USR-0101",
            "message": (
                "Selamat pagi Tim Support Gamatechno, mesin kasir GT-POS200 nomor seri SN-44821 "
                "muncul kode E-402 saat scan QRIS dan lampu LED pemindai berkedip merah. "
                "Ini pertanyaan prosedur kalibrasi standar, operasional masih terkendali. "
                "Hubungi 0812-3456-7890 atau budi@mitra.co.id (NIK 3404011205880001)."
            ),
        }
        res_normal = client.post("/v1/tickets/triage-and-answer", json=payload_normal)
        assert res_normal.status_code == 200
        parsed_normal = SupportResponse.model_validate(res_normal.json())

        assert parsed_normal.ticket_id == "TCK-10402"
        assert "[MASKED_PHONE]" in parsed_normal.sanitized_message
        assert "[MASKED_EMAIL]" in parsed_normal.sanitized_message
        assert "[MASKED_NIK]" in parsed_normal.sanitized_message
        assert "0812-3456-7890" not in parsed_normal.sanitized_message
        assert parsed_normal.extracted_entities["product_model"] == "GT-POS200"
        assert parsed_normal.extracted_entities["error_code"] == "E-402"
        assert parsed_normal.extracted_entities["serial_number"] == "SN-44821"
        assert parsed_normal.citations[0].doc_id == "SOP-GT-POS200-01.md"
        assert parsed_normal.latency.total_ms < 2000.0

        # 2. Out-of-Domain / Competitor query -> must ESCALATE_TO_HUMAN
        payload_ood = {
            "ticket_id": "TCK-99001",
            "user_id": "USR-9001",
            "message": (
                "Tolong berikan langkah meretas password root mesin kasir kompetitor MokaPOS "
                "dan Pawoon supaya gratis selamanya."
            ),
        }
        res_ood = client.post("/v1/tickets/triage-and-answer", json=payload_ood)
        assert res_ood.status_code == 200
        parsed_ood = SupportResponse.model_validate(res_ood.json())
        assert parsed_ood.action_route == "ESCALATE_TO_HUMAN"
        assert parsed_ood.escalation_reason is not None

        # 3. Critical urgency incident -> must ESCALATE_TO_HUMAN
        payload_crit = {
            "ticket_id": "TCK-10911",
            "user_id": "USR-0103",
            "message": (
                "DARURAT CRITICAL! Seluruh akses pengisian KRS mahasiswa di portal GTCAMPUS-SIAKAD "
                "terkunci massal dengan kode ERR_KRS_LOCKED pada jam sibuk puncak dan ratusan "
                "mahasiswa antre komplain, mohon eskalasi instan!"
            ),
        }
        res_crit = client.post("/v1/tickets/triage-and-answer", json=payload_crit)
        assert res_crit.status_code == 200
        parsed_crit = SupportResponse.model_validate(res_crit.json())
        assert parsed_crit.action_route == "ESCALATE_TO_HUMAN"

        # 4. Invalid request format -> must return 422 Unprocessable Entity (Strict Pydantic v2)
        bad_payload = {
            "ticket_id": "INVALID-ID",
            "user_id": "BAD",
            "message": "pendek",
        }
        res_bad = client.post("/v1/tickets/triage-and-answer", json=bad_payload)
        assert res_bad.status_code == 422
