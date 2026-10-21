"""Grounded response generator enforcing strict SOP citations and abstention guardrails."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class GroundedRAGGenerator:
    """Synthesizes grounded customer support responses strictly anchored to Top-3 SOP chunks."""

    def __init__(self, tau: float = 0.65) -> None:
        self.tau = tau

    def generate(
        self,
        ticket_id: str,
        category: str,
        urgency: str,
        extracted_entities: Dict[str, Optional[str]],
        action_route: str,
        escalation_reason: Optional[str],
        top_chunks: List[Dict[str, Any]],
        top_retrieval_confidence: float,
    ) -> Dict[str, str]:
        """Generate `suggested_reply` and `internal_triage_notes` with strict `[doc_id: section_title]` citations."""
        prod = extracted_entities.get("product_model") or "Tidak Spesifik"
        err = extracted_entities.get("error_code") or "Tidak Disebutkan"
        sn = extracted_entities.get("serial_number") or "Tidak Disebutkan"

        internal_notes = (
            f"[TRIASE OTOMATIS] Tiket={ticket_id} | Divisi={category} | Urgensi={urgency} | "
            f"Entitas=(Produk: {prod}, Kode: {err}, SN: {sn}) | "
            f"TopRerankerConf={top_retrieval_confidence:.4f} | Route={action_route}."
        )
        if escalation_reason:
            internal_notes += f" Alasan Eskalasi: {escalation_reason}"

        # Case 1: Low retrieval confidence or Out-of-Domain -> strict abstention (refuse to hallucinate)
        if action_route == "ESCALATE_TO_HUMAN" and top_retrieval_confidence < self.tau:
            suggested_reply = (
                f"Halo Pelanggan Yth., terima kasih telah menghubungi Layanan Bantuan Teknis Enterprise "
                f"(Tiket `{ticket_id}`). Berdasarkan pemeriksaan sistem keamanan pengetahuan kami, permintaan atau "
                f"kendala yang Anda sampaikan berada di luar cakupan prosedur operasi standar (SOP) otomatis kami "
                f"(Confidence Score: {top_retrieval_confidence:.2f} < {self.tau:.2f}). Demi menghindari informasi teknis "
                f"yang tidak akurat, tiket Anda telah kami eskalasikan secara langsung ke **Tim Teknisi Manusia / "
                f"Spesialis Divisi {category}** untuk peninjauan lebih lanjut."
            )
            return {
                "suggested_reply": suggested_reply,
                "internal_triage_notes": internal_notes,
            }

        # Case 2: Critical urgency with valid SOP reference -> escalate + provide verified initial SOP guidance
        primary_chunk = top_chunks[0]
        primary_citation = f"[{primary_chunk['doc_id']}: {primary_chunk['section_title']}]"
        extracted_steps = self._extract_clean_procedure_lines(primary_chunk["text"])

        if action_route == "ESCALATE_TO_HUMAN" and urgency == "Critical":
            secondary_citations = [
                f"[{c['doc_id']}: {c['section_title']}]" for c in top_chunks[1:3]
            ]
            sec_ref_str = (
                f" Rujukan pendukung: {', '.join(secondary_citations)}."
                if secondary_citations
                else ""
            )
            suggested_reply = (
                f"**[PRIORITAS DARURAT / CRITICAL — ESKALASI TEKNISI AHLI]**\n"
                f"Tiket `{ticket_id}` ({prod} / Kode `{err}`) telah masuk ke jalur **Fast-Track Escalation** "
                f"Divisi **{category}** dan teknisi on-duty sedang mengambil alih penanganan.\n\n"
                f"Sambil menunggu intervensi langsung teknisi kami, berikut panduan pemeriksaan awal resmi "
                f"berdasarkan dokumen SOP {primary_citation}:\n"
                f"{extracted_steps}\n\n"
                f"Seluruh langkah di atas bersumber langsung dari {primary_citation}.{sec_ref_str}"
            )
            return {
                "suggested_reply": suggested_reply,
                "internal_triage_notes": internal_notes,
            }

        # Case 3: AUTO_RESOLVED -> grounded step-by-step SOP resolution with strict citations
        supporting_refs = [
            f"[{c['doc_id']}: {c['section_title']}]" for c in top_chunks[:3]
        ]
        unique_refs = list(dict.fromkeys(supporting_refs))
        refs_joined = ", ".join(unique_refs)

        suggested_reply = (
            f"Halo Pelanggan Yth., berikut adalah panduan penyelesaian resmi untuk kendala perangkat/sistem "
            f"**{prod}** (Kode Indikasi: `{err}`, Nomor Seri: `{sn}`) berdasarkan prosedur standar "
            f"{primary_citation}:\n\n"
            f"{extracted_steps}\n\n"
            f"**Catatan Kepatuhan & Rujukan SOP:** Prosedur ini ditambatkan secara ketat pada dokumen resmi "
            f"{refs_joined}. Apabila kendala masih berlanjut setelah langkah di atas, silakan balas tiket "
            f"`{ticket_id}` ini untuk eskalasi ke teknisi Divisi `{category}`."
        )
        return {
            "suggested_reply": suggested_reply,
            "internal_triage_notes": internal_notes,
        }

    @staticmethod
    def _extract_clean_procedure_lines(chunk_text: str) -> str:
        """Extract numbered procedure steps or key body paragraphs from a chunk."""
        lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip()]
        body_lines = [
            ln
            for ln in lines
            if not ln.startswith("[SOP-") and not ln.startswith("## ")
        ]
        if not body_lines:
            return chunk_text.strip()
        return "\n".join(body_lines[:8])
