"""Dual-View 3-Tab Streamlit Interface for Gamatechno AI Support Copilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.api.main import execute_five_stage_pipeline, load_pipeline_components
from src.api.schemas import TicketRequest

st.set_page_config(
    page_title="Gamatechno AI Support Copilot",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model IndoBERT-LoRA, Indeks BM25/FAISS & Cross-Encoder Reranker...")
def get_cached_pipeline() -> Dict[str, Any]:
    """Load and cache all pipeline models in Streamlit session resource."""
    return load_pipeline_components("configs/config.yaml")


PRESET_SCENARIOS: Dict[str, Dict[str, str]] = {
    "Kasus 1: Error E-402 QRIS Kasir GT-POS200": {
        "ticket_id": "TCK-10402",
        "user_id": "USR-0101",
        "message": (
            "Halo Tim Support Gamatechno, mesin kasir GT-POS200 nomor seri SN-44821 di cabang "
            "kami gagal inisialisasi modul scanner QRIS dan NFC dengan kode error E-402. Lampu LED "
            "pemindai kedip merah 3 kali. Mohon panduan kalibrasinya hari ini. Kontak PIC kasir: "
            "0812-3456-7890 atau kasir.utama@tokomitra.co.id (NIK 3404011205880001)."
        ),
    },
    "Kasus 2: Klaim Garansi Unit Terkena Air RMA-12": {
        "ticket_id": "TCK-10412",
        "user_id": "USR-0102",
        "message": (
            "Siang admin SIDIG, perangkat kami dengan nomor seri SN-77319 ketumpahan air kopi dan "
            "stiker segel hologram garansi sempat robek waktu dibuka teknisi toko, di portal muncul "
            "status RMA-12. Bagaimana ketentuan klaim garansi dan prosedur perbaikannya? Hubungi WA "
            "6285711223344."
        ),
    },
    "Kasus 3: Insiden Kritis Server SIAKAD Down Saat KRS": {
        "ticket_id": "TCK-10911",
        "user_id": "USR-0103",
        "message": (
            "DARURAT CRITICAL! Seluruh akses pengisian KRS mahasiswa di portal GTCAMPUS-SIAKAD "
            "terkunci massal dengan kode ERR_KRS_LOCKED pada jam puncak batas akhir revisi KRS! "
            "Ratusan mahasiswa dan dosen wali komplain serentak, mohon eskalasi teknisi DBA sekarang "
            "juga! Kontak admin akademik: +6281398765432, email admin.siakad@univnegeri.ac.id."
        ),
    },
    "Kasus 4: Pertanyaan di Luar SOP / Out-of-Domain": {
        "ticket_id": "TCK-99001",
        "user_id": "USR-9001",
        "message": (
            "Tolong berikan langkah meretas password root mesin kasir kompetitor MokaPOS dan Pawoon "
            "serta cara install game Genshin Impact di layar kasir agar gratis selamanya."
        ),
    },
}


def main() -> None:
    """Render the 3-tab Gamatechno AI Support Copilot Streamlit application."""
    state = get_cached_pipeline()

    st.title("🛡️ Gamatechno AI Support Copilot")
    st.caption(
        "End-to-End Multi-Task Ticket Triage (IndoBERT PEFT/LoRA + Focal Loss + OOF Calibration) "
        "& Metadata-Filtered Grounded Hybrid RAG Pipeline"
    )

    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "Kasus 1: Error E-402 QRIS Kasir GT-POS200"
        preset = PRESET_SCENARIOS[st.session_state["selected_preset"]]
        st.session_state["ticket_id"] = preset["ticket_id"]
        st.session_state["user_id"] = preset["user_id"]
        st.session_state["message"] = preset["message"]

    tab1, tab2, tab3 = st.tabs(
        [
            "📨 Tab 1 — Portal Bantuan Pelanggan (Customer Simulator)",
            "🧑‍💻 Tab 2 — Dashboard Agen CS (AI Support Copilot View)",
            "📊 Tab 3 — Ruang Evaluasi & Benchmark Model (Ablation Explorer)",
        ]
    )

    with tab1:
        st.subheader("Simulasi Pengiriman Tiket Keluhan Pelanggan")
        st.write("Pilih salah satu skenario prasetel di bawah ini atau ketik keluhan bebas:")

        cols = st.columns(4)
        for idx, (sc_name, sc_data) in enumerate(PRESET_SCENARIOS.items()):
            if cols[idx].button(sc_name, use_container_width=True):
                st.session_state["selected_preset"] = sc_name
                st.session_state["ticket_id"] = sc_data["ticket_id"]
                st.session_state["user_id"] = sc_data["user_id"]
                st.session_state["message"] = sc_data["message"]

        col_id1, col_id2 = st.columns(2)
        ticket_id_in = col_id1.text_input("Ticket ID (`^TCK-\\d{4,8}$`)", key="ticket_id")
        user_id_in = col_id2.text_input("User ID (`^USR-\\d{4,6}$`)", key="user_id")
        message_in = st.text_area(
            "Pesan Keluhan Mentah Pelanggan (Boleh memuat slang, typo, kode error, nomor seri, dan PII)",
            key="message",
            height=130,
        )

        if st.button("🚀 Kirim & Proses Tiket (End-to-End Pipeline)", type="primary"):
            try:
                req = TicketRequest(
                    ticket_id=ticket_id_in.strip(),
                    user_id=user_id_in.strip(),
                    message=message_in.strip(),
                )
                resp, diag = execute_five_stage_pipeline(req, state=state)
                st.session_state["last_response"] = resp.model_dump()
                st.session_state["last_diagnostics"] = diag
                st.session_state["last_user_id"] = req.user_id
                st.success(
                    f"Tiket `{resp.ticket_id}` berhasil diproses dalam {resp.latency.total_ms:.2f} ms!"
                )
            except Exception as err:
                st.error(f"Validasi Skema TicketRequest Gagal: {err}")

        if "last_response" in st.session_state:
            resp_data = st.session_state["last_response"]
            st.divider()
            route_badge = (
                "🟢 AUTO_RESOLVED (Dijawab Otomatis Berbasis SOP)"
                if resp_data["action_route"] == "AUTO_RESOLVED"
                else "🔴 ESCALATE_TO_HUMAN (Dieskalasikan ke Teknisi Ahli)"
            )
            st.markdown(f"### Status Penanganan: **{route_badge}**")
            if resp_data.get("escalation_reason"):
                st.warning(f"**Alasan Eskalasi / Guardrail:** {resp_data['escalation_reason']}")
            st.info(resp_data["suggested_reply"])

    with tab2:
        st.subheader("AI Support Copilot — Transparansi Triase, Entitas & Hybrid RAG")
        if "last_response" not in st.session_state:
            st.info("Silakan kirimkan tiket terlebih dahulu pada **Tab 1** untuk melihat analisis Agen CS.")
        else:
            resp_data = st.session_state["last_response"]
            diag = st.session_state["last_diagnostics"]

            st.markdown("#### 1. Sanitasi Pelindungan Data Pribadi (PII Masking)")
            st.code(resp_data["sanitized_message"], language="text")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Prediksi Kategori Divisi",
                    resp_data["category"],
                    f"Confidence: {resp_data['category_confidence'] * 100:.1f}%",
                )
                cat_prob_df = pd.DataFrame(
                    list(diag["category_probs"].items()),
                    columns=["Category", "Probability"],
                )
                st.dataframe(cat_prob_df, hide_index=True, use_container_width=True)

            with c2:
                st.metric(
                    "Prediksi Tingkat Urgensi",
                    resp_data["urgency"],
                    f"Confidence: {resp_data['urgency_confidence'] * 100:.1f}%",
                )
                urg_prob_df = pd.DataFrame(
                    list(diag["urgency_probs"].items()),
                    columns=["Urgency", "Probability"],
                )
                st.dataframe(urg_prob_df, hide_index=True, use_container_width=True)

            with c3:
                st.metric(
                    "Keputusan Confidence Router",
                    resp_data["action_route"],
                    f"Top Reranker: {resp_data['top_retrieval_confidence']:.3f} (tau=0.65)",
                )
                ent_df = pd.DataFrame(
                    [
                        {"Entitas Teknis": k, "Nilai Diekstrak": v or "—"}
                        for k, v in resp_data["extracted_entities"].items()
                    ]
                )
                st.dataframe(ent_df, hide_index=True, use_container_width=True)

            st.markdown("#### 2. Perbandingan Peringkat Dokumen: BM25 vs Dense E5 vs Cross-Encoder Reranker")
            rank_rows = []
            for item in diag.get("reranked_candidates", []):
                rank_rows.append(
                    {
                        "Reranker Rank": item.get("reranker_rank"),
                        "Doc ID": item.get("doc_id"),
                        "Product": item.get("product_model"),
                        "Section Title": item.get("section_title"),
                        "BM25 Rank": item.get("bm25_rank"),
                        "BM25 Score": item.get("bm25_score"),
                        "Dense Rank": item.get("dense_rank"),
                        "Dense Cosine": item.get("dense_score"),
                        "Weighted RRF": item.get("rrf_score"),
                        "Calibrated Reranker Score": item.get("reranker_score"),
                    }
                )
            st.dataframe(pd.DataFrame(rank_rows), hide_index=True, use_container_width=True)

            st.markdown("#### 3. Kutipan Asli Pasal SOP (Strict Grounding Citations)")
            for cite in resp_data.get("citations", []):
                with st.expander(
                    f"📄 [{cite['doc_id']}: {cite['section_title']}] — Reranker Score: {cite['reranker_score']:.4f}"
                ):
                    st.write(cite["snippet"])

            st.markdown("#### 4. Umpan Balik Agen CS (Human-in-the-Loop Active Learning Queue)")
            with st.form("agent_feedback_form"):
                f_col1, f_col2 = st.columns(2)
                corr_cat = f_col1.selectbox(
                    "Koreksi Kategori Divisi",
                    ["Hardware_POS", "Billing_Payment", "Academic_Portal", "Warranty_Claim"],
                    index=["Hardware_POS", "Billing_Payment", "Academic_Portal", "Warranty_Claim"].index(
                        resp_data["category"]
                    ),
                )
                corr_urg = f_col2.selectbox(
                    "Koreksi Tingkat Urgensi",
                    ["Low", "Medium", "High", "Critical"],
                    index=["Low", "Medium", "High", "Critical"].index(resp_data["urgency"]),
                )
                agent_notes = st.text_input(
                    "Catatan Koreksi Teknisi / Agen CS",
                    value="Verifikasi manual agen CS selesai.",
                )
                submitted = st.form_submit_button("💾 Simpan Koreksi ke active_learning_queue.csv")
                if submitted:
                    state["reranker"]._append_to_active_learning_queue(
                        ticket_id=resp_data["ticket_id"],
                        user_id=st.session_state.get("last_user_id", "USR-0001"),
                        sanitized_message=resp_data["sanitized_message"],
                        predicted_category=resp_data["category"],
                        predicted_urgency=resp_data["urgency"],
                        top_retrieval_confidence=float(resp_data["top_retrieval_confidence"]),
                        escalation_reason=resp_data.get("escalation_reason") or "MANUAL_AGENT_FEEDBACK",
                        agent_corrected_category=corr_cat,
                        agent_corrected_urgency=corr_urg,
                        agent_notes=agent_notes,
                    )
                    st.success("Umpan balik berhasil disimpan ke `data/active_learning_queue.csv`!")

    with tab3:
        st.subheader("Evaluasi Ilmiah Offline & Performa Sistem Online")

        triage_bench_path = Path("reports/triage_benchmark.json")
        guardrail_path = Path("reports/guardrail_audit.json")
        cm_img_path = Path("reports/confusion_matrix_triage.png")

        if triage_bench_path.exists():
            t_bench = json.loads(triage_bench_path.read_text(encoding="utf-8"))
            st.markdown("#### 1. Benchmark Triase Multi-Task: Baseline TF-IDF+LightGBM vs IndoBERT-LoRA")
            st.dataframe(pd.DataFrame(t_bench["models"]).T, use_container_width=True)

            col_s1, col_s2 = st.columns([1.2, 1.0])
            with col_s1:
                st.markdown("##### Slicing Error Analysis Berdasarkan Karakteristik Tiket")
                st.dataframe(pd.DataFrame(t_bench["slicing_analysis"]).T, use_container_width=True)
            with col_s2:
                if cm_img_path.exists():
                    st.markdown("##### Confusion Matrix (Test Split)")
                    st.image(str(cm_img_path), use_container_width=True)

        if guardrail_path.exists():
            g_audit = json.loads(guardrail_path.read_text(encoding="utf-8"))
            st.markdown("#### 2. Studi Ablasi Retrieval Hybrid RAG & Guardrail Keamanan")
            st.dataframe(pd.DataFrame(g_audit["ablation_summary"]).T, use_container_width=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness Score", f"{g_audit['faithfulness_score']:.4f}", "Target >= 0.95")
            m2.metric("Citation Precision", f"{g_audit['citation_precision']:.4f}", "Target >= 0.92")
            m3.metric("Abstention Accuracy (OOD)", f"{g_audit['abstention_accuracy']:.4f}", "Target >= 0.90")
            m4.metric("False Auto-Resolve Rate", f"{g_audit['false_auto_resolve_rate']:.4f}", "Target <= 0.10")

        if "last_response" in st.session_state:
            st.markdown("#### 3. Rincian Waktu Eksekusi Terakhir (Runtime Latency Breakdown)")
            lat_df = pd.DataFrame([st.session_state["last_response"]["latency"]])
            st.dataframe(lat_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
