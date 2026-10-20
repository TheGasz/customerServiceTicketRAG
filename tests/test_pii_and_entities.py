"""Unit tests for PII masking and hybrid technical entity extraction."""

from __future__ import annotations

import re

import pandas as pd

from src.preprocessing.pii_masker import extract_technical_entities, mask_pii


def test_mask_pii_phone_email_and_nik() -> None:
    """Verify phone numbers (+628, 628, 08), emails, and 16-digit NIK are masked."""
    raw_text = (
        "Halo min, mesin GT-POS200 SN-12345 error E-402. Hubungi saya di "
        "+62812-3456-7890 atau 085711223344, email teknisi.toko@mitra.co.id, "
        "dan NIK pemilik 3404011205880001 segera ya."
    )
    sanitized = mask_pii(raw_text)

    assert "[MASKED_PHONE]" in sanitized
    assert "[MASKED_EMAIL]" in sanitized
    assert "[MASKED_NIK]" in sanitized
    assert "+62812-3456-7890" not in sanitized
    assert "085711223344" not in sanitized
    assert "teknisi.toko@mitra.co.id" not in sanitized
    assert "3404011205880001" not in sanitized
    # Ensure technical entities remain untouched
    assert "GT-POS200" in sanitized
    assert "SN-12345" in sanitized
    assert "E-402" in sanitized


def test_zero_pii_leakage_on_generated_tickets() -> None:
    """Ensure sanitized_message across train, val, and test splits contains zero PII."""
    nik_re = re.compile(r"\b\d{16}\b")
    email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    phone_re = re.compile(r"(?<!\d)(?:\+62|62|0)8\d{1,3}(?:[\s-]?\d{3,4}){1,3}(?!\d)")

    for split_path in [
        "data/tickets/train_tickets.csv",
        "data/tickets/val_tickets.csv",
        "data/tickets/test_tickets.csv",
    ]:
        df = pd.read_csv(split_path).fillna("")
        for msg in df["sanitized_message"]:
            assert not nik_re.search(msg), f"Leaked NIK in {split_path}: {msg}"
            assert not email_re.search(msg), f"Leaked Email in {split_path}: {msg}"
            assert not phone_re.search(msg), f"Leaked Phone in {split_path}: {msg}"


def test_extract_technical_entities_explicit_and_contextual() -> None:
    """Verify deterministic regex and contextual rules extract accurate entities."""
    sample_1 = "Mesin kasir GT-POS200 nomor seri sn-99812 muncul error e-402 saat scan QRIS."
    ent_1 = extract_technical_entities(sample_1)
    assert ent_1["product_model"] == "GT-POS200"
    assert ent_1["error_code"] == "E-402"
    assert ent_1["serial_number"] == "SN-99812"

    sample_2 = "Mahasiswa gagal isi KRS di portal SIAKAD muncul kode ERR_KRS_LOCKED."
    ent_2 = extract_technical_entities(sample_2)
    assert ent_2["product_model"] == "GTCAMPUS-SIAKAD"
    assert ent_2["error_code"] == "ERR_KRS_LOCKED"
    assert ent_2["serial_number"] is None

    sample_3 = "Unit SN-54321 ketumpahan cairan kena RMA-12 di portal klaim garansi."
    ent_3 = extract_technical_entities(sample_3)
    assert ent_3["product_model"] == "SIDIG-WARRANTY"
    assert ent_3["error_code"] == "RMA-12"
    assert ent_3["serial_number"] == "SN-54321"


def test_entity_extraction_exact_match_f1_target() -> None:
    """Verify Exact-Match tuple accuracy >= 0.95 on test split as required by PRD."""
    df = pd.read_csv("data/tickets/test_tickets.csv").fillna("")
    matches = 0
    for _, row in df.iterrows():
        extracted = extract_technical_entities(str(row["sanitized_message"]))
        pred_tuple = (
            extracted["product_model"] or "",
            extracted["error_code"] or "",
            extracted["serial_number"] or "",
        )
        gt_tuple = (
            str(row["product_model"]).upper(),
            str(row["error_code"]).upper(),
            str(row["serial_number"]).upper(),
        )
        if pred_tuple == gt_tuple:
            matches += 1

    exact_match_score = matches / len(df)
    assert exact_match_score >= 0.95, (
        f"Expected Exact-Match F1 >= 0.95, got {exact_match_score:.4f}"
    )
