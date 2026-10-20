"""PII sanitization and hybrid technical entity extraction module."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Compiled regular expressions for Indonesian PII masking
_NIK_PATTERN = re.compile(r"\b\d{16}\b")
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+62|62|0)8\d{1,3}(?:[\s-]?\d{3,4}){1,3}(?!\d)"
)

# Compiled regular expressions for technical entity extraction
_PRODUCT_MODEL_PATTERN = re.compile(
    r"\b(GT-POS200|SIDIG-WARRANTY|GTCAMPUS-SIAKAD|GTFLEET-TRACKER)\b",
    re.IGNORECASE,
)
_ERROR_CODE_PATTERN = re.compile(
    r"\b(E-\d{3,4}|ERR_[A-Z0-9_]+|RMA-\d{2}|SLA-GOLD|SSO-\d{3}|VA-\d{3}|OTA-\d{3})\b",
    re.IGNORECASE,
)
_SERIAL_NUMBER_PATTERN = re.compile(
    r"\b(SN-\d{5,10})\b",
    re.IGNORECASE,
)

# Error code to product model deterministic mapping
_ERROR_TO_PRODUCT: Dict[str, str] = {
    "E-402": "GT-POS200",
    "E-405": "GT-POS200",
    "E-501": "GT-POS200",
    "E-509": "GT-POS200",
    "RMA-08": "SIDIG-WARRANTY",
    "RMA-12": "SIDIG-WARRANTY",
    "SLA-GOLD": "SIDIG-WARRANTY",
    "RMA-19": "SIDIG-WARRANTY",
    "ERR_KRS_LOCKED": "GTCAMPUS-SIAKAD",
    "SSO-401": "GTCAMPUS-SIAKAD",
    "ERR_VA_TIMEOUT": "GTCAMPUS-SIAKAD",
    "VA-808": "GTCAMPUS-SIAKAD",
    "E-901": "GTFLEET-TRACKER",
    "E-904": "GTFLEET-TRACKER",
    "ERR_MQTT_DROP": "GTFLEET-TRACKER",
    "OTA-702": "GTFLEET-TRACKER",
}

# Contextual keyword rules for product model inference when explicit product ID is omitted
_CONTEXTUAL_PRODUCT_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "GT-POS200",
        (
            "gt-pos",
            "pos200",
            "mesin kasir",
            "terminal kasir",
            "qris",
            "nfc",
            "kepala cetak",
            "printer termal",
            "struk",
            "sqlite lokal",
            "settlement harian",
        ),
    ),
    (
        "SIDIG-WARRANTY",
        (
            "sidig",
            "klaim garansi",
            "masa garansi",
            "unit cadangan",
            "loaner unit",
            "segel rusak",
            "kena air",
            "terkena cairan",
            "penggantian mainboard",
            "rma",
        ),
    ),
    (
        "GTCAMPUS-SIAKAD",
        (
            "gtcampus",
            "siakad",
            "pengisian krs",
            "krs terkunci",
            "dosen wali",
            "sso kampus",
            "pembayaran ukt",
            "spp virtual account",
            "callback webhook",
            "mahasiswa",
        ),
    ),
    (
        "GTFLEET-TRACKER",
        (
            "gtfleet",
            "antena gnss",
            "sinyal gps",
            "telemetri bus",
            "armada",
            "sensor bahan bakar",
            "kartu tap penumpang",
            "broker mqtt",
            "mqtt",
            "firmware ota",
            "over-the-air",
        ),
    ),
)


def mask_pii(text: str) -> str:
    """Sanitize Indonesian PII (16-digit NIK, mobile phone numbers, emails) from text.

    Args:
        text: Raw customer ticket message.

    Returns:
        Sanitized string with PII replaced by `[MASKED_NIK]`, `[MASKED_PHONE]`,
        and `[MASKED_EMAIL]`.
    """
    if not text or not text.strip():
        return ""

    # Mask 16-digit NIK first to prevent partial phone regex overlap
    sanitized = _NIK_PATTERN.sub("[MASKED_NIK]", text)
    sanitized = _EMAIL_PATTERN.sub("[MASKED_EMAIL]", sanitized)
    sanitized = _PHONE_PATTERN.sub("[MASKED_PHONE]", sanitized)
    return sanitized


def extract_technical_entities(
    text: str, infer_product_from_context: bool = True
) -> Dict[str, Optional[str]]:
    """Extract `product_model`, `error_code`, and `serial_number` using hybrid rules.

    Combines deterministic regular expressions with error-code mapping and
    contextual domain keywords to identify product models even in informal tickets.

    Args:
        text: Input ticket text (raw or PII-sanitized).
        infer_product_from_context: Whether to infer `product_model` from error code
            or contextual domain keywords if not explicitly named.

    Returns:
        Dictionary with keys `product_model`, `error_code`, and `serial_number`
        (normalized to uppercase strings or `None`).
    """
    if not text or not text.strip():
        return {
            "product_model": None,
            "error_code": None,
            "serial_number": None,
        }

    prod_match = _PRODUCT_MODEL_PATTERN.search(text)
    err_match = _ERROR_CODE_PATTERN.search(text)
    sn_match = _SERIAL_NUMBER_PATTERN.search(text)

    product_model = prod_match.group(1).upper() if prod_match else None
    error_code = err_match.group(1).upper() if err_match else None
    serial_number = sn_match.group(1).upper() if sn_match else None

    if product_model is None and infer_product_from_context:
        if error_code and error_code in _ERROR_TO_PRODUCT:
            product_model = _ERROR_TO_PRODUCT[error_code]
        else:
            lowered = text.lower()
            best_product: Optional[str] = None
            best_hits = 0
            for candidate_product, keywords in _CONTEXTUAL_PRODUCT_RULES:
                hits = sum(1 for kw in keywords if kw in lowered)
                if hits > best_hits:
                    best_hits = hits
                    best_product = candidate_product
            if best_hits > 0:
                product_model = best_product

    return {
        "product_model": product_model,
        "error_code": error_code,
        "serial_number": serial_number,
    }
