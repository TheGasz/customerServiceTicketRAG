"""Unit tests verifying zero user_id leakage and disjoint template pools across splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.generate_synthetic_data import (
    EVAL_OPENERS_FORMAL,
    EVAL_OPENERS_SLANG,
    TRAIN_OPENERS_FORMAL,
    TRAIN_OPENERS_SLANG,
)
from src.preprocessing.splitter import verify_zero_user_leakage


def test_kb_sop_documents_exist_and_structured() -> None:
    """Verify all 8 SOP Markdown files exist and contain required # / ## / ### headings."""
    kb_dir = Path("data/kb_docs")
    expected_files = [
        "SOP-GT-POS200-01.md",
        "SOP-GT-POS200-02.md",
        "SOP-SIDIG-RMA-01.md",
        "SOP-SIDIG-SLA-02.md",
        "SOP-GTCAMPUS-01.md",
        "SOP-GTCAMPUS-02.md",
        "SOP-GTFLEET-01.md",
        "SOP-GTFLEET-02.md",
    ]
    for fname in expected_files:
        fpath = kb_dir / fname
        assert fpath.exists(), f"Missing SOP file: {fpath}"
        text = fpath.read_text(encoding="utf-8")
        assert "\n## " in text, f"Missing ## section heading in {fname}"
        assert "\n### " in text, f"Missing ### subsection heading in {fname}"


def test_zero_user_id_overlap_and_row_counts() -> None:
    """Verify train (840), val (180), and test (180) splits have zero user_id overlap."""
    train_df = pd.read_csv("data/tickets/train_tickets.csv")
    val_df = pd.read_csv("data/tickets/val_tickets.csv")
    test_df = pd.read_csv("data/tickets/test_tickets.csv")

    assert len(train_df) == 840
    assert len(val_df) == 180
    assert len(test_df) == 180

    verify_zero_user_leakage(train_df, val_df, test_df, user_col="user_id")

    all_users = (
        set(train_df["user_id"])
        | set(val_df["user_id"])
        | set(test_df["user_id"])
    )
    assert len(all_users) == 240


def test_disjoint_template_pools_between_train_and_eval() -> None:
    """Verify train and eval opener pools are strictly disjoint."""
    train_pool = set(TRAIN_OPENERS_FORMAL + TRAIN_OPENERS_SLANG)
    eval_pool = set(EVAL_OPENERS_FORMAL + EVAL_OPENERS_SLANG)
    assert train_pool.isdisjoint(eval_pool), (
        "Train and eval template opener pools must be strictly disjoint!"
    )
