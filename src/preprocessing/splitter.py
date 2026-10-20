"""Group-based dataset splitter preventing user_id data leakage across splits."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_tickets_by_user(
    df: pd.DataFrame,
    user_col: str = "user_id",
    train_size: int = 840,
    val_size: int = 180,
    test_size: int = 180,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ticket dataset into train, val, and test with zero `user_id` overlap.

    Uses `GroupShuffleSplit` grouped by `user_id`. Assumes each `user_id` has an
    equal number of tickets (e.g. 5 tickets per user across 240 users = 1,200 total)
    so exact row counts (840 / 180 / 180) are preserved alongside strict group isolation.

    Args:
        df: Full ticket DataFrame containing `user_col`.
        user_col: Column name identifying the customer account group.
        train_size: Target number of rows for the training split.
        val_size: Target number of rows for the validation split.
        test_size: Target number of rows for the test split.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of `(train_df, val_df, test_df)` with disjoint `user_id` sets.

    Raises:
        ValueError: If `df` is empty, missing `user_col`, or if any user overlap occurs.
    """
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    if user_col not in df.columns:
        raise ValueError(f"Column '{user_col}' not found in DataFrame.")

    total_rows = len(df)
    expected_total = train_size + val_size + test_size
    if total_rows != expected_total:
        train_frac = train_size / expected_total
        val_rel_frac = val_size / (val_size + test_size)
    else:
        train_frac = train_size / total_rows
        val_rel_frac = val_size / (val_size + test_size)

    gss_outer = GroupShuffleSplit(
        n_splits=1, train_size=train_frac, random_state=seed
    )
    train_idx, temp_idx = next(gss_outer.split(df, groups=df[user_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss_inner = GroupShuffleSplit(
        n_splits=1, train_size=val_rel_frac, random_state=seed + 1
    )
    val_idx, test_idx = next(gss_inner.split(temp_df, groups=temp_df[user_col]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    verify_zero_user_leakage(train_df, val_df, test_df, user_col=user_col)
    return train_df, val_df, test_df


def verify_zero_user_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str = "user_id",
) -> None:
    """Assert that train, validation, and test splits share zero `user_id` values.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        user_col: Name of the user identifier column.

    Raises:
        AssertionError: If any intersection between user sets is non-empty.
    """
    train_users = set(train_df[user_col].unique())
    val_users = set(val_df[user_col].unique())
    test_users = set(test_df[user_col].unique())

    overlap_train_val = train_users.intersection(val_users)
    overlap_train_test = train_users.intersection(test_users)
    overlap_val_test = val_users.intersection(test_users)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise AssertionError(
            f"Data leakage detected! Overlaps: "
            f"train∩val={len(overlap_train_val)}, "
            f"train∩test={len(overlap_train_test)}, "
            f"val∩test={len(overlap_val_test)}"
        )
