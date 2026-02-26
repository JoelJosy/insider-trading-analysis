"""
Feature pipeline — orchestrates all feature modules and writes the
final feature CSV.

Usage
-----
    # From project root:
    python -m src.features.pipeline --input data/processed/AAPL_form4.csv

    # With custom output path:
    python -m src.features.pipeline \\
        --input data/processed/AAPL_form4.csv \\
        --output data/processed/AAPL_features.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.features import (
    insider_features,
    network_features,
    preflight,
    temporal_features,
    text_features,
    trade_features,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/features.log")


# Columns that are pure identifiers or redundant after feature engineering
_DROP_COLS = [
    "xml_path",
    "footnote_text",   # replaced by extracted features
    "issuer_name",     # captured by ticker
    "is_amendment",    # all False in dataset
    "accession_number",
]

# Columns to keep for model training (excludes raw leakage / id-only cols)
NUMERIC_FEATURE_COLS = [
    "log_shares",
    "log_total_value",
    "log_shares_owned_after",
    "total_value_is_imputed",
    "trade_direction",
    "signed_value",
    "days_to_filing",
    "pct_position_traded",
    "txn_day_of_week",
    "txn_month",
    "txn_quarter",
    "role_seniority",
    "is_ceo",
    "is_cfo",
    "is_coo",
    "is_director_only",
    "is_open_market",
    "is_derivative",
    "has_plan",
    "insider_total_trades",
    "insider_avg_trade_value",
    "insider_buy_sell_ratio",
    "insider_tenure_days",
    "trades_7d",
    "trades_30d",
    "trades_90d",
    "buy_count_7d",
    "sell_count_7d",
    "buy_count_30d",
    "sell_count_30d",
    "buy_count_90d",
    "sell_count_90d",
    "net_value_7d",
    "net_value_30d",
    "net_value_90d",
    "days_since_last_trade",
    "consecutive_direction",
    "trade_frequency_90d",
    "other_insiders_72h",
    "same_dir_insiders_72h",
    "coordination_score",
    "cluster_flag",
    "footnote_length",
    "footnote_has_plan",
    "footnote_routine_score",
    "footnote_routine_hits",
    "footnote_informed_hits",
    "footnote_has_option",
    "value_bucket_num",
]

# Categorical feature columns kept alongside the numeric matrix
CATEGORICAL_FEATURE_COLS = [
    "value_bucket",
]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Read the processed Form 4 CSV with correct dtypes."""
    df = pd.read_csv(
        csv_path,
        parse_dates=["filing_date", "transaction_date", "exercise_date", "expiration_date"],
        dtype={
            "insider_cik": str,
            "issuer_cik": str,
            "transaction_code": str,
            "insider_role": str,
        },
        low_memory=False,
    )
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature modules in order.

    Execution order matters:
    1. trade_features — adds signed_value, trade_direction (needed later)
    2. insider_features — adds role flags + historical aggregates
    3. temporal_features — rolling windows (needs trade_direction + dates)
    4. network_features — coordination (needs trade_direction + dates)
    5. text_features — footnote NLP (independent)
    """
    steps = [
        ("trade",    trade_features.build),
        ("insider",  insider_features.build),
        ("temporal", temporal_features.build),
        ("network",  network_features.build),
        ("text",     text_features.build),
    ]

    for name, fn in steps:
        t0 = time.perf_counter()
        df = fn(df)
        logger.info("  [%s] done in %.2fs  (cols: %d)", name, time.perf_counter() - t0, len(df.columns))

    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in _DROP_COLS if c in df.columns]
    return df.drop(columns=to_drop)


def save(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved feature CSV → %s  (%d rows × %d cols)", out_path, len(df), len(df.columns))


def features_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric + categorical feature columns with key identifiers."""
    id_cols = ["insider_cik", "ticker", "transaction_date", "transaction_code"]
    keep = (
        id_cols
        + [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
        + [c for c in CATEGORICAL_FEATURE_COLS if c in df.columns]
    )
    return df[keep]


def build_feature_matrix(
    csv_path: str,
    output_path: str | None = None,
    report_dir: str = "reports",
    skip_preflight: bool = False,
) -> pd.DataFrame:
    """
    End-to-end function: load → preflight → feature-engineer → save.

    Returns the full enriched DataFrame (all original + new columns).
    Optionally saves the features-only matrix to `output_path`.
    """
    df = load_raw(csv_path)

    if skip_preflight:
        before = len(df)
        df = df.drop_duplicates()
        if (n := before - len(df)):
            logger.warning("Dropped %d duplicate rows", n)
    else:
        df, _ = preflight.run(df, csv_path, report_dir=report_dir)

    df = run_pipeline(df)
    df = drop_unused(df)

    if output_path:
        feat_df = features_only(df)
        save(feat_df, output_path)

    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Run the feature engineering pipeline")
    p.add_argument("--input", required=True, help="Path to processed CSV (e.g. data/processed/AAPL_form4.csv)")
    p.add_argument(
        "--output",
        default=None,
        help="Output path for features CSV. Defaults to <input_stem>_features.csv",
    )
    p.add_argument("--report-dir", default="reports", help="Directory for quality reports (default: reports/)")
    p.add_argument("--full", action="store_true", help="Save all columns, not just feature matrix")
    p.add_argument("--skip-preflight", action="store_true", help="Skip quality preflight checks (faster, less safe)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = args.input

    if args.output:
        out = args.output
    else:
        stem = Path(input_path).stem
        out = str(Path(input_path).parent / f"{stem}_features.csv")

    logger.info("=== Feature Pipeline START ===")
    logger.info("Input : %s", input_path)
    logger.info("Output: %s", out)

    df = load_raw(input_path)

    if args.skip_preflight:
        before = len(df)
        df = df.drop_duplicates()
        if (n := before - len(df)):
            logger.warning("Dropped %d duplicate rows", n)
    else:
        df, preflight_summary = preflight.run(df, input_path, report_dir=args.report_dir)
        if preflight_summary["recommendations"]:
            print("\n[Preflight] Recommendations:")
            for r in preflight_summary["recommendations"]:
                print(f"  → {r}")

    df = run_pipeline(df)
    df = drop_unused(df)

    if args.full:
        save(df, out)
    else:
        save(features_only(df), out)

    logger.info("=== Feature Pipeline DONE  ===")
    print(f"\nDone.  Feature matrix → {out}")
    print(f"Rows: {len(df):,}   Feature cols: {len([c for c in NUMERIC_FEATURE_COLS if c in df.columns])}")
