"""
Preflight checks — runs DataQualityChecker on the raw DataFrame before any
feature engineering begins, then applies targeted cleaning based on what the
report finds.

What it does
------------
1. Runs quality analysis (or loads an existing fresh report from disk)
2. Logs all warnings loudly so the user knows what they're working with
3. Drops constant columns (zero signal)
4. Calls clean_dataframe() to remove duplicates, fill missing dates,
   and cap outliers — decisions driven by actual report findings
5. Returns the cleaned DataFrame + a summary dict for the pipeline log
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.quality import DataQualityChecker, clean_dataframe
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/features.log")

# How old a cached report can be before we re-run analysis (hours)
_REPORT_MAX_AGE_HOURS = 24

# Columns we always keep regardless of what the report says
_NEVER_DROP = {
    "transaction_date", "filing_date", "insider_cik", "ticker",
    "transaction_code", "shares", "price_per_share", "total_value",
    "trade_direction",  # may not exist yet, harmless
}

# Warning substrings that are blocking (pipeline raises if found)
_BLOCKING_PATTERNS = [
    "negative values in 'price_per_share'",
    "future dates in 'transaction_date'",
]


def _load_cached_report(report_path: Path) -> dict[str, Any] | None:
    """Return the cached JSON report if it exists and is recent enough."""
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text())
        generated = datetime.fromisoformat(data["generated_at"])
        age = datetime.now() - generated
        if age < timedelta(hours=_REPORT_MAX_AGE_HOURS):
            return data
    except Exception:
        pass
    return None


def _run_fresh_report(df: pd.DataFrame, report_dir: str, stem: str) -> dict[str, Any]:
    """Run quality analysis and save JSON + Markdown reports."""
    checker = DataQualityChecker()
    report = checker.analyze(df)

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}_quality_report.json"
    md_path = out / f"{stem}_quality_report.md"
    report.to_json(str(json_path))
    report.save_markdown(str(md_path))
    logger.info("Quality report saved → %s", json_path)

    return report.to_dict()


def _check_blocking_warnings(warnings: list[str]) -> None:
    """Raise ValueError if any blocking warning is present."""
    for w in warnings:
        for pattern in _BLOCKING_PATTERNS:
            if pattern in w.lower():
                raise ValueError(f"[Preflight] Blocking data issue detected: {w}")


def _log_warnings(warnings: list[str]) -> None:
    if not warnings:
        logger.info("[Preflight] No quality warnings.")
        return
    for w in warnings:
        logger.warning("[Preflight] %s", w)


def _drop_constant_columns(df: pd.DataFrame, constant_cols: list[str]) -> pd.DataFrame:
    """Drop columns that have only one unique value — they carry no signal."""
    to_drop = [c for c in constant_cols if c in df.columns and c not in _NEVER_DROP]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info("[Preflight] Dropped %d constant columns: %s", len(to_drop), to_drop)
    return df


def _decide_outlier_method(report: dict[str, Any]) -> str:
    """
    Return 'cap' if price outlier percentage is high, else 'none'.
    Threshold: >5 % outliers in price_per_share → cap at 1st/99th percentile.
    """
    for col_stat in report.get("column_stats", []):
        if col_stat.get("name") == "price_per_share":
            if col_stat.get("outlier_pct", 0) > 5.0:
                logger.info(
                    "[Preflight] price_per_share outlier_pct=%.1f%% — enabling cap",
                    col_stat["outlier_pct"],
                )
                return "cap"
    return "none"


def _apply_recommendations(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    """Log each recommendation so it's visible in the run log."""
    recs = report.get("recommendations", [])
    if recs:
        logger.info("[Preflight] Recommendations from quality report:")
        for r in recs:
            logger.info("  → %s", r)
    return df


def run(
    df: pd.DataFrame,
    csv_path: str,
    report_dir: str = "reports",
    block_on_critical: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run preflight checks and return (cleaned_df, summary_dict).

    Parameters
    ----------
    df               : raw DataFrame (post-load, pre-feature-engineering)
    csv_path         : original CSV path — used to name the report file
    report_dir       : directory to read/write quality reports
    block_on_critical: if True, raise on blocking warnings (negative prices, etc.)
    """
    stem = Path(csv_path).stem
    report_path = Path(report_dir) / f"{stem}_quality_report.json"

    # Use cached report if fresh, otherwise regenerate
    report = _load_cached_report(report_path)
    if report is None:
        logger.info("[Preflight] Running fresh quality analysis...")
        report = _run_fresh_report(df, report_dir, stem)
    else:
        logger.info("[Preflight] Using cached quality report (%s)", report_path)

    # 1. Check for blocking issues
    warnings = report.get("warnings", [])
    _log_warnings(warnings)
    if block_on_critical:
        _check_blocking_warnings(warnings)

    # 2. Drop constant columns
    df = _drop_constant_columns(df, report.get("constant_columns", []))

    # 3. Apply clean_dataframe() with report-driven settings
    outlier_method = _decide_outlier_method(report)
    df, clean_log = clean_dataframe(
        df,
        remove_duplicates=True,
        handle_outliers=(outlier_method != "none"),
        outlier_method=outlier_method,
        fill_missing_dates=True,
    )
    for action in clean_log.get("actions", []):
        logger.info("[Preflight] clean_dataframe: %s", action)

    # 4. Surface recommendations in the log
    df = _apply_recommendations(df, report)

    summary = {
        "warnings": warnings,
        "constant_cols_dropped": [
            c for c in report.get("constant_columns", []) if c not in _NEVER_DROP
        ],
        "outlier_method": outlier_method,
        "rows_after_preflight": len(df),
        "clean_actions": clean_log.get("actions", []),
        "recommendations": report.get("recommendations", []),
    }

    logger.info(
        "[Preflight] Done. %d rows remain after cleaning.", len(df)
    )
    return df, summary
