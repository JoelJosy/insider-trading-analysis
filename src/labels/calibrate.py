"""
Phase 3 label calibration utility.

Sweeps abnormal-return thresholds and confidence-source requirements,
then writes calibration artifacts with recommended settings.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.labels.pipeline import (
    _load_input,
    add_earnings_confirmation,
    add_enforcement_confirmation,
    add_price_labels,
    combine_signals,
)
from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/labels.log")


def _parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _score_row(row: pd.Series, target_mid: float) -> float:
    informed_pct = float(row["informed_pct"])
    coverage_pct = float(row["coverage_pct"])
    spread_penalty = abs(informed_pct - target_mid)
    coverage_penalty = max(0.0, 95.0 - coverage_pct) * 0.2
    return spread_penalty + coverage_penalty


def _recommend(grid_df: pd.DataFrame, target_min: float, target_max: float) -> dict:
    target_mid = (target_min + target_max) / 2
    in_band = grid_df[(grid_df["informed_pct"] >= target_min) & (grid_df["informed_pct"] <= target_max)]
    pool = in_band if not in_band.empty else grid_df
    scored = pool.copy()
    scored["score"] = scored.apply(lambda row: _score_row(row, target_mid), axis=1)
    best = scored.sort_values(["score", "coverage_pct"], ascending=[True, False]).iloc[0]
    return {
        "abnormal_threshold": float(best["abnormal_threshold"]),
        "confidence_sources_required": int(best["confidence_sources_required"]),
        "informed_pct": float(best["informed_pct"]),
        "coverage_pct": float(best["coverage_pct"]),
    }


def run_calibration(
    input_csv: str,
    output_prefix: str,
    horizon_days: int,
    benchmark_ticker: str,
    thresholds: list[float],
    confidence_levels: list[int],
    target_min_informed_pct: float,
    target_max_informed_pct: float,
    earnings_proximity_days: int,
    earnings_csv: str | None,
    enforcement_csv: str | None,
    enforcement_lookahead_days: int,
    price_cache_dir: str,
) -> tuple[pd.DataFrame, dict]:
    df = _load_input(input_csv)

    base = add_price_labels(
        df,
        horizon_days=horizon_days,
        abnormal_threshold=min(thresholds),
        benchmark_ticker=benchmark_ticker,
        price_cache_dir=price_cache_dir,
        require_prices=True,
    )
    base = add_earnings_confirmation(base, earnings_csv=earnings_csv, proximity_days=earnings_proximity_days)
    base = add_enforcement_confirmation(
        base,
        enforcement_csv=enforcement_csv,
        lookahead_days=enforcement_lookahead_days,
    )

    rows = []
    total = len(base)
    coverage_pct = float(base["forward_return"].notna().mean() * 100)

    for threshold in thresholds:
        trial = base.copy()
        trial["price_signal"] = (trial["signed_abnormal_return"] >= threshold).astype(int)

        for conf in confidence_levels:
            labeled = combine_signals(trial, confidence_sources_required=conf)
            informed = int(labeled["informed_label"].sum())
            rows.append(
                {
                    "abnormal_threshold": threshold,
                    "confidence_sources_required": conf,
                    "rows": total,
                    "informed_rows": informed,
                    "informed_pct": (informed / total * 100) if total else 0.0,
                    "coverage_pct": coverage_pct,
                }
            )

    grid_df = pd.DataFrame(rows).sort_values(["confidence_sources_required", "abnormal_threshold"])
    recommendation = _recommend(grid_df, target_min_informed_pct, target_max_informed_pct)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "input_csv": input_csv,
        "horizon_days": horizon_days,
        "benchmark_ticker": benchmark_ticker,
        "target_informed_pct_range": [target_min_informed_pct, target_max_informed_pct],
        "threshold_candidates": thresholds,
        "confidence_candidates": confidence_levels,
        "price_coverage_pct": round(coverage_pct, 2),
        "recommendation": recommendation,
    }

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    grid_path = f"{out_prefix}_grid.csv"
    json_path = f"{out_prefix}_summary.json"
    md_path = f"{out_prefix}_summary.md"

    grid_df.to_csv(grid_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Phase 3 Calibration Report",
        f"\n**Generated:** {summary['generated_at']}",
        f"\n**Input:** {input_csv}",
        f"\n**Price Coverage:** {summary['price_coverage_pct']}%",
        "\n## Recommended Settings",
        f"- abnormal_threshold: {recommendation['abnormal_threshold']}",
        f"- confidence_sources_required: {recommendation['confidence_sources_required']}",
        f"- informed_pct (at recommendation): {recommendation['informed_pct']:.2f}%",
        "\n## Target Range",
        f"- informed_pct target: {target_min_informed_pct}% to {target_max_informed_pct}%",
        "\n## Artifacts",
        f"- Grid CSV: {grid_path}",
        f"- Summary JSON: {json_path}",
    ]
    Path(md_path).write_text("\n".join(md), encoding="utf-8")

    logger.info("Saved calibration grid -> %s", grid_path)
    logger.info("Saved calibration summary -> %s", json_path)
    return grid_df, summary


def _parse_args() -> argparse.Namespace:
    cfg = get_config().labeling
    p = argparse.ArgumentParser(description="Calibrate Phase 3 labeling thresholds")
    p.add_argument("--input", required=True, help="Input CSV (features or processed)")
    p.add_argument("--output-prefix", default=None, help="Prefix for output files under reports/")
    p.add_argument("--horizon-days", type=int, default=30)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--thresholds", default="0.05,0.10,0.15,0.20", help="Comma-separated thresholds")
    p.add_argument("--confidence-levels", default="1,2", help="Comma-separated confidence source counts")
    p.add_argument("--target-min-informed-pct", type=float, default=5.0)
    p.add_argument("--target-max-informed-pct", type=float, default=20.0)
    p.add_argument("--earnings-proximity-days", type=int, default=cfg.earnings_proximity_days)
    p.add_argument("--earnings-csv", default="data/external/earnings/earnings_announcements.csv")
    p.add_argument("--enforcement-csv", default="data/external/sec/sec_enforcement.csv")
    p.add_argument("--enforcement-lookahead-days", type=int, default=365)
    p.add_argument("--price-cache-dir", default="data/external/prices")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stem = Path(args.input).stem
    output_prefix = args.output_prefix or f"reports/{stem}_label_calibration"

    grid, summary = run_calibration(
        input_csv=args.input,
        output_prefix=output_prefix,
        horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark,
        thresholds=_parse_float_list(args.thresholds),
        confidence_levels=_parse_int_list(args.confidence_levels),
        target_min_informed_pct=args.target_min_informed_pct,
        target_max_informed_pct=args.target_max_informed_pct,
        earnings_proximity_days=args.earnings_proximity_days,
        earnings_csv=args.earnings_csv,
        enforcement_csv=args.enforcement_csv,
        enforcement_lookahead_days=args.enforcement_lookahead_days,
        price_cache_dir=args.price_cache_dir,
    )

    print("\nCalibration complete.")
    print(f"Rows evaluated: {len(grid):,}")
    print("Recommended settings:")
    print(summary["recommendation"])
