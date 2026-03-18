from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_pct(num: int, den: int) -> float:
    return float((num / den) * 100) if den else 0.0


def build_label_quality_summary(df: pd.DataFrame, input_csv: str, params: dict[str, Any]) -> dict[str, Any]:
    total = len(df)
    if "final_label" in df.columns:
        final = pd.to_numeric(df["final_label"], errors="coerce").fillna(-1).astype(int)
        opportunistic = int((final == 1).sum())
        routine = int((final == 0).sum())
        uncertain = int((final == -1).sum())
    else:
        opportunistic = int(df.get("informed_label", pd.Series(dtype=int)).fillna(0).astype(int).sum())
        routine = int(df.get("routine_label", pd.Series(dtype=int)).fillna(0).astype(int).sum())
        uncertain = max(total - opportunistic - routine, 0)

    price_cov = int(df.get("forward_return", pd.Series(dtype=float)).notna().sum()) if "forward_return" in df.columns else 0
    bench_cov = int(df.get("benchmark_return", pd.Series(dtype=float)).notna().sum()) if "benchmark_return" in df.columns else 0

    price_signal = int(df.get("price_signal", pd.Series(dtype=int)).fillna(0).astype(int).sum())
    earnings_signal = int(df.get("earnings_proximity_flag", pd.Series(dtype=int)).fillna(0).astype(int).sum())
    enforce_signal = int(df.get("enforcement_followup_flag", pd.Series(dtype=int)).fillna(0).astype(int).sum())

    source_count_dist = {}
    if "label_source_count" in df.columns:
        for key, val in df["label_source_count"].fillna(0).astype(int).value_counts().sort_index().items():
            source_count_dist[str(int(key))] = int(val)

    recommendations: list[str] = []
    opportunistic_pct = _safe_pct(opportunistic, total)
    price_cov_pct = _safe_pct(price_cov, total)

    if price_cov_pct < 90:
        recommendations.append("Price coverage is below 90%; re-run labeling later or verify market source access.")
    if opportunistic_pct < 1:
        recommendations.append("Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.")
    elif opportunistic_pct > 40:
        recommendations.append("Opportunistic share is high (>40%); review Cohen/plan rules for over-sensitive labeling.")

    if earnings_signal == 0:
        recommendations.append("No earnings confirmations found; provide earnings CSV to improve multi-source labeling.")
    if enforce_signal == 0:
        recommendations.append("No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.")

    if not recommendations:
        recommendations.append("Label distribution and coverage look healthy for this run.")

    return {
        "generated_at": datetime.now().isoformat(),
        "input_csv": input_csv,
        "params": params,
        "rows": total,
        "coverage": {
            "forward_return_rows": price_cov,
            "forward_return_pct": round(price_cov_pct, 2),
            "benchmark_return_rows": bench_cov,
            "benchmark_return_pct": round(_safe_pct(bench_cov, total), 2),
        },
        "labels": {
            "opportunistic_rows": opportunistic,
            "opportunistic_pct": round(opportunistic_pct, 2),
            "routine_rows": routine,
            "routine_pct": round(_safe_pct(routine, total), 2),
            "uncertain_rows": uncertain,
            "uncertain_pct": round(_safe_pct(uncertain, total), 2),
        },
        "signals": {
            "price_signal_rows": price_signal,
            "earnings_signal_rows": earnings_signal,
            "enforcement_signal_rows": enforce_signal,
            "label_source_count_distribution": source_count_dist,
        },
        "recommendations": recommendations,
    }


def summary_to_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    labels = summary["labels"]
    signals = summary["signals"]

    lines = [
        "# Phase 3 Label Quality Report",
        f"\n**Generated:** {summary['generated_at']}",
        f"\n**Input CSV:** {summary['input_csv']}",
        "\n---",
        "\n## Coverage",
        f"- Forward return coverage: {coverage['forward_return_rows']} rows ({coverage['forward_return_pct']}%)",
        f"- Benchmark return coverage: {coverage['benchmark_return_rows']} rows ({coverage['benchmark_return_pct']}%)",
        "\n## Label Distribution",
        f"- Opportunistic: {labels['opportunistic_rows']} ({labels['opportunistic_pct']}%)",
        f"- Routine: {labels['routine_rows']} ({labels['routine_pct']}%)",
        f"- Uncertain: {labels['uncertain_rows']} ({labels['uncertain_pct']}%)",
        "\n## Signal Counts",
        f"- Price signal rows: {signals['price_signal_rows']}",
        f"- Earnings signal rows: {signals['earnings_signal_rows']}",
        f"- Enforcement signal rows: {signals['enforcement_signal_rows']}",
        "\n### Source Count Distribution",
        "| Source Count | Rows |",
        "|---|---:|",
    ]

    for source_count, rows in signals["label_source_count_distribution"].items():
        lines.append(f"| {source_count} | {rows} |")

    lines.append("\n## Recommendations")
    for rec in summary["recommendations"]:
        lines.append(f"- {rec}")

    return "\n".join(lines)


def save_label_quality_report(summary: dict[str, Any], report_json_path: str, report_md_path: str) -> None:
    Path(report_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    Path(report_md_path).write_text(summary_to_markdown(summary), encoding="utf-8")
