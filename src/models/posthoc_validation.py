"""
Post-hoc validation for XGBoost insider trading classifier.

Compares XGBoost-flagged trades vs unflagged trades on external signals:
- signed_abnormal_return (did the trade precede abnormal stock movement?)
- abnormal_return (raw company-specific return)
- earnings_proximity_flag (was the trade near an earnings announcement?)

This is NOT model evaluation — it validates whether flagged trades
have real-world economic significance beyond the training labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]  # points to project root
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

LABEL_COL = "final_label"
TRANSACTION_CODE_COL = "transaction_code"
VALID_TRANSACTION_CODES = {"P", "S"}
VALID_LABELS = {0, 1}

FEATURE_COLS = [
    "log_shares", "log_total_value", "log_shares_owned_after",
    "total_value_is_imputed", "trade_direction", "signed_value",
    "days_to_filing", "pct_position_traded", "txn_day_of_week",
    "txn_month", "txn_quarter", "role_seniority", "is_ceo", "is_cfo",
    "is_coo", "is_director_only", "is_open_market", "is_derivative",
    "insider_total_trades", "insider_avg_trade_value",
    "insider_buy_sell_ratio", "insider_tenure_days", "trades_7d",
    "trades_30d", "trades_90d", "buy_count_7d", "sell_count_7d",
    "buy_count_30d", "sell_count_30d", "buy_count_90d", "sell_count_90d",
    "net_value_7d", "net_value_30d", "net_value_90d",
    "days_since_last_trade", "consecutive_direction", "trade_frequency_90d",
    "other_insiders_72h", "same_dir_insiders_72h", "coordination_score",
    "cluster_flag", "footnote_length", "value_bucket_num",
]

THRESHOLD = 0.5


def _load_test() -> pd.DataFrame:
    path = DATA_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, low_memory=False)
    df[TRANSACTION_CODE_COL] = df[TRANSACTION_CODE_COL].astype(str).str.upper().str.strip()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    mask = (
        df[TRANSACTION_CODE_COL].isin(VALID_TRANSACTION_CODES)
        & df[LABEL_COL].isin(VALID_LABELS)
    )
    df = df.loc[mask].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def _get_features(df: pd.DataFrame) -> np.ndarray:
    x = (
        df[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return x.values


def _ttest(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    t, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    return float(t), float(p)


def _plot_boxplot(flagged_scores: pd.Series, unflagged_scores: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [flagged_scores.dropna().values, unflagged_scores.dropna().values],
        labels=["Flagged (opportunistic)", "Unflagged (routine)"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Zero line")
    ax.set_ylabel("Signed Abnormal Return (30-day)")
    ax.set_title("Signed Abnormal Return: XGBoost Flagged vs Unflagged Trades")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved boxplot -> {out_path}")


def _plot_histogram(flagged_scores: pd.Series, unflagged_scores: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.5, 0.4, 50)
    ax.hist(unflagged_scores.dropna(), bins=bins, alpha=0.5, density=True, label="Unflagged (routine)")
    ax.hist(flagged_scores.dropna(), bins=bins, alpha=0.5, density=True, label="Flagged (opportunistic)")
    ax.axvline(flagged_scores.mean(), color="steelblue", linestyle="--", linewidth=2,
               label=f"Flagged mean: {flagged_scores.mean():.4f}")
    ax.axvline(unflagged_scores.mean(), color="orange", linestyle="--", linewidth=2,
               label=f"Unflagged mean: {unflagged_scores.mean():.4f}")
    ax.set_xlabel("Signed Abnormal Return (30-day)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Signed Abnormal Return by XGBoost Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved histogram -> {out_path}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and scaler...")
    model = joblib.load(MODELS_DIR / "xgb_model.joblib")
    scaler = joblib.load(MODELS_DIR / "xgb_scaler.joblib")

    print("Loading test set...")
    test_df = _load_test()
    print(f"Test rows: {len(test_df)}")

    x = _get_features(test_df)
    x_scaled = scaler.transform(x)
    probs = model.predict_proba(x_scaled)[:, 1]

    test_df = test_df.copy()
    test_df["xgb_prob"] = probs
    test_df["xgb_flagged"] = (probs >= THRESHOLD).astype(int)

    flagged = test_df[test_df["xgb_flagged"] == 1]
    unflagged = test_df[test_df["xgb_flagged"] == 0]

    print(f"\nFlagged as opportunistic: {len(flagged)} trades")
    print(f"Unflagged (routine):      {len(unflagged)} trades")

    # --- signed abnormal return ---
    sar_flagged = flagged["signed_abnormal_return"]
    sar_unflagged = unflagged["signed_abnormal_return"]
    t_sar, p_sar = _ttest(sar_flagged, sar_unflagged)

    # --- abnormal return ---
    ar_flagged = flagged["abnormal_return"]
    ar_unflagged = unflagged["abnormal_return"]
    t_ar, p_ar = _ttest(ar_flagged, ar_unflagged)

    # --- earnings proximity ---
    earn_flagged = flagged["earnings_proximity_flag"].mean()
    earn_unflagged = unflagged["earnings_proximity_flag"].mean()

    print("\n=== Post-hoc Validation Results ===")
    print(f"\nSigned Abnormal Return (30-day):")
    print(f"  Flagged mean:   {sar_flagged.mean():.4f}")
    print(f"  Unflagged mean: {sar_unflagged.mean():.4f}")
    print(f"  Difference:     {sar_flagged.mean() - sar_unflagged.mean():.4f}")
    print(f"  t-stat: {t_sar:.4f}, p-value: {p_sar:.4f} {'*significant*' if p_sar < 0.05 else ''}")

    print(f"\nAbnormal Return (30-day, unsigned):")
    print(f"  Flagged mean:   {ar_flagged.mean():.4f}")
    print(f"  Unflagged mean: {ar_unflagged.mean():.4f}")
    print(f"  t-stat: {t_ar:.4f}, p-value: {p_ar:.4f} {'*significant*' if p_ar < 0.05 else ''}")

    print(f"\nEarnings Proximity Flag:")
    print(f"  Flagged:   {earn_flagged:.3f} ({earn_flagged*100:.1f}% near earnings)")
    print(f"  Unflagged: {earn_unflagged:.3f} ({earn_unflagged*100:.1f}% near earnings)")

    # --- plots ---
    _plot_boxplot(sar_flagged, sar_unflagged, REPORTS_DIR / "posthoc_boxplot.png")
    _plot_histogram(sar_flagged, sar_unflagged, REPORTS_DIR / "posthoc_histogram.png")

    # --- save results ---
    results = {
        "threshold": THRESHOLD,
        "n_flagged": len(flagged),
        "n_unflagged": len(unflagged),
        "signed_abnormal_return": {
            "flagged_mean": round(float(sar_flagged.mean()), 6),
            "unflagged_mean": round(float(sar_unflagged.mean()), 6),
            "difference": round(float(sar_flagged.mean() - sar_unflagged.mean()), 6),
            "t_stat": round(t_sar, 4),
            "p_value": round(p_sar, 6),
            "significant_at_0.05": bool(p_sar < 0.05),
        },
        "abnormal_return": {
            "flagged_mean": round(float(ar_flagged.mean()), 6),
            "unflagged_mean": round(float(ar_unflagged.mean()), 6),
            "t_stat": round(t_ar, 4),
            "p_value": round(p_ar, 6),
            "significant_at_0.05": bool(p_ar < 0.05),
        },
        "earnings_proximity": {
            "flagged_pct": round(float(earn_flagged * 100), 2),
            "unflagged_pct": round(float(earn_unflagged * 100), 2),
        },
    }

    out_path = REPORTS_DIR / "posthoc_validation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()