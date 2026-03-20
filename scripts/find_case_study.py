"""
Case study candidate finder.
Finds high-confidence true positives in the test set with large abnormal returns.
Run from project root: python scripts/find_case_study.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading test set...")
test = pd.read_csv(ROOT / "data/processed/test.csv", low_memory=False)
test = test[
    test["transaction_code"].isin(["P", "S"]) &
    test["final_label"].isin([0, 1])
].copy()
test["final_label"] = test["final_label"].astype(int)
print(f"Test rows after filtering: {len(test)}")

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model and scaler...")
model = joblib.load(ROOT / "models/xgb_model.joblib")
scaler = joblib.load(ROOT / "models/xgb_scaler.joblib")

x = (
    test[FEATURE_COLS]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
test["xgb_prob"] = model.predict_proba(scaler.transform(x))[:, 1]
test["xgb_flagged"] = (test["xgb_prob"] >= 0.5).astype(int)

# ── Fuzzy score (inline) ───────────────────────────────────────────────────
txn_month = test["txn_month"].fillna(0)
pct = test["pct_position_traded"].fillna(0)
has_plan = test.get("has_plan", pd.Series(1, index=test.index)).fillna(1)
others = test["other_insiders_72h"].fillna(0)
days_filing = test["days_to_filing"].fillna(0)

near = np.where(txn_month.isin([1, 4, 7, 10]), 1.0,
        np.where(txn_month.isin([2, 5, 8, 11]), 0.5, 0.0))
pos = np.clip((pct - 0.01) / 0.09, 0, 1)
pos = np.where(pct >= 0.10, 1.0, pos)
noplan = np.where(has_plan == 0, 1.0, 0.0)
coord = np.clip(others / 3.0, 0, 1)
coord = np.where(others >= 3, 1.0, coord)
lag = np.clip((days_filing - 1) / 4.0, 0, 1)
lag = np.where(days_filing > 5, 1.0, lag)

test["fuzzy_score"] = np.clip(
    0.30 * near + 0.25 * pos + 0.25 * noplan + 0.10 * coord + 0.10 * lag,
    0.0, 1.0
)
test["fuzzy_flagged"] = (test["fuzzy_score"] >= 0.5).astype(int)
test["both_flagged"] = ((test["xgb_flagged"] == 1) & (test["fuzzy_flagged"] == 1)).astype(int)

# ── Find candidates ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP CASE STUDY CANDIDATES")
print("Filter: xgb_prob >= 0.75, true positive, |abnormal_return| > 0.05")
print("="*70)

candidates = test[
    (test["xgb_prob"] >= 0.75) &
    (test["final_label"] == 1) &
    (test["abnormal_return"].abs() > 0.05)
].copy()

display_cols = [
    "ticker", "transaction_date", "insider_cik",
    "xgb_prob", "fuzzy_score", "both_flagged",
    "abnormal_return", "signed_abnormal_return",
    "role_seniority", "is_ceo", "is_cfo",
    "pct_position_traded", "log_total_value",
    "earnings_proximity_flag", "transaction_code",
]
available = [c for c in display_cols if c in candidates.columns]

candidates_sorted = candidates.sort_values(
    "abnormal_return", key=abs, ascending=False
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

print(f"\nFound {len(candidates_sorted)} candidates. Top 15:\n")
print(candidates_sorted[available].head(15).to_string(index=False))

# ── Both flagged (strongest cases) ────────────────────────────────────────
print("\n" + "="*70)
print("STRONGEST CASES: Flagged by BOTH XGBoost AND Fuzzy Logic")
print("="*70)
both = candidates_sorted[candidates_sorted["both_flagged"] == 1]
print(f"Found {len(both)} candidates flagged by both.\n")
print(both[available].head(10).to_string(index=False))

# ── Per ticker summary ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("CANDIDATE COUNT BY TICKER")
print("="*70)
ticker_counts = (
    candidates_sorted.groupby("ticker")
    .agg(
        n_candidates=("xgb_prob", "count"),
        mean_xgb_prob=("xgb_prob", "mean"),
        max_abs_abnormal=("abnormal_return", lambda x: x.abs().max()),
        n_both_flagged=("both_flagged", "sum"),
    )
    .sort_values("n_candidates", ascending=False)
)
print(ticker_counts.to_string())

# ── Best single trade recommendation ──────────────────────────────────────
print("\n" + "="*70)
print("RECOMMENDED CASE STUDY TRADE")
print("="*70)
if len(both) > 0:
    best = both.iloc[0]
else:
    best = candidates_sorted.iloc[0]

print(f"\nTicker:              {best.get('ticker', 'N/A')}")
print(f"Transaction Date:    {best.get('transaction_date', 'N/A')}")
print(f"Transaction Code:    {best.get('transaction_code', 'N/A')}")
print(f"Insider CIK:         {best.get('insider_cik', 'N/A')}")
print(f"Role Seniority:      {best.get('role_seniority', 'N/A')} (6=CEO, 5=CFO, 4=COO/President)")
print(f"Is CEO:              {best.get('is_ceo', 'N/A')}")
print(f"Is CFO:              {best.get('is_cfo', 'N/A')}")
print(f"XGBoost Prob:        {best.get('xgb_prob', 'N/A'):.4f}")
print(f"Fuzzy Score:         {best.get('fuzzy_score', 'N/A'):.4f}")
print(f"Both Flagged:        {bool(best.get('both_flagged', 0))}")
print(f"Abnormal Return:     {best.get('abnormal_return', 'N/A'):.4f} ({best.get('abnormal_return', 0)*100:.2f}%)")
print(f"Signed Abn Return:   {best.get('signed_abnormal_return', 'N/A'):.4f}")
print(f"Pct Position Traded: {best.get('pct_position_traded', 'N/A'):.4f}")
print(f"Log Total Value:     {best.get('log_total_value', 'N/A'):.4f}")
print(f"Near Earnings:       {bool(best.get('earnings_proximity_flag', 0))}")
print(f"Cohen Label:         {best.get('cohen_label', 'N/A')}")
print(f"Final Label:         {best.get('final_label', 'N/A')}")

print("\nNext step: Look up this insider on SEC EDGAR:")
cik = best.get('insider_cik', '')
ticker = best.get('ticker', '')
print(f"  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=10")
print(f"  Search: {ticker} Form 4 filings around {best.get('transaction_date', '')}")