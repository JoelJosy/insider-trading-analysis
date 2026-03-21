import pandas as pd
import numpy as np
import joblib
from scipy import stats
from pathlib import Path

ROOT = Path(".")

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

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "INTC": "Tech",
    "IBM": "Tech", "CSCO": "Tech", "META": "Tech", "ADBE": "Tech",
    "CRM": "Tech", "ORCL": "Tech", "QCOM": "Tech", "TXN": "Tech",
    "MRNA": "Biotech", "BIIB": "Biotech", "AMGN": "Biotech", "REGN": "Biotech",
    "GILD": "Biotech", "ZTS": "Biotech",
    "LLY": "Pharma", "PFE": "Pharma", "MRK": "Pharma", "ABBV": "Pharma",
    "JNJ": "Pharma", "MDT": "Pharma", "BMY": "Pharma", "TMO": "Pharma",
    "DHR": "Pharma",
    "JPM": "Finance", "GS": "Finance", "MS": "Finance", "BAC": "Finance",
    "WFC": "Finance", "C": "Finance", "AXP": "Finance", "V": "Finance",
    "BK": "Finance", "BLK": "Finance", "SCHW": "Finance", "COF": "Finance",
    "MA": "Finance", "USB": "Finance", "MET": "Finance",
    "XOM": "Energy", "COP": "Energy", "OXY": "Energy",
    "WMT": "Consumer", "MCD": "Consumer", "KO": "Consumer", "PG": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MO": "Consumer", "PM": "Consumer",
    "KHC": "Consumer", "MDLZ": "Consumer", "PEP": "Consumer", "TGT": "Consumer",
    "CMCSA": "Consumer", "DIS": "Consumer", "CHTR": "Consumer",
    "BA": "Defense", "LMT": "Defense", "GD": "Defense", "RTX": "Defense",
    "UNH": "Healthcare", "HUM": "Healthcare", "CVS": "Healthcare",
    "ACN": "Tech", "PYPL": "Finance", "TMUS": "Telecom", "T": "Telecom",
    "VZ": "Telecom", "AMT": "Telecom", "SPG": "REIT",
    "UNP": "Industrial", "UPS": "Industrial", "FDX": "Industrial",
    "GE": "Industrial", "EMR": "Industrial", "HON": "Industrial",
    "NEE": "Utility", "DUK": "Utility", "SO": "Utility", "EXC": "Utility",
    "LIN": "Industrial", "DOW": "Industrial", "LOW": "Consumer",
    "AON": "Finance", "ALL": "Finance", "AIG": "Finance",
    "PYPL": "Finance", "F": "Industrial", "GM": "Industrial",
}

print("Loading test set and model...")
test = pd.read_csv(ROOT / "data/processed/test.csv", low_memory=False)
test = test[
    test["transaction_code"].isin(["P","S"]) &
    test["final_label"].isin([0,1])
].copy()
test["sector"] = test["ticker"].map(SECTOR_MAP).fillna("Other")

model = joblib.load(ROOT / "models/xgb_model.joblib")
scaler = joblib.load(ROOT / "models/xgb_scaler.joblib")
x = test[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
test["xgb_prob"] = model.predict_proba(scaler.transform(x))[:,1]
test["xgb_flagged"] = (test["xgb_prob"] >= 0.5).astype(int)

print("\n=== POST-HOC VALIDATION BY SECTOR ===")
print(f"{'Sector':<15} {'N_flag':>7} {'N_unflag':>9} {'Flag_AR':>10} {'Unflag_AR':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 72)

sector_results = []
for sector in sorted(test["sector"].unique()):
    sdf = test[test["sector"] == sector]
    flagged = sdf[sdf["xgb_flagged"] == 1]["abnormal_return"].dropna()
    unflagged = sdf[sdf["xgb_flagged"] == 0]["abnormal_return"].dropna()

    if len(flagged) < 10 or len(unflagged) < 10:
        continue

    t_stat, p_val = stats.ttest_ind(flagged, unflagged, equal_var=False)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"{sector:<15} {len(flagged):>7} {len(unflagged):>9} "
          f"{flagged.mean():>10.4f} {unflagged.mean():>10.4f} "
          f"{p_val:>10.4f} {sig:>5}")

    sector_results.append({
        "sector": sector, "n_flagged": len(flagged),
        "n_unflagged": len(unflagged),
        "flagged_ar_mean": round(float(flagged.mean()), 6),
        "unflagged_ar_mean": round(float(unflagged.mean()), 6),
        "p_value": round(float(p_val), 6),
        "significant": p_val < 0.05
    })

print("\n=== SIGNED ABNORMAL RETURN BY SECTOR ===")
print(f"{'Sector':<15} {'N_flag':>7} {'Flag_SAR':>10} {'Unflag_SAR':>10} {'p-value':>10} {'Sig':>5}")
print("-" * 62)

for sector in sorted(test["sector"].unique()):
    sdf = test[test["sector"] == sector]
    flagged = sdf[sdf["xgb_flagged"] == 1]["signed_abnormal_return"].dropna()
    unflagged = sdf[sdf["xgb_flagged"] == 0]["signed_abnormal_return"].dropna()

    if len(flagged) < 10 or len(unflagged) < 10:
        continue

    t_stat, p_val = stats.ttest_ind(flagged, unflagged, equal_var=False)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

    print(f"{sector:<15} {len(flagged):>7} {flagged.mean():>10.4f} "
          f"{unflagged.mean():>10.4f} {p_val:>10.4f} {sig:>5}")