import pandas as pd
import numpy as np
import joblib
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
    "CRM": "Tech", "ORCL": "Tech", "QCOM": "Tech", "TXN": "Tech", "ACN": "Tech",
    "MRNA": "Biotech", "BIIB": "Biotech", "AMGN": "Biotech", "REGN": "Biotech",
    "GILD": "Biotech",
    "LLY": "Pharma", "PFE": "Pharma", "MRK": "Pharma", "ABBV": "Pharma",
    "JNJ": "Pharma", "MDT": "Pharma", "BMY": "Pharma", "TMO": "Pharma",
    "DHR": "Pharma", "ZTS": "Pharma",
    "JPM": "Finance", "GS": "Finance", "MS": "Finance", "BAC": "Finance",
    "WFC": "Finance", "C": "Finance", "AXP": "Finance", "V": "Finance",
    "BK": "Finance", "BLK": "Finance", "SCHW": "Finance", "COF": "Finance",
    "MA": "Finance", "USB": "Finance", "MET": "Finance", "AON": "Finance",
    "ALL": "Finance", "AIG": "Finance", "PYPL": "Finance", "SPG": "Finance",
    "XOM": "Energy", "COP": "Energy", "OXY": "Energy",
    "WMT": "Consumer", "MCD": "Consumer", "KO": "Consumer", "PG": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "MO": "Consumer", "PM": "Consumer",
    "KHC": "Consumer", "MDLZ": "Consumer", "PEP": "Consumer", "TGT": "Consumer",
    "LOW": "Consumer", "DIS": "Consumer", "CMCSA": "Consumer", "CHTR": "Consumer",
    "BA": "Defense", "LMT": "Defense", "GD": "Defense", "RTX": "Defense",
    "UNH": "Healthcare", "HUM": "Healthcare", "CVS": "Healthcare",
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom", "AMT": "Telecom",
    "UNP": "Industrial", "UPS": "Industrial", "FDX": "Industrial",
    "GE": "Industrial", "EMR": "Industrial", "F": "Industrial",
    "GM": "Industrial", "LIN": "Industrial", "DOW": "Industrial",
    "NEE": "Utility", "DUK": "Utility", "SO": "Utility", "EXC": "Utility",
}

print("Loading master labeled dataset...")
master = pd.read_csv(ROOT / "data/processed/master_labeled.csv", low_memory=False)
master = master[
    master["transaction_code"].isin(["P", "S"]) &
    master["final_label"].isin([0, 1])
].copy()
master["sector"] = master["ticker"].map(SECTOR_MAP).fillna("Other")
master["transaction_date"] = pd.to_datetime(master["transaction_date"], errors="coerce")
master["txn_year"] = master["transaction_date"].dt.year

print(f"Total labeled P/S trades: {len(master):,}")
print(f"Tickers: {master['ticker'].nunique()}")

print("\n" + "="*70)
print("FLAG RATE BY SECTOR (full dataset, all years)")
print("="*70)

model = joblib.load(ROOT / "models/xgb_model.joblib")
scaler = joblib.load(ROOT / "models/xgb_scaler.joblib")

x = master[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
master["xgb_prob"] = model.predict_proba(scaler.transform(x))[:, 1]
master["xgb_flagged"] = (master["xgb_prob"] >= 0.5).astype(int)

sector_stats = master.groupby("sector").agg(
    n_trades=("xgb_flagged", "count"),
    flag_rate=("xgb_flagged", "mean"),
    mean_prob=("xgb_prob", "mean"),
    opp_rate=("final_label", "mean"),
    n_tickers=("ticker", "nunique"),
).sort_values("flag_rate", ascending=False).round(3)

print(sector_stats.to_string())

print("\n" + "="*70)
print("FLAG RATE BY YEAR (full dataset, all sectors)")
print("="*70)

year_stats = master.groupby("txn_year").agg(
    n_trades=("xgb_flagged", "count"),
    flag_rate=("xgb_flagged", "mean"),
    mean_prob=("xgb_prob", "mean"),
    opp_rate=("final_label", "mean"),
).sort_values("txn_year").round(3)

print(year_stats.to_string())

print("\n" + "="*70)
print("BIOTECH FLAG RATE BY YEAR")
print("="*70)

bio_stats = master[master["sector"] == "Biotech"].groupby("txn_year").agg(
    n_trades=("xgb_flagged", "count"),
    flag_rate=("xgb_flagged", "mean"),
    mean_prob=("xgb_prob", "mean"),
    opp_rate=("final_label", "mean"),
).sort_values("txn_year").round(3)

print(bio_stats.to_string())

print("\n" + "="*70)
print("DEFENSE FLAG RATE BY YEAR")
print("="*70)

def_stats = master[master["sector"] == "Defense"].groupby("txn_year").agg(
    n_trades=("xgb_flagged", "count"),
    flag_rate=("xgb_flagged", "mean"),
    mean_prob=("xgb_prob", "mean"),
).sort_values("txn_year").round(3)

print(def_stats.to_string())

print("\n" + "="*70)
print("FLAG RATE BY SECTOR AND YEAR (pivot)")
print("="*70)

pivot = master.groupby(["sector", "txn_year"])["xgb_flagged"].mean().unstack(
    level="txn_year"
).round(3)
pivot = pivot.reindex(sector_stats.index)
print(pivot.to_string())