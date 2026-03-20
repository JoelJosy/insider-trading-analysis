import pandas as pd
import numpy as np
import joblib

ROOT = "."
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
    "IBM": "Tech", "CSCO": "Tech", "META": "Tech",
    "MRNA": "Biotech", "BIIB": "Biotech", "AMGN": "Biotech", "REGN": "Biotech",
    "LLY": "Pharma", "PFE": "Pharma", "MRK": "Pharma", "ABBV": "Pharma",
    "JNJ": "Pharma", "MDT": "Pharma",
    "JPM": "Finance", "GS": "Finance", "MS": "Finance", "BAC": "Finance",
    "WFC": "Finance", "C": "Finance", "AXP": "Finance", "V": "Finance",
    "XOM": "Energy", "COP": "Energy",
    "WMT": "Consumer", "MCD": "Consumer", "KO": "Consumer", "PG": "Consumer",
    "BA": "Defense/Aero", "LMT": "Defense/Aero",
    "UNH": "Healthcare",
}

master = pd.read_csv(f"{ROOT}/data/processed/master_labeled.csv", low_memory=False)
master = master[master["transaction_code"].isin(["P","S"]) & master["final_label"].isin([0,1])].copy()
master["sector"] = master["ticker"].map(SECTOR_MAP).fillna("Other")
master["transaction_date"] = pd.to_datetime(master["transaction_date"], errors="coerce")
master["txn_year"] = master["transaction_date"].dt.year

model = joblib.load(f"{ROOT}/models/xgb_model.joblib")
scaler = joblib.load(f"{ROOT}/models/xgb_scaler.joblib")

x = master[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0)
master["xgb_prob"] = model.predict_proba(scaler.transform(x))[:,1]
master["xgb_flagged"] = (master["xgb_prob"] >= 0.5).astype(int)

# By sector
print("=== FLAG RATE BY SECTOR ===")
sector = master.groupby("sector").agg(
    n_trades=("xgb_flagged","count"),
    flag_rate=("xgb_flagged","mean"),
    mean_prob=("xgb_prob","mean"),
    opp_rate=("final_label","mean")
).sort_values("flag_rate", ascending=False)
print(sector.round(3).to_string())

# By year
print("\n=== FLAG RATE BY YEAR ===")
year = master.groupby("txn_year").agg(
    n_trades=("xgb_flagged","count"),
    flag_rate=("xgb_flagged","mean"),
    mean_prob=("xgb_prob","mean"),
    opp_rate=("final_label","mean")
).sort_values("txn_year")
print(year.round(3).to_string())

# Biotech by year (MRNA COVID story)
print("\n=== BIOTECH FLAG RATE BY YEAR ===")
bio = master[master["sector"]=="Biotech"].groupby("txn_year").agg(
    n_trades=("xgb_flagged","count"),
    flag_rate=("xgb_flagged","mean"),
    mean_prob=("xgb_prob","mean"),
).sort_values("txn_year")
print(bio.round(3).to_string())