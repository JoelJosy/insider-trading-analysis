import pandas as pd, joblib, numpy as np
from pathlib import Path

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
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom",
    "CMCSA": "Telecom", "CHTR": "Telecom", "AMT": "Telecom"
}

test = pd.read_csv('data/processed/test.csv', low_memory=False)
test = test[test['transaction_code'].isin(['P','S']) & test['final_label'].isin([0,1])].copy()
test['sector'] = test['ticker'].map(SECTOR_MAP)

model = joblib.load('models/xgb_model.joblib')
scaler = joblib.load('models/xgb_scaler.joblib')
x = test[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
test['xgb_prob'] = model.predict_proba(scaler.transform(x))[:,1]
test['xgb_flagged'] = (test['xgb_prob'] >= 0.5).astype(int)

telecom = test[test['sector'] == 'Telecom']
print(telecom.groupby(['ticker','xgb_flagged']).size().unstack(fill_value=0))
print("\nAbnormal return by ticker (flagged only):")
print(telecom[telecom['xgb_flagged']==1].groupby('ticker')['abnormal_return'].agg(['mean','count']))