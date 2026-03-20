import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle

ROOT = Path(".")
test = pd.read_csv(ROOT / "data/processed/test.csv")
test = test[test["transaction_code"].isin(["P","S"]) & test["final_label"].isin([0,1])].copy()

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
    "cluster_flag", "footnote_length", "value_bucket_num"
]

# XGBoost predictions
xgb = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/xgb_scaler.joblib")
x = test[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
xgb_pred = (xgb.predict_proba(scaler.transform(x))[:,1] >= 0.5).astype(int)

# Fuzzy predictions (recompute inline)
txn_month = test["txn_month"].fillna(0)
pct = test["pct_position_traded"].fillna(0)
has_plan = test.get("has_plan", pd.Series(1, index=test.index)).fillna(1)
others = test["other_insiders_72h"].fillna(0)
days_filing = test["days_to_filing"].fillna(0)

near = np.where(txn_month.isin([1,4,7,10]), 1.0, np.where(txn_month.isin([2,5,8,11]), 0.5, 0.0))
pos = np.clip((pct - 0.01) / 0.09, 0, 1); pos = np.where(pct >= 0.10, 1.0, pos)
noplan = np.where(has_plan == 0, 1.0, 0.0)
coord = np.clip(others / 3.0, 0, 1); coord = np.where(others >= 3, 1.0, coord)
lag = np.clip((days_filing - 1) / 4.0, 0, 1); lag = np.where(days_filing > 5, 1.0, lag)
fuzzy_score = 0.30*near + 0.25*pos + 0.25*noplan + 0.10*coord + 0.10*lag
fuzzy_pred = (fuzzy_score >= 0.5).astype(int)

# OR ensemble
y_true = test["final_label"].astype(int)
or_pred = np.maximum(xgb_pred, fuzzy_pred)

print(f"XGBoost alone:   F1={f1_score(y_true, xgb_pred):.3f}")
print(f"Fuzzy alone:     F1={f1_score(y_true, fuzzy_pred):.3f}")
print(f"OR ensemble:     F1={f1_score(y_true, or_pred):.3f}")

# Venn diagram counts
both = ((xgb_pred==1) & (fuzzy_pred==1) & (y_true==1)).sum()
xgb_only = ((xgb_pred==1) & (fuzzy_pred==0) & (y_true==1)).sum()
fuzzy_only = ((xgb_pred==0) & (fuzzy_pred==1) & (y_true==1)).sum()
neither = ((xgb_pred==0) & (fuzzy_pred==0) & (y_true==1)).sum()
print(f"\nTrue opportunistic caught:")
print(f"  Both:       {both}")
print(f"  XGB only:   {xgb_only}")
print(f"  Fuzzy only: {fuzzy_only}")
print(f"  Neither:    {neither}")
print(f"  Total opp:  {y_true.sum()}")

# Compute Venn numbers
xgb_total = ((xgb_pred==1) & (y_true==1)).sum()
fuzzy_total = ((fuzzy_pred==1) & (y_true==1)).sum()

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Draw circles
circle1 = Circle((3.5, 3), 2.2, fill=True, facecolor='#4C72B0', alpha=0.4, edgecolor='#4C72B0', linewidth=2)
circle2 = Circle((6.5, 3), 2.2, fill=True, facecolor='#DD8452', alpha=0.4, edgecolor='#DD8452', linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Labels inside circles
ax.text(2.2, 3, str(int(xgb_only)), ha='center', va='center', fontsize=22, fontweight='bold', color='#2d4f8a')
ax.text(5.0, 3, str(int(both)), ha='center', va='center', fontsize=22, fontweight='bold', color='#333333')
ax.text(7.8, 3, str(int(fuzzy_only)), ha='center', va='center', fontsize=22, fontweight='bold', color='#a0522d')

# Circle labels
ax.text(2.0, 5.5, 'XGBoost', ha='center', va='center', fontsize=13, fontweight='bold', color='#2d4f8a')
ax.text(8.0, 5.5, 'Fuzzy Logic', ha='center', va='center', fontsize=13, fontweight='bold', color='#a0522d')

# Neither label
ax.text(5.0, 0.4, f'Missed by both: {int(neither)}', ha='center', va='center', fontsize=11, color='#666666')

# Title
ax.set_title(
    f'True Opportunistic Trades Caught (Test Set, n={int(y_true.sum())})\n'
    f'OR Ensemble F1={f1_score(y_true, or_pred):.3f}  |  '
    f'XGB F1={f1_score(y_true, xgb_pred):.3f}  |  '
    f'Fuzzy F1={f1_score(y_true, fuzzy_pred):.3f}',
    fontsize=11, pad=15
)

plt.tight_layout()
plt.savefig('reports/venn_complementarity.png', dpi=160, bbox_inches='tight')
plt.close()
print("\nVenn diagram saved to reports/venn_complementarity.png")