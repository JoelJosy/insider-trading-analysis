import pandas as pd
import glob

# Load all labeled files
files = glob.glob("data/processed/*_form4_features_labeled.csv")
dfs = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    dfs.append(df)

master = pd.concat(dfs, ignore_index=True)
master['txn_year'] = pd.to_datetime(
    master['transaction_date'], errors='coerce'
).dt.year

# Drop 2025-2026 rows
master = master[master['txn_year'] <= 2024]

# Save master
master.to_csv("data/processed/master_labeled.csv", index=False)

# Create splits — P/S only for training
ps = master[master['transaction_code'].isin(['P', 'S'])].copy()
ps = ps[ps['final_label'].isin([0, 1])]  # drop uncertain

train = ps[ps['txn_year'] <= 2022]
val   = ps[ps['txn_year'] == 2023]
test  = ps[ps['txn_year'] == 2024]

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv",     index=False)
test.to_csv("data/processed/test.csv",   index=False)

print(f"Master (all, excl 2025+): {len(master):,} rows")
print(f"\nP/S only, labeled splits:")
print(f"  Train (≤2022): {len(train):,} rows | "
      f"opp: {(train['final_label']==1).sum()} | "
      f"routine: {(train['final_label']==0).sum()}")
print(f"  Val   (2023):  {len(val):,} rows | "
      f"opp: {(val['final_label']==1).sum()} | "
      f"routine: {(val['final_label']==0).sum()}")
print(f"  Test  (2024):  {len(test):,} rows | "
      f"opp: {(test['final_label']==1).sum()} | "
      f"routine: {(test['final_label']==0).sum()}")