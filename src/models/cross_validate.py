"""
Walk-forward cross validation for XGBoost on insider trading data.

Runs three temporal windows with fixed hyperparameters to evaluate
model performance across different time periods.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]

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

WINDOWS = [
    {"name": "window1", "train_end": 2020, "val_year": 2021, "test_year": 2022},
    {"name": "window2", "train_end": 2021, "val_year": 2022, "test_year": 2023},
    {"name": "window3", "train_end": 2022, "val_year": 2023, "test_year": 2024},
]


def load_and_filter_data() -> pd.DataFrame:
    df = pd.read_csv(
        ROOT_DIR / "data" / "processed" / "master_labeled.csv",
        low_memory=False,
    )
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["txn_year"] = df["transaction_date"].dt.year
    df = df[df["transaction_code"].isin(["P", "S"])].copy()
    df = df[df["final_label"].isin([0, 1])].copy()
    return df


def fill_nan_inf(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def save_splits(window_name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out = ROOT_DIR / "data" / "processed" / "splits" / window_name
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"  Splits saved -> data/processed/splits/{window_name}/")


def run_cross_validation() -> None:
    print("\nLoading data...")
    master = load_and_filter_data()
    print(f"Loaded {len(master):,} rows after filtering (P/S, labeled)")

    results: dict = {"windows": [], "summary": {}}
    auc_pr_scores: list[float] = []
    f1_scores: list[float] = []

    for window in WINDOWS:
        print(f"\n{'='*70}")
        print(
            f"Running {window['name'].upper()}: train_end={window['train_end']}, "
            f"val_year={window['val_year']}, test_year={window['test_year']}"
        )
        print("=" * 70)

        train = master[master["txn_year"] <= window["train_end"]].copy()
        val = master[master["txn_year"] == window["val_year"]].copy()
        test = master[master["txn_year"] == window["test_year"]].copy()

        if len(train) < 100 or len(val) < 100 or len(test) < 100:
            print(
                f"Skipping: insufficient rows "
                f"(train={len(train)}, val={len(val)}, test={len(test)})"
            )
            continue

        # Save splits to disk
        save_splits(window["name"], train, val, test)

        X_train = fill_nan_inf(train[FEATURE_COLS].copy())
        X_val = fill_nan_inf(val[FEATURE_COLS].copy())
        X_test = fill_nan_inf(test[FEATURE_COLS].copy())

        y_train = train["final_label"].values
        y_val = val["final_label"].values
        y_test = test["final_label"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        routine_count = int((y_train == 0).sum())
        opportunistic_count = int((y_train == 1).sum())
        scale_pos_weight = (
            routine_count / opportunistic_count if opportunistic_count > 0 else 1.0
        )

        train_dmatrix = xgb.DMatrix(X_train_scaled, label=y_train)
        val_dmatrix = xgb.DMatrix(X_val_scaled, label=y_val)
        test_dmatrix = xgb.DMatrix(X_test_scaled, label=y_test)

        params = {
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "seed": 42,  # XGBoost uses seed, not random_state
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
        }

        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=100,
            evals=[(val_dmatrix, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        y_pred_proba = bst.predict(test_dmatrix)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Use average_precision_score to match other models
        auc_pr = float(average_precision_score(y_test, y_pred_proba))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        precision_val = float(precision_score(y_test, y_pred, zero_division=0))
        recall_val = float(recall_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        window_result = {
            "name": window["name"],
            "train_end": window["train_end"],
            "val_year": window["val_year"],
            "test_year": window["test_year"],
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "scale_pos_weight": float(scale_pos_weight),
            "auc_pr": auc_pr,
            "f1": f1,
            "precision": precision_val,
            "recall": recall_val,
            "confusion_matrix": cm,
        }
        results["windows"].append(window_result)
        auc_pr_scores.append(auc_pr)
        f1_scores.append(f1)

        print(f"\nData split:")
        print(
            f"  Train: {len(train):>6,} rows | Routine: {routine_count:>5,} | "
            f"Opportunistic: {opportunistic_count:>5,}"
        )
        print(f"  Val:   {len(val):>6,} rows | scale_pos_weight: {scale_pos_weight:.4f}")
        print(f"  Test:  {len(test):>6,} rows")
        print(f"\nTest set metrics:")
        print(f"  AUC-PR:    {auc_pr:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision_val:.4f}")
        print(f"  Recall:    {recall_val:.4f}")
        print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    if auc_pr_scores:
        summary = {
            "mean_auc_pr": float(np.mean(auc_pr_scores)),
            "std_auc_pr": float(np.std(auc_pr_scores)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
        }
        results["summary"] = summary

        print(f"\n{'='*70}")
        print("CROSS-VALIDATION SUMMARY (across all windows)")
        print("=" * 70)
        print(f"Mean AUC-PR: {summary['mean_auc_pr']:.4f} ± {summary['std_auc_pr']:.4f}")
        print(f"Mean F1:     {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

    output_path = ROOT_DIR / "reports" / "cross_validation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to reports/cross_validation_results.json")


if __name__ == "__main__":
    run_cross_validation()