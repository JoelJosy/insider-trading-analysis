"""Fuzzy-logic suspicion scoring for SEC Form 4 insider-trade anomaly detection.

This module:
- Loads train/validation/test datasets.
- Computes a continuous fuzzy suspicion score for every row (including uncertain labels).
- Evaluates on open-market (P/S) test rows with final_label in {0, 1}.
- Reports AUC-PR, F1, precision, recall, and confusion matrix.
- Saves score distribution plots, full scored dataset, model artifact, and JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

FEATURE_COLS = [
    "log_shares",
    "log_total_value",
    "log_shares_owned_after",
    "total_value_is_imputed",
    "trade_direction",
    "signed_value",
    "days_to_filing",
    "pct_position_traded",
    "txn_day_of_week",
    "txn_month",
    "txn_quarter",
    "role_seniority",
    "is_ceo",
    "is_cfo",
    "is_coo",
    "is_director_only",
    "is_open_market",
    "is_derivative",
    "insider_total_trades",
    "insider_avg_trade_value",
    "insider_buy_sell_ratio",
    "insider_tenure_days",
    "trades_7d",
    "trades_30d",
    "trades_90d",
    "buy_count_7d",
    "sell_count_7d",
    "buy_count_30d",
    "sell_count_30d",
    "buy_count_90d",
    "sell_count_90d",
    "net_value_7d",
    "net_value_30d",
    "net_value_90d",
    "days_since_last_trade",
    "consecutive_direction",
    "trade_frequency_90d",
    "other_insiders_72h",
    "same_dir_insiders_72h",
    "coordination_score",
    "cluster_flag",
    "footnote_length",
    "value_bucket_num",
]

LEAKAGE_COLS = {
    "future_close",
    "forward_return",
    "benchmark_return",
    "abnormal_return",
    "signed_abnormal_return",
    "price_signal",
    "earnings_proximity_flag",
    "enforcement_followup_flag",
    "has_plan",
    "footnote_has_plan",
    "footnote_routine_score",
    "footnote_routine_hits",
    "footnote_informed_hits",
    "footnote_has_option",
    "informed_label",
    "routine_label",
    "cohen_label",
    "plan_override",
}

RANDOM_STATE = 42
THRESHOLD = 0.5

FUZZY_WEIGHTS = {
    "near_earnings": 0.30,
    "position_size": 0.25,
    "no_plan": 0.25,
    "coordination": 0.10,
    "filing_lag": 0.10,
}


def _ensure_output_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Loading {dataset_name} dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {dataset_name} shape: {df.shape}")
    return df


def _safe_series(df: pd.DataFrame, column: str, default: float | int, dataset_name: str) -> pd.Series:
    if column not in df.columns:
        print(f"[WARN] {dataset_name}: missing '{column}'. Using default={default}.")
        return pd.Series(default, index=df.index)
    return df[column]


def near_earnings_membership(txn_month: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(txn_month), errors="coerce").fillna(0).to_numpy(dtype=float)
    membership = np.zeros_like(values, dtype=float)

    membership[np.isin(values, [1, 4, 7, 10])] = 1.0
    membership[np.isin(values, [2, 5, 8, 11])] = 0.5
    membership[np.isin(values, [3, 6, 9, 12])] = 0.0

    return membership


def position_size_membership(pct: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(pct), errors="coerce").fillna(0).to_numpy(dtype=float)
    membership = np.zeros_like(values, dtype=float)

    membership[values >= 0.10] = 1.0
    middle_mask = (values > 0.01) & (values < 0.10)
    membership[middle_mask] = (values[middle_mask] - 0.01) / 0.09

    return np.clip(membership, 0.0, 1.0)


def no_plan_membership(has_plan: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(has_plan), errors="coerce").fillna(1).to_numpy(dtype=float)
    membership = np.where(values == 0, 1.0, 0.0)
    return membership


def coordination_membership(n_others: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(n_others), errors="coerce").fillna(0).to_numpy(dtype=float)
    membership = np.zeros_like(values, dtype=float)

    membership[values >= 3] = 1.0
    middle_mask = (values > 0) & (values < 3)
    membership[middle_mask] = values[middle_mask] / 3.0

    return np.clip(membership, 0.0, 1.0)


def filing_lag_membership(days: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(days), errors="coerce").fillna(0).to_numpy(dtype=float)
    membership = np.zeros_like(values, dtype=float)

    membership[values > 5] = 1.0
    middle_mask = (values > 1) & (values <= 5)
    membership[middle_mask] = (values[middle_mask] - 1.0) / 4.0

    return np.clip(membership, 0.0, 1.0)


def compute_fuzzy_scores(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Computing fuzzy scores for {dataset_name}...")
    working = df.copy()

    txn_month = _safe_series(working, "txn_month", default=0, dataset_name=dataset_name)
    pct_position_traded = _safe_series(working, "pct_position_traded", default=0.0, dataset_name=dataset_name)
    has_plan = _safe_series(working, "has_plan", default=1, dataset_name=dataset_name)
    other_insiders_72h = _safe_series(working, "other_insiders_72h", default=0, dataset_name=dataset_name)
    days_to_filing = _safe_series(working, "days_to_filing", default=0, dataset_name=dataset_name)

    near_earnings_score = near_earnings_membership(txn_month)
    position_size_score = position_size_membership(pct_position_traded)
    no_plan_score = no_plan_membership(has_plan)
    coordination_score = coordination_membership(other_insiders_72h)
    filing_lag_score = filing_lag_membership(days_to_filing)

    fuzzy_score = (
        FUZZY_WEIGHTS["near_earnings"] * near_earnings_score
        + FUZZY_WEIGHTS["position_size"] * position_size_score
        + FUZZY_WEIGHTS["no_plan"] * no_plan_score
        + FUZZY_WEIGHTS["coordination"] * coordination_score
        + FUZZY_WEIGHTS["filing_lag"] * filing_lag_score
    )

    working["fuzzy_suspicion_score"] = np.clip(fuzzy_score, 0.0, 1.0)
    return working


def _filter_eval_rows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Creating evaluation subset for {dataset_name} (transaction_code in P/S and final_label in 0/1)...")
    eval_df = df.copy()

    if "transaction_code" in eval_df.columns:
        eval_df = eval_df[eval_df["transaction_code"].isin(["P", "S"])]
    else:
        print(f"[WARN] {dataset_name}: missing 'transaction_code'; skipping P/S filter.")

    if "final_label" not in eval_df.columns:
        raise ValueError(f"{dataset_name}: missing required column 'final_label' for evaluation.")

    eval_df = eval_df[eval_df["final_label"].isin([0, 1])].copy()
    print(f"[INFO] Evaluation subset shape: {eval_df.shape}")
    return eval_df


def _compute_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float = THRESHOLD) -> dict[str, float | list[list[int]]]:
    y_true_num = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_score_num = pd.to_numeric(y_score, errors="coerce").fillna(0.0).clip(0, 1)
    y_pred = (y_score_num >= threshold).astype(int)

    if y_true_num.nunique() > 1:
        auc_pr = float(average_precision_score(y_true_num, y_score_num))
    else:
        print("[WARN] Only one class present in y_true. AUC-PR set to NaN.")
        auc_pr = float("nan")

    metrics: dict[str, float | list[list[int]]] = {
        "auc_pr": auc_pr,
        "f1": float(f1_score(y_true_num, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_num, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_num, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_num, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics


def _save_score_distribution_plot(eval_df: pd.DataFrame, output_path: Path) -> None:
    routine_scores = eval_df.loc[eval_df["final_label"] == 0, "fuzzy_suspicion_score"]
    opportunistic_scores = eval_df.loc[eval_df["final_label"] == 1, "fuzzy_suspicion_score"]

    plt.figure(figsize=(9, 5))
    plt.hist(routine_scores, bins=30, alpha=0.6, label="Routine (label=0)", density=True)
    plt.hist(opportunistic_scores, bins=30, alpha=0.6, label="Opportunistic (label=1)", density=True)
    plt.title("Fuzzy Suspicion Score Distribution (Test Set)")
    plt.xlabel("fuzzy_suspicion_score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved fuzzy score distribution plot: {output_path}")


def main() -> None:
    leakage_found = LEAKAGE_COLS.intersection(set(FEATURE_COLS))
    if leakage_found:
        raise ValueError(f"Leakage columns found in FEATURE_COLS: {leakage_found}")

    print("[INFO] Starting fuzzy-logic scoring pipeline...")
    _ensure_output_directories()

    train_df = _load_csv(TRAIN_PATH, "train")
    val_df = _load_csv(VAL_PATH, "validation")
    test_df = _load_csv(TEST_PATH, "test")

    train_scored = compute_fuzzy_scores(train_df, "train")
    train_scored["dataset_split"] = "train"

    val_scored = compute_fuzzy_scores(val_df, "validation")
    val_scored["dataset_split"] = "val"

    test_scored = compute_fuzzy_scores(test_df, "test")
    test_scored["dataset_split"] = "test"

    full_scored = pd.concat([train_scored, val_scored, test_scored], axis=0, ignore_index=True)

    output_scored_path = Path("data/processed/master_fuzzy_scored.csv")
    output_scored_path.parent.mkdir(parents=True, exist_ok=True)
    full_scored.to_csv(output_scored_path, index=False)
    print(f"[INFO] Saved full fuzzy-scored dataset: {output_scored_path}")

    eval_df = _filter_eval_rows(test_scored, "test")

    print("[INFO] Evaluating fuzzy model on test subset...")
    metrics = _compute_metrics(
        y_true=eval_df["final_label"],
        y_score=eval_df["fuzzy_suspicion_score"],
        threshold=THRESHOLD,
    )
    print(
        "[RESULT] "
        f"AUC-PR={metrics['auc_pr']:.6f}, "
        f"F1={metrics['f1']:.6f}, "
        f"Precision={metrics['precision']:.6f}, "
        f"Recall={metrics['recall']:.6f}"
    )
    print(f"[RESULT] Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    _save_score_distribution_plot(eval_df, REPORTS_DIR / "fuzzy_score_distribution.png")

    if "transaction_code" in full_scored.columns:
        print("[INFO] Mean fuzzy score by transaction_code:")
        score_by_code = (
            full_scored.groupby("transaction_code")["fuzzy_suspicion_score"]
            .mean()
            .sort_values(ascending=False)
        )
        print(score_by_code.to_string())
    else:
        print("[WARN] full_scored missing 'transaction_code'; cannot compute mean score by transaction type.")

    mean_score_routine = float(eval_df.loc[eval_df["final_label"] == 0, "fuzzy_suspicion_score"].mean())
    mean_score_opportunistic = float(eval_df.loc[eval_df["final_label"] == 1, "fuzzy_suspicion_score"].mean())

    print("[INFO] Saving fuzzy model artifact...")
    fuzzy_artifact = {
        "weights": FUZZY_WEIGHTS,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(fuzzy_artifact, MODELS_DIR / "fuzzy_model.joblib")

    results = {
        "model": "fuzzy",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "threshold_used": THRESHOLD,
        "mean_score_routine": mean_score_routine,
        "mean_score_opportunistic": mean_score_opportunistic,
    }

    results_path = REPORTS_DIR / "model_results_fuzzy.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[INFO] Saved results JSON: {results_path}")

    print("[INFO] Fuzzy-logic pipeline completed successfully.")


if __name__ == "__main__":
    main()
