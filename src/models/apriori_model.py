"""Apriori association-rule mining for interpretable opportunistic-trade signals.

This module:
- Loads train/test datasets and filters to open-market, labeled transactions.
- Builds binary transactional features using train-set statistics only.
- Mines frequent itemsets and association rules with Apriori.
- Keeps rules with consequent {'opportunistic'} and scores test rows by max rule confidence.
- Evaluates with AUC-PR, F1, precision, recall, and confusion matrix.
- Saves top rules, model artifact, and JSON summary for reproducible standalone execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.5


def _ensure_output_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Loading {dataset_name} dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] {dataset_name} shape before filtering: {df.shape}")
    return df


def _filter_open_market_labeled(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Filtering {dataset_name} to open-market labeled rows...")

    filtered = df.copy()

    if "transaction_code" in filtered.columns:
        filtered = filtered[filtered["transaction_code"].isin(["P", "S"])]
    else:
        print(f"[WARN] {dataset_name}: missing 'transaction_code'; skipping P/S filter.")

    if "final_label" not in filtered.columns:
        raise ValueError(f"{dataset_name}: missing required column 'final_label'.")

    filtered = filtered[filtered["final_label"].isin([0, 1])].copy()
    print(f"[INFO] {dataset_name} shape after filtering: {filtered.shape}")
    return filtered


def _safe_numeric_series(df: pd.DataFrame, column: str, default: float, dataset_name: str) -> pd.Series:
    if column not in df.columns:
        print(f"[WARN] {dataset_name}: missing '{column}'. Using default={default}.")
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _compute_train_statistics(train_df: pd.DataFrame) -> dict[str, float]:
    if "log_total_value" in train_df.columns:
        median_log_total_value = float(pd.to_numeric(train_df["log_total_value"], errors="coerce").median())
    else:
        print("[WARN] train: missing 'log_total_value'. median_log_total_value set to NaN.")
        median_log_total_value = float("nan")

    stats = {
        "median_log_total_value": median_log_total_value,
    }
    print(f"[INFO] Train statistics: {stats}")
    return stats


def _build_binary_frame(df: pd.DataFrame, stats: dict[str, float], dataset_name: str) -> pd.DataFrame:
    print(f"[INFO] Building binary Apriori basket for {dataset_name}...")

    pct_position = _safe_numeric_series(df, "pct_position_traded", default=0.0, dataset_name=dataset_name)
    txn_month = _safe_numeric_series(df, "txn_month", default=0.0, dataset_name=dataset_name)
    trade_direction = _safe_numeric_series(df, "trade_direction", default=1.0, dataset_name=dataset_name)
    role_seniority = _safe_numeric_series(df, "role_seniority", default=0.0, dataset_name=dataset_name)
    other_insiders_72h = _safe_numeric_series(df, "other_insiders_72h", default=0.0, dataset_name=dataset_name)
    log_total_value = _safe_numeric_series(df, "log_total_value", default=0.0, dataset_name=dataset_name)
    days_to_filing = _safe_numeric_series(df, "days_to_filing", default=0.0, dataset_name=dataset_name)

    median_log_total_value = stats.get("median_log_total_value", float("nan"))
    if np.isnan(median_log_total_value):
        print("[WARN] median_log_total_value is NaN; 'large_dollar' will be all zeros.")
        large_dollar = pd.Series(0, index=df.index, dtype=int)
    else:
        large_dollar = (log_total_value > median_log_total_value).astype(int)

    basket = pd.DataFrame(
        {
            "large_trade": (pct_position > 0.05).astype(int),
            "near_earnings": txn_month.isin([1, 4, 7, 10]).astype(int),
            "sell_transaction": (trade_direction == -1).astype(int),
            "senior_insider": (role_seniority >= 4).astype(int),
            "coordinated": (other_insiders_72h >= 2).astype(int),
            "large_dollar": large_dollar.astype(int),
            "late_filing": (days_to_filing > 2).astype(int),
            "opportunistic": (df["final_label"] == 1).astype(int),
        },
        index=df.index,
    )

    return basket


def _format_itemset(itemset: frozenset) -> str:
    return " & ".join(sorted(itemset))


def _mine_rules(train_basket: pd.DataFrame) -> pd.DataFrame:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        print("[ERROR] Missing dependency 'mlxtend'. Install it with: pip install mlxtend")
        raise SystemExit(1) from exc

    print(f"[INFO] Running Apriori with min_support={MIN_SUPPORT}...")
    frequent_itemsets = apriori(
        train_basket.astype(bool),
        min_support=MIN_SUPPORT,
        use_colnames=True,
    )
    print(f"[INFO] Frequent itemsets found: {len(frequent_itemsets)}")

    if frequent_itemsets.empty:
        return pd.DataFrame()

    print(f"[INFO] Generating association rules with min_confidence={MIN_CONFIDENCE}...")
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["consequents"] == frozenset({"opportunistic"})].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules


def _print_and_save_top_rules(rules: pd.DataFrame, top_n: int = 15) -> tuple[pd.DataFrame, list[dict[str, float | str]]]:
    top_rules_path = REPORTS_DIR / "apriori_rules.csv"

    if rules.empty:
        print("[WARN] No rules found with consequent {'opportunistic'}.")
        empty_columns = ["antecedents", "consequents", "support", "confidence", "lift"]
        pd.DataFrame(columns=empty_columns).to_csv(top_rules_path, index=False)
        print(f"[INFO] Saved empty rules file: {top_rules_path}")
        return pd.DataFrame(columns=empty_columns), []

    top_rules = rules.head(top_n).copy()
    top_rules["antecedents"] = top_rules["antecedents"].apply(_format_itemset)
    top_rules["consequents"] = top_rules["consequents"].apply(_format_itemset)

    print("[INFO] Top Apriori rules (by lift):")
    for _, row in top_rules.iterrows():
        print(
            "  "
            f"antecedents={row['antecedents']} | "
            f"support={row['support']:.4f} | "
            f"confidence={row['confidence']:.4f} | "
            f"lift={row['lift']:.4f}"
        )

    top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_csv(
        top_rules_path,
        index=False,
    )
    print(f"[INFO] Saved top rules CSV: {top_rules_path}")

    top_rules_json = [
        {
            "antecedents": str(row["antecedents"]),
            "consequents": str(row["consequents"]),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
        }
        for _, row in top_rules.iterrows()
    ]

    return top_rules, top_rules_json


def _score_with_rules(test_basket: pd.DataFrame, rules: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(len(test_basket), dtype=float)
    if rules.empty:
        return scores

    for _, row in rules.iterrows():
        antecedents = list(row["antecedents"])
        confidence = float(row["confidence"])

        if not antecedents:
            continue

        missing = [item for item in antecedents if item not in test_basket.columns]
        if missing:
            print(f"[WARN] Skipping rule with missing antecedent columns in test set: {missing}")
            continue

        fires = test_basket[antecedents].all(axis=1).to_numpy()
        scores = np.where(fires, np.maximum(scores, confidence), scores)

    return np.clip(scores, 0.0, 1.0)


def _compute_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_true_num = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_score_num = pd.Series(y_score, index=y_true.index).fillna(0.0).clip(0, 1)
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


def main() -> None:
    leakage_found = LEAKAGE_COLS.intersection(set(FEATURE_COLS))
    if leakage_found:
        raise ValueError(f"Leakage columns found in FEATURE_COLS: {leakage_found}")

    print("[INFO] Starting Apriori association-rule pipeline...")
    _ensure_output_directories()

    train_raw = _load_csv(TRAIN_PATH, "train")
    test_raw = _load_csv(TEST_PATH, "test")

    train_df = _filter_open_market_labeled(train_raw, "train")
    test_df = _filter_open_market_labeled(test_raw, "test")

    stats = _compute_train_statistics(train_df)

    train_basket = _build_binary_frame(train_df, stats, "train")
    test_basket = _build_binary_frame(test_df, stats, "test")

    rules = _mine_rules(train_basket)
    top_rules_df, top_rules_json = _print_and_save_top_rules(rules, top_n=15)

    print("[INFO] Scoring test transactions using max confidence of fired rules...")
    scores = _score_with_rules(test_basket.drop(columns=["opportunistic"], errors="ignore"), rules)

    print("[INFO] Evaluating Apriori model on test set...")
    metrics = _compute_metrics(y_true=test_df["final_label"], y_score=scores, threshold=0.5)
    print(
        "[RESULT] "
        f"AUC-PR={metrics['auc_pr']:.6f}, "
        f"F1={metrics['f1']:.6f}, "
        f"Precision={metrics['precision']:.6f}, "
        f"Recall={metrics['recall']:.6f}"
    )
    print(f"[RESULT] Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    print("[INFO] Saving Apriori model artifact...")
    model_artifact = {
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "train_statistics": stats,
        "rules": rules,
        "binary_columns": [
            "large_trade",
            "near_earnings",
            "sell_transaction",
            "senior_insider",
            "coordinated",
            "large_dollar",
            "late_filing",
        ],
    }
    joblib.dump(model_artifact, MODELS_DIR / "apriori_model.joblib")

    results = {
        "model": "apriori",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "n_rules_found": int(len(rules)),
        "top_rules": top_rules_json,
    }

    results_path = REPORTS_DIR / "model_results_apriori.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[INFO] Saved results JSON: {results_path}")

    print("[INFO] Apriori pipeline completed successfully.")


if __name__ == "__main__":
    main()
