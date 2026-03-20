from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

SPLITS = ("train", "val", "test")
LABEL_COL = "final_label"
TRANSACTION_CODE_COL = "transaction_code"
VALID_TRANSACTION_CODES = {"P", "S"}
VALID_LABELS = {0, 1}
BASELINE_AUC_PR = 0.6650  

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

FEATURE_GROUPS = {
    "trade": [
        "log_shares",
        "log_total_value",
        "log_shares_owned_after",
        "total_value_is_imputed",
        "signed_value",
        "pct_position_traded",
        "value_bucket_num",
        "trade_direction",
    ],
    "temporal": [
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
    ],
    "insider_profile": [
        "role_seniority",
        "is_ceo",
        "is_cfo",
        "is_coo",
        "is_director_only",
        "insider_total_trades",
        "insider_avg_trade_value",
        "insider_buy_sell_ratio",
        "insider_tenure_days",
    ],
    "network": [
        "other_insiders_72h",
        "same_dir_insiders_72h",
        "coordination_score",
        "cluster_flag",
    ],
    "timing": ["txn_day_of_week", "txn_month", "txn_quarter", "days_to_filing"],
    "text": ["footnote_length"],
    "market_context": ["is_open_market", "is_derivative", "total_value_is_imputed"],
}

# Best params from grid search
XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 7,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.4436,  # 3121/2162 from new train split
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "early_stopping_rounds": 20,
    "n_jobs": -1,
}


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_feature_setup() -> None:
    leakage_overlap = sorted(set(FEATURE_COLS) & LEAKAGE_COLS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns present in feature list: {leakage_overlap}")


def _load_split(split_name: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required split file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = {TRANSACTION_CODE_COL, LABEL_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    filtered = df.copy()
    filtered[TRANSACTION_CODE_COL] = (
        filtered[TRANSACTION_CODE_COL].astype(str).str.upper().str.strip()
    )
    filtered[LABEL_COL] = pd.to_numeric(filtered[LABEL_COL], errors="coerce")

    mask = (
        filtered[TRANSACTION_CODE_COL].isin(VALID_TRANSACTION_CODES)
        & filtered[LABEL_COL].isin(VALID_LABELS)
    )
    filtered = filtered.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"{csv_path} has no rows after applying transaction/label filters")

    filtered[LABEL_COL] = filtered[LABEL_COL].astype(int)
    return filtered


def _prepare_features_and_labels(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = df[LABEL_COL].astype(int)
    return x, y


def _evaluate(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _fit_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    x_train_raw, y_train = _prepare_features_and_labels(train_df, feature_cols)
    x_val_raw, y_val = _prepare_features_and_labels(val_df, feature_cols)
    x_test_raw, y_test = _prepare_features_and_labels(test_df, feature_cols)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    y_prob = model.predict_proba(x_test)[:, 1]
    return _evaluate(y_test, y_prob)


def _print_sorted_table(rows: list[dict]) -> None:
    print("\nAblation results sorted by AUC-PR drop (descending):")
    print(f"{'group_removed':<18} {'features_left':>13} {'auc_pr':>10} {'drop':>10}")
    print("-" * 56)
    for row in rows:
        print(
            f"{str(row['group']):<18} "
            f"{int(row['features_left']):>13d} "
            f"{float(row['auc_pr']):>10.4f} "
            f"{float(row['auc_pr_drop']):>10.4f}"
        )


def _save_drop_chart(rows: list[dict], output_path: Path) -> None:
    labels = [str(row["group"]) for row in rows]
    drops = [float(row["auc_pr_drop"]) for row in rows]
    colors = ["#d9534f" if d > 0 else "#5cb85c" for d in drops]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, drops, color=colors)
    plt.gca().invert_yaxis()
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel(f"AUC-PR Drop vs Baseline ({BASELINE_AUC_PR:.4f})")
    plt.title("XGBoost Ablation Study: Feature Group Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved ablation chart: {output_path}")


def main() -> None:
    _ensure_output_dirs()
    _validate_feature_setup()

    print("Loading split datasets...")
    datasets = {split: _load_split(split) for split in SPLITS}
    print(
        f"Train: {len(datasets['train'])} | "
        f"Val: {len(datasets['val'])} | "
        f"Test: {len(datasets['test'])}"
    )

    print(f"\nBaseline AUC-PR: {BASELINE_AUC_PR:.4f}")
    print("Running ablations (remove one feature group at a time)...\n")

    ablation_rows: list[dict] = []
    for group_name, group_features in FEATURE_GROUPS.items():
        remaining_features = [col for col in FEATURE_COLS if col not in set(group_features)]
        print(
            f"Ablating '{group_name}' "
            f"({len(group_features)} removed, {len(remaining_features)} kept)..."
        )

        metrics = _fit_and_evaluate(
            datasets["train"],
            datasets["val"],
            datasets["test"],
            remaining_features,
        )

        auc_pr_drop = BASELINE_AUC_PR - metrics["auc_pr"]
        ablation_rows.append(
            {
                "group": group_name,
                "removed_features": group_features,
                "features_left": len(remaining_features),
                "auc_pr": metrics["auc_pr"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "auc_pr_drop": auc_pr_drop,
            }
        )
        print(
            f"  AUC-PR={metrics['auc_pr']:.4f}  "
            f"drop={auc_pr_drop:+.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}"
        )

    sorted_rows = sorted(
        ablation_rows,
        key=lambda row: float(row["auc_pr_drop"]),
        reverse=True,
    )
    _print_sorted_table(sorted_rows)

    print("\nRunning trade-only minimum viable feature experiment...")
    trade_only_features = FEATURE_GROUPS["trade"]
    trade_only_metrics = _fit_and_evaluate(
        datasets["train"],
        datasets["val"],
        datasets["test"],
        trade_only_features,
    )
    trade_only_drop = BASELINE_AUC_PR - trade_only_metrics["auc_pr"]
    print(
        f"trade_only -> "
        f"AUC-PR={trade_only_metrics['auc_pr']:.4f}  "
        f"drop={trade_only_drop:+.4f}  "
        f"F1={trade_only_metrics['f1']:.4f}  "
        f"P={trade_only_metrics['precision']:.4f}  "
        f"R={trade_only_metrics['recall']:.4f}"
    )

    output_json = REPORTS_DIR / "ablation_results.json"
    output_plot = REPORTS_DIR / "ablation_auc_pr_drop.png"
    _save_drop_chart(sorted_rows, output_plot)

    output = {
        "baseline_auc_pr": BASELINE_AUC_PR,
        "xgb_params": XGB_PARAMS,
        "ablation_results": sorted_rows,
        "trade_only_baseline": {
            "features_used": trade_only_features,
            "features_count": len(trade_only_features),
            "auc_pr": trade_only_metrics["auc_pr"],
            "f1": trade_only_metrics["f1"],
            "precision": trade_only_metrics["precision"],
            "recall": trade_only_metrics["recall"],
            "auc_pr_drop": trade_only_drop,
        },
    }

    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"\nSaved ablation results: {output_json}")


if __name__ == "__main__":
    main()