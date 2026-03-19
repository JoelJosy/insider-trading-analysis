from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

SPLITS = ("train", "val", "test")
LABEL_COL = "final_label"
TRANSACTION_CODE_COL = "transaction_code"
VALID_TRANSACTION_CODES = {"P", "S"}
VALID_LABELS = {0, 1}

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

LEAKAGE_COLS = {
    "future_close", "forward_return", "benchmark_return",
    "abnormal_return", "signed_abnormal_return", "price_signal",
    "earnings_proximity_flag", "enforcement_followup_flag",
    "has_plan", "footnote_has_plan", "footnote_routine_score",
    "footnote_routine_hits", "footnote_informed_hits", "footnote_has_option",
    "informed_label", "routine_label", "cohen_label", "plan_override",
}


def _ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


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

    mask = filtered[TRANSACTION_CODE_COL].isin(VALID_TRANSACTION_CODES) & filtered[LABEL_COL].isin(
        VALID_LABELS
    )
    filtered = filtered.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"{csv_path} has no rows after applying transaction/label filters")

    filtered[LABEL_COL] = filtered[LABEL_COL].astype(int)
    return filtered


def _prepare_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = (
        df[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = df[LABEL_COL].astype(int)
    return x, y


def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    class_counts = y_train.value_counts()
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))
    if negative_count == 0 or positive_count == 0:
        raise ValueError("Train split must contain both labels 0 and 1")
    return float(negative_count / positive_count)


def _parameter_grid() -> list[dict[str, float | int]]:
    max_depth = [3, 5, 7]
    learning_rate = [0.05, 0.1]
    n_estimators = [100, 200]
    subsample = [0.8, 1.0]

    grid = []
    for depth, lr, n_est, sub in itertools.product(
        max_depth, learning_rate, n_estimators, subsample
    ):
        grid.append(
            {
                "max_depth": depth,
                "learning_rate": lr,
                "n_estimators": n_est,
                "subsample": sub,
            }
        )
    return grid


def _fit_single_model(
    params: dict[str, float | int],
    scale_pos_weight: float,
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        subsample=float(params["subsample"]),
        colsample_bytree=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def _grid_search(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_val: np.ndarray,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> tuple[XGBClassifier, dict[str, float | int], float]:
    best_model: XGBClassifier | None = None
    best_params: dict[str, float | int] | None = None
    best_auc_pr = -np.inf

    for params in _parameter_grid():
        model = _fit_single_model(params, scale_pos_weight, x_train, y_train, x_val, y_val)
        val_prob = model.predict_proba(x_val)[:, 1]
        val_auc_pr = float(average_precision_score(y_val, val_prob))

        if val_auc_pr > best_auc_pr:
            best_auc_pr = val_auc_pr
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("Grid search failed to produce a valid XGBoost model")

    return best_model, best_params, best_auc_pr


def _evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_pred = (y_prob >= threshold).astype(int)
    auc_pr = float(average_precision_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def _save_shap_plots(
    model: XGBClassifier,
    x_test_scaled: np.ndarray,
    feature_names: list[str],
    bar_path: Path,
    beeswarm_path: Path,
) -> None:
    x_test_df = pd.DataFrame(x_test_scaled, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test_df)

    shap.summary_plot(shap_values, x_test_df, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()

    shap.summary_plot(shap_values, x_test_df, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160)
    plt.close()


def main() -> None:
    _ensure_output_dirs()

    leakage_overlap = sorted(set(FEATURE_COLS) & LEAKAGE_COLS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns present in feature list: {leakage_overlap}")

    datasets = {split: _load_split(split) for split in SPLITS}
    x_train_raw, y_train = _prepare_features_and_labels(datasets["train"])
    x_val_raw, y_val = _prepare_features_and_labels(datasets["val"])
    x_test_raw, y_test = _prepare_features_and_labels(datasets["test"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    scale_pos_weight = _compute_scale_pos_weight(y_train)

    best_model, best_params, best_val_auc_pr = _grid_search(
        x_train, y_train, x_val, y_val, scale_pos_weight
    )

    test_prob = best_model.predict_proba(x_test)[:, 1]
    metrics = _evaluate(y_test, test_prob, threshold=0.5)

    _save_shap_plots(
        best_model,
        x_test,
        FEATURE_COLS,
        REPORTS_DIR / "shap_xgb.png",
        REPORTS_DIR / "shap_xgb_beeswarm.png",
    )

    joblib.dump(best_model, MODELS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "xgb_scaler.joblib")

    results = {
        "model": "xgboost",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "best_params": {
            **best_params,
            "scale_pos_weight": scale_pos_weight,
            "val_auc_pr": best_val_auc_pr,
        },
    }

    with (REPORTS_DIR / "model_results_xgb.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("XGBoost pipeline complete.")
    print(f"Train rows: {len(y_train)} | Val rows: {len(y_val)} | Test rows: {len(y_test)}")
    print(f"Best params: {best_params}")
    print(f"Best val AUC-PR: {best_val_auc_pr:.4f}")
    print(
        "Test metrics -> "
        f"AUC-PR: {metrics['auc_pr']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}"
    )
    print(f"Test confusion matrix [ [tn, fp], [fn, tp] ]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
