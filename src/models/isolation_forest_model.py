from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

SPLITS = ("train", "val", "test")
LABEL_COL = "final_label"
TRANSACTION_CODE_COL = "transaction_code"
VALID_TRANSACTION_CODES = {"P", "S"}
SUPERVISED_VALID_LABELS = {0, 1}

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

    out = df.copy()
    out[TRANSACTION_CODE_COL] = out[TRANSACTION_CODE_COL].astype(str).str.upper().str.strip()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    out = out.loc[out[TRANSACTION_CODE_COL].isin(VALID_TRANSACTION_CODES)].copy()

    if out.empty:
        raise ValueError(f"{csv_path} has no rows after transaction code filtering")

    return out


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return (
        df[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _supervised_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[LABEL_COL].isin(SUPERVISED_VALID_LABELS)].copy()


def _anomaly_score(model: IsolationForest, x_scaled: np.ndarray) -> np.ndarray:
    raw = -model.decision_function(x_scaled)
    return raw


def _fit_score_normalizer(train_scores: np.ndarray) -> tuple[float, float]:
    min_val = float(np.min(train_scores))
    max_val = float(np.max(train_scores))
    return min_val, max_val


def _apply_score_normalizer(scores: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    denom = max_val - min_val
    if denom <= 0:
        return np.zeros_like(scores, dtype=float)
    normalized = (scores - min_val) / denom
    return np.clip(normalized, 0.0, 1.0)


def _tune_contamination(
    x_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    y_val: pd.Series,
) -> tuple[IsolationForest, float, float, tuple[float, float]]:
    best_model: IsolationForest | None = None
    best_contamination = 0.05
    best_val_auc_pr = -np.inf
    best_norm_params = (0.0, 1.0)

    for contamination in np.arange(0.05, 0.401, 0.05):
        model = IsolationForest(
            n_estimators=200,
            contamination=float(round(contamination, 2)),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train_scaled)

        train_scores_raw = _anomaly_score(model, x_train_scaled)
        norm_min, norm_max = _fit_score_normalizer(train_scores_raw)

        val_scores_raw = _anomaly_score(model, x_val_scaled)
        val_scores = _apply_score_normalizer(val_scores_raw, norm_min, norm_max)
        val_auc_pr = float(average_precision_score(y_val, val_scores))

        if val_auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_auc_pr
            best_contamination = float(round(contamination, 2))
            best_model = model
            best_norm_params = (norm_min, norm_max)

    if best_model is None:
        raise RuntimeError("Contamination tuning failed to produce a model")

    return best_model, best_contamination, best_val_auc_pr, best_norm_params


def _tune_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.1, 0.901, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def _evaluate(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    preds = (scores >= threshold).astype(int)
    auc_pr = float(average_precision_score(y_true, scores))
    f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    precision = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()

    return {
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def _plot_score_distribution(
    y_test: pd.Series,
    test_scores: np.ndarray,
    output_path: Path,
) -> None:
    routine_scores = test_scores[y_test.values == 0]
    opportunistic_scores = test_scores[y_test.values == 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0.0, 1.0, 40)
    ax.hist(routine_scores, bins=bins, alpha=0.55, density=True, label="Routine (label=0)")
    ax.hist(opportunistic_scores, bins=bins, alpha=0.55, density=True, label="Opportunistic (label=1)")
    ax.set_xlabel("Anomaly Score (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Isolation Forest Anomaly Score Distribution (Test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_feature_importance_proxy(
    x_train_raw: pd.DataFrame,
    train_scores: np.ndarray,
    output_path: Path,
) -> None:
    pseudo_labels = (train_scores > 0.5).astype(int)

    if np.unique(pseudo_labels).size < 2:
        threshold = float(np.median(train_scores))
        pseudo_labels = (train_scores > threshold).astype(int)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train_raw, pseudo_labels)

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    top_20 = importances.sort_values(ascending=False).head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_20.index, top_20.values)
    ax.set_xlabel("Random Forest Importance")
    ax.set_title("Top 20 Feature Importance Proxy for Isolation Forest")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()

    leakage_overlap = sorted(set(FEATURE_COLS) & LEAKAGE_COLS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns present in feature list: {leakage_overlap}")

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")

    train_unsup_df = train_df.copy()
    val_sup_df = _supervised_view(val_df)
    test_sup_df = _supervised_view(test_df)

    if val_sup_df.empty or test_sup_df.empty:
        raise ValueError("Validation/Test splits have no rows with final_label in {0, 1}")

    x_train_raw = _prepare_features(train_unsup_df)
    x_val_raw = _prepare_features(val_sup_df)
    x_test_raw = _prepare_features(test_sup_df)

    y_val = val_sup_df[LABEL_COL].astype(int)
    y_test = test_sup_df[LABEL_COL].astype(int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    best_model, best_contamination, best_val_auc_pr, norm_params = _tune_contamination(
        x_train_scaled,
        x_val_scaled,
        y_val,
    )
    norm_min, norm_max = norm_params

    val_scores = _apply_score_normalizer(_anomaly_score(best_model, x_val_scaled), norm_min, norm_max)
    best_threshold, best_val_f1 = _tune_threshold(y_val, val_scores)

    train_scores = _apply_score_normalizer(
        _anomaly_score(best_model, x_train_scaled),
        norm_min,
        norm_max,
    )
    test_scores = _apply_score_normalizer(
        _anomaly_score(best_model, x_test_scaled),
        norm_min,
        norm_max,
    )
    metrics = _evaluate(y_test, test_scores, best_threshold)

    _plot_score_distribution(
        y_test,
        test_scores,
        REPORTS_DIR / "iforest_score_distribution.png",
    )
    _plot_feature_importance_proxy(
        x_train_raw,
        train_scores,
        REPORTS_DIR / "iforest_feature_importance.png",
    )

    joblib.dump(best_model, MODELS_DIR / "isolation_forest_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "isolation_forest_scaler.joblib")

    results = {
        "model": "isolation_forest",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "best_contamination": best_contamination,
        "best_threshold": best_threshold,
        "confusion_matrix": metrics["confusion_matrix"],
        "val_auc_pr": best_val_auc_pr,
        "val_f1": best_val_f1,
    }
    with (REPORTS_DIR / "model_results_isolation_forest.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("Isolation Forest pipeline complete.")
    print(f"Train rows (unsupervised): {len(train_unsup_df)}")
    print(f"Val rows (supervised eval): {len(val_sup_df)} | Test rows (supervised eval): {len(test_sup_df)}")
    print(f"Best contamination: {best_contamination:.2f} | Best val AUC-PR: {best_val_auc_pr:.4f}")
    print(f"Best threshold (val F1): {best_threshold:.2f} | Val F1: {best_val_f1:.4f}")
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
