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
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


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
    "has_plan",
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
    "footnote_has_plan",
    "footnote_routine_score",
    "footnote_routine_hits",
    "footnote_informed_hits",
    "footnote_has_option",
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

    features = (
        df[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    labels = df[LABEL_COL].astype(int)
    return features, labels


def _get_class_priors(y_train: pd.Series) -> list[float]:
    priors = y_train.value_counts(normalize=True).sort_index()
    prior_0 = float(priors.get(0, 0.0))
    prior_1 = float(priors.get(1, 0.0))
    if prior_0 <= 0.0 or prior_1 <= 0.0:
        raise ValueError("Train split must contain both classes 0 and 1 for GaussianNB priors")
    return [prior_0, prior_1]


def _tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    # thresholds = np.arange(0.1, 0.901, 0.05)
    thresholds = np.arange(0.05, 0.951, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def _evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
    y_pred = (y_prob >= threshold).astype(int)

    auc_pr = float(average_precision_score(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def _plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"GaussianNB (AUC-PR={auc_pr:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Naive Bayes Precision-Recall Curve (Test)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_feature_separation(
    x_train_unscaled: pd.DataFrame,
    y_train: pd.Series,
    out_path: Path,
) -> None:
    class_0 = x_train_unscaled[y_train == 0]
    class_1 = x_train_unscaled[y_train == 1]

    mean_0 = class_0.mean(axis=0)
    mean_1 = class_1.mean(axis=0)
    var_0 = class_0.var(axis=0, ddof=1)
    var_1 = class_1.var(axis=0, ddof=1)
    pooled_std = np.sqrt((var_0 + var_1) / 2.0).replace(0, np.nan)

    separation = ((mean_1 - mean_0).abs() / pooled_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    top_20 = separation.sort_values(ascending=False).head(20)
    top_20 = top_20.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_20.index, top_20.values)
    ax.set_xlabel("|mean_class1 - mean_class0| / pooled_std")
    ax.set_title("Top 20 Feature Separation Scores (Train)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()

    leakage_overlap = sorted(set(FEATURE_COLS) & LEAKAGE_COLS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns present in feature list: {leakage_overlap}")
    print(f"Leakage check passed — {len(FEATURE_COLS)} features verified clean.")

    datasets = {split: _load_split(split) for split in SPLITS}
    x_train_raw, y_train = _prepare_features_and_labels(datasets["train"])
    x_val_raw, y_val = _prepare_features_and_labels(datasets["val"])
    x_test_raw, y_test = _prepare_features_and_labels(datasets["test"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    priors = _get_class_priors(y_train)
    model = GaussianNB(priors=priors)
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    best_threshold, best_val_f1 = _tune_threshold(y_val, val_prob)

    test_prob = model.predict_proba(x_test)[:, 1]
    metrics = _evaluate(y_test, test_prob, best_threshold)

    _plot_pr_curve(y_test, test_prob, REPORTS_DIR / "nb_pr_curve.png")
    _plot_feature_separation(x_train_raw, y_train, REPORTS_DIR / "nb_feature_separation.png")

    joblib.dump(model, MODELS_DIR / "naive_bayes_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "naive_bayes_scaler.joblib")

    results = {
        "model": "gaussian_naive_bayes",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "best_threshold": best_threshold,
        "confusion_matrix": metrics["confusion_matrix"],
        "val_best_f1": best_val_f1,
    }

    with (REPORTS_DIR / "model_results_naive_bayes.json").open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print("Naive Bayes pipeline complete.")
    print(f"Train rows: {len(y_train)} | Val rows: {len(y_val)} | Test rows: {len(y_test)}")
    print(f"Best threshold (val): {best_threshold:.2f} (val F1={best_val_f1:.4f})")
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
