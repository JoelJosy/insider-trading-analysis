"""K-means clustering for insider behavioral anomaly detection on SEC Form 4 trades.

This module:
- Loads pre-split train/validation/test datasets.
- Filters to open-market trades (P/S) with deterministic labels (0/1).
- Builds insider-level behavioral profiles and clusters insiders with K-means.
- Maps cluster membership back to transactions and derives suspicion scores.
- Evaluates performance on the test set with AUC-PR, F1, precision, recall, and confusion matrix.
- Saves plots, model artifacts, and JSON results for reproducible collaborator workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
TEST_PATH = Path("data/processed/test.csv")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

RANDOM_STATE = 42

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

INSIDER_PROFILE_FEATURES = [
    "insider_total_trades",
    "insider_avg_trade_value",
    "insider_buy_sell_ratio",
    "insider_tenure_days",
    "role_seniority",
    "trade_frequency_90d",
    "pct_position_traded",
    "buy_count_90d",
    "sell_count_90d",
]


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
    print(f"[INFO] Filtering {dataset_name} to transaction_code in ['P', 'S'] and final_label in [0, 1]...")

    filtered = df.copy()

    if "transaction_code" in filtered.columns:
        filtered = filtered[filtered["transaction_code"].isin(["P", "S"])]
    else:
        print(f"[WARN] {dataset_name}: missing 'transaction_code'; skipping open-market filter.")

    if "final_label" not in filtered.columns:
        raise ValueError(f"{dataset_name}: missing required column 'final_label'.")

    filtered = filtered[filtered["final_label"].isin([0, 1])].copy()
    print(f"[INFO] {dataset_name} shape after filtering: {filtered.shape}")
    return filtered


def _available_feature_columns(df: pd.DataFrame, expected: list[str], dataset_name: str) -> list[str]:
    available = [col for col in expected if col in df.columns]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        print(f"[WARN] {dataset_name}: missing feature columns will be skipped: {missing}")
    if not available:
        raise ValueError(f"{dataset_name}: none of the required profile features are available.")
    return available


def _aggregate_insider_profiles(
    df: pd.DataFrame,
    feature_columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    if "insider_cik" not in df.columns:
        raise ValueError(f"{dataset_name}: missing required column 'insider_cik'.")

    working = df.copy()
    for col in feature_columns:
        if col not in working.columns:
            print(f"[WARN] {dataset_name}: feature '{col}' missing. Filling with NaN for aggregation.")
            working[col] = np.nan

    agg_spec: dict[str, str] = {col: "mean" for col in feature_columns}
    profiles = working.groupby("insider_cik", dropna=True, as_index=False).agg(agg_spec)

    if "informed_label" in working.columns:
        informed_rate = (
            working.groupby("insider_cik", dropna=True, as_index=False)["informed_label"].mean()
            .rename(columns={"informed_label": "insider_informed_rate"})
        )
        profiles = profiles.merge(informed_rate, on="insider_cik", how="left")
    else:
        print(f"[WARN] {dataset_name}: 'informed_label' missing. Cluster informed rates may be NaN.")
        profiles["insider_informed_rate"] = np.nan

    print(f"[INFO] {dataset_name}: built {len(profiles)} insider profiles.")
    return profiles


def _pick_elbow_k(k_values: list[int], inertias: list[float]) -> int:
    if len(k_values) != len(inertias):
        raise ValueError("k_values and inertias length mismatch.")

    points = np.column_stack([np.array(k_values, dtype=float), np.array(inertias, dtype=float)])
    first = points[0]
    last = points[-1]

    line = last - first
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(k_values[0])

    line_unit = line / line_norm
    vectors = points - first
    projection = np.outer(np.dot(vectors, line_unit), line_unit)
    orthogonal = vectors - projection
    distances = np.linalg.norm(orthogonal, axis=1)

    elbow_idx = int(np.argmax(distances))
    return int(k_values[elbow_idx])


def _plot_elbow(k_values: list[int], inertias: list[float], optimal_k: int, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.axvline(optimal_k, color="red", linestyle="--", label=f"Selected k={optimal_k}")
    plt.title("K-means Elbow Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved elbow plot: {output_path}")


def _plot_pca_clusters(
    scaled_features: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    pca = PCA(n_components=2, svd_solver="randomized", random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.8,
        s=35,
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("K-means Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved cluster PCA plot: {output_path}")

    return components


def _plot_pca_labels(
    pca_components: np.ndarray,
    informed_values: pd.Series,
    output_path: Path,
) -> None:
    color_values = pd.to_numeric(informed_values, errors="coerce").fillna(0.0).to_numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        c=color_values,
        cmap="coolwarm",
        alpha=0.8,
        s=35,
    )
    plt.colorbar(scatter, label="Mean informed_label")
    plt.title("PCA Projection Colored by informed_label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"[INFO] Saved informed-label PCA plot: {output_path}")


def _assign_clusters_to_transactions(
    transactions: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
    model: KMeans,
    train_feature_means: pd.Series,
    cluster_informed_rate: dict[int, float],
    dataset_name: str,
) -> pd.DataFrame:
    profiles = _aggregate_insider_profiles(transactions, feature_columns, dataset_name)

    profile_features = profiles[feature_columns].copy()
    profile_features = profile_features.fillna(train_feature_means)
    scaled = scaler.transform(profile_features)

    profiles["kmeans_cluster"] = model.predict(scaled)

    cluster_map = profiles.set_index("insider_cik")["kmeans_cluster"]
    output = transactions.copy()
    output["kmeans_cluster"] = output["insider_cik"].map(cluster_map)

    fallback_score = float(np.nanmean(list(cluster_informed_rate.values()))) if cluster_informed_rate else 0.0
    output["suspicion_score"] = (
        output["kmeans_cluster"].map(cluster_informed_rate).fillna(fallback_score).astype(float)
    )

    return output


def _compute_classification_metrics(y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5) -> dict[str, object]:
    y_true_num = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
    y_score_num = pd.to_numeric(y_score, errors="coerce").fillna(0.0).clip(0, 1)
    y_pred = (y_score_num >= threshold).astype(int)

    if y_true_num.nunique() > 1:
        auc_pr = float(average_precision_score(y_true_num, y_score_num))
    else:
        print("[WARN] Only one class present in y_true. AUC-PR set to NaN.")
        auc_pr = float("nan")

    metrics = {
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

    print("[INFO] Starting K-means insider clustering pipeline...")
    _ensure_output_directories()

    train_raw = _load_csv(TRAIN_PATH, "train")
    val_raw = _load_csv(VAL_PATH, "validation")
    test_raw = _load_csv(TEST_PATH, "test")

    train_df = _filter_open_market_labeled(train_raw, "train")
    val_df = _filter_open_market_labeled(val_raw, "validation")
    test_df = _filter_open_market_labeled(test_raw, "test")

    available_features = _available_feature_columns(train_df, INSIDER_PROFILE_FEATURES, "train")

    train_profiles = _aggregate_insider_profiles(train_df, available_features, "train")
    train_features = train_profiles[available_features].copy()
    train_feature_means = train_features.mean(numeric_only=True)
    train_features = train_features.fillna(train_feature_means)

    print("[INFO] Fitting StandardScaler on train insider profiles...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    print("[INFO] Searching for optimal k with elbow method (k=2..10)...")
    k_values = list(range(2, 11))
    inertias: list[float] = []
    for k in k_values:
        candidate = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        candidate.fit(train_scaled)
        inertias.append(float(candidate.inertia_))

    optimal_k = _pick_elbow_k(k_values, inertias)
    print(f"[INFO] Selected optimal k={optimal_k}")

    _plot_elbow(k_values, inertias, optimal_k, REPORTS_DIR / "kmeans_elbow.png")

    print("[INFO] Fitting final KMeans model...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=20)
    train_clusters = kmeans.fit_predict(train_scaled)
    train_profiles["kmeans_cluster"] = train_clusters

    if "informed_label" in train_df.columns:
        cluster_assignment_map = train_profiles.set_index("insider_cik")["kmeans_cluster"]
        train_for_cluster_rates = train_df.copy()
        train_for_cluster_rates["kmeans_cluster"] = train_for_cluster_rates["insider_cik"].map(
            cluster_assignment_map
        )
        cluster_informed_rate_series = (
            train_for_cluster_rates.groupby("kmeans_cluster")["informed_label"].mean().fillna(0.0)
        )
    else:
        print("[WARN] train: missing 'informed_label'. Falling back to insider-level informed rates.")
        cluster_informed_rate_series = (
            train_profiles.groupby("kmeans_cluster")["insider_informed_rate"].mean().fillna(0.0)
        )

    cluster_informed_rate = {
        int(cluster_id): float(rate)
        for cluster_id, rate in cluster_informed_rate_series.items()
    }
    suspicious_cluster = int(cluster_informed_rate_series.idxmax()) if not cluster_informed_rate_series.empty else -1
    print(f"[INFO] Suspicious cluster (highest informed mean): {suspicious_cluster}")

    print("[INFO] Building and printing cluster profiles...")
    profile_agg_spec: dict[str, str] = {col: "mean" for col in available_features}
    cluster_profiles_df = train_profiles.groupby("kmeans_cluster", as_index=False).agg(profile_agg_spec)
    cluster_profiles_df["informed_rate"] = (
        cluster_profiles_df["kmeans_cluster"].map(cluster_informed_rate_series).fillna(0.0)
    )
    print(cluster_profiles_df.to_string(index=False))

    print("[INFO] Generating PCA scatter plots...")
    pca_components = _plot_pca_clusters(
        scaled_features=train_scaled,
        cluster_labels=train_clusters,
        output_path=REPORTS_DIR / "kmeans_clusters.png",
    )
    _plot_pca_labels(
        pca_components=pca_components,
        informed_values=train_profiles["insider_informed_rate"],
        output_path=REPORTS_DIR / "kmeans_clusters_labels.png",
    )

    print("[INFO] Mapping cluster membership back to transaction-level data...")
    train_scored = _assign_clusters_to_transactions(
        train_df,
        available_features,
        scaler,
        kmeans,
        train_feature_means,
        cluster_informed_rate,
        "train",
    )
    _ = _assign_clusters_to_transactions(
        val_df,
        available_features,
        scaler,
        kmeans,
        train_feature_means,
        cluster_informed_rate,
        "validation",
    )
    test_scored = _assign_clusters_to_transactions(
        test_df,
        available_features,
        scaler,
        kmeans,
        train_feature_means,
        cluster_informed_rate,
        "test",
    )

    print("[INFO] Evaluating on test transactions...")
    metrics = _compute_classification_metrics(
        y_true=test_scored["final_label"],
        y_score=test_scored["suspicion_score"],
        threshold=0.5,
    )

    print(
        "[RESULT] "
        f"AUC-PR={metrics['auc_pr']:.6f}, "
        f"F1={metrics['f1']:.6f}, "
        f"Precision={metrics['precision']:.6f}, "
        f"Recall={metrics['recall']:.6f}"
    )
    print(f"[RESULT] Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    print("[INFO] Saving model artifacts...")
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "kmeans_scaler.joblib")

    results = {
        "model": "kmeans",
        "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "n_clusters": int(optimal_k),
        "suspicious_cluster": suspicious_cluster,
        "cluster_profiles": cluster_profiles_df.to_dict(orient="records"),
        "feature_columns_used": available_features,
    }

    results_path = REPORTS_DIR / "model_results_kmeans.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[INFO] Saved results JSON: {results_path}")

    print("[INFO] K-means pipeline completed successfully.")


if __name__ == "__main__":
    main()
