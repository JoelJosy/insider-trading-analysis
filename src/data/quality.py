import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/quality.log")


@dataclass
class ColumnStats:
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_count: int = 0
    outlier_pct: float = 0.0
    top_values: Optional[Dict[str, int]] = None


@dataclass
class DataQualityReport:
    generated_at: str
    source_file: Optional[str] = None
    total_rows: int = 0
    total_columns: int = 0
    memory_usage_mb: float = 0.0
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    column_stats: List[ColumnStats] = field(default_factory=list)
    duplicate_rows: int = 0
    missing_value_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    transaction_type_distribution: Optional[Dict[str, int]] = None
    insider_role_distribution: Optional[Dict[str, int]] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Data Quality Report",
            f"\n**Generated:** {self.generated_at}",
            f"\n**Source:** {self.source_file or 'DataFrame'}",
            "\n---\n",
            "## Dataset Overview\n",
            f"- **Total Rows:** {self.total_rows:,}",
            f"- **Total Columns:** {self.total_columns}",
            f"- **Memory Usage:** {self.memory_usage_mb:.2f} MB",
            f"- **Date Range:** {self.date_min} to {self.date_max}",
            f"- **Duplicate Rows:** {self.duplicate_rows:,}",
            "\n---\n",
            "## Column Statistics\n",
            "| Column | Type | Count | Missing % | Unique |",
            "|--------|------|-------|-----------|--------|",
        ]
        for col in self.column_stats[:20]:
            lines.append(f"| {col.name} | {col.dtype} | {col.count:,} | {col.missing_pct:.1f}% | {col.unique:,} |")
        if len(self.column_stats) > 20:
            lines.append(f"\n*... and {len(self.column_stats) - 20} more columns*")

        numeric_cols = [c for c in self.column_stats if c.mean is not None]
        if numeric_cols:
            lines += ["\n---\n", "## Numeric Column Distributions\n",
                      "| Column | Mean | Std | Min | Median | Max | Outliers % |",
                      "|--------|------|-----|-----|--------|-----|------------|"]
            for col in numeric_cols[:10]:
                lines.append(f"| {col.name} | {col.mean:.2f} | {col.std:.2f} | {col.min:.2f} | {col.median:.2f} | {col.max:.2f} | {col.outlier_pct:.1f}% |")

        for attr, title in [("transaction_type_distribution", "Transaction Type"), ("insider_role_distribution", "Insider Role")]:
            dist = getattr(self, attr)
            if dist:
                lines += ["\n---\n", f"## {title} Distribution\n", "| Type | Count |", "|------|-------|"]
                for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"| {k} | {v:,} |")

        if self.warnings:
            lines += ["\n---\n", "## Warnings\n"] + [f"- {w}" for w in self.warnings]
        if self.recommendations:
            lines += ["\n---\n", "## Recommendations\n"] + [f"- {r}" for r in self.recommendations]

        return "\n".join(lines)

    def save_markdown(self, path: str) -> None:
        Path(path).write_text(self.to_markdown())


class DataQualityChecker:
    def __init__(self, outlier_threshold: float = 3.0):
        self.outlier_threshold = outlier_threshold
        self.logger = setup_logger(self.__class__.__name__, "logs/quality.log")

    def analyze(self, df: pd.DataFrame, source_file: Optional[str] = None) -> DataQualityReport:
        report = DataQualityReport(
            generated_at=datetime.now().isoformat(),
            source_file=source_file,
            total_rows=len(df),
            total_columns=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 ** 2),
            duplicate_rows=int(df.duplicated().sum()),
        )
        for col in ["transaction_date", "filing_date"]:
            if col in df.columns:
                valid = pd.to_datetime(df[col], errors="coerce").dropna()
                if not valid.empty:
                    report.date_min = str(valid.min().date())
                    report.date_max = str(valid.max().date())
                    break

        for col in df.columns:
            s = self._analyze_column(df[col])
            report.column_stats.append(s)
            if s.missing_pct > 50:
                report.missing_value_columns.append(col)
            if s.unique == 1 and s.count > 0:
                report.constant_columns.append(col)
            if s.unique > len(df) * 0.9 and s.dtype == "object":
                report.high_cardinality_columns.append(col)

        if "transaction_code" in df.columns:
            report.transaction_type_distribution = df["transaction_code"].value_counts().to_dict()
        if "insider_role" in df.columns:
            report.insider_role_distribution = df["insider_role"].value_counts().to_dict()

        self._add_warnings(report, df)
        self._add_recommendations(report, df)
        return report

    def _analyze_column(self, series: pd.Series) -> ColumnStats:
        n = len(series)
        missing = int(series.isna().sum())
        s = ColumnStats(
            name=series.name, dtype=str(series.dtype), count=n,
            missing=missing, missing_pct=(missing / n * 100) if n else 0,
            unique=series.nunique(),
        )
        if pd.api.types.is_bool_dtype(series):
            s.top_values = {str(k): v for k, v in series.value_counts().head(10).items()}
        elif pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if not valid.empty:
                s.mean, s.std, s.min = float(valid.mean()), float(valid.std()), float(valid.min())
                s.q25, s.median, s.q75, s.max = (
                    float(valid.quantile(0.25)), float(valid.median()),
                    float(valid.quantile(0.75)), float(valid.max()),
                )
                if len(valid) > 2 and valid.std() > 0:
                    s.skewness = float(stats.skew(valid.tolist()))
                    s.kurtosis = float(stats.kurtosis(valid.tolist()))
                outliers = self._outliers_iqr(valid)
                s.outlier_count = int(outliers.sum())
                s.outlier_pct = float(outliers.sum() / len(valid) * 100)
        elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            s.top_values = series.value_counts().head(10).to_dict()
        return s

    def _outliers_iqr(self, series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

    def _add_warnings(self, report: DataQualityReport, df: pd.DataFrame) -> None:
        if report.duplicate_rows:
            report.warnings.append(f"Found {report.duplicate_rows:,} duplicate rows ({report.duplicate_rows/report.total_rows*100:.1f}%)")
        if report.missing_value_columns:
            report.warnings.append(f"High missing values (>50%) in: {', '.join(report.missing_value_columns)}")
        if report.constant_columns:
            report.warnings.append(f"Constant columns: {', '.join(report.constant_columns)}")
        for col in ["price_per_share", "shares", "total_value"]:
            if col in df.columns:
                neg = int((df[col] < 0).sum())
                if neg:
                    report.warnings.append(f"Found {neg} negative values in '{col}'")
        if "price_per_share" in df.columns:
            high = int((df["price_per_share"] > 10000).sum())
            if high:
                report.warnings.append(f"Found {high} suspiciously high prices (>$10,000/share)")
        for col in ["transaction_date", "filing_date"]:
            if col in df.columns:
                future = int((pd.to_datetime(df[col], errors="coerce") > pd.Timestamp.now()).sum())
                if future:
                    report.warnings.append(f"Found {future} future dates in '{col}'")

    def _add_recommendations(self, report: DataQualityReport, df: pd.DataFrame) -> None:
        if report.duplicate_rows:
            report.recommendations.append("Remove duplicates with df.drop_duplicates()")
        price_stats = next((c for c in report.column_stats if c.name == "price_per_share"), None)
        if price_stats and price_stats.outlier_pct > 5:
            report.recommendations.append("Consider winsorizing or log-transforming price_per_share")
        high_missing = [c.name for c in report.column_stats if c.missing_pct > 20 and c.name not in ("footnote_text", "officer_title")]
        if high_missing:
            report.recommendations.append(f"Implement imputation for: {', '.join(high_missing)}")
        if report.transaction_type_distribution:
            total = sum(report.transaction_type_distribution.values())
            if max(report.transaction_type_distribution.values()) / total > 0.7:
                report.recommendations.append("Transaction types are imbalanced - consider stratified sampling")
        if "has_10b5_1_plan" in df.columns and df["has_10b5_1_plan"].mean() * 100 < 5:
            report.recommendations.append("Low 10b5-1 plan coverage - review footnote parsing logic")


def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    outlier_method: str = "cap",
    fill_missing_dates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log = {"original_rows": len(df), "actions": []}
    df = df.copy()

    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        if (n := before - len(df)):
            log["actions"].append(f"Removed {n} duplicate rows")

    if fill_missing_dates and "transaction_date" in df.columns and "filing_date" in df.columns:
        missing = int(df["transaction_date"].isna().sum())
        if missing:
            df["transaction_date"] = df["transaction_date"].fillna(df["filing_date"])
            log["actions"].append(f"Filled {missing} missing transaction_dates from filing_date")

    if handle_outliers and outlier_method != "none":
        for col in ("price_per_share", "shares", "total_value"):
            if col not in df.columns:
                continue
            valid = df[col].notna() & (df[col] > 0)
            data = df.loc[valid, col]
            if len(data) < 10:
                continue
            q1, q99 = data.quantile(0.01), data.quantile(0.99)
            if outlier_method == "cap":
                n_out = int(((df[col] < q1) | (df[col] > q99)).sum())
                df.loc[valid, col] = data.clip(q1, q99)
                log["actions"].append(f"Capped {n_out} outliers in {col}")
            elif outlier_method == "remove":
                before = len(df)
                df = df[~((df[col] < q1) | (df[col] > q99))]
                log["actions"].append(f"Removed {before - len(df)} rows with outliers in {col}")

    if "price_per_share" in df.columns and "is_derivative" in df.columns:
        neg = (df["price_per_share"] < 0) & (~df["is_derivative"])
        if (n := int(neg.sum())):
            df = df[~neg]
            log["actions"].append(f"Removed {n} rows with negative prices")

    log["final_rows"] = len(df)
    log["rows_removed"] = log["original_rows"] - log["final_rows"]
    return df, log


def generate_quality_report(
    csv_path: str,
    output_dir: str = "reports",
    save_json: bool = True,
    save_markdown: bool = True,
) -> DataQualityReport:
    df = pd.read_csv(csv_path, parse_dates=["filing_date", "transaction_date"])
    report = DataQualityChecker().analyze(df, source_file=csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    if save_json:
        report.to_json(str(out / f"{stem}_quality_report.json"))
    if save_markdown:
        report.save_markdown(str(out / f"{stem}_quality_report.md"))
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Data quality analysis")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    report = generate_quality_report(args.csv, args.output)
    print(report.to_markdown())
