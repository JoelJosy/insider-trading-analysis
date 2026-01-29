"""
Data Quality Module for Insider Trading Analysis.

Provides comprehensive data quality checks, outlier detection,
and generates quality reports for the parsed data.
"""

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

# Setup logger
logger = setup_logger(__name__, "logs/quality.log")


@dataclass
class ColumnStats:
    """Statistics for a single column."""
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    
    # Numeric stats (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Outlier info
    outlier_count: int = 0
    outlier_pct: float = 0.0
    
    # Categorical stats
    top_values: Optional[Dict[str, int]] = None


@dataclass
class DataQualityReport:
    """Complete data quality report."""
    generated_at: str
    source_file: Optional[str] = None
    
    # Dataset overview
    total_rows: int = 0
    total_columns: int = 0
    memory_usage_mb: float = 0.0
    
    # Date range
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    
    # Column statistics
    column_stats: List[ColumnStats] = field(default_factory=list)
    
    # Data quality issues
    duplicate_rows: int = 0
    missing_value_columns: List[str] = field(default_factory=list)
    high_cardinality_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    
    # Distribution info
    transaction_type_distribution: Optional[Dict[str, int]] = None
    insider_role_distribution: Optional[Dict[str, int]] = None
    
    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save report as JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
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
            "## Column Statistics\n"
        ]
        
        # Create table for column stats
        lines.append("| Column | Type | Count | Missing % | Unique |")
        lines.append("|--------|------|-------|-----------|--------|")
        
        for col in self.column_stats[:20]:  # Limit to first 20
            lines.append(
                f"| {col.name} | {col.dtype} | {col.count:,} | "
                f"{col.missing_pct:.1f}% | {col.unique:,} |"
            )
        
        if len(self.column_stats) > 20:
            lines.append(f"\n*... and {len(self.column_stats) - 20} more columns*")
        
        # Numeric distributions
        numeric_cols = [c for c in self.column_stats if c.mean is not None]
        if numeric_cols:
            lines.extend([
                "\n---\n",
                "## Numeric Column Distributions\n",
                "| Column | Mean | Std | Min | Median | Max | Outliers % |",
                "|--------|------|-----|-----|--------|-----|------------|"
            ])
            
            for col in numeric_cols[:10]:
                lines.append(
                    f"| {col.name} | {col.mean:.2f} | {col.std:.2f} | "
                    f"{col.min:.2f} | {col.median:.2f} | {col.max:.2f} | "
                    f"{col.outlier_pct:.1f}% |"
                )
        
        # Transaction type distribution
        if self.transaction_type_distribution:
            lines.extend([
                "\n---\n",
                "## Transaction Type Distribution\n",
                "| Type | Count |",
                "|------|-------|"
            ])
            for k, v in sorted(self.transaction_type_distribution.items(), 
                               key=lambda x: x[1], reverse=True):
                lines.append(f"| {k} | {v:,} |")
        
        # Insider role distribution
        if self.insider_role_distribution:
            lines.extend([
                "\n---\n",
                "## Insider Role Distribution\n",
                "| Role | Count |",
                "|------|-------|"
            ])
            for k, v in sorted(self.insider_role_distribution.items(),
                               key=lambda x: x[1], reverse=True):
                lines.append(f"| {k} | {v:,} |")
        
        # Warnings
        if self.warnings:
            lines.extend([
                "\n---\n",
                "## Warnings\n"
            ])
            for warning in self.warnings:
                lines.append(f"- {warning}")
        
        # Recommendations
        if self.recommendations:
            lines.extend([
                "\n---\n",
                "## Recommendations\n"
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    def save_markdown(self, path: str) -> None:
        """Save report as markdown file."""
        with open(path, 'w') as f:
            f.write(self.to_markdown())


class DataQualityChecker:
    """
    Performs comprehensive data quality checks on insider trading data.
    """
    
    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize the quality checker.

        Args:
            outlier_threshold: Z-score threshold for outlier detection.
        """
        self.outlier_threshold = outlier_threshold
        self.logger = setup_logger(self.__class__.__name__, "logs/quality.log")
    
    def analyze(
        self, 
        df: pd.DataFrame, 
        source_file: Optional[str] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive analysis on a DataFrame.

        Args:
            df: DataFrame to analyze.
            source_file: Optional source file path.

        Returns:
            DataQualityReport with all findings.
        """
        report = DataQualityReport(
            generated_at=datetime.now().isoformat(),
            source_file=source_file,
            total_rows=len(df),
            total_columns=len(df.columns),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024**2)
        )
        
        # Date range
        date_cols = ['transaction_date', 'filing_date']
        for col in date_cols:
            if col in df.columns:
                valid_dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if not valid_dates.empty:
                    report.date_min = str(valid_dates.min().date())
                    report.date_max = str(valid_dates.max().date())
                    break
        
        # Duplicate detection
        report.duplicate_rows = df.duplicated().sum()
        
        # Column-by-column analysis
        for col in df.columns:
            col_stats = self._analyze_column(df[col])
            report.column_stats.append(col_stats)
            
            # Track issues
            if col_stats.missing_pct > 50:
                report.missing_value_columns.append(col)
            
            if col_stats.unique == 1 and col_stats.count > 0:
                report.constant_columns.append(col)
            
            if col_stats.unique > len(df) * 0.9 and col_stats.dtype == 'object':
                report.high_cardinality_columns.append(col)
        
        # Transaction type distribution
        if 'transaction_code' in df.columns:
            report.transaction_type_distribution = df['transaction_code'].value_counts().to_dict()
        
        # Insider role distribution
        if 'insider_role' in df.columns:
            report.insider_role_distribution = df['insider_role'].value_counts().to_dict()
        
        # Generate warnings and recommendations
        self._generate_warnings(report, df)
        self._generate_recommendations(report, df)
        
        return report
    
    def _analyze_column(self, series: pd.Series) -> ColumnStats:
        """Analyze a single column."""
        col_stats = ColumnStats(
            name=series.name,
            dtype=str(series.dtype),
            count=len(series),
            missing=series.isna().sum(),
            missing_pct=(series.isna().sum() / len(series)) * 100 if len(series) > 0 else 0,
            unique=series.nunique()
        )
        
        # Boolean analysis (treat as categorical)
        if pd.api.types.is_bool_dtype(series):
            top_values = series.value_counts().head(10).to_dict()
            col_stats.top_values = {str(k): v for k, v in top_values.items()}

        # Numeric analysis
        elif pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if not valid.empty:
                col_stats.mean = valid.mean()
                col_stats.std = valid.std()
                col_stats.min = valid.min()
                col_stats.q25 = valid.quantile(0.25)
                col_stats.median = valid.median()
                col_stats.q75 = valid.quantile(0.75)
                col_stats.max = valid.max()
                
                if len(valid) > 2:
                    col_stats.skewness = float(stats.skew(valid.tolist()))
                    col_stats.kurtosis = float(stats.kurtosis(valid.tolist()))
                
                # Outlier detection using IQR
                outliers = self._detect_outliers_iqr(valid)
                col_stats.outlier_count = outliers.sum()
                col_stats.outlier_pct = (outliers.sum() / len(valid)) * 100
        
        # Categorical analysis
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            top_values = series.value_counts().head(10).to_dict()
            col_stats.top_values = top_values
        
        return col_stats
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return z_scores > self.outlier_threshold
    
    def _generate_warnings(self, report: DataQualityReport, df: pd.DataFrame) -> None:
        """Generate warnings based on data quality issues."""
        
        if report.duplicate_rows > 0:
            pct = (report.duplicate_rows / report.total_rows) * 100
            report.warnings.append(
                f"Found {report.duplicate_rows:,} duplicate rows ({pct:.1f}%)"
            )
        
        if report.missing_value_columns:
            report.warnings.append(
                f"High missing values (>50%) in columns: {', '.join(report.missing_value_columns)}"
            )
        
        if report.constant_columns:
            report.warnings.append(
                f"Constant columns (single value): {', '.join(report.constant_columns)}"
            )
        
        # Check for negative prices or shares
        for col in ['price_per_share', 'shares', 'total_value']:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    report.warnings.append(
                        f"Found {neg_count} negative values in '{col}'"
                    )
        
        # Check for unrealistic prices
        if 'price_per_share' in df.columns:
            high_price = (df['price_per_share'] > 10000).sum()
            if high_price > 0:
                report.warnings.append(
                    f"Found {high_price} suspiciously high prices (>$10,000/share)"
                )
        
        # Check for future dates
        for col in ['transaction_date', 'filing_date']:
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors='coerce')
                future_dates = (dates > pd.Timestamp.now()).sum()
                if future_dates > 0:
                    report.warnings.append(
                        f"Found {future_dates} future dates in '{col}'"
                    )
    
    def _generate_recommendations(
        self, 
        report: DataQualityReport, 
        df: pd.DataFrame
    ) -> None:
        """Generate recommendations based on data quality issues."""
        
        if report.duplicate_rows > 0:
            report.recommendations.append(
                "Remove duplicate rows before analysis using df.drop_duplicates()"
            )
        
        # Check price outliers
        price_stats = next(
            (c for c in report.column_stats if c.name == 'price_per_share'), 
            None
        )
        if price_stats and price_stats.outlier_pct > 5:
            report.recommendations.append(
                "Consider capping extreme prices using winsorization or log transformation"
            )
        
        # Check missing values
        high_missing = [
            c.name for c in report.column_stats 
            if c.missing_pct > 20 and c.name not in ['footnote_text', 'officer_title']
        ]
        if high_missing:
            report.recommendations.append(
                f"Implement imputation strategy for: {', '.join(high_missing)}"
            )
        
        # Check class imbalance for transaction codes
        if report.transaction_type_distribution:
            total = sum(report.transaction_type_distribution.values())
            max_pct = max(report.transaction_type_distribution.values()) / total
            if max_pct > 0.7:
                report.recommendations.append(
                    "Transaction types are imbalanced - consider stratified sampling or SMOTE for modeling"
                )
        
        # 10b5-1 plan coverage
        if 'has_10b5_1_plan' in df.columns:
            plan_pct = df['has_10b5_1_plan'].mean() * 100
            if plan_pct < 5:
                report.recommendations.append(
                    f"Only {plan_pct:.1f}% of trades have 10b5-1 plans identified - "
                    "review footnote parsing logic"
                )


def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_outliers: bool = True,
    outlier_method: str = 'cap',  # 'cap', 'remove', or 'none'
    fill_missing_dates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean a DataFrame according to best practices.

    Args:
        df: DataFrame to clean.
        remove_duplicates: Whether to remove duplicate rows.
        handle_outliers: Whether to handle outliers in numeric columns.
        outlier_method: 'cap' (winsorize), 'remove', or 'none'.
        fill_missing_dates: Whether to fill missing transaction_date with filing_date.

    Returns:
        Tuple of (cleaned DataFrame, cleaning summary dict).
    """
    cleaning_log = {
        'original_rows': len(df),
        'actions': []
    }
    
    df_clean = df.copy()
    
    # Remove duplicates
    if remove_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = before - len(df_clean)
        if removed > 0:
            cleaning_log['actions'].append(f"Removed {removed} duplicate rows")
    
    # Fill missing transaction dates
    if fill_missing_dates and 'transaction_date' in df_clean.columns:
        missing = df_clean['transaction_date'].isna().sum()
        if missing > 0 and 'filing_date' in df_clean.columns:
            df_clean['transaction_date'] = df_clean['transaction_date'].fillna(
                df_clean['filing_date']
            )
            cleaning_log['actions'].append(
                f"Filled {missing} missing transaction_dates with filing_date"
            )
    
    # Handle outliers in key numeric columns
    if handle_outliers and outlier_method != 'none':
        numeric_cols = ['price_per_share', 'shares', 'total_value']
        
        for col in numeric_cols:
            if col not in df_clean.columns:
                continue
            
            valid_mask = df_clean[col].notna() & (df_clean[col] > 0)
            valid_data = df_clean.loc[valid_mask, col]
            
            if len(valid_data) < 10:
                continue
            
            Q1 = valid_data.quantile(0.01)
            Q99 = valid_data.quantile(0.99)
            
            if outlier_method == 'cap':
                before_outliers = ((df_clean[col] < Q1) | (df_clean[col] > Q99)).sum()
                df_clean.loc[valid_mask, col] = df_clean.loc[valid_mask, col].clip(Q1, Q99)
                cleaning_log['actions'].append(
                    f"Capped {before_outliers} outliers in {col} to [1st, 99th] percentile"
                )
            
            elif outlier_method == 'remove':
                outlier_mask = (df_clean[col] < Q1) | (df_clean[col] > Q99)
                before = len(df_clean)
                df_clean = df_clean[~outlier_mask]
                removed = before - len(df_clean)
                cleaning_log['actions'].append(
                    f"Removed {removed} rows with outliers in {col}"
                )
    
    # Remove rows with clearly invalid data
    # - Negative prices (except for some derivative positions)
    if 'price_per_share' in df_clean.columns:
        neg_mask = (df_clean['price_per_share'] < 0) & (df_clean['is_derivative'] == False)
        removed = neg_mask.sum()
        if removed > 0:
            df_clean = df_clean[~neg_mask]
            cleaning_log['actions'].append(f"Removed {removed} rows with negative prices")
    
    cleaning_log['final_rows'] = len(df_clean)
    cleaning_log['rows_removed'] = cleaning_log['original_rows'] - cleaning_log['final_rows']
    
    return df_clean, cleaning_log


def generate_quality_report(
    csv_path: str,
    output_dir: str = "reports",
    save_json: bool = True,
    save_markdown: bool = True
) -> DataQualityReport:
    """
    Generate a data quality report for a CSV file.

    Args:
        csv_path: Path to the CSV file.
        output_dir: Directory to save reports.
        save_json: Whether to save JSON report.
        save_markdown: Whether to save markdown report.

    Returns:
        DataQualityReport instance.
    """
    logger.info(f"Generating quality report for: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['filing_date', 'transaction_date'])
    
    # Analyze
    checker = DataQualityChecker()
    report = checker.analyze(df, source_file=csv_path)
    
    # Save reports
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(csv_path).stem
    
    if save_json:
        json_path = output_path / f"{base_name}_quality_report.json"
        report.to_json(str(json_path))
        logger.info(f"Saved JSON report: {json_path}")
    
    if save_markdown:
        md_path = output_path / f"{base_name}_quality_report.md"
        report.save_markdown(str(md_path))
        logger.info(f"Saved Markdown report: {md_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data quality analysis")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output", type=str, default="reports", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    
    args = parser.parse_args()
    
    if args.report:
        report = generate_quality_report(args.csv, args.output)
        print(report.to_markdown())
