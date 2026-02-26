# Data Quality Report

**Generated:** 2026-02-26T13:45:23.272124

**Source:** data/processed/AAPL_form4.csv

---

## Dataset Overview

- **Total Rows:** 750
- **Total Columns:** 31
- **Memory Usage:** 1.97 MB
- **Date Range:** 2019-12-27 to 2026-02-01
- **Duplicate Rows:** 26

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 750 | 0.0% | 255 |
| filing_date | datetime64[ns] | 750 | 0.0% | 100 |
| issuer_cik | int64 | 750 | 0.0% | 1 |
| issuer_name | object | 750 | 0.0% | 1 |
| ticker | object | 750 | 0.0% | 1 |
| insider_name | object | 750 | 0.0% | 17 |
| insider_cik | int64 | 750 | 0.0% | 17 |
| insider_role | object | 750 | 0.0% | 5 |
| is_director | bool | 750 | 0.0% | 2 |
| is_officer | bool | 750 | 0.0% | 2 |
| is_ten_percent_owner | bool | 750 | 0.0% | 1 |
| officer_title | object | 750 | 22.0% | 6 |
| has_10b5_1_plan | bool | 750 | 0.0% | 2 |
| footnote_text | object | 750 | 1.9% | 148 |
| is_amendment | bool | 750 | 0.0% | 1 |
| xml_path | object | 750 | 0.0% | 255 |
| security_title | object | 750 | 0.0% | 3 |
| transaction_date | datetime64[ns] | 750 | 0.0% | 140 |
| transaction_code | object | 750 | 0.0% | 5 |
| shares | float64 | 750 | 0.0% | 401 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| issuer_cik | 320193.00 | 0.00 | 320193.00 | 320193.00 | 320193.00 | 0.0% |
| insider_cik | 1473578.34 | 228875.79 | 1051401.00 | 1496686.00 | 2078476.00 | 0.0% |
| shares | 72582.05 | 296282.49 | 165.00 | 16457.50 | 5040000.00 | 8.8% |
| price_per_share | 133.09 | 109.78 | 0.00 | 150.00 | 503.43 | 0.0% |
| total_value | 10142487.02 | 30608974.45 | 27934.50 | 2950875.30 | 397025647.20 | 4.6% |
| shares_owned_after | 423420.40 | 1000426.05 | 0.00 | 66477.00 | 8319726.00 | 10.0% |
| conversion_price | 38.90 | 14.20 | 28.86 | 38.90 | 48.95 | 0.0% |
| underlying_shares | 66697.49 | 293855.25 | 281.00 | 16612.00 | 5040000.00 | 7.2% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 337 |
| S | 207 |
| A | 111 |
| F | 73 |
| G | 22 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 325 |
| Director | 165 |
| CFO | 100 |
| COO | 88 |
| CEO | 72 |

---

## Warnings

- Found 26 duplicate rows (3.5%)
- High missing values (>50%) in: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares