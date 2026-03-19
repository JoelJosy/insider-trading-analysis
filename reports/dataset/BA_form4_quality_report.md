# Data Quality Report

**Generated:** 2026-03-18T23:02:25.493127

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 557
- **Total Columns:** 32
- **Memory Usage:** 0.74 MB
- **Date Range:** 2020-02-23 to 2025-12-01
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 557 | 0.0% | 500 |
| filing_date | datetime64[ns] | 557 | 0.0% | 97 |
| issuer_cik | object | 557 | 0.0% | 1 |
| issuer_name | object | 557 | 0.0% | 1 |
| ticker | object | 557 | 0.0% | 1 |
| insider_name | object | 557 | 0.0% | 54 |
| insider_cik | object | 557 | 0.0% | 54 |
| insider_role | object | 557 | 0.0% | 5 |
| is_director | bool | 557 | 0.0% | 2 |
| is_officer | bool | 557 | 0.0% | 2 |
| is_ten_percent_owner | bool | 557 | 0.0% | 1 |
| officer_title | object | 557 | 45.4% | 42 |
| has_10b5_1_plan | bool | 557 | 0.0% | 1 |
| footnote_text | object | 557 | 1.6% | 86 |
| is_amendment | bool | 557 | 0.0% | 1 |
| xml_path | object | 557 | 0.0% | 500 |
| security_title | object | 557 | 0.0% | 4 |
| transaction_date | datetime64[ns] | 557 | 0.0% | 97 |
| transaction_code | object | 557 | 0.0% | 6 |
| shares | float64 | 557 | 0.0% | 425 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 4903.34 | 12378.18 | 12.07 | 580.27 | 121023.00 | 13.1% |
| price_per_share | 61.79 | 99.26 | 0.00 | 0.00 | 332.45 | 0.0% |
| total_value | 1858015.74 | 4185539.59 | 3102.59 | 269126.49 | 27745281.85 | 16.3% |
| shares_owned_after | 25042.27 | 29912.79 | 0.00 | 15805.00 | 196903.36 | 7.4% |
| conversion_price | 35.41 | 86.94 | 0.00 | 0.00 | 280.73 | 14.5% |
| underlying_shares | 4880.82 | 15275.77 | 108.14 | 450.67 | 121023.00 | 15.1% |
| close_price_on_txn_date | 204.85 | 38.33 | 126.05 | 203.37 | 317.90 | 9.6% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 389 |
| F | 129 |
| S | 16 |
| P | 10 |
| M | 8 |
| D | 5 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Director | 253 |
| Officer | 185 |
| CEO | 97 |
| CFO | 19 |
| COO | 3 |

---

## Warnings

- High missing values (>50%) in: total_value, acquired_disposed, direct_indirect, exercise_date, expiration_date
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, has_10b5_1_plan, is_amendment

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic