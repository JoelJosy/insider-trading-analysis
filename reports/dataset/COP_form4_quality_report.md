# Data Quality Report

**Generated:** 2026-03-18T21:55:37.441438

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 946
- **Total Columns:** 31
- **Memory Usage:** 1.67 MB
- **Date Range:** 2019-09-30 to 2025-12-19
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 946 | 0.0% | 484 |
| filing_date | datetime64[ns] | 946 | 0.0% | 160 |
| issuer_cik | object | 946 | 0.0% | 1 |
| issuer_name | object | 946 | 0.0% | 1 |
| ticker | object | 946 | 0.0% | 1 |
| insider_name | object | 946 | 0.0% | 35 |
| insider_cik | object | 946 | 0.0% | 34 |
| insider_role | object | 946 | 0.0% | 6 |
| is_director | bool | 946 | 0.0% | 2 |
| is_officer | bool | 946 | 0.0% | 2 |
| is_ten_percent_owner | bool | 946 | 0.0% | 1 |
| officer_title | object | 946 | 31.0% | 13 |
| has_10b5_1_plan | bool | 946 | 0.0% | 1 |
| footnote_text | object | 946 | 1.6% | 169 |
| is_amendment | bool | 946 | 0.0% | 1 |
| xml_path | object | 946 | 0.0% | 484 |
| security_title | object | 946 | 0.0% | 5 |
| transaction_date | datetime64[ns] | 946 | 0.0% | 169 |
| transaction_code | object | 946 | 0.0% | 7 |
| shares | float64 | 946 | 0.0% | 537 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 25143.01 | 80936.29 | 60.00 | 4127.00 | 832519.00 | 11.2% |
| price_per_share | 54.05 | 44.71 | 0.00 | 58.08 | 135.63 | 0.0% |
| total_value | 1687848.94 | 7085375.40 | 3581.30 | 220025.04 | 76426823.70 | 14.2% |
| shares_owned_after | 48268.34 | 113718.06 | 0.00 | 15209.00 | 832519.00 | 6.9% |
| conversion_price | 52.91 | 13.38 | 33.12 | 54.80 | 69.25 | 0.0% |
| underlying_shares | 19748.51 | 67521.94 | 60.00 | 2401.00 | 819900.00 | 14.5% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 361 |
| A | 318 |
| F | 146 |
| D | 60 |
| S | 38 |
| G | 12 |
| P | 11 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 382 |
| Director | 215 |
| Other | 201 |
| CEO | 77 |
| CFO | 48 |
| COO | 23 |

---

## Warnings

- High missing values (>50%) in: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, has_10b5_1_plan, is_amendment

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic