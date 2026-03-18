# Data Quality Report

**Generated:** 2026-03-18T15:18:21.589929

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 336
- **Total Columns:** 32
- **Memory Usage:** 0.50 MB
- **Date Range:** 2021-12-15 to 2024-12-13
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 336 | 0.0% | 217 |
| filing_date | datetime64[ns] | 336 | 0.0% | 74 |
| issuer_cik | object | 336 | 0.0% | 2 |
| issuer_name | object | 336 | 0.0% | 2 |
| ticker | object | 336 | 0.0% | 2 |
| insider_name | object | 336 | 0.0% | 31 |
| insider_cik | object | 336 | 0.0% | 31 |
| insider_role | object | 336 | 0.0% | 5 |
| is_director | bool | 336 | 0.0% | 2 |
| is_officer | bool | 336 | 0.0% | 2 |
| is_ten_percent_owner | bool | 336 | 0.0% | 1 |
| officer_title | object | 336 | 15.2% | 7 |
| has_10b5_1_plan | bool | 336 | 0.0% | 1 |
| footnote_text | object | 336 | 1.8% | 27 |
| is_amendment | bool | 336 | 0.0% | 1 |
| xml_path | object | 336 | 0.0% | 217 |
| security_title | object | 336 | 0.0% | 4 |
| transaction_date | datetime64[ns] | 336 | 0.0% | 79 |
| transaction_code | object | 336 | 0.0% | 8 |
| shares | float64 | 336 | 0.0% | 208 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 110039.21 | 1491809.45 | 3.00 | 3477.00 | 27349211.00 | 11.0% |
| price_per_share | 25.49 | 17.37 | 0.00 | 27.96 | 59.05 | 0.0% |
| total_value | 4664542.40 | 67620570.60 | 153.53 | 110643.84 | 1230714495.00 | 11.2% |
| shares_owned_after | 124796.08 | 181619.96 | 0.00 | 49557.97 | 691116.00 | 11.3% |
| conversion_price | 36.56 | 9.02 | 25.60 | 42.30 | 46.94 | 0.0% |
| underlying_shares | 30246.65 | 70347.24 | 3.00 | 1345.95 | 644640.00 | 11.3% |
| close_price_on_txn_date | 34.36 | 8.54 | 24.80 | 29.16 | 57.19 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 223 |
| F | 47 |
| M | 30 |
| D | 17 |
| S | 8 |
| I | 6 |
| P | 4 |
| U | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 226 |
| CEO | 55 |
| Director | 50 |
| CFO | 4 |
| Other | 1 |

---

## Warnings

- High missing values (>50%) in: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date
- Constant columns: is_ten_percent_owner, has_10b5_1_plan, is_amendment, direct_indirect

---

## Recommendations

- Implement imputation for: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic