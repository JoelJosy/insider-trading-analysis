# Data Quality Report

**Generated:** 2026-03-18T23:02:16.836995

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,464
- **Total Columns:** 32
- **Memory Usage:** 4.13 MB
- **Date Range:** 2018-01-19 to 2025-11-25
- **Duplicate Rows:** 3

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,464 | 0.0% | 500 |
| filing_date | datetime64[ns] | 1,464 | 0.0% | 173 |
| issuer_cik | object | 1,464 | 0.0% | 42 |
| issuer_name | object | 1,464 | 0.0% | 42 |
| ticker | object | 1,464 | 0.0% | 44 |
| insider_name | object | 1,464 | 0.0% | 42 |
| insider_cik | object | 1,464 | 0.0% | 42 |
| insider_role | object | 1,464 | 0.0% | 7 |
| is_director | bool | 1,464 | 0.0% | 2 |
| is_officer | bool | 1,464 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,464 | 0.0% | 2 |
| officer_title | object | 1,464 | 68.3% | 20 |
| has_10b5_1_plan | bool | 1,464 | 0.0% | 2 |
| footnote_text | object | 1,464 | 0.0% | 344 |
| is_amendment | bool | 1,464 | 0.0% | 1 |
| xml_path | object | 1,464 | 0.0% | 500 |
| security_title | object | 1,464 | 0.0% | 39 |
| transaction_date | datetime64[ns] | 1,464 | 16.7% | 292 |
| transaction_code | object | 1,464 | 0.0% | 10 |
| shares | float64 | 1,464 | 0.0% | 813 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 486411.82 | 3708338.11 | 1.00 | 3609.00 | 66471429.00 | 16.2% |
| price_per_share | 127.79 | 197.20 | 0.00 | 16.77 | 751.21 | 1.1% |
| total_value | 5465640.86 | 44240384.67 | 2.20 | 236810.66 | 1143531187.20 | 11.5% |
| shares_owned_after | 5691440.64 | 15791937.12 | 0.00 | 105794.00 | 95882920.00 | 17.3% |
| conversion_price | 69.25 | 21.30 | 0.00 | 75.00 | 78.78 | 9.5% |
| underlying_shares | 788100.00 | 4650556.48 | 0.00 | 3200.00 | 69574201.35 | 15.8% |
| close_price_on_txn_date | 247.32 | 222.80 | 2.40 | 278.12 | 802.32 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 631 |
| P | 230 |
| A | 214 |
| M | 142 |
| C | 116 |
| F | 74 |
| J | 30 |
| G | 21 |
| X | 4 |
| D | 2 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Director | 448 |
| 10%_Owner | 342 |
| Other | 328 |
| Officer | 246 |
| CEO | 39 |
| COO | 34 |
| CFO | 27 |

---

## Warnings

- Found 3 duplicate rows (0.2%)
- High missing values (>50%) in: officer_title, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares, close_price_on_txn_date
- Low 10b5-1 plan coverage - review footnote parsing logic