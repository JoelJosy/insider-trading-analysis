# Data Quality Report

**Generated:** 2026-03-18T21:56:21.553045

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 460
- **Total Columns:** 32
- **Memory Usage:** 0.76 MB
- **Date Range:** 2019-03-18 to 2025-11-10
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 460 | 0.0% | 259 |
| filing_date | datetime64[ns] | 460 | 0.0% | 84 |
| issuer_cik | object | 460 | 0.0% | 3 |
| issuer_name | object | 460 | 0.0% | 3 |
| ticker | object | 460 | 0.0% | 3 |
| insider_name | object | 460 | 0.0% | 43 |
| insider_cik | object | 460 | 0.0% | 41 |
| insider_role | object | 460 | 0.0% | 5 |
| is_director | bool | 460 | 0.0% | 2 |
| is_officer | bool | 460 | 0.0% | 2 |
| is_ten_percent_owner | bool | 460 | 0.0% | 2 |
| officer_title | object | 460 | 25.2% | 27 |
| has_10b5_1_plan | bool | 460 | 0.0% | 2 |
| footnote_text | object | 460 | 1.3% | 99 |
| is_amendment | bool | 460 | 0.0% | 1 |
| xml_path | object | 460 | 0.0% | 259 |
| security_title | object | 460 | 0.0% | 6 |
| transaction_date | datetime64[ns] | 460 | 0.0% | 101 |
| transaction_code | object | 460 | 0.0% | 8 |
| shares | float64 | 460 | 0.0% | 303 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 643069.66 | 11853259.85 | 3.00 | 2353.00 | 253516000.00 | 15.0% |
| price_per_share | 66.28 | 46.62 | 0.00 | 78.33 | 128.80 | 0.0% |
| total_value | 2350975.16 | 6112403.64 | 341.04 | 220000.00 | 65946928.00 | 12.0% |
| shares_owned_after | 117955.35 | 743491.29 | 0.00 | 21578.77 | 12955016.00 | 8.3% |
| conversion_price | 88.32 | 29.77 | 39.29 | 84.71 | 129.22 | 0.0% |
| underlying_shares | 56983.06 | 543334.52 | 36.34 | 1980.73 | 8333333.00 | 16.5% |
| close_price_on_txn_date | 92.73 | 20.63 | 60.49 | 94.79 | 126.58 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 206 |
| M | 144 |
| S | 49 |
| F | 47 |
| P | 10 |
| C | 2 |
| J | 1 |
| G | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 311 |
| Director | 112 |
| CFO | 17 |
| CEO | 16 |
| 10%_Owner | 4 |

---

## Warnings

- High missing values (>50%) in: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date
- Constant columns: is_amendment

---

## Recommendations

- Implement imputation for: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares