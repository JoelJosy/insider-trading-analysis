# Data Quality Report

**Generated:** 2026-03-18T21:55:27.382756

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,471
- **Total Columns:** 31
- **Memory Usage:** 2.77 MB
- **Date Range:** 2015-06-02 to 2025-12-01
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,471 | 0.0% | 500 |
| filing_date | datetime64[ns] | 1,471 | 0.0% | 180 |
| issuer_cik | object | 1,471 | 0.0% | 3 |
| issuer_name | object | 1,471 | 0.0% | 3 |
| ticker | object | 1,471 | 0.0% | 3 |
| insider_name | object | 1,471 | 0.0% | 49 |
| insider_cik | object | 1,471 | 0.0% | 49 |
| insider_role | object | 1,471 | 0.0% | 5 |
| is_director | bool | 1,471 | 0.0% | 2 |
| is_officer | bool | 1,471 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,471 | 0.0% | 2 |
| officer_title | object | 1,471 | 12.1% | 41 |
| has_10b5_1_plan | bool | 1,471 | 0.0% | 2 |
| footnote_text | object | 1,471 | 7.5% | 163 |
| is_amendment | bool | 1,471 | 0.0% | 1 |
| xml_path | object | 1,471 | 0.0% | 500 |
| security_title | object | 1,471 | 0.0% | 5 |
| transaction_date | datetime64[ns] | 1,471 | 0.0% | 210 |
| transaction_code | object | 1,471 | 0.0% | 7 |
| shares | float64 | 1,471 | 0.0% | 816 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 80570.41 | 2819459.88 | 1.00 | 1055.00 | 107975968.00 | 12.9% |
| price_per_share | 77.58 | 123.62 | 0.00 | 0.00 | 440.00 | 0.0% |
| total_value | 717043.18 | 3520395.51 | 494.44 | 136661.40 | 58338000.00 | 14.0% |
| shares_owned_after | 206711.63 | 2131553.78 | 0.00 | 5620.00 | 24391273.00 | 13.7% |
| conversion_price | 1.26 | 13.93 | 0.00 | 0.00 | 301.85 | 1.5% |
| underlying_shares | 3296.02 | 6358.81 | 1.00 | 1482.50 | 80522.00 | 15.7% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 592 |
| F | 329 |
| A | 257 |
| J | 177 |
| S | 80 |
| P | 23 |
| G | 13 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 1,070 |
| Director | 165 |
| CEO | 113 |
| CFO | 110 |
| 10%_Owner | 13 |

---

## Warnings

- High missing values (>50%) in: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares