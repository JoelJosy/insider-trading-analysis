# Data Quality Report

**Generated:** 2026-03-18T21:56:42.154354

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 3,264
- **Total Columns:** 31
- **Memory Usage:** 13.45 MB
- **Date Range:** 2019-11-22 to 2025-12-30
- **Duplicate Rows:** 5

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 3,264 | 0.0% | 500 |
| filing_date | datetime64[ns] | 3,264 | 0.0% | 281 |
| issuer_cik | object | 3,264 | 0.0% | 2 |
| issuer_name | object | 3,264 | 0.0% | 2 |
| ticker | object | 3,264 | 0.0% | 2 |
| insider_name | object | 3,264 | 0.0% | 25 |
| insider_cik | object | 3,264 | 0.0% | 25 |
| insider_role | object | 3,264 | 0.0% | 6 |
| is_director | bool | 3,264 | 0.0% | 2 |
| is_officer | bool | 3,264 | 0.0% | 2 |
| is_ten_percent_owner | bool | 3,264 | 0.0% | 2 |
| officer_title | object | 3,264 | 45.4% | 16 |
| has_10b5_1_plan | bool | 3,264 | 0.0% | 2 |
| footnote_text | object | 3,264 | 1.8% | 258 |
| is_amendment | bool | 3,264 | 0.0% | 1 |
| xml_path | object | 3,264 | 0.0% | 500 |
| security_title | object | 3,264 | 0.0% | 8 |
| transaction_date | datetime64[ns] | 3,264 | 0.0% | 346 |
| transaction_code | object | 3,264 | 0.0% | 8 |
| shares | float64 | 3,264 | 0.0% | 1,172 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 22251.08 | 489138.66 | 1.00 | 534.00 | 20421899.00 | 13.3% |
| price_per_share | 527.79 | 315.82 | 0.00 | 600.06 | 1206.70 | 0.0% |
| total_value | 6024967.85 | 189072647.16 | 462.15 | 300994.40 | 10412105205.15 | 11.7% |
| shares_owned_after | 64915.37 | 384676.99 | 0.00 | 22452.50 | 20421899.00 | 11.4% |
| conversion_price | 401.01 | 215.39 | 0.00 | 380.95 | 888.34 | 0.0% |
| underlying_shares | 22984.34 | 90124.80 | 1.00 | 3500.00 | 1108015.00 | 12.7% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 2,105 |
| M | 657 |
| F | 220 |
| A | 202 |
| G | 70 |
| C | 5 |
| I | 4 |
| W | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Director | 1,474 |
| Officer | 1,201 |
| CFO | 487 |
| CEO | 95 |
| Other | 4 |
| 10%_Owner | 3 |

---

## Warnings

- Found 5 duplicate rows (0.2%)
- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: conversion_price, exercise_date, expiration_date, underlying_shares