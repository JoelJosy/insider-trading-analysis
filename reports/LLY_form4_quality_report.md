# Data Quality Report

**Generated:** 2026-03-18T21:55:52.832246

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,412
- **Total Columns:** 31
- **Memory Usage:** 3.86 MB
- **Date Range:** 2022-08-15 to 2025-12-29
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,412 | 0.0% | 500 |
| filing_date | datetime64[ns] | 1,412 | 0.0% | 202 |
| issuer_cik | object | 1,412 | 0.0% | 3 |
| issuer_name | object | 1,412 | 0.0% | 3 |
| ticker | object | 1,412 | 0.0% | 3 |
| insider_name | object | 1,412 | 0.0% | 35 |
| insider_cik | object | 1,412 | 0.0% | 35 |
| insider_role | object | 1,412 | 0.0% | 6 |
| is_director | bool | 1,412 | 0.0% | 2 |
| is_officer | bool | 1,412 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,412 | 0.0% | 2 |
| officer_title | object | 1,412 | 82.8% | 26 |
| has_10b5_1_plan | bool | 1,412 | 0.0% | 2 |
| footnote_text | object | 1,412 | 2.3% | 183 |
| is_amendment | bool | 1,412 | 0.0% | 1 |
| xml_path | object | 1,412 | 0.0% | 500 |
| security_title | object | 1,412 | 0.0% | 3 |
| transaction_date | datetime64[ns] | 1,412 | 0.0% | 209 |
| transaction_code | object | 1,412 | 0.0% | 7 |
| shares | float64 | 1,412 | 0.0% | 1,054 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 31701.71 | 799407.93 | 4.67 | 3269.00 | 29992668.00 | 10.3% |
| price_per_share | 3139.41 | 47584.83 | 0.00 | 780.21 | 1032319.00 | 0.3% |
| total_value | 83701651.23 | 1785006503.43 | 1861.75 | 2388281.81 | 57664293912.00 | 9.8% |
| shares_owned_after | 61814254.70 | 46895633.93 | 0.00 | 94092644.00 | 103854110.00 | 0.0% |
| underlying_shares | 6192.59 | 8401.86 | 184.00 | 3490.00 | 47786.00 | 10.7% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 924 |
| A | 341 |
| M | 76 |
| F | 38 |
| G | 21 |
| P | 11 |
| I | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Other | 766 |
| Director | 270 |
| Officer | 209 |
| 10%_Owner | 133 |
| CEO | 26 |
| CFO | 8 |

---

## Warnings

- High missing values (>50%) in: officer_title, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment
- Found 4 suspiciously high prices (>$10,000/share)

---

## Recommendations

- Implement imputation for: conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic