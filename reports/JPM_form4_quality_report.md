# Data Quality Report

**Generated:** 2026-03-18T21:55:47.053947

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 932
- **Total Columns:** 31
- **Memory Usage:** 1.58 MB
- **Date Range:** 2019-03-01 to 2025-11-12
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 932 | 0.0% | 487 |
| filing_date | datetime64[ns] | 932 | 0.0% | 118 |
| issuer_cik | object | 932 | 0.0% | 3 |
| issuer_name | object | 932 | 0.0% | 3 |
| ticker | object | 932 | 0.0% | 3 |
| insider_name | object | 932 | 0.0% | 29 |
| insider_cik | object | 932 | 0.0% | 29 |
| insider_role | object | 932 | 0.0% | 6 |
| is_director | bool | 932 | 0.0% | 2 |
| is_officer | bool | 932 | 0.0% | 2 |
| is_ten_percent_owner | bool | 932 | 0.0% | 2 |
| officer_title | object | 932 | 18.2% | 20 |
| has_10b5_1_plan | bool | 932 | 0.0% | 2 |
| footnote_text | object | 932 | 9.1% | 127 |
| is_amendment | bool | 932 | 0.0% | 1 |
| xml_path | object | 932 | 0.0% | 487 |
| security_title | object | 932 | 0.0% | 6 |
| transaction_date | datetime64[ns] | 932 | 0.0% | 138 |
| transaction_code | object | 932 | 0.0% | 8 |
| shares | float64 | 932 | 0.0% | 530 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 33062.87 | 99190.75 | 14.00 | 12688.00 | 1639344.00 | 10.3% |
| price_per_share | 74.93 | 94.15 | 0.00 | 0.00 | 315.43 | 0.0% |
| total_value | 3974457.97 | 14733510.20 | 51.38 | 620793.40 | 223095000.00 | 9.9% |
| shares_owned_after | 642058.52 | 4694065.50 | 0.00 | 45026.00 | 52036586.00 | 15.0% |
| conversion_price | 60.63 | 50.86 | 3.77 | 46.58 | 159.09 | 30.0% |
| underlying_shares | 220120.84 | 3013950.49 | 753.00 | 19358.63 | 52036572.00 | 8.4% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 386 |
| A | 253 |
| S | 118 |
| F | 117 |
| G | 42 |
| P | 10 |
| X | 4 |
| J | 2 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 344 |
| CEO | 340 |
| Director | 158 |
| COO | 49 |
| CFO | 29 |
| 10%_Owner | 12 |

---

## Warnings

- High missing values (>50%) in: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment, exercise_date

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares