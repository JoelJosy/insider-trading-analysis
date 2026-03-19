# Data Quality Report

**Generated:** 2026-03-18T21:57:03.623502

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 710
- **Total Columns:** 31
- **Memory Usage:** 0.84 MB
- **Date Range:** 2016-11-30 to 2025-12-18
- **Duplicate Rows:** 15

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 710 | 0.0% | 455 |
| filing_date | datetime64[ns] | 710 | 0.0% | 149 |
| issuer_cik | object | 710 | 0.0% | 2 |
| issuer_name | object | 710 | 0.0% | 2 |
| ticker | object | 710 | 0.0% | 2 |
| insider_name | object | 710 | 0.0% | 65 |
| insider_cik | object | 710 | 0.0% | 65 |
| insider_role | object | 710 | 0.0% | 3 |
| is_director | bool | 710 | 0.0% | 2 |
| is_officer | bool | 710 | 0.0% | 2 |
| is_ten_percent_owner | bool | 710 | 0.0% | 2 |
| officer_title | object | 710 | 20.4% | 16 |
| has_10b5_1_plan | bool | 710 | 0.0% | 1 |
| footnote_text | object | 710 | 11.1% | 148 |
| is_amendment | bool | 710 | 0.0% | 1 |
| xml_path | object | 710 | 0.0% | 455 |
| security_title | object | 710 | 0.0% | 6 |
| transaction_date | datetime64[ns] | 710 | 0.0% | 165 |
| transaction_code | object | 710 | 0.0% | 7 |
| shares | float64 | 710 | 0.0% | 324 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 118088.81 | 984236.28 | 4.00 | 4536.00 | 13530723.00 | 12.1% |
| price_per_share | 51.09 | 43.34 | 0.00 | 62.97 | 119.01 | 0.0% |
| total_value | 2991894.60 | 15609999.84 | 2890.75 | 456189.39 | 216127427.50 | 12.9% |
| shares_owned_after | 401005.28 | 883931.57 | 0.00 | 219850.00 | 13530723.00 | 7.6% |
| conversion_price | 2.44 | 0.53 | 2.25 | 2.25 | 3.75 | 12.5% |
| underlying_shares | 8133931.88 | 4767467.99 | 2489643.00 | 6500000.00 | 13530723.00 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 266 |
| F | 172 |
| G | 166 |
| S | 64 |
| P | 34 |
| D | 6 |
| I | 2 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 512 |
| Director | 145 |
| CEO | 53 |

---

## Warnings

- Found 15 duplicate rows (2.1%)
- High missing values (>50%) in: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: has_10b5_1_plan, is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: price_per_share, total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic