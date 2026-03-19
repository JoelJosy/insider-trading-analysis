# Data Quality Report

**Generated:** 2026-03-18T21:56:56.535386

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 791
- **Total Columns:** 31
- **Memory Usage:** 1.73 MB
- **Date Range:** 2023-06-16 to 2025-12-26
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 791 | 0.0% | 478 |
| filing_date | datetime64[ns] | 791 | 0.0% | 172 |
| issuer_cik | object | 791 | 0.0% | 2 |
| issuer_name | object | 791 | 0.0% | 2 |
| ticker | object | 791 | 0.0% | 2 |
| insider_name | object | 791 | 0.0% | 28 |
| insider_cik | object | 791 | 0.0% | 28 |
| insider_role | object | 791 | 0.0% | 4 |
| is_director | bool | 791 | 0.0% | 2 |
| is_officer | bool | 791 | 0.0% | 2 |
| is_ten_percent_owner | bool | 791 | 0.0% | 2 |
| officer_title | object | 791 | 62.1% | 4 |
| has_10b5_1_plan | bool | 791 | 0.0% | 2 |
| footnote_text | object | 791 | 0.4% | 292 |
| is_amendment | bool | 791 | 0.0% | 1 |
| xml_path | object | 791 | 0.0% | 478 |
| security_title | object | 791 | 0.0% | 5 |
| transaction_date | datetime64[ns] | 791 | 0.0% | 230 |
| transaction_code | object | 791 | 0.0% | 7 |
| shares | float64 | 791 | 0.0% | 414 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 17931672.90 | 216139044.67 | 10.18 | 25000.00 | 3002673393.00 | 15.0% |
| price_per_share | 76.18 | 62.41 | 0.00 | 79.83 | 177.44 | 0.0% |
| total_value | 45236758.33 | 93224726.00 | 997.63 | 1968128.33 | 769784848.10 | 10.8% |
| shares_owned_after | 189347098.17 | 250314125.53 | 0.00 | 1415632.82 | 656395261.00 | 15.0% |
| conversion_price | 10.00 | nan | 10.00 | 10.00 | 10.00 | 0.0% |
| underlying_shares | 15870411.00 | 0.00 | 15870411.00 | 15870411.00 | 15870411.00 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 414 |
| A | 142 |
| F | 103 |
| J | 103 |
| G | 25 |
| X | 3 |
| P | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| 10%_Owner | 298 |
| Officer | 263 |
| Director | 193 |
| CEO | 37 |

---

## Warnings

- High missing values (>50%) in: officer_title, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment, conversion_price, expiration_date, underlying_shares

---

## Recommendations

- Implement imputation for: total_value, conversion_price, exercise_date, expiration_date, underlying_shares