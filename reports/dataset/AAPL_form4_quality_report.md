# Data Quality Report

**Generated:** 2026-03-18T21:50:08.930099

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,402
- **Total Columns:** 31
- **Memory Usage:** 3.41 MB
- **Date Range:** 2016-04-01 to 2026-02-01
- **Duplicate Rows:** 40

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,402 | 0.0% | 486 |
| filing_date | datetime64[ns] | 1,402 | 0.0% | 212 |
| issuer_cik | object | 1,402 | 0.0% | 1 |
| issuer_name | object | 1,402 | 0.0% | 2 |
| ticker | object | 1,402 | 0.0% | 1 |
| insider_name | object | 1,402 | 0.0% | 25 |
| insider_cik | object | 1,402 | 0.0% | 25 |
| insider_role | object | 1,402 | 0.0% | 5 |
| is_director | bool | 1,402 | 0.0% | 2 |
| is_officer | bool | 1,402 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,402 | 0.0% | 1 |
| officer_title | object | 1,402 | 19.3% | 8 |
| has_10b5_1_plan | bool | 1,402 | 0.0% | 2 |
| footnote_text | object | 1,402 | 2.7% | 300 |
| is_amendment | bool | 1,402 | 0.0% | 1 |
| xml_path | object | 1,402 | 0.0% | 486 |
| security_title | object | 1,402 | 0.0% | 4 |
| transaction_date | datetime64[ns] | 1,402 | 0.0% | 301 |
| transaction_code | object | 1,402 | 0.0% | 5 |
| shares | float64 | 1,402 | 0.0% | 684 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 56758.77 | 221835.74 | 100.00 | 16467.50 | 5040000.00 | 5.9% |
| price_per_share | 120.15 | 98.37 | 0.00 | 142.85 | 503.43 | 1.1% |
| total_value | 7182652.51 | 22375238.46 | 15576.00 | 2249296.86 | 397025647.20 | 6.3% |
| shares_owned_after | 301235.57 | 777303.09 | 0.00 | 59064.00 | 8319726.00 | 16.0% |
| conversion_price | 24.15 | 8.45 | 13.02 | 23.83 | 48.95 | 6.7% |
| underlying_shares | 55087.74 | 224781.05 | 281.00 | 17656.50 | 5040000.00 | 4.5% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 617 |
| S | 397 |
| A | 190 |
| F | 147 |
| G | 51 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 694 |
| Director | 271 |
| CFO | 182 |
| COO | 148 |
| CEO | 107 |

---

## Warnings

- Found 40 duplicate rows (2.9%)
- High missing values (>50%) in: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares