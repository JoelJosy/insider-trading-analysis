# Data Quality Report

**Generated:** 2026-03-18T21:55:14.870112

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 2,940
- **Total Columns:** 31
- **Memory Usage:** 8.23 MB
- **Date Range:** 2011-07-26 to 2025-12-29
- **Duplicate Rows:** 89

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 2,940 | 0.0% | 476 |
| filing_date | datetime64[ns] | 2,940 | 0.0% | 184 |
| issuer_cik | object | 2,940 | 0.0% | 64 |
| issuer_name | object | 2,940 | 0.0% | 66 |
| ticker | object | 2,940 | 0.0% | 65 |
| insider_name | object | 2,940 | 0.0% | 39 |
| insider_cik | object | 2,940 | 0.0% | 39 |
| insider_role | object | 2,940 | 0.0% | 6 |
| is_director | bool | 2,940 | 0.0% | 2 |
| is_officer | bool | 2,940 | 0.0% | 2 |
| is_ten_percent_owner | bool | 2,940 | 0.0% | 2 |
| officer_title | object | 2,940 | 50.7% | 21 |
| has_10b5_1_plan | bool | 2,940 | 0.0% | 1 |
| footnote_text | object | 2,940 | 0.1% | 243 |
| is_amendment | bool | 2,940 | 0.0% | 1 |
| xml_path | object | 2,940 | 0.0% | 476 |
| security_title | object | 2,940 | 0.0% | 62 |
| transaction_date | datetime64[ns] | 2,940 | 0.0% | 452 |
| transaction_code | object | 2,940 | 0.0% | 8 |
| shares | float64 | 2,940 | 0.0% | 1,013 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 112754.45 | 875770.11 | 0.00 | 2674.00 | 20249064.00 | 9.7% |
| price_per_share | 587.54 | 3735.87 | 0.00 | 13.45 | 47500.00 | 2.4% |
| total_value | 7899125.31 | 64920043.39 | 8.39 | 3877.23 | 2023286474.88 | 18.9% |
| shares_owned_after | 15513475.56 | 115718443.15 | 0.00 | 15557.00 | 1020161313.00 | 14.2% |
| conversion_price | 1.73 | 4.21 | 0.00 | 0.00 | 11.50 | 15.0% |
| underlying_shares | 33066.28 | 45494.98 | 9.00 | 20589.00 | 413659.00 | 8.5% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 860 |
| S | 764 |
| P | 597 |
| F | 328 |
| A | 176 |
| J | 96 |
| D | 70 |
| G | 49 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Other | 1,395 |
| Officer | 1,204 |
| CEO | 181 |
| CFO | 64 |
| 10%_Owner | 55 |
| Director | 41 |

---

## Warnings

- Found 89 duplicate rows (3.0%)
- High missing values (>50%) in: officer_title, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: has_10b5_1_plan, is_amendment
- Found 41 suspiciously high prices (>$10,000/share)

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic