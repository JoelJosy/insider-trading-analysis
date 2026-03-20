# Phase 3 Label Quality Report

**Generated:** 2026-03-20T13:36:28.331067

**Input CSV:** data/processed/META_form4_features.csv

---

## Coverage
- Forward return coverage: 3667 rows (100.0%)
- Benchmark return coverage: 3667 rows (100.0%)

## Label Distribution
- Opportunistic: 11 (0.3%)
- Routine: 2865 (78.13%)
- Uncertain: 791 (21.57%)

## Signal Counts
- Price signal rows: 893
- Earnings signal rows: 2561
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 873 |
| 1 | 2134 |
| 2 | 660 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.