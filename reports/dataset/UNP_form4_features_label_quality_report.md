# Phase 3 Label Quality Report

**Generated:** 2026-03-21T05:22:23.028915

**Input CSV:** data/processed/UNP_form4_features.csv

---

## Coverage
- Forward return coverage: 253 rows (100.0%)
- Benchmark return coverage: 253 rows (100.0%)

## Label Distribution
- Opportunistic: 0 (0.0%)
- Routine: 5 (1.98%)
- Uncertain: 248 (98.02%)

## Signal Counts
- Price signal rows: 34
- Earnings signal rows: 197
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 56 |
| 1 | 163 |
| 2 | 34 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.