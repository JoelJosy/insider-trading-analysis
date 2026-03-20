# Phase 3 Label Quality Report

**Generated:** 2026-03-20T16:22:05.912045

**Input CSV:** data/processed/AMZN_form4_features.csv

---

## Coverage
- Forward return coverage: 2485 rows (100.0%)
- Benchmark return coverage: 2485 rows (100.0%)

## Label Distribution
- Opportunistic: 1 (0.04%)
- Routine: 1787 (71.91%)
- Uncertain: 697 (28.05%)

## Signal Counts
- Price signal rows: 236
- Earnings signal rows: 2362
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 109 |
| 1 | 2154 |
| 2 | 222 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.