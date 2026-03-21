# Phase 3 Label Quality Report

**Generated:** 2026-03-21T05:20:00.310160

**Input CSV:** data/processed/T_form4_features.csv

---

## Coverage
- Forward return coverage: 433 rows (100.0%)
- Benchmark return coverage: 433 rows (100.0%)

## Label Distribution
- Opportunistic: 1 (0.23%)
- Routine: 0 (0.0%)
- Uncertain: 432 (99.77%)

## Signal Counts
- Price signal rows: 123
- Earnings signal rows: 339
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 83 |
| 1 | 238 |
| 2 | 112 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.