# Phase 3 Label Quality Report

**Generated:** 2026-03-20T16:05:52.774585

**Input CSV:** data/processed/ACN_form4_features.csv

---

## Coverage
- Forward return coverage: 755 rows (100.0%)
- Benchmark return coverage: 755 rows (100.0%)

## Label Distribution
- Opportunistic: 4 (0.53%)
- Routine: 235 (31.13%)
- Uncertain: 516 (68.34%)

## Signal Counts
- Price signal rows: 156
- Earnings signal rows: 396
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 269 |
| 1 | 420 |
| 2 | 66 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.