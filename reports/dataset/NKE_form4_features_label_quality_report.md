# Phase 3 Label Quality Report

**Generated:** 2026-03-20T22:16:27.970358

**Input CSV:** data/processed/NKE_form4_features.csv

---

## Coverage
- Forward return coverage: 872 rows (100.0%)
- Benchmark return coverage: 872 rows (100.0%)

## Label Distribution
- Opportunistic: 0 (0.0%)
- Routine: 215 (24.66%)
- Uncertain: 657 (75.34%)

## Signal Counts
- Price signal rows: 170
- Earnings signal rows: 714
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 106 |
| 1 | 648 |
| 2 | 118 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.