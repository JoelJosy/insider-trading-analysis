# Phase 3 Label Quality Report

**Generated:** 2026-03-20T16:30:40.845398

**Input CSV:** data/processed/BLK_form4_features.csv

---

## Coverage
- Forward return coverage: 214 rows (100.0%)
- Benchmark return coverage: 214 rows (100.0%)

## Label Distribution
- Opportunistic: 87 (40.65%)
- Routine: 0 (0.0%)
- Uncertain: 127 (59.35%)

## Signal Counts
- Price signal rows: 22
- Earnings signal rows: 68
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 133 |
| 1 | 72 |
| 2 | 9 |

## Recommendations
- Opportunistic share is high (>40%); review Cohen/plan rules for over-sensitive labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.