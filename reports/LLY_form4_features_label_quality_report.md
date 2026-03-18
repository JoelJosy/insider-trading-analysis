# Phase 3 Label Quality Report

**Generated:** 2026-03-18T22:58:09.624544

**Input CSV:** data/processed/LLY_form4_features.csv

---

## Coverage
- Forward return coverage: 1410 rows (100.0%)
- Benchmark return coverage: 1410 rows (100.0%)

## Label Distribution
- Opportunistic: 885 (62.77%)
- Routine: 48 (3.4%)
- Uncertain: 477 (33.83%)

## Signal Counts
- Price signal rows: 280
- Earnings signal rows: 0
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 1130 |
| 1 | 280 |

## Recommendations
- Opportunistic share is high (>40%); review Cohen/plan rules for over-sensitive labeling.
- No earnings confirmations found; provide earnings CSV to improve multi-source labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.