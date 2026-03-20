# Phase 3 Label Quality Report

**Generated:** 2026-03-20T13:35:49.596553

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
- Earnings signal rows: 1151
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 193 |
| 1 | 1003 |
| 2 | 214 |

## Recommendations
- Opportunistic share is high (>40%); review Cohen/plan rules for over-sensitive labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.