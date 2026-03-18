# Phase 3 Label Quality Report

**Generated:** 2026-03-18T22:57:53.219606

**Input CSV:** data/processed/JPM_form4_features.csv

---

## Coverage
- Forward return coverage: 920 rows (100.0%)
- Benchmark return coverage: 920 rows (100.0%)

## Label Distribution
- Opportunistic: 85 (9.24%)
- Routine: 35 (3.8%)
- Uncertain: 800 (86.96%)

## Signal Counts
- Price signal rows: 149
- Earnings signal rows: 0
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 771 |
| 1 | 149 |

## Recommendations
- No earnings confirmations found; provide earnings CSV to improve multi-source labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.