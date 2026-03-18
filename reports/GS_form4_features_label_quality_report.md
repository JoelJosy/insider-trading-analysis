# Phase 3 Label Quality Report

**Generated:** 2026-03-18T23:04:06.139052

**Input CSV:** data/processed/GS_form4_features.csv

---

## Coverage
- Forward return coverage: 555 rows (78.06%)
- Benchmark return coverage: 555 rows (78.06%)

## Label Distribution
- Opportunistic: 237 (33.33%)
- Routine: 36 (5.06%)
- Uncertain: 438 (61.6%)

## Signal Counts
- Price signal rows: 119
- Earnings signal rows: 0
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 592 |
| 1 | 119 |

## Recommendations
- Price coverage is below 90%; re-run labeling later or verify market source access.
- No earnings confirmations found; provide earnings CSV to improve multi-source labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.