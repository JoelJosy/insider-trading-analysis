# Phase 3 Label Quality Report

**Generated:** 2026-03-20T17:11:45.952160

**Input CSV:** data/processed/DOW_form4_features.csv

---

## Coverage
- Forward return coverage: 592 rows (89.16%)
- Benchmark return coverage: 664 rows (100.0%)

## Label Distribution
- Opportunistic: 28 (4.22%)
- Routine: 0 (0.0%)
- Uncertain: 636 (95.78%)

## Signal Counts
- Price signal rows: 90
- Earnings signal rows: 474
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 188 |
| 1 | 388 |
| 2 | 88 |

## Recommendations
- Price coverage is below 90%; re-run labeling later or verify market source access.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.