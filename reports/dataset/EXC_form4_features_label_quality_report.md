# Phase 3 Label Quality Report

**Generated:** 2026-03-20T17:23:54.693151

**Input CSV:** data/processed/EXC_form4_features.csv

---

## Coverage
- Forward return coverage: 1280 rows (100.0%)
- Benchmark return coverage: 1280 rows (100.0%)

## Label Distribution
- Opportunistic: 6 (0.47%)
- Routine: 26 (2.03%)
- Uncertain: 1248 (97.5%)

## Signal Counts
- Price signal rows: 238
- Earnings signal rows: 768
- Enforcement signal rows: 0

### Source Count Distribution
| Source Count | Rows |
|---|---:|
| 0 | 393 |
| 1 | 768 |
| 2 | 119 |

## Recommendations
- Opportunistic share is very low (<1%); review Cohen/plan rules for over-conservative labeling.
- No enforcement confirmations found; provide SEC enforcement CSV for higher-confidence labels.