# LLM Eval Report — `run_003`
Generated: 2026-06-10 03:09 UTC

## Summary

| Metric | Value |
|--------|-------|
| Total test cases | 10 |
| Passed | 4 (40.0%) |
| Avg composite score | 5.69 / 10 |
| Avg ROUGE-1 | 0.5963 |
| Avg ROUGE-L | 0.5217 |
| Avg LLM judge | 5.58 / 10 |
| Avg rubric | 5.89 / 10 |
| Pass threshold | 6.0 / 10 |

## Regression Check

✅ **Score OK — dropped 0.0% (within 5.0% threshold)**

- Previous score: 5.69
- Current score: 5.69
- Change: +0.0%

## Per-test Results

| ID | Category | Composite | ROUGE-1 | Judge | Rubric | Pass |
|----|----------|-----------|---------|-------|--------|------|
| sre_001 | sre | 8.00 | 0.819 | 7.8 | 8.4 | ✅ |
| sre_002 | sre | 7.49 | 0.792 | 7.9 | 6.7 | ✅ |
| sre_003 | sre | 4.85 | 0.568 | 5.4 | 3.8 | ❌ |
| rag_001 | rag | 4.80 | 0.452 | 4.2 | 5.9 | ❌ |
| rag_002 | rag | 6.40 | 0.704 | 6.5 | 6.2 | ✅ |
| general_001 | general | 5.72 | 0.585 | 5.0 | 7.1 | ❌ |
| general_002 | general | 6.16 | 0.584 | 5.6 | 7.2 | ✅ |
| sre_004 | sre | 3.83 | 0.452 | 4.2 | 3.2 | ❌ |
| general_003 | general | 4.85 | 0.475 | 4.2 | 6.0 | ❌ |
| sre_005 | sre | 4.83 | 0.533 | 5.0 | 4.5 | ❌ |

## Score Weights

| Component | Weight |
|-----------|--------|
| LLM Judge | 40% |
| Rubric | 35% |
| ROUGE | 25% |

---
_P09 · LLM Eval Framework · Staff SRE + AI Engineer Portfolio_