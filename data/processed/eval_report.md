# LLM Eval Report — `run_20260530_120000_abc123`
Generated: 2026-05-30 12:00 UTC

## Summary

| Metric | Value |
|--------|-------|
| Total test cases | 10 |
| Passed | 8 (80.0%) |
| Avg composite score | 6.84 / 10 |
| Avg ROUGE-1 | 0.5821 |
| Avg ROUGE-L | 0.4932 |
| Avg LLM judge | 7.20 / 10 |
| Avg rubric | 6.95 / 10 |
| Pass threshold | 6.0 / 10 |

## Regression Check

✅ **Score OK — dropped 0.0% (within 5.0% threshold)**

## Per-test Results

| ID | Category | Composite | ROUGE-1 | Judge | Rubric | Pass |
|----|----------|-----------|---------|-------|--------|------|
| sre_001 | sre | 7.42 | 0.681 | 8.0 | 7.8 | ✅ |
| sre_002 | sre | 7.15 | 0.624 | 7.5 | 7.2 | ✅ |
| sre_003 | sre | 6.21 | 0.512 | 6.5 | 6.4 | ✅ |
| rag_001 | rag | 5.84 | 0.423 | 6.0 | 6.1 | ❌ |
| rag_002 | rag | 7.63 | 0.698 | 8.0 | 7.5 | ✅ |
| general_001 | general | 7.21 | 0.645 | 7.5 | 7.3 | ✅ |
| general_002 | general | 7.48 | 0.712 | 7.8 | 7.2 | ✅ |
| sre_004 | sre | 6.92 | 0.598 | 7.2 | 7.0 | ✅ |
| general_003 | general | 7.15 | 0.658 | 7.5 | 7.0 | ✅ |
| sre_005 | sre | 5.43 | 0.487 | 5.5 | 5.8 | ❌ |

## Score Weights

| Component | Weight |
|-----------|--------|
| LLM Judge | 40% |
| Rubric | 35% |
| ROUGE | 25% |

---
_P09 · LLM Eval Framework · Staff SRE + AI Engineer Portfolio_
