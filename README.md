# P09 · LLM Eval Framework

Generic evaluation framework for LLM responses using three complementary methods.
Part of the [Staff SRE · AI Engineer Portfolio](https://github.com/amarshiv86).

## Where things live

| What | Where |
|------|-------|
| Eval framework code | This repo (`src/`) |
| Interactive demo | [HF Space](https://huggingface.co/spaces/amarshiv86/p09-llm-eval) |
| Test cases + reports | [HF Dataset](https://huggingface.co/datasets/amarshiv86/p09-llm-eval-dataset) |

## Eval methods

| Method | Weight | Description |
|--------|--------|-------------|
| LLM-as-judge | 40% | Local model scores 0-10 with reasoning |
| Custom rubric | 35% | Domain-specific dimensions (general/SRE/RAG) |
| ROUGE | 25% | Text overlap baseline |

## SRE additions
- **Score history:** SQLite DB tracks every run — regression visible over time
- **CI gate:** GitHub Actions blocks deploy if composite drops >5%
- **Custom rubric:** SRE category checks for kubectl commands, escalation paths, actionability
- **Regression detection:** Compares current run against last run automatically

## Run locally

```bash
git clone https://github.com/amarshiv86/p09-llm-eval
cd p09-llm-eval
pip install -r requirements.txt

# Run eval on test cases
python -m src.run_eval --dataset data/raw/test_cases.jsonl

# CI gate check
python -m src.reporter --ci
```

## Project structure

```
p09-llm-eval/
├── src/
│   ├── evaluator.py     # ROUGE + LLM judge + rubric scoring
│   ├── db.py            # SQLite score history + regression detection
│   ├── reporter.py      # Markdown report + CI gate
│   └── run_eval.py      # Main runner script
├── tests/
│   └── test_evaluator.py  # 20 unit tests
├── hf_space/            # → HF Space
│   ├── app.py           # Interactive Gradio eval runner
│   ├── README.md        # sdk_version: 5.29.0
│   └── requirements.txt
├── data/
│   ├── raw/test_cases.jsonl        # 10 test cases (general/SRE/RAG)
│   └── processed/eval_report.md   # Sample report
├── .github/workflows/
│   ├── ci.yml                  # Tests + lint + eval gate
│   ├── deploy-hf-space.yml
│   └── deploy-hf-dataset.yml
└── requirements.txt
```

## Stack

`ROUGE` · `SQLite` · `Gradio 5` · `GitHub Actions CI gate` · `Custom rubric scoring`
