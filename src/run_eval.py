"""
P09 · Main Eval Runner
Runs full evaluation pipeline:
  load test cases → evaluate → save to DB → check regression → generate report

Usage:
    python src/run_eval.py                          # evaluate default dataset
    python src/run_eval.py --dataset data/raw/sre_test_cases.jsonl
    python src/run_eval.py --run-id my-run-001
    python src/run_eval.py --ci                     # CI gate mode (exit 1 on regression)
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone

from .db import check_regression, save_run
from .evaluator import LLMEvaluator, TestCase
from .reporter import generate_report, run_ci_gate


def load_test_cases(path: str) -> list[TestCase]:
    cases = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(TestCase(
                id=data.get("id", f"tc_{i:03d}"),
                question=data["question"],
                reference_answer=data["reference_answer"],
                generated_answer=data["generated_answer"],
                category=data.get("category", "general"),
                metadata=data.get("metadata", {}),
            ))
    return cases


def main():
    parser = argparse.ArgumentParser(description="LLM Eval Framework")
    parser.add_argument("--dataset", default="data/raw/test_cases.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--db-path", default="data/processed/eval_history.db")
    parser.add_argument("--report-path", default="data/processed/eval_report.md")
    parser.add_argument("--model-name", default="heuristic")
    parser.add_argument("--ci", action="store_true", help="CI gate mode")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--regression-threshold", type=float, default=5.0)
    args = parser.parse_args()

    if args.ci:
        sys.exit(run_ci_gate(args.db_path))

    run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"Run ID: {run_id}")
    print(f"Dataset: {args.dataset}")

    print("Loading test cases...")
    test_cases = load_test_cases(args.dataset)
    print(f"Loaded {len(test_cases)} test cases")

    evaluator = LLMEvaluator(pipe=None, pass_threshold=args.threshold)

    print("Evaluating...")
    results = evaluator.evaluate_batch(test_cases)

    summary = evaluator.summary(results)
    summary["model_name"] = args.model_name
    summary["dataset"] = args.dataset

    print("\nResults:")
    print(f"  Pass rate: {summary['pass_rate']}%")
    print(f"  Avg composite: {summary['avg_composite']:.2f}/10")
    print(f"  Avg ROUGE-1: {summary['avg_rouge1']:.4f}")
    print(f"  Avg judge: {summary['avg_llm_judge']:.2f}/10")

    results_dicts = [r.to_dict() for r in results]
    save_run(run_id, summary, results_dicts, args.db_path)
    print(f"Saved to DB: {args.db_path}")

    regression = check_regression(
        summary["avg_composite"],
        args.db_path,
        args.regression_threshold,
    )
    print(f"\nRegression: {regression['reason']}")

    generate_report(run_id, summary, results_dicts, regression, args.report_path)
    print(f"Report: {args.report_path}")

    if regression["has_regression"]:
        print("\n⚠️  Regression detected!")
        sys.exit(1)

    print("\n✅ Eval complete — no regression")


if __name__ == "__main__":
    main()
