"""
P09 · Eval Reporter
Generates markdown report and CI gate decision.
"""

import os
import sys
from datetime import datetime, timezone

from .db import check_regression, get_recent_runs


def generate_report(
    run_id: str,
    summary: dict,
    results: list[dict],
    regression_info: dict,
    output_path: str = "data/processed/eval_report.md",
) -> str:
    """Generate a markdown eval report."""

    passed_emoji = "✅" if regression_info.get("has_regression") is False else "⚠️"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# LLM Eval Report — `{run_id}`",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total test cases | {summary.get('total', 0)} |",
        f"| Passed | {summary.get('passed', 0)} ({summary.get('pass_rate', 0)}%) |",
        f"| Avg composite score | {summary.get('avg_composite', 0):.2f} / 10 |",
        f"| Avg ROUGE-1 | {summary.get('avg_rouge1', 0):.4f} |",
        f"| Avg ROUGE-L | {summary.get('avg_rougeL', 0):.4f} |",
        f"| Avg LLM judge | {summary.get('avg_llm_judge', 0):.2f} / 10 |",
        f"| Avg rubric | {summary.get('avg_rubric', 0):.2f} / 10 |",
        f"| Pass threshold | {summary.get('threshold', 6.0)} / 10 |",
        "",
        "## Regression Check",
        "",
        f"{passed_emoji} **{regression_info.get('reason', 'N/A')}**",
        "",
    ]

    if regression_info.get("previous") is not None:
        lines += [
            f"- Previous score: {regression_info['previous']}",
            f"- Current score: {regression_info['current']}",
            f"- Change: {regression_info.get('drop_pct', 0):+.1f}%",
            "",
        ]

    lines += [
        "## Per-test Results",
        "",
        "| ID | Category | Composite | ROUGE-1 | Judge | Rubric | Pass |",
        "|----|----------|-----------|---------|-------|--------|------|",
    ]

    for r in results:
        pass_icon = "✅" if r.get("passed") else "❌"
        lines.append(
            f"| {r.get('test_case_id', '?')} "
            f"| {r.get('category', '?')} "
            f"| {r.get('composite_score', 0):.2f} "
            f"| {r.get('rouge1', 0):.3f} "
            f"| {r.get('llm_judge_score', 0):.1f} "
            f"| {r.get('rubric_score', 0):.1f} "
            f"| {pass_icon} |"
        )

    lines += [
        "",
        "## Score Weights",
        "",
        "| Component | Weight |",
        "|-----------|--------|",
        "| LLM Judge | 40% |",
        "| Rubric | 35% |",
        "| ROUGE | 25% |",
        "",
        "---",
        "_P09 · LLM Eval Framework · Staff SRE + AI Engineer Portfolio_",
    ]

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def run_ci_gate(db_path: str = "data/processed/eval_history.db") -> int:
    """
    CI gate — reads latest run from DB and checks for regression.
    Returns exit code: 0 = pass, 1 = regression detected.
    Called by GitHub Actions eval workflow.
    """
    from .db import get_recent_runs

    runs = get_recent_runs(1, db_path)
    if not runs:
        print("No eval runs found — skipping gate")
        return 0

    latest = runs[0]
    regression = check_regression(latest["avg_composite"], db_path)

    print(f"\n── CI Gate ────────────────────────────")
    print(f"Run ID:   {latest['run_id']}")
    print(f"Score:    {latest['avg_composite']:.2f} / 10")
    print(f"Pass rate: {latest['pass_rate']}%")
    print(f"Status:   {regression['reason']}")
    print(f"───────────────────────────────────────\n")

    if regression["has_regression"]:
        print("❌ CI GATE FAILED — score regression detected")
        print("   Deployment blocked. Fix the model or update test cases.")
        return 1

    print("✅ CI GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run_ci_gate())
