"""
P09 · LLM Eval Framework — HuggingFace Space Demo
Interactive eval runner — paste Q&A pairs, get scores back.
Local inference, no external API calls.
gradio==5.29.0 + audioop-lts for Python 3.13 compatibility.
"""

import json
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from src.evaluator import LLMEvaluator, TestCase, RUBRIC_DIMENSIONS

# ── Evaluator (no model — heuristic judge for Space demo) ─────────────────────
evaluator = LLMEvaluator(pipe=None, pass_threshold=6.0)

CATEGORY_OPTIONS = ["general", "sre", "rag"]

SAMPLE_CASES = [
    {
        "label": "SRE — CrashLoopBackOff",
        "category": "sre",
        "question": "What steps should I take when a pod is in CrashLoopBackOff?",
        "reference": "1. Check pod logs: kubectl logs <pod> --previous. 2. Describe the pod: kubectl describe pod <pod>. 3. Exit code 137 = OOMKilled — increase memory limits. 4. Check liveness probe configuration.",
        "generated": "Check pod logs with kubectl logs --previous. Describe the pod to see exit codes. If exit code 137 the pod is OOM killed — increase memory. Check liveness probes.",
    },
    {
        "label": "RAG — Capital of France",
        "category": "rag",
        "question": "What is the capital of France?",
        "reference": "Paris is the capital of France. It is the largest city and serves as the political, economic, and cultural center.",
        "generated": "Paris.",
    },
    {
        "label": "General — What is Docker?",
        "category": "general",
        "question": "What is Docker?",
        "reference": "Docker is a platform for developing and running applications in containers. Containers package an application and its dependencies for consistent execution across environments.",
        "generated": "Docker is a containerization tool that packages applications into lightweight portable containers.",
    },
    {
        "label": "SRE — Error Budget (good answer)",
        "category": "sre",
        "question": "How do I calculate error budget for a 99.9% SLO?",
        "reference": "Error budget = 1 - 0.999 = 0.001 = 0.1%. For 30 days: 0.001 * 30 * 24 * 60 = 43.2 minutes allowed downtime. Track burn rate to know if you are consuming it too fast.",
        "generated": "Error budget = 1 - SLO = 0.1%. Over 30 days that is 43.2 minutes of allowed downtime. Monitor burn rate — if > 1 you are consuming budget faster than it replenishes.",
    },
]


def evaluate(question: str, reference: str, generated: str, category: str) -> tuple:
    if not question.strip() or not reference.strip() or not generated.strip():
        return "⚠️ Fill in all three fields.", "", "", ""

    tc = TestCase(
        id="demo_001",
        question=question,
        reference_answer=reference,
        generated_answer=generated,
        category=category,
    )

    result = evaluator.evaluate_one(tc)

    # ── Summary card ──────────────────────────────────────────────────────────
    pass_icon = "✅ PASSED" if result.passed else "❌ FAILED"
    summary_md = f"""## {pass_icon} — Composite: **{result.composite_score:.1f} / 10**

| Metric | Score |
|--------|-------|
| 🏆 Composite | **{result.composite_score:.2f} / 10** |
| 🤖 LLM Judge | {result.llm_judge_score:.1f} / 10 |
| 📋 Rubric | {result.rubric_score:.1f} / 10 |
| 📊 ROUGE-1 | {result.rouge1:.4f} |
| 📊 ROUGE-L | {result.rougeL:.4f} |
| ⏱️ Latency | {result.latency_ms}ms |

**Judge reasoning:** {result.llm_judge_reasoning}
"""

    # ── Rubric breakdown ──────────────────────────────────────────────────────
    rubric_lines = [f"## Rubric Breakdown — `{category}` category\n"]
    for dim, info in result.rubric_breakdown.items():
        bar_filled = int(info["score"] * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        rubric_lines.append(
            f"**{dim.title()}** ({info['description']})\n"
            f"`{bar}` {info['score_10']:.1f}/10 (weight: {int(info['weight']*100)}%)\n"
        )
    rubric_md = "\n".join(rubric_lines)

    # ── ROUGE detail ──────────────────────────────────────────────────────────
    rouge_md = f"""## ROUGE Scores

| Metric | Score | What it means |
|--------|-------|---------------|
| ROUGE-1 | {result.rouge1:.4f} | Unigram overlap |
| ROUGE-2 | {result.rouge2:.4f} | Bigram overlap |
| ROUGE-L | {result.rougeL:.4f} | Longest common subsequence |

ROUGE measures text overlap between generated and reference answer.
Higher = more similar wording. Not a perfect quality signal — used as one of three components.
"""

    raw_json = json.dumps(result.to_dict(), indent=2)
    return summary_md, rubric_md, rouge_md, raw_json


def load_sample(sample_label: str) -> tuple:
    for s in SAMPLE_CASES:
        if s["label"] == sample_label:
            return s["question"], s["reference"], s["generated"], s["category"]
    return "", "", "", "general"


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="P09 · LLM Eval Framework", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📊 P09 · LLM Eval Framework
    **Staff SRE + AI Engineer Portfolio**

    Evaluate LLM responses using three methods:
    - 🤖 **LLM-as-judge** — scores quality 0-10
    - 📋 **Custom rubric** — domain-specific scoring (general / SRE / RAG)
    - 📊 **ROUGE** — text overlap baseline

    Scores are stored in SQLite for regression tracking. A CI gate blocks deploys if score drops >5%.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("**Load a sample:**")
            sample_dd = gr.Dropdown(
                choices=[s["label"] for s in SAMPLE_CASES],
                label="Sample cases",
                value=None,
            )

            category = gr.Dropdown(
                choices=CATEGORY_OPTIONS,
                value="general",
                label="Category (affects rubric dimensions)",
            )
            question = gr.Textbox(label="Question", lines=2)
            reference = gr.Textbox(label="Reference Answer", lines=4)
            generated = gr.Textbox(label="Generated Answer (to evaluate)", lines=4)

            eval_btn = gr.Button("📊 Evaluate", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=3):
            summary_out = gr.Markdown()
            with gr.Accordion("📋 Rubric Breakdown", open=True):
                rubric_out = gr.Markdown()
            with gr.Accordion("📊 ROUGE Detail", open=False):
                rouge_out = gr.Markdown()
            with gr.Accordion("🔧 Raw JSON", open=False):
                json_out = gr.Code(language="json")

    with gr.Accordion("📖 How scoring works", open=False):
        gr.Markdown("""
        ## Composite Score = LLM Judge (40%) + Rubric (35%) + ROUGE (25%)

        **LLM Judge:** A language model scores the answer 0-10 based on accuracy,
        completeness and quality vs the reference. Uses heuristic fallback in this demo.

        **Custom Rubric:** Dimension-based scoring tailored to the category:
        - **General:** accuracy, completeness, clarity, conciseness
        - **SRE:** accuracy, actionability (has kubectl commands?), completeness, escalation path
        - **RAG:** faithfulness, relevance, completeness, hallucination risk

        **ROUGE:** Text overlap between generated and reference. Useful baseline
        but not sufficient alone — a paraphrase scores low on ROUGE but may be better.

        **Pass threshold:** 6.0/10. Runs stored in SQLite.
        CI gate in GitHub Actions blocks deploy if composite drops >5%.

        **SRE angle:** This framework is used to eval P08 agent answers and P07 fine-tuned model.
        P10 Observability Dashboard pulls scores from this DB.
        """)

    gr.Markdown("""
    ---
    [GitHub Repo](https://github.com/amarshiv86/p09-llm-eval) ·
    [Staff SRE Portfolio](https://github.com/amarshiv86)
    """)

    sample_dd.change(
        fn=load_sample,
        inputs=[sample_dd],
        outputs=[question, reference, generated, category],
    )
    eval_btn.click(
        fn=evaluate,
        inputs=[question, reference, generated, category],
        outputs=[summary_out, rubric_out, rouge_out, json_out],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", "general", "", "", "", ""),
        outputs=[question, reference, generated, category,
                 summary_out, rubric_out, rouge_out, json_out],
    )

demo.launch()
