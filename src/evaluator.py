"""
P09 · LLM Eval Framework — Core Evaluator
Evaluates LLM responses using three methods:
1. ROUGE scores (text overlap, no model needed)
2. LLM-as-judge (local Qwen 0.5B, no API cost)
3. Custom SRE rubric (domain-specific scoring)
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rouge_score import rouge_scorer


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class TestCase:
    id: str
    question: str
    reference_answer: str
    generated_answer: str
    category: str = "general"  # general | sre | rag
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    test_case_id: str
    question: str
    category: str
    rouge1: float
    rouge2: float
    rougeL: float
    llm_judge_score: float        # 0-10
    llm_judge_reasoning: str
    rubric_score: float           # 0-10
    rubric_breakdown: dict
    composite_score: float        # weighted average
    passed: bool                  # True if above threshold
    latency_ms: int
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "category": self.category,
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "llm_judge_score": self.llm_judge_score,
            "llm_judge_reasoning": self.llm_judge_reasoning,
            "rubric_score": self.rubric_score,
            "rubric_breakdown": self.rubric_breakdown,
            "composite_score": self.composite_score,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


# ── ROUGE scorer ──────────────────────────────────────────────────────────────
def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ── Custom SRE rubric ─────────────────────────────────────────────────────────
RUBRIC_DIMENSIONS = {
    "general": {
        "accuracy": {"weight": 0.35, "description": "Factually correct"},
        "completeness": {"weight": 0.25, "description": "Covers all key points"},
        "clarity": {"weight": 0.20, "description": "Clear and well-structured"},
        "conciseness": {"weight": 0.20, "description": "Not unnecessarily verbose"},
    },
    "sre": {
        "accuracy": {"weight": 0.30, "description": "Technically correct for SRE"},
        "actionability": {"weight": 0.30, "description": "Provides specific commands/steps"},
        "completeness": {"weight": 0.20, "description": "Covers diagnosis and resolution"},
        "escalation": {"weight": 0.20, "description": "Mentions escalation/severity"},
    },
    "rag": {
        "faithfulness": {"weight": 0.35, "description": "Answer grounded in context"},
        "relevance": {"weight": 0.30, "description": "Directly answers the question"},
        "completeness": {"weight": 0.20, "description": "Uses available context well"},
        "hallucination": {"weight": 0.15, "description": "No fabricated information"},
    },
}


def compute_rubric_score(
    question: str,
    prediction: str,
    reference: str,
    category: str = "general",
) -> dict:
    """
    Heuristic rubric scoring — no model needed.
    Scores each dimension based on text analysis.
    """
    category = category if category in RUBRIC_DIMENSIONS else "general"
    dimensions = RUBRIC_DIMENSIONS[category]

    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    pred_words = set(pred_lower.split())
    ref_words = set(ref_lower.split())

    # Word overlap ratio
    overlap = len(pred_words & ref_words) / max(len(ref_words), 1)

    # Length ratio (penalize too short or too long)
    length_ratio = len(prediction) / max(len(reference), 1)
    length_score = 1.0 if 0.5 <= length_ratio <= 2.0 else max(0, 1 - abs(length_ratio - 1))

    # Has numbered steps (good for SRE)
    has_steps = bool(re.search(r"\d+\.", prediction))

    # Has specific commands (kubectl, grep, etc.)
    has_commands = bool(re.search(
        r"kubectl|grep|curl|systemctl|docker|helm|terraform|prometheus|psql",
        pred_lower
    ))

    breakdown = {}
    scores = {}

    for dim, info in dimensions.items():
        if dim == "accuracy":
            scores[dim] = min(1.0, overlap * 1.5)
        elif dim == "completeness":
            scores[dim] = overlap
        elif dim == "clarity":
            scores[dim] = length_score * (0.8 + 0.2 * has_steps)
        elif dim == "conciseness":
            scores[dim] = 1.0 if length_ratio <= 1.5 else max(0.3, 1.5 / length_ratio)
        elif dim == "actionability":
            scores[dim] = (0.5 * has_steps + 0.5 * has_commands)
        elif dim == "escalation":
            has_escalation = any(w in pred_lower for w in ["escalat", "page", "alert", "oncall", "on-call", "sever"])
            scores[dim] = 0.8 if has_escalation else 0.4
        elif dim == "faithfulness":
            scores[dim] = min(1.0, overlap * 1.3)
        elif dim == "relevance":
            q_words = set(question.lower().split())
            scores[dim] = len(pred_words & q_words) / max(len(q_words), 1)
        elif dim == "hallucination":
            # Lower score = more hallucination risk (words in prediction not in reference)
            novel_words = pred_words - ref_words - set(question.lower().split())
            hallucination_ratio = len(novel_words) / max(len(pred_words), 1)
            scores[dim] = max(0, 1 - hallucination_ratio * 0.5)
        else:
            scores[dim] = overlap

        scores[dim] = round(min(1.0, max(0.0, scores[dim])), 3)
        breakdown[dim] = {
            "score": scores[dim],
            "score_10": round(scores[dim] * 10, 1),
            "weight": info["weight"],
            "description": info["description"],
        }

    # Weighted average
    total = sum(scores[d] * dimensions[d]["weight"] for d in dimensions)
    return {
        "total_score": round(total * 10, 2),  # 0-10 scale
        "breakdown": breakdown,
        "category": category,
    }


# ── LLM-as-judge ─────────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Score this answer from 0-10.

Question: {question}

Reference answer: {reference}

Generated answer: {generated}

Score from 0-10 where:
0-3: Wrong or irrelevant
4-6: Partially correct, missing key points
7-8: Mostly correct with minor gaps
9-10: Accurate, complete, well-structured

Respond with ONLY: SCORE: <number>
REASON: <one sentence>"""


def llm_judge(
    question: str,
    prediction: str,
    reference: str,
    pipe=None,
) -> dict[str, Any]:
    """
    LLM-as-judge using local model.
    Falls back to heuristic if no model provided.
    """
    if pipe is None:
        # Heuristic fallback — used in CI (no model loaded)
        rouge = compute_rouge(prediction, reference)
        score = round((rouge["rouge1"] + rouge["rougeL"]) * 5, 1)
        return {
            "score": min(10.0, score),
            "reasoning": "Heuristic score based on ROUGE (no judge model loaded)",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference[:500],
        generated=prediction[:500],
    )

    formatted = (
        f"<|im_start|>system\nYou are an expert evaluator.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    try:
        output = pipe(formatted, return_full_text=False)[0]["generated_text"]
        output = output.split("<|im_end|>")[0].strip()

        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", output)
        reason_match = re.search(r"REASON:\s*(.+)", output)

        score = float(score_match.group(1)) if score_match else 5.0
        score = min(10.0, max(0.0, score))
        reasoning = reason_match.group(1).strip() if reason_match else output[:100]

        return {"score": round(score, 1), "reasoning": reasoning}

    except Exception as e:
        return {"score": 5.0, "reasoning": f"Judge error: {str(e)[:50]}"}


# ── Composite scorer ──────────────────────────────────────────────────────────
COMPOSITE_WEIGHTS = {
    "rouge": 0.25,
    "llm_judge": 0.40,
    "rubric": 0.35,
}

PASS_THRESHOLD = 6.0  # out of 10


def compute_composite(
    rouge_scores: dict,
    llm_judge_score: float,
    rubric_score: float,
) -> float:
    rouge_avg = (rouge_scores["rouge1"] + rouge_scores["rougeL"]) * 5  # 0-10
    composite = (
        rouge_avg * COMPOSITE_WEIGHTS["rouge"]
        + llm_judge_score * COMPOSITE_WEIGHTS["llm_judge"]
        + rubric_score * COMPOSITE_WEIGHTS["rubric"]
    )
    return round(composite, 2)


# ── Main evaluator ────────────────────────────────────────────────────────────
class LLMEvaluator:
    def __init__(self, pipe=None, pass_threshold: float = PASS_THRESHOLD):
        self.pipe = pipe
        self.pass_threshold = pass_threshold

    def evaluate_one(self, test_case: TestCase) -> EvalResult:
        start = time.time()

        rouge = compute_rouge(test_case.generated_answer, test_case.reference_answer)
        judge = llm_judge(
            test_case.question,
            test_case.generated_answer,
            test_case.reference_answer,
            self.pipe,
        )
        rubric = compute_rubric_score(
            test_case.question,
            test_case.generated_answer,
            test_case.reference_answer,
            test_case.category,
        )
        composite = compute_composite(rouge, judge["score"], rubric["total_score"])

        return EvalResult(
            test_case_id=test_case.id,
            question=test_case.question,
            category=test_case.category,
            rouge1=rouge["rouge1"],
            rouge2=rouge["rouge2"],
            rougeL=rouge["rougeL"],
            llm_judge_score=judge["score"],
            llm_judge_reasoning=judge["reasoning"],
            rubric_score=rubric["total_score"],
            rubric_breakdown=rubric["breakdown"],
            composite_score=composite,
            passed=composite >= self.pass_threshold,
            latency_ms=int((time.time() - start) * 1000),
        )

    def evaluate_batch(self, test_cases: list[TestCase]) -> list[EvalResult]:
        return [self.evaluate_one(tc) for tc in test_cases]

    def summary(self, results: list[EvalResult]) -> dict:
        if not results:
            return {}
        n = len(results)
        return {
            "total": n,
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": round(sum(1 for r in results if r.passed) / n * 100, 1),
            "avg_composite": round(sum(r.composite_score for r in results) / n, 2),
            "avg_rouge1": round(sum(r.rouge1 for r in results) / n, 4),
            "avg_rougeL": round(sum(r.rougeL for r in results) / n, 4),
            "avg_llm_judge": round(sum(r.llm_judge_score for r in results) / n, 2),
            "avg_rubric": round(sum(r.rubric_score for r in results) / n, 2),
            "threshold": self.pass_threshold,
        }
