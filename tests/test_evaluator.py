import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluator import (
    LLMEvaluator,
    TestCase,
    compute_rouge,
    compute_rubric_score,
    llm_judge,
)
from src.db import init_db, save_run, get_recent_runs, check_regression


# ── ROUGE tests ───────────────────────────────────────────────────────────────
class TestROUGE:
    def test_identical_text_scores_one(self):
        text = "The quick brown fox jumps over the lazy dog"
        scores = compute_rouge(text, text)
        assert scores["rouge1"] == 1.0
        assert scores["rougeL"] == 1.0

    def test_empty_prediction_scores_zero(self):
        scores = compute_rouge("", "some reference text here")
        assert scores["rouge1"] == 0.0

    def test_partial_overlap(self):
        pred = "The cat sat on the mat"
        ref = "The cat sat on the floor"
        scores = compute_rouge(pred, ref)
        assert 0 < scores["rouge1"] < 1.0

    def test_returns_all_metrics(self):
        scores = compute_rouge("hello world", "hello there world")
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores


# ── Rubric tests ──────────────────────────────────────────────────────────────
class TestRubric:
    def test_sre_category_has_actionability(self):
        result = compute_rubric_score(
            "What to do when pod crashes?",
            "Run kubectl logs <pod> --previous to check logs",
            "Check logs with kubectl logs --previous",
            "sre",
        )
        assert "actionability" in result["breakdown"]
        assert result["total_score"] > 0

    def test_rag_category_has_faithfulness(self):
        result = compute_rubric_score(
            "What is RAG?",
            "RAG is Retrieval Augmented Generation",
            "RAG stands for Retrieval Augmented Generation",
            "rag",
        )
        assert "faithfulness" in result["breakdown"]

    def test_score_between_zero_and_ten(self):
        result = compute_rubric_score("Q", "Reference answer", "Generated answer")
        assert 0 <= result["total_score"] <= 10

    def test_unknown_category_falls_back_to_general(self):
        result = compute_rubric_score("Q", "R", "G", "unknown_category")
        assert result["category"] == "general"

    def test_better_answer_scores_higher(self):
        good = compute_rubric_score(
            "How to debug CrashLoopBackOff?",
            "Run kubectl logs <pod> --previous, check exit codes, check resource limits",
            "Run kubectl logs <pod> --previous to see crash. Check exit codes. Exit 137 = OOMKilled, increase limits.",
            "sre",
        )
        bad = compute_rubric_score(
            "How to debug CrashLoopBackOff?",
            "Run kubectl logs <pod> --previous, check exit codes, check resource limits",
            "I don't know",
            "sre",
        )
        assert good["total_score"] > bad["total_score"]


# ── LLM judge heuristic tests ─────────────────────────────────────────────────
class TestLLMJudge:
    def test_returns_score_and_reasoning(self):
        result = llm_judge("Q", "Generated answer", "Reference answer", pipe=None)
        assert "score" in result
        assert "reasoning" in result
        assert 0 <= result["score"] <= 10

    def test_heuristic_fallback_when_no_pipe(self):
        result = llm_judge("Q", "same text", "same text", pipe=None)
        assert result["score"] > 0

    def test_perfect_match_scores_higher(self):
        text = "Paris is the capital of France"
        perfect = llm_judge("Capital of France?", text, text, pipe=None)
        bad = llm_judge("Capital of France?", "I am not sure", text, pipe=None)
        assert perfect["score"] >= bad["score"]


# ── Evaluator tests ───────────────────────────────────────────────────────────
class TestEvaluator:
    def setup_method(self):
        self.evaluator = LLMEvaluator(pipe=None, pass_threshold=6.0)

    def test_evaluate_one_returns_result(self):
        tc = TestCase(
            id="test_001",
            question="What is SLO?",
            reference_answer="SLO is Service Level Objective — a reliability target",
            generated_answer="An SLO is a Service Level Objective, a target for reliability",
            category="general",
        )
        result = self.evaluator.evaluate_one(tc)
        assert result.test_case_id == "test_001"
        assert 0 <= result.composite_score <= 10
        assert result.latency_ms >= 0

    def test_evaluate_batch(self):
        cases = [
            TestCase(f"tc_{i}", f"Q{i}", f"Ref {i}", f"Gen {i}")
            for i in range(3)
        ]
        results = self.evaluator.evaluate_batch(cases)
        assert len(results) == 3

    def test_summary(self):
        cases = [
            TestCase(f"tc_{i}", f"Q{i}", f"Reference answer {i}", f"Generated answer {i}")
            for i in range(5)
        ]
        results = self.evaluator.evaluate_batch(cases)
        summary = self.evaluator.summary(results)
        assert summary["total"] == 5
        assert "pass_rate" in summary
        assert "avg_composite" in summary

    def test_good_answer_passes(self):
        tc = TestCase(
            id="pass_test",
            question="What is kubectl?",
            reference_answer="kubectl is the command-line tool for Kubernetes cluster management",
            generated_answer="kubectl is the CLI tool for managing Kubernetes clusters and resources",
            category="general",
        )
        result = self.evaluator.evaluate_one(tc)
        assert result.composite_score > 0


# ── DB tests ──────────────────────────────────────────────────────────────────
class TestDatabase:
    def test_init_creates_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            init_db(db_path)
            runs = get_recent_runs(5, db_path)
            assert isinstance(runs, list)
        finally:
            os.unlink(db_path)

    def test_save_and_retrieve_run(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            summary = {
                "total": 5, "passed": 4, "pass_rate": 80.0,
                "avg_composite": 7.2, "avg_rouge1": 0.6,
                "avg_rougeL": 0.5, "avg_llm_judge": 7.0,
                "avg_rubric": 6.8, "model_name": "test",
                "dataset": "test.jsonl",
            }
            save_run("run_001", summary, [], db_path)
            runs = get_recent_runs(5, db_path)
            assert len(runs) == 1
            assert runs[0]["run_id"] == "run_001"
            assert runs[0]["avg_composite"] == 7.2
        finally:
            os.unlink(db_path)

    def test_regression_detection(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            base_summary = {"total": 5, "passed": 4, "pass_rate": 80.0,
                           "avg_composite": 8.0, "avg_rouge1": 0.7,
                           "avg_rougeL": 0.6, "avg_llm_judge": 8.0,
                           "avg_rubric": 7.5, "model_name": "v1", "dataset": "test"}
            save_run("run_001", base_summary, [], db_path)

            bad_summary = {**base_summary, "avg_composite": 5.0, "model_name": "v2"}
            save_run("run_002", bad_summary, [], db_path)

            result = check_regression(5.0, db_path, threshold_drop_pct=5.0)
            assert result["has_regression"] is True
            assert result["drop_pct"] > 5.0
        finally:
            os.unlink(db_path)

    def test_no_regression_when_score_stable(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            summary = {"total": 5, "passed": 4, "pass_rate": 80.0,
                      "avg_composite": 7.5, "avg_rouge1": 0.6,
                      "avg_rougeL": 0.5, "avg_llm_judge": 7.5,
                      "avg_rubric": 7.0, "model_name": "v1", "dataset": "test"}
            save_run("run_001", summary, [], db_path)
            save_run("run_002", {**summary, "avg_composite": 7.4}, [], db_path)

            result = check_regression(7.4, db_path, threshold_drop_pct=5.0)
            assert result["has_regression"] is False
        finally:
            os.unlink(db_path)
