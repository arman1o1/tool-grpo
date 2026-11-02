"""
Unit tests for Tool-GRPO Reasoning & Tool-Use Evaluator.
"""

import os
import tempfile
import pytest

from src.verifier import MathVerifier
from src.sandbox import PythonSandbox
from src.evaluator import (
    ReasoningEvaluator,
    calculate_pass_at_k,
    cluster_and_majority_vote,
    classify_failure,
)
from src.eval_reporter import generate_html_eval_dashboard


def test_calculate_pass_at_k_bounds():
    # When c = 0, pass@k is 0.0
    assert calculate_pass_at_k(n=4, c=0, k=1) == 0.0
    assert calculate_pass_at_k(n=4, c=0, k=2) == 0.0

    # When c = n, pass@k is 100.0
    assert calculate_pass_at_k(n=4, c=4, k=1) == 100.0
    assert calculate_pass_at_k(n=4, c=4, k=4) == 100.0

    # When n < k, returns 0.0
    assert calculate_pass_at_k(n=2, c=1, k=4) == 0.0

    # Known mathematical value: n=4, c=2, k=1 -> 50.0%
    assert calculate_pass_at_k(n=4, c=2, k=1) == 50.0

    # Known mathematical value: n=4, c=2, k=2 -> 1 - (2/4 * 1/3) = 1 - 1/6 = 5/6 = 83.33%
    assert abs(calculate_pass_at_k(n=4, c=2, k=2) - 83.33) < 0.1


def test_cluster_and_majority_vote_algebraic_equivalence():
    verifier = MathVerifier()

    # Three votes for 0.5 (one as fraction, one as decimal), one vote for 10
    preds = ["1/2", "0.5", "1/2", "10"]
    winner, is_corr, votes = cluster_and_majority_vote(preds, verifier, r"\boxed{\frac{1}{2}}")

    assert winner in ["1/2", "0.5"]
    assert is_corr is True
    assert votes == 3


def test_classify_failure_modes():
    verifier = MathVerifier()

    # 1. Missing answer tag
    t1 = {"completion": "<think>Just thinking without answering.</think>"}
    assert classify_failure(t1, "42", verifier) == "FORMAT_MISSING_ANSWER"

    # 2. Syntax error
    t2 = {
        "completion": "<think>Run broken code</think><code>def f(</code>\n<observation>\nSyntaxError: invalid syntax\n</observation>\n<answer>0</answer>",
        "exec_failures": 1,
        "succ_execs": 0,
        "repairs": 0
    }
    assert classify_failure(t2, "42", verifier) == "EXECUTION_SYNTAX_ERROR"

    # 3. Grounding hallucination: Observation was 42 (the true answer), but model said 100
    t3 = {
        "completion": "<think>Compute</think><code>print(42)</code>\n<observation>\n42\n</observation>\n<think>I will guess 100.</think><answer>100</answer>",
        "exec_failures": 0,
        "succ_execs": 1,
        "repairs": 0
    }
    assert classify_failure(t3, "42", verifier) == "GROUNDING_HALLUCINATION"

    # 4. Pure logic error
    t4 = {
        "completion": "<think>Compute</think><code>print(15)</code>\n<observation>\n15\n</observation>\n<think>Result is 15.</think><answer>15</answer>",
        "exec_failures": 0,
        "succ_execs": 1,
        "repairs": 0
    }
    assert classify_failure(t4, "42", verifier) == "LOGIC_INCORRECT"


def test_reasoning_evaluator_aggregation():
    evaluator = ReasoningEvaluator()

    # Problem 1: 3/4 correct rollouts
    t_p1 = [
        {"completion": "<answer>10</answer>", "tool_calls": 1, "succ_execs": 1, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>10</answer>", "tool_calls": 1, "succ_execs": 1, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>10</answer>", "tool_calls": 1, "succ_execs": 1, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>99</answer>", "tool_calls": 1, "succ_execs": 1, "exec_failures": 0, "repairs": 0},
    ]
    r1 = evaluator.evaluate_problem_samples("p-01", "q1", "10", t_p1, category="arithmetic")

    # Problem 2: 0/4 correct rollouts
    t_p2 = [
        {"completion": "<answer>0</answer>", "tool_calls": 0, "succ_execs": 0, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>1</answer>", "tool_calls": 0, "succ_execs": 0, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>2</answer>", "tool_calls": 0, "succ_execs": 0, "exec_failures": 0, "repairs": 0},
        {"completion": "<answer>3</answer>", "tool_calls": 0, "succ_execs": 0, "exec_failures": 0, "repairs": 0},
    ]
    r2 = evaluator.evaluate_problem_samples("p-02", "q2", "50", t_p2, category="algebra")

    summary = evaluator.aggregate_benchmark("TEST_BENCH", [r1, r2], k_values=[1, 4])

    assert summary.total_problems == 2
    assert summary.pass_at_1 == 50.0  # r1's first sample is correct, r2 is not
    assert summary.majority_at_k[4] == 50.0  # r1 majority is 10 (correct), r2 majority is wrong
    assert summary.tool_invocation_rate == 50.0
    assert "arithmetic" in summary.category_metrics
    assert "algebra" in summary.category_metrics


def test_html_eval_dashboard_generation():
    sample_report = [{
        "benchmark_name": "GSM8K_MOCK",
        "total_problems": 1,
        "pass_at_1": 100.0,
        "pass_at_k": {"1": 100.0, "4": 100.0},
        "majority_at_k": {"4": 100.0},
        "tool_invocation_rate": 100.0,
        "tool_success_rate": 100.0,
        "reflexion_recovery_rate": 0.0,
        "avg_tool_calls_per_problem": 1.0,
        "failure_taxonomy": {},
        "category_metrics": {"arithmetic": {"count": 1, "pass_at_1": 100.0, "majority_correct": 100.0}},
        "sample_results": []
    }]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_html = os.path.join(tmp_dir, "test_eval_report.html")
        generate_html_eval_dashboard(sample_report, out_html)

        assert os.path.exists(out_html)
        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()
        assert "RLVR EVAL SUITE" in html
        assert "GSM8K_MOCK" in html
