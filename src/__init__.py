"""
Tool-GRPO: Interleaved Code Execution via Multi-Turn Group Relative Policy Optimization.
"""

from .sandbox import PythonSandbox
from .verifier import MathVerifier, extract_boxed, latex_to_sympy_str
from .rewards import CompositeRewardEvaluator
from .rollout import MultiTurnRolloutEngine
from .evaluator import (
    ReasoningEvaluator,
    BenchmarkSummary,
    EvaluationSampleResult,
    calculate_pass_at_k,
    cluster_and_majority_vote,
    classify_failure,
)
from .eval_reporter import generate_html_eval_dashboard

__all__ = [
    "PythonSandbox",
    "MathVerifier",
    "extract_boxed",
    "latex_to_sympy_str",
    "CompositeRewardEvaluator",
    "MultiTurnRolloutEngine",
    "ReasoningEvaluator",
    "BenchmarkSummary",
    "EvaluationSampleResult",
    "calculate_pass_at_k",
    "cluster_and_majority_vote",
    "classify_failure",
    "generate_html_eval_dashboard",
]
