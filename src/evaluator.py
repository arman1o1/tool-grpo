"""
Tool-GRPO: Rigorous Reasoning & Tool-Use Evaluation Framework
Provides multi-metric evaluation (Pass@1, Pass@k, Cons@k/Maj@k), failure taxonomy diagnosis,
and multi-benchmark aggregation (GSM8K, MATH-500).
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable

from .sandbox import PythonSandbox
from .verifier import MathVerifier


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased Pass@k estimator (Chen et al., 2021 - HumanEval).
    n: total samples generated per problem
    c: number of correct samples
    k: k-value (k <= n)
    """
    if n < k or n <= 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 100.0

    # 1 - prod_{j=1}^k (n - c - j + 1) / (n - j + 1)
    prod = 1.0
    for j in range(1, k + 1):
        prod *= (n - c - j + 1) / (n - j + 1)
    return round(max(0.0, min(1.0, 1.0 - prod)) * 100.0, 2)


def cluster_and_majority_vote(
    predictions: List[str],
    verifier: MathVerifier,
    ground_truth: str
) -> Tuple[Optional[str], bool, int]:
    """
    Groups predictions into equivalence clusters using MathVerifier,
    determines the majority candidate, and checks if it matches ground truth.
    Returns: (majority_prediction, is_correct, vote_count)
    """
    if not predictions:
        return None, False, 0

    valid_preds = [p for p in predictions if p is not None and str(p).strip()]
    if not valid_preds:
        return None, False, 0

    # Cluster predictions by verifier equivalence
    clusters: List[Dict[str, Any]] = []  # {"rep": str, "count": int, "members": List[str]}

    for pred in valid_preds:
        matched = False
        for c in clusters:
            if verifier.verify(f"<answer>{pred}</answer>", f"<answer>{c['rep']}</answer>"):
                c["count"] += 1
                c["members"].append(pred)
                matched = True
                break
        if not matched:
            clusters.append({"rep": pred, "count": 1, "members": [pred]})

    # Sort clusters by vote count descending
    clusters.sort(key=lambda x: x["count"], reverse=True)
    winner = clusters[0]["rep"]
    votes = clusters[0]["count"]

    is_correct = verifier.verify(f"<answer>{winner}</answer>", ground_truth)
    return winner, is_correct, votes


def classify_failure(
    trajectory: Dict[str, Any],
    ground_truth: str,
    verifier: MathVerifier
) -> str:
    """
    Diagnoses root-cause failure mode of an unsuccessful trajectory:
    - FORMAT_MISSING_ANSWER
    - EXECUTION_SYNTAX_ERROR
    - EXECUTION_TIMEOUT
    - EXECUTION_RUNTIME_ERROR
    - GROUNDING_HALLUCINATION
    - LOGIC_INCORRECT
    - VERIFIER_PARSE_FAIL
    """
    text = trajectory.get("completion", "") or trajectory.get("text", "")

    # 1. Format checks
    if "</answer>" not in text.lower() and r"\boxed{" not in text:
        return "FORMAT_MISSING_ANSWER"

    pred = verifier.extract_predicted_answer(text)
    if pred is None:
        return "VERIFIER_PARSE_FAIL"

    # 2. Execution error checks
    exec_failures = trajectory.get("exec_failures", 0)
    succ_execs = trajectory.get("succ_execs", 0)
    repairs = trajectory.get("repairs", 0)

    # Check raw observations or error traces if available
    obs_matches = re.findall(r"<observation>(.*?)</observation>", text, re.DOTALL | re.IGNORECASE)
    has_syntax_err = any("syntaxerror" in o.lower() for o in obs_matches)
    has_timeout = any("timeouterror" in o.lower() or "timeout" in o.lower() for o in obs_matches)
    has_runtime_err = any("runtimeerror" in o.lower() or "error:" in o.lower() for o in obs_matches)

    # If the last code execution failed and was never repaired
    if exec_failures > 0 and repairs == 0 and succ_execs == 0:
        if has_syntax_err:
            return "EXECUTION_SYNTAX_ERROR"
        if has_timeout:
            return "EXECUTION_TIMEOUT"
        return "EXECUTION_RUNTIME_ERROR"

    # 3. Grounding / Hallucination check
    # If code produced clean output, but model reported a completely unrelated answer
    if obs_matches:
        last_obs = obs_matches[-1].strip()
        obs_num = verifier.extract_predicted_answer(f"<answer>{last_obs}</answer>")
        # If observation had a valid number, but answer disagreed with observation
        if obs_num is not None and pred != obs_num:
            # Check if observation was actually the ground truth
            if verifier.verify(f"<answer>{obs_num}</answer>", ground_truth):
                return "GROUNDING_HALLUCINATION"

    # 4. Pure mathematical / algorithmic logic error
    return "LOGIC_INCORRECT"


@dataclass
class EvaluationSampleResult:
    problem_id: str
    question: str
    ground_truth: str
    category: str = "general"
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Optional[str]] = field(default_factory=list)
    correct_mask: List[bool] = field(default_factory=list)
    majority_prediction: Optional[str] = None
    majority_correct: bool = False
    majority_votes: int = 0
    failure_mode: Optional[str] = None
    tool_calls: int = 0
    exec_failures: int = 0
    succ_execs: int = 0
    repairs: int = 0
    avg_tokens: float = 0.0


@dataclass
class BenchmarkSummary:
    benchmark_name: str
    total_problems: int
    pass_at_1: float
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    majority_at_k: Dict[int, float] = field(default_factory=dict)
    tool_invocation_rate: float = 0.0
    tool_success_rate: float = 0.0
    reflexion_recovery_rate: float = 0.0
    avg_tool_calls_per_problem: float = 0.0
    avg_tokens_per_trajectory: float = 0.0
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)


class ReasoningEvaluator:
    """
    Standardized, rigorous evaluation harness for reasoning models with tool execution.
    """
    def __init__(
        self,
        sandbox: Optional[PythonSandbox] = None,
        verifier: Optional[MathVerifier] = None,
    ):
        self.sandbox = sandbox or PythonSandbox()
        self.verifier = verifier or MathVerifier()

    def evaluate_problem_samples(
        self,
        problem_id: str,
        question: str,
        ground_truth: str,
        trajectories: List[Dict[str, Any]],
        category: str = "general"
    ) -> EvaluationSampleResult:
        """
        Evaluates a set of sampled rollouts for a single problem.
        """
        preds = []
        correct_mask = []
        tot_tools = 0
        tot_fail = 0
        tot_succ = 0
        tot_repairs = 0
        tot_tokens = 0

        for traj in trajectories:
            completion = traj.get("completion", "") or traj.get("text", "")
            pred = self.verifier.extract_predicted_answer(completion)
            is_corr = self.verifier.verify(completion, ground_truth)
            preds.append(pred)
            correct_mask.append(is_corr)

            tot_tools += traj.get("tool_calls", 0)
            tot_fail += traj.get("exec_failures", 0)
            tot_succ += traj.get("succ_execs", 0)
            tot_repairs += traj.get("repairs", 0)
            tot_tokens += len(traj.get("tokens", [])) if "tokens" in traj else len(completion.split())

        num_samples = len(trajectories)
        avg_tokens = tot_tokens / num_samples if num_samples > 0 else 0.0

        maj_pred, maj_corr, votes = cluster_and_majority_vote(preds, self.verifier, ground_truth)

        # Failure diagnosis (if all samples or majority failed)
        failure_mode = None
        if not maj_corr and trajectories:
            failure_mode = classify_failure(trajectories[0], ground_truth, self.verifier)

        return EvaluationSampleResult(
            problem_id=problem_id,
            question=question,
            ground_truth=ground_truth,
            category=category,
            trajectories=trajectories,
            predictions=preds,
            correct_mask=correct_mask,
            majority_prediction=maj_pred,
            majority_correct=maj_corr,
            majority_votes=votes,
            failure_mode=failure_mode,
            tool_calls=tot_tools,
            exec_failures=tot_fail,
            succ_execs=tot_succ,
            repairs=tot_repairs,
            avg_tokens=round(avg_tokens, 1)
        )

    def aggregate_benchmark(
        self,
        benchmark_name: str,
        results: List[EvaluationSampleResult],
        k_values: List[int] = [1, 4, 8]
    ) -> BenchmarkSummary:
        """
        Computes aggregate metrics, pass@k, majority@k, and failure taxonomy.
        """
        n_problems = len(results)
        if n_problems == 0:
            return BenchmarkSummary(
                benchmark_name=benchmark_name,
                total_problems=0,
                pass_at_1=0.0
            )

        # Pass@1 (Greedy / First sample)
        pass_1_correct = sum(1 for r in results if r.correct_mask and r.correct_mask[0])
        pass_1 = round((pass_1_correct / n_problems) * 100.0, 2)

        # Pass@k & Maj@k
        pass_at_k = {}
        majority_at_k = {}

        # Determine max samples per problem across results
        samples_per_prob = len(results[0].correct_mask) if results[0].correct_mask else 1

        for k in k_values:
            if k <= samples_per_prob:
                # Pass@k estimator
                scores_k = []
                for r in results:
                    c = sum(1 for x in r.correct_mask[:samples_per_prob] if x)
                    n = samples_per_prob
                    scores_k.append(calculate_pass_at_k(n, c, k))
                pass_at_k[k] = round(sum(scores_k) / len(scores_k), 2)

                # Maj@k
                maj_k_correct = 0
                for r in results:
                    sub_preds = r.predictions[:k]
                    _, corr, _ = cluster_and_majority_vote(sub_preds, self.verifier, r.ground_truth)
                    if corr:
                        maj_k_correct += 1
                majority_at_k[k] = round((maj_k_correct / n_problems) * 100.0, 2)

        # Tool metrics
        probs_with_tools = sum(1 for r in results if r.tool_calls > 0)
        tool_invocation_rate = round((probs_with_tools / n_problems) * 100.0, 2)

        tot_tool_calls = sum(r.tool_calls for r in results)
        tot_succ = sum(r.succ_execs for r in results)
        tot_fail = sum(r.exec_failures for r in results)
        tot_execs = tot_succ + tot_fail
        tool_success_rate = round((tot_succ / tot_execs * 100.0), 2) if tot_execs > 0 else 0.0

        probs_with_failures = [r for r in results if r.exec_failures > 0]
        repaired_probs = sum(1 for r in probs_with_failures if r.repairs > 0 and r.majority_correct)
        reflexion_recovery_rate = round((repaired_probs / len(probs_with_failures) * 100.0), 2) if probs_with_failures else 0.0

        avg_tool_calls = round(tot_tool_calls / (n_problems * samples_per_prob), 2)
        avg_tokens = round(sum(r.avg_tokens for r in results) / n_problems, 1)

        # Failure Taxonomy
        taxonomy = Counter(r.failure_mode for r in results if r.failure_mode)

        # Category Breakdown
        cat_map = defaultdict(list)
        for r in results:
            cat_map[r.category].append(r)

        category_metrics = {}
        for cat, items in cat_map.items():
            cat_p1 = sum(1 for x in items if x.correct_mask and x.correct_mask[0]) / len(items) * 100.0
            cat_maj = sum(1 for x in items if x.majority_correct) / len(items) * 100.0
            category_metrics[cat] = {
                "count": len(items),
                "pass_at_1": round(cat_p1, 2),
                "majority_correct": round(cat_maj, 2),
            }

        return BenchmarkSummary(
            benchmark_name=benchmark_name,
            total_problems=n_problems,
            pass_at_1=pass_1,
            pass_at_k=pass_at_k,
            majority_at_k=majority_at_k,
            tool_invocation_rate=tool_invocation_rate,
            tool_success_rate=tool_success_rate,
            reflexion_recovery_rate=reflexion_recovery_rate,
            avg_tool_calls_per_problem=avg_tool_calls,
            avg_tokens_per_trajectory=avg_tokens,
            failure_taxonomy=dict(taxonomy),
            category_metrics=category_metrics,
            sample_results=[asdict(r) for r in results]
        )
