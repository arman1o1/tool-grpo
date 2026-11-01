import re
from typing import Dict, Any, Optional
try:
    from .verifier import MathVerifier
except ImportError:
    from verifier import MathVerifier

class CompositeRewardEvaluator:
    """
    Computes shaped rewards for Tool-GRPO multi-turn trajectory optimization:
    1. Outcome correctness (+1.0 / 0.0) verified symbolically/numerically.
    2. Format compliance (+0.1 / -0.1) for valid XML tags and <answer>...</answer>.
    3. Tool execution bonus (+0.05 per successful code execution) & error penalty (-0.1 per exception).
    4. Length efficiency penalty (-0.02 * length / max_length).
    """
    def __init__(self, verifier: Optional[MathVerifier] = None):
        self.verifier = verifier or MathVerifier()

    def format_reward(self, text: str) -> float:
        """Penalizes unclosed XML tags or missing <answer> blocks."""
        score = 0.0
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE))
        if not has_answer:
            return -0.2

        score += 0.1
        # Check tag balancing for think, code, observation
        for tag in ["think", "code", "observation"]:
            opens = len(re.findall(f"<{tag}>", text, re.IGNORECASE))
            closes = len(re.findall(f"</{tag}>", text, re.IGNORECASE))
            if opens != closes:
                score -= 0.05
        return score

    def evaluate(
        self,
        completion: str,
        ground_truth: str,
        execution_failures: int = 0,
        successful_executions: int = 0,
        reflexion_repairs: int = 0,
        token_count: int = 0,
        max_tokens: int = 1024,
    ) -> Dict[str, float]:
        """
        Calculates individual reward components and the aggregated scalar reward.
        """
        is_correct = self.verifier.verify(completion, ground_truth)
        r_outcome = 1.0 if is_correct else 0.0
        
        r_format = self.format_reward(completion)
        
        # Tool usage shaping: reward successful executions, penalize syntax/runtime errors
        r_exec = (0.05 * min(successful_executions, 2)) - (0.1 * min(execution_failures, 3))

        # Reflexion repair bonus for recovering after an error
        r_reflexion = 0.15 * min(reflexion_repairs, 2)

        # Mild length penalty for excessive token usage
        r_length = -0.02 * (token_count / max_tokens) if max_tokens > 0 else 0.0

        total_reward = r_outcome + r_format + r_exec + r_reflexion + r_length

        return {
            "total_reward": round(total_reward, 4),
            "r_outcome": r_outcome,
            "r_format": round(r_format, 4),
            "r_exec": round(r_exec, 4),
            "r_reflexion": round(r_reflexion, 4),
            "r_length": round(r_length, 4),
            "is_correct": 1.0 if is_correct else 0.0,
        }

if __name__ == "__main__":
    evaluator = CompositeRewardEvaluator()
    sample = "<think>Compute 12 * 12</think><code>print(12*12)</code><observation>144</observation><answer>144</answer>"
    scores = evaluator.evaluate(
        sample,
        "#### 144",
        execution_failures=0,
        successful_executions=1,
        token_count=50
    )
    assert scores["is_correct"] == 1.0
    assert scores["total_reward"] > 0.9, f"Unexpected score: {scores}"
    print(f"[+] CompositeRewardEvaluator self-test passed: {scores}")
