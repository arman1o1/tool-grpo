import pytest
from src.rewards import CompositeRewardEvaluator
from src.verifier import MathVerifier

def test_reward_perfect_trajectory():
    evaluator = CompositeRewardEvaluator()
    sample = "<think>Step 1</think><code>print(40 + 2)</code><observation>42</observation><answer>42</answer>"
    scores = evaluator.evaluate(
        completion=sample,
        ground_truth="#### 42",
        execution_failures=0,
        successful_executions=1,
        token_count=30,
        max_tokens=1000,
    )
    assert scores["is_correct"] == 1.0
    assert scores["r_outcome"] == 1.0
    assert scores["r_format"] > 0.0
    assert scores["r_exec"] > 0.0
    assert scores["total_reward"] > 1.0

def test_reward_reflexion_repair_bonus():
    evaluator = CompositeRewardEvaluator()
    sample = "<think>Turn 1</think><code>print(1/0)</code><observation>ZeroDivisionError</observation><think>Repair</think><code>print(42)</code><observation>42</observation><answer>42</answer>"
    scores = evaluator.evaluate(
        completion=sample,
        ground_truth="#### 42",
        execution_failures=1,
        successful_executions=1,
        reflexion_repairs=1,
        token_count=80,
    )
    assert scores["is_correct"] == 1.0
    assert scores["r_reflexion"] == 0.15
    assert scores["total_reward"] > 1.0

def test_reward_missing_answer_tag():
    evaluator = CompositeRewardEvaluator()
    sample = "<think>Thinking about 42...</think> The answer is 42."
    scores = evaluator.evaluate(
        completion=sample,
        ground_truth="#### 42",
        execution_failures=0,
        successful_executions=0,
    )
    assert scores["r_format"] < 0.0

def test_reward_unhandled_execution_penalties():
    evaluator = CompositeRewardEvaluator()
    sample = "<think>Compute</think><code>print(1/0)</code><answer>0</answer>"
    scores = evaluator.evaluate(
        completion=sample,
        ground_truth="#### 100",
        execution_failures=2,
        successful_executions=0,
    )
    assert scores["is_correct"] == 0.0
    assert scores["r_exec"] < 0.0
    assert scores["total_reward"] < 0.0

def test_reward_length_scaling():
    evaluator = CompositeRewardEvaluator()
    sample = "<answer>42</answer>"
    scores_short = evaluator.evaluate(sample, "#### 42", token_count=10, max_tokens=1000)
    scores_long = evaluator.evaluate(sample, "#### 42", token_count=800, max_tokens=1000)
    assert scores_short["total_reward"] > scores_long["total_reward"]
