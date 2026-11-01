import pytest
from src.rollout import MultiTurnRolloutEngine
from src.sandbox import PythonSandbox

class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]

def test_rollout_single_turn_no_tools():
    engine = MultiTurnRolloutEngine()
    prompt = "Problem: 2+2?"
    
    def agent_gen(ctx):
        return "<think>Simple math</think><answer>4</answer>"

    res = engine.rollout_with_callable(prompt, agent_gen, DummyTokenizer())
    assert res["tool_call_count"] == 0
    assert "4" in res["completion"]
    assert res["num_policy_tokens"] > 0
    assert res["num_env_tokens"] == len(prompt)

def test_rollout_interleaved_multi_turn_execution():
    engine = MultiTurnRolloutEngine()
    prompt = "Problem: 12 * 12?"

    def agent_gen(ctx):
        if "<code>" not in ctx:
            return "<think>Let's run code</think><code>print(12 * 12)</code>"
        elif "<observation>" in ctx and "<answer>" not in ctx:
            return "<think>Got 144 from python</think><answer>144</answer>"
        return ""

    res = engine.rollout_with_callable(prompt, agent_gen, DummyTokenizer())
    assert res["tool_call_count"] == 1
    assert res["successful_executions"] == 1
    assert res["reflexion_repairs"] == 0
    assert "144" in res["completion"]
    assert "<observation>" in res["trajectory_text"]

    tokens = res["tokens"]
    mask = res["loss_mask"]
    assert len(tokens) == len(mask)
    assert sum(mask) == res["num_policy_tokens"]
    assert len(mask) - sum(mask) == res["num_env_tokens"]

def test_rollout_reflexion_self_correction_recovery():
    engine = MultiTurnRolloutEngine(max_turns=4)
    prompt = "Problem: Calculate 10 / 2."

    # Agent introduces intentional ZeroDivisionError on Turn 1, then self-corrects on Turn 2
    def agent_self_repair(ctx):
        if "<code>" not in ctx:
            return "<think>Initial buggy attempt</think><code>print(10 / 0)</code>"
        elif "ZeroDivisionError" in ctx and "10 // 2" not in ctx:
            return "<think>Caught ZeroDivisionError! Self-correcting denominator...</think><code>print(10 // 2)</code>"
        elif "<observation>\n5\n</observation>" in ctx and "<answer>" not in ctx:
            return "<think>Verified recovered result.</think><answer>5</answer>"
        return ""

    res = engine.rollout_with_callable(prompt, agent_self_repair, DummyTokenizer())
    assert res["tool_call_count"] == 2
    assert res["execution_failures"] == 1
    assert res["successful_executions"] == 1
    assert res["reflexion_repairs"] == 1
    assert "5" in res["completion"]
