import re
import torch
from typing import List, Dict, Any, Tuple, Optional, Callable

try:
    from .sandbox import PythonSandbox
except ImportError:
    from sandbox import PythonSandbox

class MultiTurnRolloutEngine:
    """
    Orchestrates multi-turn interleaved trajectory generation with an execution sandbox.
    Produces token-level loss masks so GRPO policy optimization only computes gradients on
    model-generated tokens (excluding prompts and environment <observation> return tokens).
    """
    def __init__(
        self,
        sandbox: Optional[PythonSandbox] = None,
        max_turns: int = 3,
        code_stop_tag: str = "</code>",
        answer_stop_tag: str = "</answer>",
    ):
        self.sandbox = sandbox or PythonSandbox()
        self.max_turns = max_turns
        self.code_stop_tag = code_stop_tag
        self.answer_stop_tag = answer_stop_tag
        self.code_regex = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)

    def extract_latest_code(self, text: str) -> str:
        matches = self.code_regex.findall(text)
        return matches[-1].strip() if matches else ""

    def rollout_with_callable(
        self,
        prompt: str,
        generation_fn: Callable[[str], str],
        tokenizer,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Executes an interleaved multi-turn rollout using a string generation callback.
        Ideal for unit testing, mock simulations, and API-based sampling.
        """
        current_text = prompt
        full_tokens: List[int] = tokenizer.encode(prompt, add_special_tokens=True)
        loss_mask: List[int] = [0] * len(full_tokens)

        turns = 0
        execution_failures = 0
        successful_executions = 0
        tool_call_count = 0
        had_failure = False
        reflexion_repairs = 0

        while turns < self.max_turns:
            turns += 1
            chunk = generation_fn(current_text)
            if not chunk:
                break

            chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
            current_text += chunk
            full_tokens.extend(chunk_tokens)
            loss_mask.extend([1] * len(chunk_tokens))

            if self.answer_stop_tag.lower() in chunk.lower():
                break

            if self.code_stop_tag.lower() in chunk.lower():
                tool_call_count += 1
                code_to_run = self.extract_latest_code(current_text)
                success, output = self.sandbox.execute(code_to_run)
                if success:
                    successful_executions += 1
                    if had_failure:
                        reflexion_repairs += 1
                        had_failure = False
                else:
                    execution_failures += 1
                    had_failure = True

                obs_text = f"\n<observation>\n{output}\n</observation>\n"
                obs_tokens = tokenizer.encode(obs_text, add_special_tokens=False)

                current_text += obs_text
                full_tokens.extend(obs_tokens)
                loss_mask.extend([0] * len(obs_tokens))

            if len(full_tokens) >= max_new_tokens + len(loss_mask[:len(prompt)]):
                break

        completion_only = current_text[len(prompt):]
        return {
            "trajectory_text": current_text,
            "completion": completion_only,
            "tokens": full_tokens,
            "loss_mask": loss_mask,
            "tool_call_count": tool_call_count,
            "execution_failures": execution_failures,
            "successful_executions": successful_executions,
            "reflexion_repairs": reflexion_repairs,
            "num_policy_tokens": sum(loss_mask),
            "num_env_tokens": len(loss_mask) - sum(loss_mask),
        }

    def rollout_with_model(
        self,
        model,
        tokenizer,
        prompt: str,
        device: str = "cpu",
        temperature: float = 0.7,
        max_turn_tokens: int = 128,
        max_total_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Executes a real multi-turn interleaved rollout using a HuggingFace causal LM.
        Dynamically stops at </code> or </answer>, executes sandbox code, injects
        <observation>...</observation>, and records 0/1 token-level loss masks.
        """
        full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        curr_text = prompt
        loss_mask = [0] * len(full_tokens)

        turns = 0
        execution_failures = 0
        successful_executions = 0
        tool_calls = 0
        had_failure = False
        reflexion_repairs = 0

        while turns < self.max_turns and len(full_tokens) < max_total_tokens:
            turns += 1
            input_ids = torch.tensor([full_tokens], device=device)

            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_turn_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_ids = gen_out[0][len(full_tokens):].tolist()
            if not new_ids:
                break

            chunk_str = tokenizer.decode(new_ids, skip_special_tokens=True)
            curr_text += chunk_str
            full_tokens.extend(new_ids)
            loss_mask.extend([1] * len(new_ids))

            if self.answer_stop_tag.lower() in chunk_str.lower():
                break

            if self.code_stop_tag.lower() in chunk_str.lower():
                tool_calls += 1
                code_matches = self.code_regex.findall(curr_text[len(prompt):])
                if code_matches:
                    code_to_run = code_matches[-1].strip()
                    ok, output = self.sandbox.execute(code_to_run)
                    if ok:
                        successful_executions += 1
                        if had_failure:
                            reflexion_repairs += 1
                            had_failure = False
                    else:
                        execution_failures += 1
                        had_failure = True

                    obs = f"\n<observation>\n{output}\n</observation>\n"
                    obs_ids = tokenizer.encode(obs, add_special_tokens=False)
                    curr_text += obs
                    full_tokens.extend(obs_ids)
                    loss_mask.extend([0] * len(obs_ids))

        completion_only = curr_text[len(prompt):]
        return {
            "trajectory_text": curr_text,
            "completion": completion_only,
            "tokens": full_tokens,
            "loss_mask": loss_mask,
            "tool_call_count": tool_calls,
            "execution_failures": execution_failures,
            "successful_executions": successful_executions,
            "reflexion_repairs": reflexion_repairs,
            "num_policy_tokens": sum(loss_mask),
            "num_env_tokens": len(loss_mask) - sum(loss_mask),
        }

if __name__ == "__main__":
    class MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return [hash(w) % 10000 for w in text.split()]

    engine = MultiTurnRolloutEngine()
    prompt = "Problem: What is 15 * 15?"

    def mock_agent(ctx):
        if "<code>" not in ctx:
            return "<think>Calculate 15*15</think><code>print(15 * 15)</code>"
        elif "<observation>" in ctx and "<answer>" not in ctx:
            return "<think>Output is 225</think><answer>225</answer>"
        return ""

    res = engine.rollout_with_callable(prompt, mock_agent, MockTokenizer())
    assert res["tool_call_count"] == 1
    assert res["successful_executions"] == 1
    assert "225" in res["completion"]
    assert res["num_policy_tokens"] > 0
    assert res["num_env_tokens"] > 0
    print("[+] MultiTurnRolloutEngine self-test passed successfully.")
