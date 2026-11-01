"""
Tool-GRPO: Production Colab Two-Stage RLVR Pipeline (SFT Warm-Up + Multi-Turn GRPO)
Solves the RL cold-start exploration barrier and aligns Qwen2.5-Coder-1.5B with verified code execution.
"""

import os
import sys
import re
import json
import time
import subprocess
import tempfile
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional

print("=" * 70, flush=True)
print("  TOOL-GRPO: TWO-STAGE RLVR PIPELINE (SFT WARM-UP + MULTI-TURN GRPO)", flush=True)
print("=" * 70, flush=True)

# -------------------------------------------------------------
# 1. Isolated Python Sandbox
# -------------------------------------------------------------
class PythonSandbox:
    def __init__(self, timeout: float = 3.0, max_chars: int = 500):
        self.timeout = timeout
        self.max_chars = max_chars

    def execute(self, code: str) -> Tuple[bool, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name
        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            if res.returncode == 0:
                output = out if out else "[Executed with no stdout]"
                return True, output[:self.max_chars]
            else:
                err_line = err.splitlines()[-1] if err else "Execution Error"
                return False, f"RuntimeError: {err_line}"[:self.max_chars]
        except subprocess.TimeoutExpired:
            return False, f"TimeoutError: Exceeded {self.timeout}s limit"
        except Exception as e:
            return False, f"Error: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

# -------------------------------------------------------------
# 2. Math Verifier (SymPy + Fractions + Float Equivalence)
# -------------------------------------------------------------
def extract_boxed(text: str) -> Optional[str]:
    if not text:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None

class MathVerifier:
    def __init__(self):
        self.answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def extract_predicted(self, completion: str) -> Optional[str]:
        if not completion:
            return None
        matches = self.answer_tag_pattern.findall(completion)
        if matches:
            cand = matches[-1].strip()
            inner = extract_boxed(cand)
            return inner if inner else cand
        
        boxed = extract_boxed(completion)
        if boxed:
            return boxed

        numbers = re.findall(r"[-+]?\d*\.?\d+", completion)
        return numbers[-1].strip() if numbers else None

    def extract_gt(self, solution: str) -> str:
        if not solution:
            return ""
        if "####" in solution:
            gt = solution.split("####")[-1].strip()
        else:
            gt = solution.strip()
        boxed = extract_boxed(gt)
        return boxed if boxed else gt

    def normalize(self, val: str) -> str:
        if not val:
            return ""
        c = val.replace("$", "").replace(",", "").replace("%", "").strip()
        return c[:-1].strip() if c.endswith(".") else c

    def verify(self, completion: str, gt_raw: str) -> bool:
        pred = self.extract_predicted(completion)
        if pred is None:
            return False
        gt = self.extract_gt(gt_raw)
        norm_p = self.normalize(pred)
        norm_g = self.normalize(gt)

        if norm_p == norm_g:
            return True
        try:
            return abs(float(norm_p) - float(norm_g)) < 1e-4
        except ValueError:
            pass

        try:
            from sympy import simplify
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            t = standard_transformations + (implicit_multiplication_application,)
            diff = simplify(parse_expr(norm_p, transformations=t) - parse_expr(norm_g, transformations=t))
            return bool(diff == 0)
        except Exception:
            return False

# -------------------------------------------------------------
# 3. Shaped Composite Reward Evaluator
# -------------------------------------------------------------
class CompositeRewardEvaluator:
    def __init__(self, verifier: MathVerifier):
        self.verifier = verifier

    def evaluate(self, completion: str, gt: str, exec_failures: int, succ_execs: int, repairs: int, token_len: int) -> Dict[str, float]:
        is_corr = self.verifier.verify(completion, gt)
        r_outcome = 1.0 if is_corr else 0.0

        has_ans = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL | re.IGNORECASE))
        r_format = 0.1 if has_ans else -0.1

        r_exec = (0.05 * min(succ_execs, 2)) - (0.1 * min(exec_failures, 2))
        r_reflexion = 0.15 * min(repairs, 1)
        r_len = -0.02 * (token_len / 512.0)

        total = r_outcome + r_format + r_exec + r_reflexion + r_len
        return {
            "total_reward": total,
            "r_outcome": r_outcome,
            "r_format": r_format,
            "r_exec": r_exec,
            "is_correct": 1.0 if is_corr else 0.0,
        }

# -------------------------------------------------------------
# 4. Multi-Turn In-Context Dialogue Formatter
# -------------------------------------------------------------
def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert mathematical problem solver with access to an isolated Python execution environment.\n"
                "When you need to perform calculations, write Python code inside <code>...</code> blocks.\n"
                "The environment returns the output in <observation>...</observation>.\n"
                "Put your final numerical answer strictly inside <answer>...</answer> tags."
            )
        },
        {
            "role": "user",
            "content": "A florist has 48 roses. She sells 18 in the morning and 12 in the afternoon. How many remain?"
        },
        {
            "role": "assistant",
            "content": "<think>Remaining roses: 48 - 18 - 12.</think><code>print(48 - 18 - 12)</code>\n<observation>\n18\n</observation>\n<think>Calculation verified: 18 remain.</think><answer>18</answer>"
        },
        {
            "role": "user",
            "content": "What is 15 percent of 80?"
        },
        {
            "role": "assistant",
            "content": "<think>Compute 0.15 * 80.</think><code>print(int(0.15 * 80))</code>\n<observation>\n12\n</observation>\n<think>15% of 80 is 12.</think><answer>12</answer>"
        },
        {
            "role": "user",
            "content": f"Solve this problem step-by-step:\n{question}"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -------------------------------------------------------------
# 5. Interleaved Rollout Engine
# -------------------------------------------------------------
def run_interleaved_rollout(
    model,
    tokenizer,
    sandbox: PythonSandbox,
    prompt: str,
    device: str,
    max_turns: int = 3,
    max_total_tokens: int = 400,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    curr_text = prompt
    loss_mask = [0] * len(full_tokens)

    turns = 0
    exec_failures = 0
    succ_execs = 0
    had_failure = False
    repairs = 0
    tool_calls = 0

    while turns < max_turns and len(full_tokens) < max_total_tokens:
        turns += 1
        input_ids = torch.tensor([full_tokens], device=device)

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=160,
                temperature=temperature if temperature > 0 else None,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = gen_out[0][len(full_tokens):].tolist()
        if not new_ids:
            break

        chunk_str = tokenizer.decode(new_ids, skip_special_tokens=True)
        curr_text += chunk_str
        full_tokens.extend(new_ids)
        loss_mask.extend([1] * len(new_ids))

        if "</answer>" in chunk_str.lower():
            break

        if "</code>" in chunk_str.lower():
            tool_calls += 1
            code_matches = re.findall(r"<code>(.*?)</code>", curr_text[len(prompt):], re.DOTALL | re.IGNORECASE)
            if code_matches:
                code_to_run = code_matches[-1].strip()
                ok, output = sandbox.execute(code_to_run)
                if ok:
                    succ_execs += 1
                    if had_failure:
                        repairs += 1
                        had_failure = False
                else:
                    exec_failures += 1
                    had_failure = True

                obs = f"\n<observation>\n{output}\n</observation>\n"
                obs_ids = tokenizer.encode(obs, add_special_tokens=False)
                curr_text += obs
                full_tokens.extend(obs_ids)
                loss_mask.extend([0] * len(obs_ids))

    completion_only = curr_text[len(prompt):]
    return {
        "text": curr_text,
        "completion": completion_only,
        "tokens": full_tokens,
        "loss_mask": loss_mask,
        "tool_calls": tool_calls,
        "exec_failures": exec_failures,
        "succ_execs": succ_execs,
        "repairs": repairs,
    }

# -------------------------------------------------------------
# 6. SFT Warm-Up Dataset (Synthetic Interleaved Trajectories)
# -------------------------------------------------------------
WARMUP_TRAJECTORIES = [
    {
        "q": "Natalia sold clips to 48 of her friends in April, and half as many in May. How many clips altogether?",
        "solution": "<think>April: 48, May: 48 // 2. Total: 48 + 24.</think><code>april = 48\nmay = april // 2\nprint(april + may)</code>\n<observation>\n72\n</observation>\n<think>Total clips sold is 72.</think><answer>72</answer>"
    },
    {
        "q": "Weng earns $12 an hour babysitting. Yesterday she worked for 50 minutes. How much did she earn in dollars?",
        "solution": "<think>Hourly rate: 12. Minutes: 50. Earnings = 12 * (50/60).</think><code>print(int(12 * (50 / 60)))</code>\n<observation>\n10\n</observation>\n<think>Earned 10 dollars.</think><answer>10</answer>"
    },
    {
        "q": "Betty picked 16 strawberries. Her brother picked twice as many. How many strawberries in total?",
        "solution": "<think>Betty: 16, Brother: 16 * 2. Total: 16 + 32.</think><code>print(16 + 16 * 2)</code>\n<observation>\n48\n</observation>\n<think>Total strawberries: 48.</think><answer>48</answer>"
    },
    {
        "q": "A baker has 60 loaves of bread. 35 are sold in the morning. He bakes 20 more. How many loaves now?",
        "solution": "<think>Remaining after morning: 60 - 35 = 25. Add 20.</think><code>print(60 - 35 + 20)</code>\n<observation>\n45\n</observation>\n<think>Baker has 45 loaves.</think><answer>45</answer>"
    },
    {
        "q": "John buys 3 shirts for $15 each and 2 pairs of jeans for $30 each. How much in total?",
        "solution": "<think>Shirts: 3 * 15, Jeans: 2 * 30. Total.</think><code>print(3 * 15 + 2 * 30)</code>\n<observation>\n105\n</observation>\n<think>Total is $105.</think><answer>105</answer>"
    },
    {
        "q": "A car travels at 60 mph for 3 hours, then 40 mph for 2 hours. What is the total distance traveled in miles?",
        "solution": "<think>Distance = 60 * 3 + 40 * 2.</think><code>print(60 * 3 + 40 * 2)</code>\n<observation>\n260\n</observation>\n<think>Total distance is 260 miles.</think><answer>260</answer>"
    },
    {
        "q": "A classroom has 24 students. 1/3 wear glasses. How many students do NOT wear glasses?",
        "solution": "<think>Glasses = 24 / 3 = 8. No glasses = 24 - 8.</think><code>print(24 - 24 // 3)</code>\n<observation>\n16\n</observation>\n<think>16 students do not wear glasses.</think><answer>16</answer>"
    },
    {
        "q": "Calculate 15 squared minus 12 squared.",
        "solution": "<think>Compute 15^2 - 12^2.</think><code>print(15**2 - 12**2)</code>\n<observation>\n81\n</observation>\n<think>Result is 81.</think><answer>81</answer>"
    }
]

# -------------------------------------------------------------
# 7. Main Execution: Warm-Up, GRPO, and Comparison
# -------------------------------------------------------------
def main():
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    SAVE_DIR = "/content/tool_grpo_v2_model"
    REPORT_PATH = "/content/benchmark_comparison.json"
    os.makedirs(SAVE_DIR, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading base model in bfloat16...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    policy_model = get_peft_model(base_model, lora_config)
    policy_model.print_trainable_parameters()

    sandbox = PythonSandbox(timeout=3.0)
    verifier = MathVerifier()
    evaluator = CompositeRewardEvaluator(verifier=verifier)

    print("[*] Loading GSM8K benchmark test split...", flush=True)
    dataset = load_dataset("openai/gsm8k", "main")
    eval_samples = dataset["test"].select(range(10))

    def evaluate(model, split, label="Eval"):
        print(f"\n--- Running {label} on {len(split)} problems ---", flush=True)
        correct = 0
        tools = 0
        for i, item in enumerate(split):
            prompt = build_prompt(tokenizer, item["question"])
            res = run_interleaved_rollout(model, tokenizer, sandbox, prompt, device, max_turns=2, temperature=0.0)
            is_corr = verifier.verify(res["completion"], item["answer"])
            if is_corr:
                correct += 1
            tools += res["tool_calls"]
            pred = verifier.extract_predicted(res["completion"])
            gt = verifier.extract_gt(item["answer"])
            print(f"[{label} #{i+1:02d}/{len(split)}] Correct: {str(is_corr):5s} | Pred: {str(pred):10s} | GT: {str(gt):10s} | Tools: {res['tool_calls']}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pass_1 = (correct / len(split)) * 100
        print(f"[{label} Result] Pass@1: {pass_1:.1f}% | Avg Tools: {tools/len(split):.2f}", flush=True)
        return pass_1

    # Stage 0: Initial Zero-Shot Baseline
    print("\n" + "=" * 55, flush=True)
    print("  STAGE 0: ZERO-SHOT BASELINE EVALUATION", flush=True)
    print("=" * 55, flush=True)
    baseline_pass_1 = evaluate(policy_model, eval_samples, label="Baseline")

    # Stage 1: SFT Warm-Up (Solves Exploration Cold Start)
    print("\n" + "=" * 55, flush=True)
    print("  STAGE 1: INTERLEAVED SFT WARM-UP (8 Trajectories)", flush=True)
    print("=" * 55, flush=True)
    sft_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5)
    policy_model.train()

    for epoch in range(2):
        total_sft_loss = 0.0
        for item in WARMUP_TRAJECTORIES:
            prompt = build_prompt(tokenizer, item["q"])
            full_text = prompt + item["solution"]
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            # Mask out prompt tokens from cross-entropy loss
            labels = list(full_ids)
            for k in range(prompt_len):
                labels[k] = -100

            input_t = torch.tensor([full_ids], device=device)
            target_t = torch.tensor([labels], device=device)

            logits = policy_model(input_t).logits[:, :-1, :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_t[:, 1:].reshape(-1),
                ignore_index=-100
            )

            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()
            total_sft_loss += loss.item()

        print(f"  [SFT Epoch {epoch+1}/2] Mean Cross-Entropy Loss: {total_sft_loss / len(WARMUP_TRAJECTORIES):.4f}", flush=True)

    print("[*] SFT Warm-Up Complete. Creating frozen reference model for GRPO...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Stage 2: Multi-Turn GRPO Reinforcement Learning
    print("\n" + "=" * 55, flush=True)
    print("  STAGE 2: MULTI-TURN GRPO RLVR ALIGNMENT (G=4)", flush=True)
    print("=" * 55, flush=True)

    grpo_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    train_samples = dataset["train"].select(range(10))
    GROUP_SIZE = 4
    KL_COEFF = 0.04
    training_history = []
    start_time = time.time()

    policy_model.train()
    for step, item in enumerate(train_samples):
        prompt = build_prompt(tokenizer, item["question"])
        gt = item["answer"]

        trajectories = []
        rewards = []

        for g in range(GROUP_SIZE):
            rollout = run_interleaved_rollout(
                policy_model, tokenizer, sandbox, prompt, device, max_turns=2, temperature=0.7
            )
            score_dict = evaluator.evaluate(
                rollout["completion"], gt,
                rollout["exec_failures"], rollout["succ_execs"], rollout["repairs"],
                len(rollout["tokens"])
            )
            trajectories.append(rollout)
            rewards.append(score_dict["total_reward"])

        # Advantage Normalization
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = r_tensor.std(unbiased=False)
        mean = r_tensor.mean()
        adv = (r_tensor - mean) / (std + 1e-4) if std > 1e-4 else (r_tensor - mean)

        grpo_optimizer.zero_grad()
        step_loss = 0.0
        step_kl = 0.0

        for i in range(GROUP_SIZE):
            traj = trajectories[i]
            token_ids = torch.tensor([traj["tokens"]], device=device)
            mask = torch.tensor([traj["loss_mask"]], device=device, dtype=torch.float32)

            if mask.sum() < 1:
                continue

            logits = policy_model(token_ids).logits[:, :-1, :]
            targets = token_ids[:, 1:]
            mask_shifted = mask[:, 1:]

            log_probs = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                ref_logits = ref_model(token_ids).logits[:, :-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)

            log_ratio = ref_log_probs - log_probs
            kl = torch.exp(log_ratio) - log_ratio - 1.0

            policy_token_loss = -log_probs * adv[i]
            total_loss = (policy_token_loss + KL_COEFF * kl) * mask_shifted
            traj_loss = total_loss.sum() / (mask_shifted.sum() + 1e-6) / GROUP_SIZE
            traj_loss.backward()

            step_loss += traj_loss.item()
            step_kl += (kl * mask_shifted).sum().item() / (mask_shifted.sum().item() + 1e-6)

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        grpo_optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_r = sum(rewards) / len(rewards)
        correct_rollouts = sum(1 for r in rewards if r >= 0.9)
        print(f"[GRPO Step {step+1:02d}/{len(train_samples)}] Correct: {correct_rollouts}/{GROUP_SIZE} | Avg Reward: {avg_r:+.3f} | Loss: {step_loss:.4f} | KL: {step_kl/GROUP_SIZE:.4f}", flush=True)

        training_history.append({
            "step": step + 1,
            "correct_rollouts": correct_rollouts,
            "avg_reward": round(avg_r, 4),
            "loss": round(step_loss, 4),
            "kl": round(step_kl / GROUP_SIZE, 4),
        })

    duration = time.time() - start_time
    print(f"[*] GRPO alignment completed in {duration:.1f}s", flush=True)

    # Stage 3: Post-Training Evaluation
    print("\n" + "=" * 55, flush=True)
    print("  STAGE 3: POST-TRAINING EVALUATION (After Tool-GRPO)", flush=True)
    print("=" * 55, flush=True)
    policy_model.eval()
    post_pass_1 = evaluate(policy_model, eval_samples, label="Post-GRPO")

    # Save Checkpoint & Comparison Report
    policy_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    report = {
        "model": MODEL_ID,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "method": "Two-Stage SFT Warm-Up + Multi-Turn Interleaved Tool-GRPO",
        "baseline_pass_1": baseline_pass_1,
        "post_trained_pass_1": post_pass_1,
        "pass_1_delta": round(post_pass_1 - baseline_pass_1, 2),
        "training_duration_seconds": round(duration, 1),
        "training_history": training_history,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("              FINAL HEAD-TO-HEAD COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model Architecture   : {MODEL_ID}", flush=True)
    print(f"  Baseline Pass@1      : {baseline_pass_1:.1f}%", flush=True)
    print(f"  Post-GRPO Pass@1     : {post_pass_1:.1f}%", flush=True)
    print(f"  Accuracy Delta (Δ)   : {post_pass_1 - baseline_pass_1:+.1f}%", flush=True)
    print(f"  Report saved to      : {REPORT_PATH}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
