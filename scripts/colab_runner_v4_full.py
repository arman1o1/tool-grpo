"""
Tool-GRPO: Production Full-Scale RLVR Training & Rigorous Evaluation Suite
Trains Qwen2.5-Coder-1.5B with 30-step Multi-Turn GRPO (SFT warm-up with Reflexion priors)
and runs large-scale multi-sample evaluation on GSM8K (N=40) and MATH-500 (N=25) at k=4.
"""

import os
import sys
import subprocess
import json
import time
import re
import tempfile
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict

# Ensure torchao incompatibility is cleared
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)
except Exception:
    pass

import torch
import torch.nn.functional as F

print("=" * 70, flush=True)
print("  TOOL-GRPO: FULL TRAINING RUN + RIGOROUS EVALUATION SUITE", flush=True)
print("=" * 70, flush=True)

# -------------------------------------------------------------
# 1. Sandboxed Python Executor
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
# 2. Enhanced Math Verifier (SymPy + Float Tolerance + Radicals)
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

def latex_to_sympy_str(expr: str) -> str:
    cleaned = expr.strip()
    boxed = extract_boxed(cleaned)
    if boxed is not None:
        cleaned = boxed
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", cleaned)
    cleaned = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", cleaned)
    cleaned = cleaned.replace(r"\cdot", "*").replace(r"\times", "*")
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace("$", "").replace(",", "")
    return cleaned.strip()

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
        matches = self.answer_tag_pattern.findall(solution)
        if matches:
            return matches[-1].strip()
        if "####" in solution:
            return solution.split("####")[-1].strip()
        boxed = extract_boxed(solution)
        if boxed:
            return boxed
        return solution.strip()

    def verify(self, completion: str, ground_truth: str) -> bool:
        pred = self.extract_predicted(completion)
        gt = self.extract_gt(ground_truth)
        if pred is None or not gt:
            return False

        pred_clean = pred.strip().replace("$", "").replace(",", "").lower()
        gt_clean = gt.strip().replace("$", "").replace(",", "").lower()

        if pred_clean == gt_clean:
            return True

        # Float / Decimal Equivalence
        try:
            p_val = float(pred_clean)
            g_val = float(gt_clean)
            if math.isclose(p_val, g_val, rel_tol=1e-4, abs_tol=1e-4):
                return True
        except ValueError:
            pass

        # Fraction Equivalence
        try:
            from fractions import Fraction
            if "/" in pred_clean or "/" in gt_clean:
                p_frac = Fraction(pred_clean)
                g_frac = Fraction(gt_clean)
                return p_frac == g_frac
        except Exception:
            pass

        # SymPy Symbolic & Radical Equivalence
        try:
            import sympy
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            t = standard_transformations + (implicit_multiplication_application,)
            p_clean = latex_to_sympy_str(pred)
            g_clean = latex_to_sympy_str(gt)
            p_expr = parse_expr(p_clean, transformations=t)
            g_expr = parse_expr(g_clean, transformations=t)
            diff = sympy.simplify(p_expr - g_expr)
            if diff == 0:
                return True
            try:
                if abs(float(diff)) < 1e-4:
                    return True
            except Exception:
                pass
        except Exception:
            pass

        return False

# -------------------------------------------------------------
# 3. Metrics & Diagnostic Taxonomy
# -------------------------------------------------------------
def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k or n <= 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 100.0
    prod = 1.0
    for j in range(1, k + 1):
        prod *= (n - c - j + 1) / (n - j + 1)
    return round(max(0.0, min(1.0, 1.0 - prod)) * 100.0, 2)

def cluster_and_majority_vote(predictions: List[str], verifier: MathVerifier, ground_truth: str):
    valid_preds = [p for p in predictions if p is not None and str(p).strip()]
    if not valid_preds:
        return None, False, 0

    clusters = []
    for pred in valid_preds:
        matched = False
        for c in clusters:
            if verifier.verify(f"<answer>{pred}</answer>", c['rep']):
                c["count"] += 1
                c["members"].append(pred)
                matched = True
                break
        if not matched:
            clusters.append({"rep": pred, "count": 1, "members": [pred]})

    clusters.sort(key=lambda x: x["count"], reverse=True)
    winner = clusters[0]["rep"]
    votes = clusters[0]["count"]
    is_corr = verifier.verify(f"<answer>{winner}</answer>", ground_truth)
    return winner, is_corr, votes

def classify_failure(text: str, ground_truth: str, verifier: MathVerifier, exec_failures: int, succ_execs: int) -> str:
    if "</answer>" not in text.lower() and r"\boxed{" not in text:
        return "FORMAT_MISSING_ANSWER"
    pred = verifier.extract_predicted(text)
    if pred is None:
        return "VERIFIER_PARSE_FAIL"
    if exec_failures > 0 and succ_execs == 0:
        if "syntaxerror" in text.lower():
            return "EXECUTION_SYNTAX_ERROR"
        if "timeouterror" in text.lower():
            return "EXECUTION_TIMEOUT"
        return "EXECUTION_RUNTIME_ERROR"
    return "LOGIC_INCORRECT"

# -------------------------------------------------------------
# 4. Prompt Builder & Rollout Engine with Reflexion Feedback
# -------------------------------------------------------------
def build_prompt(tokenizer, question: str) -> str:
    system_text = (
        "You are an expert mathematical reasoning assistant with access to a Python execution tool.\n"
        "Guidelines:\n"
        "1. Write clear step-by-step reasoning inside <think>...</think> tags.\n"
        "2. To run Python, emit <code>...</code>. The environment will return <observation>...</observation>.\n"
        "3. If an error occurs, analyze the traceback, fix your code, and run it again.\n"
        "4. Provide your final definitive answer inside <answer>...</answer> tags."
    )
    demo_q = "A store has 40 apples. 15 are sold in the morning and 10 in the afternoon. How many remain?"
    demo_ans = (
        "<think>Remaining = 40 - 15 - 10.</think>\n"
        "<code>print(40 - 15 - 10)</code>\n"
        "<observation>\n15\n</observation>\n"
        "<think>Total remaining is 15.</think>\n"
        "<answer>15</answer>"
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": demo_q},
        {"role": "assistant", "content": demo_ans},
        {"role": "user", "content": f"Problem: {question}\nSolve step-by-step using Python when helpful."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_rollout(
    model, tokenizer, sandbox, prompt: str, device: str,
    max_turns: int = 3, max_tokens: int = 400, temperature: float = 0.7
) -> Dict[str, Any]:
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    curr_text = prompt
    loss_mask = [0] * len(full_tokens)

    turns = 0
    tool_calls = 0
    exec_failures = 0
    succ_execs = 0
    repairs = 0
    had_failure = False

    while turns < max_turns and len(full_tokens) < max_tokens:
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
        chunk = tokenizer.decode(new_ids, skip_special_tokens=True)
        curr_text += chunk
        full_tokens.extend(new_ids)
        loss_mask.extend([1] * len(new_ids))

        if "</answer>" in chunk.lower():
            break

        if "</code>" in chunk.lower():
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
                    obs = f"\n<observation>\n{output}\n</observation>\n"
                else:
                    exec_failures += 1
                    had_failure = True
                    # Explicit Reflexion feedback to guide the model to fix its code
                    obs = f"\n<observation>\n{output}\n[Please fix the code in a new <code> block and rerun]\n</observation>\n"

                obs_ids = tokenizer.encode(obs, add_special_tokens=False)
                curr_text += obs
                full_tokens.extend(obs_ids)
                loss_mask.extend([0] * len(obs_ids))

    return {
        "text": curr_text,
        "completion": curr_text[len(prompt):],
        "tokens": full_tokens,
        "loss_mask": loss_mask,
        "tool_calls": tool_calls,
        "exec_failures": exec_failures,
        "succ_execs": succ_execs,
        "repairs": repairs,
    }

# -------------------------------------------------------------
# 5. Composite Reward
# -------------------------------------------------------------
class RewardEvaluator:
    def __init__(self, verifier: MathVerifier):
        self.verifier = verifier

    def evaluate(self, completion: str, gt: str, exec_failures: int, succ_execs: int, repairs: int) -> float:
        r = 0.0
        # Outcome
        if self.verifier.verify(completion, gt):
            r += 1.0
        # Format
        if "<answer>" in completion and "</answer>" in completion:
            r += 0.1
        # Execution
        r += min(0.2, succ_execs * 0.1)
        if exec_failures > 0 and repairs == 0:
            r -= min(0.2, exec_failures * 0.1)
        # Reflexion repair reward
        if repairs > 0:
            r += 0.25
        return round(r, 4)

# -------------------------------------------------------------
# 6. SFT Warm-Up Trajectories (Including SymPy and Reflexion)
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
        "q": "Compute the sum 1/2 + 1/4 + 1/8 as an exact simplified fraction.",
        "solution": "<think>Use Python fractions for exact rational arithmetic.</think><code>from fractions import Fraction\nres = Fraction(1, 2) + Fraction(1, 4) + Fraction(1, 8)\nprint(f'{res.numerator}/{res.denominator}')</code>\n<observation>\n7/8\n</observation>\n<think>The exact fraction is 7/8.</think><answer>7/8</answer>"
    },
    {
        "q": "Simplify the radical expression sqrt(117).",
        "solution": "<think>Use SymPy for exact radical simplification.</think><code>import sympy\nprint(sympy.sqrt(117))</code>\n<observation>\n3*sqrt(13)\n</observation>\n<think>In exact radical form it is 3\\sqrt{13}.</think><answer>3\\sqrt{13}</answer>"
    },
    {
        # Reflexion repair demonstration
        "q": "Find the sum of all integers from 1 to 10 inclusive.",
        "solution": "<think>First attempt.</think><code>prnt(sum(range(1, 11)))</code>\n<observation>\nRuntimeError: name 'prnt' is not defined\n[Please fix the code in a new <code> block and rerun]\n</observation>\n<think>I had a typo 'prnt'. Fixing it to 'print'.</think><code>print(sum(range(1, 11)))</code>\n<observation>\n55\n</observation>\n<think>The sum is 55.</think><answer>55</answer>"
    },
    {
        "q": "Calculate 15 squared minus 12 squared.",
        "solution": "<think>Compute 15^2 - 12^2.</think><code>print(15**2 - 12**2)</code>\n<observation>\n81\n</observation>\n<think>Result is 81.</think><answer>81</answer>"
    }
]

# -------------------------------------------------------------
# 7. Main Full Pipeline: 30 GRPO Steps + Rigorous Evaluation
# -------------------------------------------------------------
def main():
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    REPORT_PATH = "/content/full_run_benchmark_results.json"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading base model in bfloat16...", flush=True)
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

    sandbox = PythonSandbox(timeout=3.0)
    verifier = MathVerifier()
    reward_evaluator = RewardEvaluator(verifier=verifier)

    # Stage 1: SFT Warm-Up
    print("\n" + "=" * 60, flush=True)
    print("  STAGE 1: SFT WARM-UP (SymPy + Reflexion Priors)", flush=True)
    print("=" * 60, flush=True)
    sft_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-5)
    policy_model.train()

    for epoch in range(2):
        sft_loss = 0.0
        for item in WARMUP_TRAJECTORIES:
            prompt = build_prompt(tokenizer, item["q"])
            full_text = prompt + item["solution"]
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

            labels = list(full_ids)
            for k in range(prompt_len):
                labels[k] = -100

            inp = torch.tensor([full_ids], device=device)
            tgt = torch.tensor([labels], device=device)

            logits = policy_model(inp).logits[:, :-1, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=-100)

            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()
            sft_loss += loss.item()

        print(f"  [SFT Epoch {epoch+1}/2] Mean Cross-Entropy Loss: {sft_loss / len(WARMUP_TRAJECTORIES):.4f}", flush=True)

    # Frozen Reference Model
    print("[*] Creating frozen reference model for GRPO...", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Stage 2: Full GRPO Training (30 Steps)
    print("\n" + "=" * 60, flush=True)
    print("  STAGE 2: MULTI-TURN GRPO TRAINING (30 steps, G=4 = 120 rollouts)", flush=True)
    print("=" * 60, flush=True)

    gsm_dataset = load_dataset("openai/gsm8k", "main")
    train_samples = gsm_dataset["train"].select(range(30))
    grpo_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    GROUP_SIZE = 4
    KL_COEFF = 0.04

    policy_model.train()
    start_train = time.time()

    for step, item in enumerate(train_samples):
        prompt = build_prompt(tokenizer, item["question"])
        gt = item["answer"]

        trajectories = []
        rewards = []

        for _ in range(GROUP_SIZE):
            rollout = run_rollout(policy_model, tokenizer, sandbox, prompt, device, temperature=0.7)
            r = reward_evaluator.evaluate(
                rollout["completion"], gt, rollout["exec_failures"], rollout["succ_execs"], rollout["repairs"]
            )
            trajectories.append(rollout)
            rewards.append(r)

        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = r_tensor.std(unbiased=False)
        mean = r_tensor.mean()
        adv = (r_tensor - mean) / (std + 1e-4) if std > 1e-4 else (r_tensor - mean)

        grpo_optimizer.zero_grad()
        step_loss = 0.0

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

            total_loss = (-log_probs * adv[i] + KL_COEFF * kl) * mask_shifted
            traj_loss = total_loss.sum() / (mask_shifted.sum() + 1e-6) / GROUP_SIZE
            traj_loss.backward()
            step_loss += traj_loss.item()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        grpo_optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_r = sum(rewards) / len(rewards)
        correct_n = sum(1 for r in rewards if r >= 0.9)
        print(f"[GRPO Step {step+1:02d}/30] Correct: {correct_n}/{GROUP_SIZE} | Avg Reward: {avg_r:+.3f} | Loss: {step_loss:.4f}", flush=True)

    print(f"[*] GRPO alignment completed in {time.time() - start_train:.1f}s", flush=True)

    # -------------------------------------------------------------
    # Stage 3: Rigorous Benchmark Evaluation Suite (GSM8K N=40, MATH-500 N=25)
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("  STAGE 3: FULL RIGOROUS EVALUATION (GSM8K N=40 + MATH-500 N=25, k=4)", flush=True)
    print("=" * 60, flush=True)

    policy_model.eval()

    gsm_test = gsm_dataset["test"].select(range(40))
    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_test = math_dataset.select(range(25))

    benchmarks_data = {
        "GSM8K": [
            {"id": f"gsm8k-{i+1:02d}", "question": r["question"], "ground_truth": r["answer"], "category": "GSM8K"}
            for i, r in enumerate(gsm_test)
        ],
        "MATH-500": [
            {"id": f"math500-{i+1:02d}", "question": r["problem"], "ground_truth": r["answer"], "category": r.get("subject", "MATH")}
            for i, r in enumerate(math_test)
        ]
    }

    K_SAMPLES = 4
    all_benchmark_summaries = []

    for b_name, problems in benchmarks_data.items():
        print(f"\n--- Running Full Evaluation on {b_name} ({len(problems)} problems, k={K_SAMPLES} rollouts/prob) ---", flush=True)
        results = []
        b_start = time.time()

        for idx, prob in enumerate(problems):
            prompt = build_prompt(tokenizer, prob["question"])
            trajectories = []
            preds = []
            correct_mask = []

            for s in range(K_SAMPLES):
                temp = 0.0 if s == 0 else 0.7
                roll = run_rollout(policy_model, tokenizer, sandbox, prompt, device, temperature=temp)
                trajectories.append({
                    "completion": roll["completion"],
                    "tool_calls": roll["tool_calls"],
                    "exec_failures": roll["exec_failures"],
                    "succ_execs": roll["succ_execs"],
                    "repairs": roll["repairs"],
                    "tokens": roll["tokens"],
                })
                pred = verifier.extract_predicted(roll["completion"])
                is_c = verifier.verify(roll["completion"], prob["ground_truth"])
                preds.append(pred)
                correct_mask.append(is_c)

            maj_winner, maj_corr, votes = cluster_and_majority_vote(preds, verifier, prob["ground_truth"])
            fail_mode = None
            if not maj_corr and trajectories:
                fail_mode = classify_failure(
                    trajectories[0]["completion"], prob["ground_truth"], verifier,
                    trajectories[0]["exec_failures"], trajectories[0]["succ_execs"]
                )

            res = {
                "problem_id": prob["id"],
                "question": prob["question"],
                "ground_truth": verifier.extract_gt(prob["ground_truth"]),
                "category": prob["category"],
                "predictions": preds,
                "correct_mask": correct_mask,
                "majority_prediction": maj_winner,
                "majority_correct": maj_corr,
                "majority_votes": votes,
                "failure_mode": fail_mode,
                "tool_calls": sum(t["tool_calls"] for t in trajectories),
                "exec_failures": sum(t["exec_failures"] for t in trajectories),
                "succ_execs": sum(t["succ_execs"] for t in trajectories),
                "repairs": sum(t["repairs"] for t in trajectories),
                "trajectories": trajectories,
            }
            results.append(res)

            status = "PASS" if correct_mask[0] else "FAIL"
            maj_s = f"MAJ: {'WIN' if maj_corr else 'FAIL'} ({votes}/{K_SAMPLES}v)"
            print(f"[{b_name} #{idx+1:02d}/{len(problems)}] {status:4s} | {maj_s:14s} | Pred: {str(maj_winner):10s} | GT: {str(res['ground_truth']):10s} | Tools: {res['tool_calls']}", flush=True)

        elapsed_b = time.time() - b_start
        n_p = len(results)
        p1 = round(sum(1 for r in results if r["correct_mask"][0]) / n_p * 100.0, 2)
        maj_p = round(sum(1 for r in results if r["majority_correct"]) / n_p * 100.0, 2)

        pass_at_k = {}
        for k in [1, 2, K_SAMPLES]:
            scores = [calculate_pass_at_k(K_SAMPLES, sum(1 for c in r["correct_mask"] if c), k) for r in results]
            pass_at_k[k] = round(sum(scores) / len(scores), 2)

        probs_with_tools = sum(1 for r in results if r["tool_calls"] > 0)
        tool_invocation_rate = round(probs_with_tools / n_p * 100.0, 2)

        tot_succ = sum(r["succ_execs"] for r in results)
        tot_fail = sum(r["exec_failures"] for r in results)
        tot_execs = tot_succ + tot_fail
        tool_succ_rate = round(tot_succ / tot_execs * 100.0, 2) if tot_execs > 0 else 0.0

        probs_with_fail = [r for r in results if r["exec_failures"] > 0]
        repaired = sum(1 for r in probs_with_fail if r["repairs"] > 0 and r["majority_correct"])
        reflexion_rate = round(repaired / len(probs_with_fail) * 100.0, 2) if probs_with_fail else 0.0

        taxonomy = dict(Counter(r["failure_mode"] for r in results if r["failure_mode"]))

        cat_map = defaultdict(list)
        for r in results:
            cat_map[r["category"]].append(r)
        category_metrics = {}
        for c_name, c_items in cat_map.items():
            c_p1 = sum(1 for x in c_items if x["correct_mask"][0]) / len(c_items) * 100.0
            c_maj = sum(1 for x in c_items if x["majority_correct"]) / len(c_items) * 100.0
            category_metrics[c_name] = {
                "count": len(c_items),
                "pass_at_1": round(c_p1, 2),
                "majority_correct": round(c_maj, 2),
            }

        summary = {
            "benchmark_name": b_name,
            "total_problems": n_p,
            "pass_at_1": p1,
            "pass_at_k": pass_at_k,
            "majority_at_k": {K_SAMPLES: maj_p},
            "tool_invocation_rate": tool_invocation_rate,
            "tool_success_rate": tool_succ_rate,
            "reflexion_recovery_rate": reflexion_rate,
            "avg_tool_calls_per_problem": round(sum(r["tool_calls"] for r in results) / (n_p * K_SAMPLES), 2),
            "failure_taxonomy": taxonomy,
            "category_metrics": category_metrics,
            "elapsed_seconds": round(elapsed_b, 1),
            "sample_results": results,
        }
        all_benchmark_summaries.append(summary)

        print(f"\n[{b_name} Full Summary]", flush=True)
        print(f"  Pass@1 (Greedy)        : {p1}%", flush=True)
        print(f"  Pass@4 (Unbiased)      : {pass_at_k.get(K_SAMPLES, p1)}%", flush=True)
        print(f"  Maj@4 (Self-Consist)   : {maj_p}%", flush=True)
        print(f"  Tool Invocation Rate   : {tool_invocation_rate}%", flush=True)
        print(f"  Tool Success Rate      : {tool_succ_rate}%", flush=True)
        print(f"  Reflexion Recovery Rate: {reflexion_rate}%", flush=True)
        print(f"  Failure Taxonomy       : {taxonomy}", flush=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_summaries, f, indent=2)

    print(f"\n[*] Full benchmark report successfully saved to: {REPORT_PATH}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
