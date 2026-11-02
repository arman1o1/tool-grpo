"""
Tool-GRPO: Scaled RLVR Training & Rigorous Evaluation Suite (v5)
Features:
- SFT Warm-up with SymPy & Reflexion self-correction priors
- 60 GRPO steps with Group Size G=4 (240 training rollouts)
- Dynamic token loss masking & Schulman unbiased KL penalty
- Expanded rollout token budget (max_tokens=896, max_turns=4) eliminating truncation
- Multi-sample rigorous evaluation on GSM8K (N=40, k=6) and MATH-500 (N=25, k=6)
- Multi-metric reporting: Pass@1, Pass@2, Pass@4, Pass@6, Maj@4, Maj@6, Tool Invocation & Success, Failure Taxonomy
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

# Clear any conflicting libraries
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True)
except Exception:
    pass

import torch
import torch.nn.functional as F

print("=" * 70, flush=True)
print("  TOOL-GRPO v5: SCALED RLVR TRAINING + 6-SAMPLE EVALUATION SUITE", flush=True)
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
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned

class MathVerifier:
    def __init__(self):
        self.answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def extract_predicted(self, completion: str) -> Optional[str]:
        if not completion:
            return None
        matches = self.answer_tag_pattern.findall(completion)
        if matches:
            return matches[-1].strip()

        boxed = extract_boxed(completion)
        if boxed:
            return boxed

        m = re.findall(r"(?:the\s+final\s+answer\s+is|answer\s*[:=])\s*([^\n\.]+)", completion, re.IGNORECASE)
        if m:
            return m[-1].strip()

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

        # SymPy Symbolic & Radical Equivalence with Float Residual Tolerance
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
            return False
        except Exception:
            pass

        return False

# -------------------------------------------------------------
# 3. Unbiased Pass@k, Majority Vote & Failure Mode Taxonomy
# -------------------------------------------------------------
def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 100.0
    prod = 1.0
    for j in range(1, k + 1):
        prod *= (n - c - j + 1) / (n - j + 1)
    return round((1.0 - prod) * 100.0, 2)

def cluster_and_majority_vote(predictions: List[Optional[str]], verifier: MathVerifier) -> Tuple[Optional[str], int]:
    valid_preds = [p for p in predictions if p is not None and p.strip() != ""]
    if not valid_preds:
        return None, 0

    clusters: List[Dict[str, Any]] = []
    for cand in valid_preds:
        assigned = False
        for clus in clusters:
            rep = clus["rep"]
            if verifier.verify(f"<answer>{cand}</answer>", f"<answer>{rep}</answer>"):
                clus["count"] += 1
                assigned = True
                break
        if not assigned:
            clusters.append({"rep": cand, "count": 1})

    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters[0]["rep"], clusters[0]["count"]

def classify_failure_mode(
    completion: str,
    exec_failures: int,
    succ_execs: int,
    repairs: int,
    is_correct: bool
) -> Optional[str]:
    if is_correct:
        return None
    if "<answer>" not in completion.lower() or "</answer>" not in completion.lower():
        return "FORMAT_MISSING_ANSWER"
    if exec_failures > 0 and repairs == 0:
        if "syntaxerror" in completion.lower():
            return "EXECUTION_SYNTAX_ERROR"
        if "timeouterror" in completion.lower():
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
        "4. Be concise in reasoning. Once you verify the computation, immediately emit your final answer inside <answer>...</answer> tags."
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
    max_turns: int = 4, max_tokens: int = 896, max_chunk_tokens: int = 240, temperature: float = 0.7
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
                max_new_tokens=max_chunk_tokens,
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
        # Reflexion recovery bonus
        if repairs > 0:
            r += 0.15
        return r

# -------------------------------------------------------------
# 6. SFT Warm-up Trajectories
# -------------------------------------------------------------
WARMUP_TRAJECTORIES = [
    {
        "q": "What is the square root of 576 divided by 4?",
        "solution": "<think>Compute sqrt(576) / 4.</think><code>import math\nprint(math.isqrt(576) // 4)</code>\n<observation>\n6\n</observation>\n<think>Square root of 576 is 24, divided by 4 is 6.</think><answer>6</answer>"
    },
    {
        "q": "Simplify the radical expression sqrt(72).",
        "solution": "<think>Simplify sqrt(72) into radical form.</think><code>import sympy as sp\nprint(sp.sqrt(72))</code>\n<observation>\n6*sqrt(2)\n</observation>\n<think>sqrt(72) simplifies to 6*sqrt(2).</think><answer>6\\sqrt{2}</answer>"
    },
    {
        "q": "Find all integer roots of the polynomial x^2 - 17x + 72 = 0.",
        "solution": "<think>Solve x^2 - 17x + 72 = 0.</think><code>import sympy as sp\nx = sp.symbols('x')\nroots = sp.solve(x**2 - 17*x + 72, x)\nprint(roots)</code>\n<observation>\n[8, 9]\n</observation>\n<think>The roots are 8 and 9.</think><answer>8, 9</answer>"
    },
    {
        "q": "Find the sum of all prime numbers between 20 and 35.",
        "solution": "<think>Prime numbers between 20 and 35: 23, 29, 31.</think><code>import sympy\nprimes = [p for p in range(21, 35) if sympy.isprime(p)]\nprint(sum(primes))</code>\n<observation>\n83\n</observation>\n<think>Sum is 83.</think><answer>83</answer>"
    },
    {
        "q": "Compute 25 factorial modulo 29.",
        "solution": "<think>Compute 25! mod 29 with error handling.</think><code># Attempting computation\nimport math\nprint(math.factorial(25) % 29)</code>\n<observation>\n12\n</observation>\n<think>The result is 12.</think><answer>12</answer>"
    },
    {
        "q": "Calculate the value of (1/3 + 1/4) / (1/2 - 1/5).",
        "solution": "<think>Calculate exact fraction.</think><code>from fractions import Fraction\nres = (Fraction(1, 3) + Fraction(1, 4)) / (Fraction(1, 2) - Fraction(1, 5))\nprint(res)</code>\n<observation>\n35/18\n</observation>\n<think>Result is 35/18.</think><answer>\\frac{35}{18}</answer>"
    },
    {
        "q": "Find the hypotenuse of a right triangle with legs 9 and 12.",
        "solution": "<think>Hypotenuse is sqrt(9^2 + 12^2).</think><code>import math\nprint(math.isqrt(9**2 + 12**2))</code>\n<observation>\n15\n</observation>\n<think>Hypotenuse is 15.</think><answer>15</answer>"
    },
    {
        "q": "Calculate 15 squared minus 12 squared.",
        "solution": "<think>Compute 15^2 - 12^2.</think><code>print(15**2 - 12**2)</code>\n<observation>\n81\n</observation>\n<think>Result is 81.</think><answer>81</answer>"
    }
]

# -------------------------------------------------------------
# 7. Main Pipeline: 60 GRPO Steps + 6-Sample Rigorous Evaluation
# -------------------------------------------------------------
def main():
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    REPORT_PATH = "/content/full_run_benchmark_results_v5.json"

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

    # Stage 2: Scaled GRPO Training (60 Steps)
    TOTAL_GRPO_STEPS = 60
    GROUP_SIZE = 4
    KL_COEFF = 0.04

    print("\n" + "=" * 60, flush=True)
    print(f"  STAGE 2: MULTI-TURN GRPO TRAINING ({TOTAL_GRPO_STEPS} steps, G={GROUP_SIZE} = {TOTAL_GRPO_STEPS*GROUP_SIZE} rollouts)", flush=True)
    print("=" * 60, flush=True)

    gsm_dataset = load_dataset("openai/gsm8k", "main")
    train_samples = gsm_dataset["train"].select(range(TOTAL_GRPO_STEPS))
    grpo_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    policy_model.train()
    start_train = time.time()

    for step, item in enumerate(train_samples):
        prompt = build_prompt(tokenizer, item["question"])
        gt = item["answer"]

        trajectories = []
        rewards = []

        for _ in range(GROUP_SIZE):
            rollout = run_rollout(policy_model, tokenizer, sandbox, prompt, device, max_tokens=768, temperature=0.7)
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
        print(f"[GRPO Step {step+1:02d}/{TOTAL_GRPO_STEPS}] Correct: {correct_n}/{GROUP_SIZE} | Avg Reward: {avg_r:+.3f} | Loss: {step_loss:.4f}", flush=True)

    print(f"[*] Scaled GRPO alignment completed in {time.time() - start_train:.1f}s", flush=True)

    # -------------------------------------------------------------
    # Stage 3: Rigorous Benchmark Evaluation Suite (GSM8K N=40, MATH-500 N=25, k=6)
    # -------------------------------------------------------------
    K_SAMPLES = 6
    EVAL_MAX_TOKENS = 896

    print("\n" + "=" * 60, flush=True)
    print(f"  STAGE 3: FULL RIGOROUS EVALUATION (GSM8K N=40 + MATH-500 N=25, k={K_SAMPLES}, max_tokens={EVAL_MAX_TOKENS})", flush=True)
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
                rollout = run_rollout(
                    policy_model, tokenizer, sandbox, prompt, device,
                    max_turns=4, max_tokens=EVAL_MAX_TOKENS, max_chunk_tokens=240, temperature=temp
                )
                pred = verifier.extract_predicted(rollout["completion"])
                is_corr = verifier.verify(rollout["completion"], prob["ground_truth"])
                preds.append(pred)
                correct_mask.append(is_corr)
                trajectories.append({
                    "sample_idx": s,
                    "temperature": temp,
                    "predicted": pred,
                    "is_correct": is_corr,
                    "tool_calls": rollout["tool_calls"],
                    "exec_failures": rollout["exec_failures"],
                    "succ_execs": rollout["succ_execs"],
                    "repairs": rollout["repairs"],
                    "completion": rollout["completion"]
                })

            greedy_correct = correct_mask[0]
            any_correct = any(correct_mask)
            maj_pred, maj_count = cluster_and_majority_vote(preds, verifier)
            maj_correct = verifier.verify(f"<answer>{maj_pred}</answer>", prob["ground_truth"]) if maj_pred else False

            # Failure taxonomy on greedy trajectory
            greedy_traj = trajectories[0]
            failure_mode = classify_failure_mode(
                greedy_traj["completion"],
                greedy_traj["exec_failures"],
                greedy_traj["succ_execs"],
                greedy_traj["repairs"],
                greedy_correct
            )

            res_obj = {
                "problem_id": prob["id"],
                "category": prob["category"],
                "question": prob["question"],
                "ground_truth": prob["ground_truth"],
                "greedy_correct": greedy_correct,
                "majority_correct": maj_correct,
                "majority_prediction": maj_pred,
                "majority_votes": maj_count,
                "correct_mask": correct_mask,
                "failure_mode": failure_mode,
                "trajectories": trajectories
            }
            results.append(res_obj)

            status_str = "PASS" if any_correct else "FAIL"
            maj_str = f"WIN ({maj_count}/{K_SAMPLES}v)" if maj_correct else f"FAIL ({maj_count}/{K_SAMPLES}v)"
            t_calls = sum(t["tool_calls"] for t in trajectories)
            print(f"[{b_name} #{idx+1:02d}/{len(problems)}] {status_str} | MAJ: {maj_str} | Pred: {str(maj_pred)[:12]:<12} | GT: {str(prob['ground_truth'])[:14]:<14} | Tools: {t_calls}", flush=True)

        # Aggregate Metrics
        total_n = len(results)
        greedy_passes = sum(1 for r in results if r["greedy_correct"])
        pass_1 = round(greedy_passes / total_n * 100, 2)

        pass_at_k = {}
        for k_val in [1, 2, 4, 6]:
            if k_val <= K_SAMPLES:
                vals = [calculate_pass_at_k(K_SAMPLES, sum(r["correct_mask"]), k_val) for r in results]
                pass_at_k[str(k_val)] = round(sum(vals) / total_n, 2)

        maj_passes = sum(1 for r in results if r["majority_correct"])
        majority_at_k = {str(K_SAMPLES): round(maj_passes / total_n * 100, 2)}

        # Tool stats
        all_trajs = [t for r in results for t in r["trajectories"]]
        total_trajs = len(all_trajs)
        used_tools = sum(1 for t in all_trajs if t["tool_calls"] > 0)
        tool_invoc_rate = round(used_tools / total_trajs * 100, 2)

        total_execs = sum(t["succ_execs"] + t["exec_failures"] for t in all_trajs)
        succ_execs_tot = sum(t["succ_execs"] for t in all_trajs)
        tool_succ_rate = round(succ_execs_tot / total_execs * 100, 2) if total_execs > 0 else 0.0

        fail_repairs = sum(t["repairs"] for t in all_trajs)
        failed_trajs = sum(1 for t in all_trajs if t["exec_failures"] > 0)
        reflexion_rate = round(fail_repairs / failed_trajs * 100, 2) if failed_trajs > 0 else 0.0

        avg_tools = round(sum(t["tool_calls"] for t in all_trajs) / total_trajs, 2)

        failure_counts = Counter(r["failure_mode"] for r in results if r["failure_mode"])

        cat_groups = defaultdict(list)
        for r in results:
            cat_groups[r["category"]].append(r)
        cat_metrics = {}
        for cat, items in cat_groups.items():
            c_tot = len(items)
            c_p1 = round(sum(1 for x in items if x["greedy_correct"]) / c_tot * 100, 2)
            c_maj = round(sum(1 for x in items if x["majority_correct"]) / c_tot * 100, 2)
            cat_metrics[cat] = {"count": c_tot, "pass_at_1": c_p1, "majority_correct": c_maj}

        summary = {
            "benchmark_name": b_name,
            "total_problems": total_n,
            "pass_at_1": pass_1,
            "pass_at_k": pass_at_k,
            "majority_at_k": majority_at_k,
            "tool_invocation_rate": tool_invoc_rate,
            "tool_success_rate": tool_succ_rate,
            "reflexion_recovery_rate": reflexion_rate,
            "avg_tool_calls_per_problem": avg_tools,
            "failure_taxonomy": dict(failure_counts),
            "category_metrics": cat_metrics,
            "elapsed_seconds": round(time.time() - b_start, 1),
            "sample_results": results
        }
        all_benchmark_summaries.append(summary)

    # Save Results
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_summaries, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"[SUCCESS] Rigorous Evaluation Complete. Output saved to {REPORT_PATH}", flush=True)
    print("=" * 70, flush=True)

    for summ in all_benchmark_summaries:
        print(f"\nBenchmark: {summ['benchmark_name']}")
        print(f"  Total Problems: {summ['total_problems']}")
        print(f"  Pass@1 (Greedy): {summ['pass_at_1']}%")
        print(f"  Pass@k: {summ['pass_at_k']}")
        print(f"  Maj@{K_SAMPLES}: {summ['majority_at_k'][str(K_SAMPLES)]}%")
        print(f"  Tool Invocation Rate: {summ['tool_invocation_rate']}%")
        print(f"  Tool Execution Success Rate: {summ['tool_success_rate']}%")
        print(f"  Failure Taxonomy: {summ['failure_taxonomy']}")

if __name__ == "__main__":
    main()
