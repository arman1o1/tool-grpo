"""
Tool-GRPO: Colab High-Throughput Evaluation Suite (GSM8K + MATH-500)
Runs rigorous multi-sample evaluation (Pass@1, Pass@k, Maj@k) with interleaved tool execution on GPU.
Generates comprehensive JSON reports and interactive HTML dashboard.
"""

import os
import sys
import json
import time
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional
import torch

# -------------------------------------------------------------
# 1. Isolated Python Sandbox
# -------------------------------------------------------------
import subprocess
import tempfile

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
# 2. Math Verifier (SymPy + Fractions + Balanced Boxed)
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
            if abs(p_val - g_val) < 1e-4:
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

        # SymPy Symbolic Equivalence
        try:
            import sympy
            p_sym = sympy.sympify(pred_clean)
            g_sym = sympy.sympify(gt_clean)
            diff = sympy.simplify(p_sym - g_sym)
            if diff == 0:
                return True
        except Exception:
            pass

        return False

# -------------------------------------------------------------
# 3. Estimators & Failure Taxonomy
# -------------------------------------------------------------
def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n < k or n <= 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
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
            if verifier.verify(f"<answer>{pred}</answer>", f"<answer>{c['rep']}</answer>"):
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
# 4. Prompt Builder & Interleaved Rollout
# -------------------------------------------------------------
def build_prompt(tokenizer, question: str) -> str:
    system_text = (
        "You are an expert mathematical reasoning assistant with access to a Python execution tool.\n"
        "Guidelines:\n"
        "1. Write clear reasoning inside <think>...</think> tags.\n"
        "2. To run Python, emit <code>...</code>. The environment will return <observation>...</observation>.\n"
        "3. Provide your final definitive answer inside <answer>...</answer> tags."
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

def run_rollout(model, tokenizer, sandbox, prompt: str, device: str, temperature: float = 0.7, max_turns: int = 3):
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    curr_text = prompt
    tool_calls = 0
    exec_failures = 0
    succ_execs = 0
    repairs = 0
    had_failure = False

    for _ in range(max_turns):
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

    return {
        "text": curr_text,
        "completion": curr_text[len(prompt):],
        "tool_calls": tool_calls,
        "exec_failures": exec_failures,
        "succ_execs": succ_execs,
        "repairs": repairs,
        "tokens": len(full_tokens),
    }

# -------------------------------------------------------------
# 5. Main Benchmark Runner
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/content/tool_grpo_v2_model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--gsm8k_samples", type=int, default=50)
    parser.add_argument("--math_samples", type=int, default=25)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--output_json", type=str, default="/content/rigorous_eval_report.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from datasets import load_dataset

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"  TOOL-GRPO RIGOROUS BENCHMARK EVAL SUITE (Device: {device})", flush=True)
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading model: {args.base_model}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )

    if os.path.exists(args.model_path):
        print(f"[*] Applying fine-tuned LoRA adapter from: {args.model_path}...", flush=True)
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        print("[*] No adapter path found. Evaluating base model.", flush=True)
        model = base_model

    model.eval()
    sandbox = PythonSandbox()
    verifier = MathVerifier()

    benchmarks = {}
    if args.gsm8k_samples > 0:
        print(f"[*] Loading GSM8K test split ({args.gsm8k_samples} samples)...", flush=True)
        gsm_ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.gsm8k_samples))
        benchmarks["GSM8K"] = [
            {"id": f"gsm8k-{i+1:03d}", "question": r["question"], "ground_truth": r["answer"], "category": "GSM8K"}
            for i, r in enumerate(gsm_ds)
        ]

    if args.math_samples > 0:
        print(f"[*] Loading MATH-500 test split ({args.math_samples} samples)...", flush=True)
        math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(args.math_samples))
        benchmarks["MATH-500"] = [
            {"id": f"math500-{i+1:03d}", "question": r["problem"], "ground_truth": r["answer"], "category": r.get("subject", "MATH")}
            for i, r in enumerate(math_ds)
        ]

    all_summaries = []

    for b_name, problems in benchmarks.items():
        print(f"\n=======================================================", flush=True)
        print(f"  BENCHMARK: {b_name} (N={len(problems)}, k={args.k_samples} rollouts/prob)", flush=True)
        print(f"=======================================================", flush=True)

        sample_results = []
        start_time = time.time()

        for idx, prob in enumerate(problems):
            prompt = build_prompt(tokenizer, prob["question"])
            trajectories = []
            preds = []
            correct_mask = []

            for s in range(args.k_samples):
                # Sample 0 is greedy (temp=0.0) for pure Pass@1; samples 1..k use temp=0.7
                temp = 0.0 if s == 0 else 0.7
                rollout = run_rollout(model, tokenizer, sandbox, prompt, device, temperature=temp)
                trajectories.append(rollout)
                pred = verifier.extract_predicted(rollout["completion"])
                is_corr = verifier.verify(rollout["completion"], prob["ground_truth"])
                preds.append(pred)
                correct_mask.append(is_corr)

            maj_winner, maj_corr, votes = cluster_and_majority_vote(preds, verifier, prob["ground_truth"])
            fail_mode = None
            if not maj_corr and trajectories:
                fail_mode = classify_failure(
                    trajectories[0]["completion"], prob["ground_truth"], verifier,
                    trajectories[0]["exec_failures"], trajectories[0]["succ_execs"]
                )

            tot_tools = sum(t["tool_calls"] for t in trajectories)
            tot_fail = sum(t["exec_failures"] for t in trajectories)
            tot_succ = sum(t["succ_execs"] for t in trajectories)
            tot_repairs = sum(t["repairs"] for t in trajectories)

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
                "tool_calls": tot_tools,
                "exec_failures": tot_fail,
                "succ_execs": tot_succ,
                "repairs": tot_repairs,
                "trajectories": trajectories,
            }
            sample_results.append(res)

            status_str = "PASS" if correct_mask[0] else "FAIL"
            maj_str = f"MAJ: {'WIN' if maj_corr else 'FAIL'} ({votes}/{args.k_samples}v)"
            print(f"[{b_name} #{idx+1:03d}/{len(problems)}] {status_str} | {maj_str} | Pred: {str(maj_winner):8s} | GT: {str(res['ground_truth']):8s} | Tools: {tot_tools}", flush=True)

        elapsed = time.time() - start_time
        n_p = len(sample_results)
        p1 = round(sum(1 for r in sample_results if r["correct_mask"][0]) / n_p * 100.0, 2)
        maj_p = round(sum(1 for r in sample_results if r["majority_correct"]) / n_p * 100.0, 2)

        p_k = {}
        for k in [1, 2, args.k_samples]:
            if k <= args.k_samples:
                scores = [calculate_pass_at_k(args.k_samples, sum(1 for c in r["correct_mask"] if c), k) for r in sample_results]
                p_k[k] = round(sum(scores) / len(scores), 2)

        tot_tools = sum(r["tool_calls"] for r in sample_results)
        tot_succ = sum(r["succ_execs"] for r in sample_results)
        tot_fail = sum(r["exec_failures"] for r in sample_results)
        tot_execs = tot_succ + tot_fail
        tool_succ_rate = round(tot_succ / tot_execs * 100.0, 2) if tot_execs > 0 else 0.0

        b_summary = {
            "benchmark_name": b_name,
            "total_problems": n_p,
            "pass_at_1": p1,
            "pass_at_k": p_k,
            "majority_at_k": {args.k_samples: maj_p},
            "tool_invocation_rate": round(sum(1 for r in sample_results if r["tool_calls"] > 0) / n_p * 100.0, 2),
            "tool_success_rate": tool_succ_rate,
            "avg_tool_calls_per_problem": round(tot_tools / (n_p * args.k_samples), 2),
            "elapsed_seconds": round(elapsed, 1),
            "sample_results": sample_results,
        }
        all_summaries.append(b_summary)

        print(f"\n[{b_name} Final Summary]", flush=True)
        print(f"  Pass@1 (Greedy)       : {p1}%", flush=True)
        print(f"  Pass@{args.k_samples} (Estimator)   : {p_k.get(args.k_samples, p1)}%", flush=True)
        print(f"  Maj@{args.k_samples} (Self-Consist) : {maj_p}%", flush=True)
        print(f"  Tool Invocations      : {b_summary['tool_invocation_rate']}%", flush=True)
        print(f"  Tool Success Rate     : {tool_succ_rate}%", flush=True)
        print(f"  Elapsed Time          : {elapsed:.1f}s", flush=True)

    with open(args.output_json, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n[*] Full report successfully saved to: {args.output_json}", flush=True)

if __name__ == "__main__":
    main()
