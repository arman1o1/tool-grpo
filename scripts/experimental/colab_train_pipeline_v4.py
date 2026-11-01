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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

print("=" * 70, flush=True)
print("  TOOL-GRPO v4: 1-SHOT INTERLEAVED REASONING WITH STRICT COMPLETION SLICING", flush=True)
print("=" * 70, flush=True)

# -------------------------------------------------------------
# 1. Isolated Python Sandbox
# -------------------------------------------------------------
class PythonSandbox:
    def __init__(self, timeout: float = 3.0, max_chars: int = 400):
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
                output = out if out else "[Executed successfully with no stdout]"
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
# 2. Math Verifier & Reward Evaluator (Evaluates ONLY Completion)
# -------------------------------------------------------------
class MathVerifier:
    def __init__(self):
        self.answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")

    def extract_predicted(self, completion: str) -> Optional[str]:
        # Strictly look within model generation
        matches = self.answer_tag_pattern.findall(completion)
        if matches:
            return matches[-1].strip()
        boxed = self.boxed_pattern.findall(completion)
        if boxed:
            return boxed[-1].strip()
        
        # Fallback: extract last number in the completion
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
        return numbers[-1] if numbers else None

    def extract_gt(self, solution: str) -> str:
        if "####" in solution:
            return solution.split("####")[-1].strip()
        boxed = self.boxed_pattern.findall(solution)
        if boxed:
            return boxed[-1].strip()
        return solution.strip()

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
            return False

class CompositeRewardEvaluator:
    def __init__(self, verifier: MathVerifier):
        self.verifier = verifier

    def evaluate(self, completion: str, ground_truth: str, exec_failures: int, token_len: int) -> Dict[str, float]:
        is_correct = self.verifier.verify(completion, ground_truth)
        r_outcome = 1.0 if is_correct else 0.0

        # Format reward: check if model produced <answer>
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
        r_format = 0.1 if has_answer else -0.1

        # Execution penalty for broken code
        r_exec = -0.1 * min(exec_failures, 2)

        # Mild length penalty
        r_length = -0.05 * (token_len / 384.0)

        total = r_outcome + r_format + r_exec + r_length
        return {
            "total_reward": total,
            "r_outcome": r_outcome,
            "r_format": r_format,
            "r_exec": r_exec,
            "is_correct": 1.0 if is_correct else 0.0
        }

# -------------------------------------------------------------
# 3. Model & Tokenizer Initialization
# -------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "/content/tool_grpo_v4_release"
os.makedirs(SAVE_DIR, exist_ok=True)
progress_file = f"{SAVE_DIR}/live_progress.json"

print(f"[*] Loading tokenizer for {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[*] Loading base model {MODEL_ID} in bfloat16 on {DEVICE}...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True
)

print("[*] Creating reference frozen copy for KL penalty...", flush=True)
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

print("[*] Applying LoRA configuration to policy model...", flush=True)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
policy_model = get_peft_model(base_model, lora_config)
policy_model.print_trainable_parameters()

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=2e-5)
sandbox = PythonSandbox(timeout=3.0)
verifier = MathVerifier()
reward_evaluator = CompositeRewardEvaluator(verifier=verifier)

# 1-Shot Exemplar in System Prompt
def build_prompt(question: str) -> str:
    system_content = (
        "You are an expert mathematical problem solver with access to a Python execution environment.\n"
        "Guidelines:\n"
        "1. Break down the reasoning in <think>...</think>.\n"
        "2. To calculate or verify equations, write executable Python code in <code>...</code> blocks.\n"
        "3. You will receive the output in <observation>...</observation>.\n"
        "4. Enclose your final numerical result inside <answer>...</answer> tags.\n\n"
        "Example Dialogue:\n"
        "User: A florist has 24 roses and gives 8 to a customer. How many roses are left?\n"
        "Assistant: <think>Roses left: 24 - 8.</think><code>print(24 - 8)</code>\n"
        "<observation>\n16\n</observation>\n"
        "<think>Calculation verified.</think><answer>16</answer>"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Solve this problem:\n{question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -------------------------------------------------------------
# 4. Multi-Turn Rollout Engine
# -------------------------------------------------------------
def run_interleaved_rollout(
    model,
    prompt: str,
    max_turns: int = 2,
    max_tokens: int = 350,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    curr_text = prompt
    full_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    loss_mask = [0] * len(full_tokens)

    turns = 0
    exec_failures = 0
    tool_calls = 0

    code_stop_tag = "</code>"
    ans_stop_tag = "</answer>"

    while turns < max_turns and len(full_tokens) < max_tokens:
        turns += 1
        input_ids = torch.tensor([full_tokens], device=DEVICE)
        
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
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

        if ans_stop_tag in chunk_str:
            break

        if code_stop_tag in chunk_str:
            tool_calls += 1
            code_matches = re.findall(r"<code>(.*?)</code>", curr_text[len(prompt):], re.DOTALL)
            if code_matches:
                code_to_run = code_matches[-1].strip()
                ok, output = sandbox.execute(code_to_run)
                if not ok:
                    exec_failures += 1
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
    }

# -------------------------------------------------------------
# 5. Dataset & Benchmark Evaluation
# -------------------------------------------------------------
print("[*] Loading GSM8K benchmark split...", flush=True)
dataset = load_dataset("openai/gsm8k", "main")
train_samples = dataset["train"].select(range(20))
eval_samples = dataset["test"].select(range(12))

def evaluate_benchmark(model, split, label="Eval"):
    print(f"\n--- Running {label} on {len(split)} problems ---", flush=True)
    correct = 0
    total = len(split)
    tool_usage = 0

    for i, item in enumerate(split):
        prompt = build_prompt(item["question"])
        res = run_interleaved_rollout(model, prompt, max_turns=2, temperature=0.0)
        
        # Verify strictly on the generated completion!
        is_corr = verifier.verify(res["completion"], item["answer"])
        if is_corr:
            correct += 1
        tool_usage += res["tool_calls"]
        ans_pred = verifier.extract_predicted(res["completion"])
        ans_gt = verifier.extract_gt(item["answer"])
        print(f"[{label} #{i+1:02d}/{total}] Correct: {str(is_corr):5s} | Pred: {str(ans_pred):12s} | GT: {str(ans_gt):10s} | Tools: {res['tool_calls']}", flush=True)
        torch.cuda.empty_cache()

    pass_1 = (correct / total) * 100
    print(f"[{label} Complete] Pass@1: {pass_1:.1f}% | Avg Tools: {tool_usage/total:.2f}", flush=True)
    return pass_1

# Baseline Evaluation
print("\n" + "=" * 50, flush=True)
print("  STEP 1: BASELINE EVALUATION (Before GRPO)", flush=True)
print("=" * 50, flush=True)
baseline_pass_1 = evaluate_benchmark(policy_model, eval_samples, label="Baseline")

# -------------------------------------------------------------
# 6. GRPO Training Loop
# -------------------------------------------------------------
print("\n" + "=" * 50, flush=True)
print("  STEP 2: TRAINING WITH GRPO (Group Size G=4)", flush=True)
print("=" * 50, flush=True)

GROUP_SIZE = 4
KL_COEFF = 0.04
training_history = []

policy_model.train()
start_time = time.time()

for step, item in enumerate(train_samples):
    prompt = build_prompt(item["question"])
    ground_truth = item["answer"]

    trajectories = []
    rewards = []
    for g in range(GROUP_SIZE):
        rollout = run_interleaved_rollout(policy_model, prompt, max_turns=2, temperature=0.7)
        # Evaluate reward strictly on the generated completion!
        score_dict = reward_evaluator.evaluate(
            rollout["completion"],
            ground_truth,
            rollout["exec_failures"],
            len(rollout["tokens"])
        )
        trajectories.append(rollout)
        rewards.append(score_dict["total_reward"])

    # Advantage normalization
    r_tensor = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
    std = r_tensor.std(unbiased=False)
    mean = r_tensor.mean()
    adv = (r_tensor - mean) / (std + 1e-4) if std > 1e-4 else (r_tensor - mean)

    optimizer.zero_grad()
    step_loss = 0.0
    step_kl = 0.0

    for i in range(GROUP_SIZE):
        traj = trajectories[i]
        token_ids = torch.tensor([traj["tokens"]], device=DEVICE)
        mask = torch.tensor([traj["loss_mask"]], device=DEVICE, dtype=torch.float32)

        if mask.sum() < 1:
            continue

        logits = policy_model(token_ids).logits[:, :-1, :]
        target_ids = token_ids[:, 1:]
        mask_shifted = mask[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_logits = ref_model(token_ids).logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1).gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        log_ratio = ref_log_probs - token_log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1.0

        policy_token_loss = -token_log_probs * adv[i]
        total_token_loss = policy_token_loss + KL_COEFF * kl

        traj_loss = (total_token_loss * mask_shifted).sum() / (mask_shifted.sum() + 1e-6)
        traj_loss = traj_loss / GROUP_SIZE
        traj_loss.backward()

        step_loss += traj_loss.item()
        step_kl += (kl * mask_shifted).sum().item() / (mask_shifted.sum().item() + 1e-6)

    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.empty_cache()

    avg_r = sum(rewards) / len(rewards)
    correct_count = sum(1 for r in rewards if r >= 0.9)
    print(
        f"[Step {step+1:02d}/{len(train_samples)}] Correct Rollouts: {correct_count}/{GROUP_SIZE} | "
        f"Avg Reward: {avg_r:+.3f} | Loss: {step_loss:.4f} | KL: {step_kl/GROUP_SIZE:.4f}",
        flush=True
    )
    
    step_record = {
        "step": step + 1,
        "correct_rollouts": correct_count,
        "avg_reward": round(avg_r, 4),
        "loss": round(step_loss, 4),
        "kl": round(step_kl / GROUP_SIZE, 4)
    }
    training_history.append(step_record)
    with open(progress_file, "w") as f:
        json.dump(training_history, f, indent=2)

    # Checkpoint every 5 steps
    if (step + 1) % 5 == 0:
        policy_model.save_pretrained(f"{SAVE_DIR}/checkpoint_step_{step+1}")
        print(f"[*] Checkpoint saved at step {step+1}", flush=True)

training_duration = time.time() - start_time
print(f"[*] Training finished in {training_duration:.1f}s ({training_duration/60:.1f} min)", flush=True)

# -------------------------------------------------------------
# 7. Post-Training Evaluation
# -------------------------------------------------------------
print("\n" + "=" * 50, flush=True)
print("  STEP 3: POST-TRAINING EVALUATION (After Tool-GRPO)", flush=True)
print("=" * 50, flush=True)
policy_model.eval()
post_pass_1 = evaluate_benchmark(policy_model, eval_samples, label="Post-GRPO")

# -------------------------------------------------------------
# 8. Save Final Artifacts
# -------------------------------------------------------------
print(f"[*] Saving final LoRA adapter to {SAVE_DIR}/lora_weights...", flush=True)
policy_model.save_pretrained(f"{SAVE_DIR}/lora_weights")
tokenizer.save_pretrained(f"{SAVE_DIR}/lora_weights")

report = {
    "model": MODEL_ID,
    "hardware": "NVIDIA A100-SXM4-40GB",
    "method": "Multi-Turn Interleaved Tool-GRPO (RLVR)",
    "baseline_pass_1": baseline_pass_1,
    "post_trained_pass_1": post_pass_1,
    "pass_1_delta": round(post_pass_1 - baseline_pass_1, 2),
    "training_duration_seconds": round(training_duration, 1),
    "training_history": training_history,
}

report_path = f"{SAVE_DIR}/benchmark_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60, flush=True)
print("             EXPERIMENT SUMMARY REPORT", flush=True)
print("=" * 60, flush=True)
print(f"  Model Architecture   : {MODEL_ID}", flush=True)
print(f"  Baseline Pass@1      : {baseline_pass_1:.1f}%", flush=True)
print(f"  Post-GRPO Pass@1     : {post_pass_1:.1f}%", flush=True)
print(f"  Accuracy Delta (Δ)   : {post_pass_1 - baseline_pass_1:+.1f}%", flush=True)
print(f"  Report saved to      : {report_path}", flush=True)
print("=" * 60, flush=True)
print("[+] Execution finished successfully.", flush=True)
