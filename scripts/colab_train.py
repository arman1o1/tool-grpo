"""
Tool-GRPO: Colab / Cloud GPU Training Pipeline
Trains Qwen2.5-Coder-1.5B with Group Relative Policy Optimization (GRPO)
for interleaved Python scratchpad code execution during mathematical reasoning.
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

# Ensure project src is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.sandbox import PythonSandbox
from src.verifier import MathVerifier
from src.rewards import CompositeRewardEvaluator
from src.rollout import MultiTurnRolloutEngine

def build_interleaved_prompt(tokenizer, question: str) -> str:
    """Formats mathematical problem with a 1-shot exemplar for interleaved code reasoning."""
    system_prompt = (
        "You are an expert mathematical problem solver with access to an isolated Python code execution sandbox.\n"
        "Guidelines:\n"
        "1. Deconstruct the problem inside <think>...</think> tags.\n"
        "2. When computation, formula simplification, or algebraic evaluation is needed, write executable Python code inside <code>...</code> blocks.\n"
        "3. The environment will return stdout in <observation>...</observation>.\n"
        "4. Enclose your final numerical or simplified symbolic answer strictly inside <answer>...</answer> tags.\n\n"
        "Example Dialogue:\n"
        "User: A florist has 48 roses. She sells half to a customer and then gives 6 to her assistant. How many remain?\n"
        "Assistant: <think>Calculate remaining roses: 48 / 2 - 6.</think><code>roses = 48 // 2 - 6\nprint(roses)</code>\n"
        "<observation>\n18\n</observation>\n"
        "<think>Verification confirmed.</think><answer>18</answer>"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Solve this math problem step-by-step:\n{question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def train_colab(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    group_size: int = 4,
    learning_rate: float = 2e-5,
    kl_coeff: float = 0.04,
    max_steps: int = 20,
    save_dir: str = "./checkpoints/tool-grpo-release",
):
    print("=" * 70)
    print("  TOOL-GRPO: CLOUD MULTI-TURN RLVR PIPELINE")
    print(f"  Model:      {model_name}")
    print(f"  Group Size: {group_size}")
    print(f"  Save Dir:   {save_dir}")
    print("=" * 70)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError:
        print("[-] Required packages missing. Run: pip install -r requirements.txt")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    print(f"[*] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading base model {model_name} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    print("[*] Instantiating frozen reference model for KL penalty...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    policy_model = get_peft_model(base_model, lora_config)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    sandbox = PythonSandbox(timeout_seconds=3.0)
    verifier = MathVerifier()
    evaluator = CompositeRewardEvaluator(verifier=verifier)
    rollout_engine = MultiTurnRolloutEngine(sandbox=sandbox, max_turns=3)

    print("[*] Loading GSM8K benchmark dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_samples = dataset["train"].select(range(min(max_steps, len(dataset["train"]))))

    training_history = []
    policy_model.train()
    start_time = time.time()

    print("\n[*] Starting GRPO Training Loop...")
    for step, item in enumerate(train_samples):
        prompt = build_interleaved_prompt(tokenizer, item["question"])
        gt = item["answer"]

        trajectories = []
        rewards = []

        for g in range(group_size):
            rollout = rollout_engine.rollout_with_model(
                policy_model,
                tokenizer,
                prompt,
                device=device,
                temperature=0.7,
            )
            score_dict = evaluator.evaluate(
                rollout["completion"],
                gt,
                execution_failures=rollout["execution_failures"],
                successful_executions=rollout["successful_executions"],
                token_count=len(rollout["tokens"])
            )
            trajectories.append(rollout)
            rewards.append(score_dict["total_reward"])

        # Group Advantage Normalization
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = r_tensor.std(unbiased=False)
        mean = r_tensor.mean()
        adv = (r_tensor - mean) / (std + 1e-4) if std > 1e-4 else (r_tensor - mean)

        optimizer.zero_grad()
        step_loss = 0.0
        step_kl = 0.0

        for i in range(group_size):
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
            total_loss = (policy_token_loss + kl_coeff * kl) * mask_shifted
            traj_loss = total_loss.sum() / (mask_shifted.sum() + 1e-6) / group_size
            traj_loss.backward()

            step_loss += traj_loss.item()
            step_kl += (kl * mask_shifted).sum().item() / (mask_shifted.sum().item() + 1e-6)

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

        avg_r = sum(rewards) / len(rewards)
        print(f"[Step {step+1:02d}/{len(train_samples)}] Avg Reward: {avg_r:+.3f} | Loss: {step_loss:.4f} | KL: {step_kl/group_size:.4f}")
        training_history.append({
            "step": step + 1,
            "avg_reward": round(avg_r, 4),
            "loss": round(step_loss, 4),
            "kl": round(step_kl / group_size, 4),
        })

    # Save final artifacts
    policy_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(PROJECT_ROOT, "results", "live_progress.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n[+] Training complete in {time.time() - start_time:.1f}s. Artifacts saved to {save_dir}")

if __name__ == "__main__":
    train_colab()
