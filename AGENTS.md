# Tool-GRPO Project Guidelines

## Rollout Token Budgeting
- **Minimum Token Ceiling:** Multi-turn tool reasoning requires at least 896 tokens per rollout (`max_tokens >= 896`). Setting lower limits (e.g. 400 tokens) causes severe false-negative evaluations (`FORMAT_MISSING_ANSWER`) due to premature truncation before `</answer>`.
- **Chunk Sizing:** Allow 200–256 tokens per turn chunk to accommodate step-by-step thinking and sandbox code blocks.

## RLVR Alignment & Tool-Use Aversion Guardrails
- **Curriculum Mixture:** Avoid training exclusively on simple arithmetic (e.g., pure GSM8K) for >30 steps. When models can solve problems in raw text, positive reinforcement on text-only trajectories coupled with runtime execution penalties causes **tool-use aversion collapse** (unlearning tool calls).
- **Domain Interleaving:** Always mix complex symbolic problems (MATH-500) where text-only generation is guaranteed to fail, or enforce a minimum tool invocation reward floor during training.
- **Verification Tolerances:** Numerical evaluation of symbolic radicals and fractions must include a floating-point residual tolerance (`abs(float(diff)) < 1e-4`) in addition to symbolic zero simplification.
