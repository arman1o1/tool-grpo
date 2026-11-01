import pytest
import torch
from train import compute_group_advantages, compute_grpo_step_loss

def test_advantage_normalization_zero_variance():
    rewards = [1.0, 1.0, 1.0, 1.0]
    adv = compute_group_advantages(rewards)
    assert torch.allclose(adv, torch.zeros(4))

def test_advantage_normalization_spread():
    rewards = [2.0, 0.0]
    adv = compute_group_advantages(rewards)
    # Mean is 1.0, std is 1.0 -> [1.0, -1.0]
    assert adv[0] > 0
    assert adv[1] < 0
    assert torch.isclose(adv.mean(), torch.tensor(0.0), atol=1e-5)

def test_grpo_loss_backward_gradient():
    vocab_size = 50
    seq_len = 10
    policy_logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    ref_logits = torch.randn(1, seq_len, vocab_size)
    token_ids = torch.randint(0, vocab_size, (1, seq_len))
    
    # Mask out first 3 tokens (prompt)
    loss_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)

    loss_dict = compute_grpo_step_loss(
        policy_logits=policy_logits,
        ref_logits=ref_logits,
        token_ids=token_ids,
        loss_mask=loss_mask,
        advantage=1.5,
        kl_coeff=0.04,
    )

    loss = loss_dict["loss"]
    assert loss.requires_grad
    loss.backward()

    assert policy_logits.grad is not None
    # Gradients on the first 2 target tokens (which correspond to masked prompt) should be zero
    assert torch.all(policy_logits.grad[:, :2, :] == 0.0)
