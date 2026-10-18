import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from efficientlm.rl_demo import reinforce_step


def test_reinforce_increases_reward_on_toy_task():
    """Reward = fraction of even token ids. A few dozen updates should push it up."""
    torch.manual_seed(0)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=2,
                                      bos_token_id=0, eos_token_id=0))
    policy = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"],
                                             fan_in_fan_out=True, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=5e-3)
    prompt = torch.tensor([[1, 2, 3]])
    reward_fn = lambda seqs: (seqs[:, 3:] % 2 == 0).float().mean(dim=1)

    rewards = [reinforce_step(policy, opt, prompt, reward_fn, num_samples=16, new_tokens=8,
                              beta=0.0, pad_id=0)["reward"] for _ in range(60)]
    assert sum(rewards[-10:]) / 10 > sum(rewards[:10]) / 10
