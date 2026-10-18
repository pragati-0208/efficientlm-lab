"""Tiny RLHF-style demo (optional / 'desirable' skill): steer a language model with RL.

  policy        : distilgpt2 + LoRA adapters (via `peft`)
  reward model  : our fine-tuned DistilBERT sentiment classifier -> P(positive)
  algorithm     : REINFORCE with a mean baseline and a KL penalty to the frozen base model
                  (the base model = the same network with adapters switched off)

Honest note: this is the simplest policy-gradient recipe, not PPO. It is meant to show the
mechanics of reward -> advantage -> policy update -> KL regularisation. Expect noisy curves.

    python -m efficientlm.rl_demo --clf_ckpt results/full_ft/seed0/model.pt --steps 150
"""
import argparse
import random

import torch


def sequence_logprobs(model, seqs, prompt_len):
    """Log-prob of each generated token given everything before it -> (B, new_tokens)."""
    logits = model(input_ids=seqs).logits[:, prompt_len - 1:-1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    tokens = seqs[:, prompt_len:]
    return logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def reinforce_step(policy, optimizer, prompt_ids, reward_fn, num_samples=8, new_tokens=20,
                   beta=0.05, pad_id=0):
    """One REINFORCE update. `policy` must be a PEFT model (needs .disable_adapter()).
    reward_fn: LongTensor (B, L_total) -> FloatTensor (B,)"""
    policy.eval()  # no dropout noise: sampling and re-scoring use the same distribution
    prompt_len = prompt_ids.size(1)
    with torch.no_grad():
        seqs = policy.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), do_sample=True, top_k=0,
            max_new_tokens=new_tokens, min_new_tokens=new_tokens, num_return_sequences=num_samples,
            pad_token_id=pad_id,
        )
        rewards = reward_fn(seqs).float()
        with policy.disable_adapter():
            ref_lp = sequence_logprobs(policy, seqs, prompt_len)

    lp = sequence_logprobs(policy, seqs, prompt_len)  # with gradient
    kl = (lp.detach() - ref_lp).sum(dim=1)  # sample estimate of KL(policy || base)
    shaped = rewards - beta * kl
    adv = shaped - shaped.mean()  # baseline = batch mean -> lower variance
    loss = -(adv.detach() * lp.sum(dim=1)).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return {"reward": rewards.mean().item(), "kl": kl.mean().item(), "loss": loss.item()}


PROMPTS = [
    "The movie was", "I think this film is", "Honestly, the acting felt", "After watching it, I",
    "The story of this film", "My overall opinion:", "The director's latest work is",
]


def main():
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--clf_ckpt", required=True, help="state_dict of the fine-tuned DistilBERT classifier")
    ap.add_argument("--clf_name", default="distilbert-base-uncased")
    ap.add_argument("--policy_name", default="distilgpt2")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/rl_reward_curve.png")
    args = ap.parse_args()

    from .utils import get_device, set_seed

    set_seed(args.seed)
    device = get_device()

    clf_tok = AutoTokenizer.from_pretrained(args.clf_name)
    clf = AutoModelForSequenceClassification.from_pretrained(args.clf_name, num_labels=2)
    clf.load_state_dict(torch.load(args.clf_ckpt, map_location="cpu"))
    clf.to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.policy_name)
    base = AutoModelForCausalLM.from_pretrained(args.policy_name)
    policy = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                                             fan_in_fan_out=True, lora_dropout=0.0, task_type="CAUSAL_LM"))
    policy.to(device)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    @torch.no_grad()
    def reward_fn(seqs):
        texts = tok.batch_decode(seqs, skip_special_tokens=True)
        enc = clf_tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt",
                      return_token_type_ids=False).to(device)
        return torch.softmax(clf(**enc).logits, dim=-1)[:, 1]

    curve = []
    for step in range(1, args.steps + 1):
        prompt = tok(random.choice(PROMPTS), return_tensors="pt").input_ids.to(device)
        s = reinforce_step(policy, opt, prompt, reward_fn, beta=args.beta, pad_id=tok.eos_token_id)
        curve.append(s["reward"])
        if step % 10 == 0:
            recent = sum(curve[-10:]) / 10
            print(f"step {step:4d}  reward(avg last 10)={recent:.3f}  kl={s['kl']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3.5))
        plt.plot(curve, alpha=0.4, label="per step")
        k = 10
        smooth = [sum(curve[max(0, i - k + 1):i + 1]) / len(curve[max(0, i - k + 1):i + 1]) for i in range(len(curve))]
        plt.plot(smooth, label=f"moving avg ({k})")
        plt.xlabel("step")
        plt.ylabel("mean P(positive)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print("saved", args.out)
    except ImportError:
        pass

    prompt = tok("The movie was", return_tensors="pt").input_ids.to(device)
    out = policy.generate(input_ids=prompt, max_new_tokens=25, do_sample=True, top_k=0,
                          pad_token_id=tok.eos_token_id)
    print("sample:", tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
