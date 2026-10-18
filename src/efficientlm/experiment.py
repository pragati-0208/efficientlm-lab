"""Run controlled experiments.

Methods
  full_ft        fine-tune all of DistilBERT (also serves as the KD teacher)
  lora_r<R>      our from-scratch LoRA, rank R, on the attention q/v projections
  lora_peft_r<R> the same recipe with the HuggingFace `peft` library (cross-check)
  student_ce     small BERT (4 layers, H=256) trained with the normal CE loss
  student_kd     the same small BERT distilled from the full_ft teacher

Controls: identical data split (data_seed), identical epochs/batch size, several model seeds,
results saved with config + git hash + environment so every number can be traced back.
"""
import argparse
import json
import os

import torch

from .config import Config
from .data import load_sst2, make_collate, make_loaders, make_synthetic, stack_collate
from .distill import DistillTrainer
from .lora import apply_lora, merge_lora, save_adapter
from .summarize import aggregate, load_runs, to_markdown
from .trainer import Trainer, evaluate_model, measure_latency
from .utils import count_params, env_info, get_device, git_hash, model_size_mb, save_json, set_seed, timer

SMOKE_VOCAB = 100


def build_model(name, kind, smoke=False, num_labels=2):
    """kind: 'teacher' (DistilBERT-like) or 'student' (BERT-like)."""
    if smoke:  # tiny random-init models: no downloads needed
        from transformers import BertConfig, BertForSequenceClassification
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        if kind == "teacher":
            return DistilBertForSequenceClassification(DistilBertConfig(
                vocab_size=SMOKE_VOCAB, dim=64, n_layers=2, n_heads=2, hidden_dim=128,
                max_position_embeddings=64, num_labels=num_labels))
        return BertForSequenceClassification(BertConfig(
            vocab_size=SMOKE_VOCAB, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, max_position_embeddings=64, num_labels=num_labels))
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)


def apply_peft_lora(model, r, alpha, dropout, targets):
    from peft import LoraConfig, TaskType, get_peft_model

    conf = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout,
                      target_modules=list(targets), modules_to_save=["pre_classifier", "classifier"])
    return get_peft_model(model, conf)


def get_loaders(cfg, smoke, seed):
    if smoke:
        tr, dv, te = make_synthetic(SMOKE_VOCAB, seed=cfg.data_seed)
        return make_loaders(tr, dv, te, stack_collate, cfg.batch_size, seed)
    tr, dv, te, tok = load_sst2(cfg.max_len, cfg.train_size, cfg.dev_size, cfg.data_seed, cfg.teacher_model)
    return make_loaders(tr, dv, te, make_collate(tok), cfg.batch_size, seed)


def load_teacher(cfg, seed, smoke, log):
    path = os.path.join(cfg.out_dir, "full_ft", f"seed{seed}", "model.pt")
    if not os.path.exists(path):
        meta = os.path.join(cfg.out_dir, "full_ft", f"seed{seed}", "metrics.json")
        if os.path.exists(meta):
            os.remove(meta)  # metrics without a checkpoint are useless for KD -> retrain
        log(f"  teacher checkpoint missing -> training full_ft (seed {seed}) first")
        run_one("full_ft", seed, cfg, smoke, log)
    teacher = build_model(cfg.teacher_model, "teacher", smoke)
    teacher.load_state_dict(torch.load(path, map_location="cpu"))
    return teacher


def run_one(method, seed, cfg, smoke=False, log=print):
    run_dir = os.path.join(cfg.out_dir, method, f"seed{seed}")
    meta_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(meta_path):
        log(f"[skip] {method} seed {seed} already done")
        with open(meta_path) as f:
            return json.load(f)
    os.makedirs(run_dir, exist_ok=True)
    log(f"[run] {method} seed {seed}")

    set_seed(seed)
    device = get_device()
    train, dev, test = get_loaders(cfg, smoke, seed)
    common = dict(device=device, epochs=cfg.epochs, weight_decay=cfg.weight_decay,
                  warmup_ratio=cfg.warmup_ratio, grad_clip=cfg.grad_clip, amp=cfg.amp, log=log)
    is_own_lora = method.startswith("lora_r")
    adapter_kb = None

    if method == "full_ft":
        model = build_model(cfg.teacher_model, "teacher", smoke)
        trainer = Trainer(model, train, dev, lr=cfg.lr_full, **common)
    elif method.startswith("lora"):
        r = int(method.split("_r")[-1])
        model = build_model(cfg.teacher_model, "teacher", smoke)
        alpha = cfg.lora_alpha_factor * r
        if is_own_lora:
            model = apply_lora(model, cfg.lora_targets, r, alpha, cfg.lora_dropout)
        else:
            model = apply_peft_lora(model, r, alpha, cfg.lora_dropout, cfg.lora_targets)
        trainer = Trainer(model, train, dev, lr=cfg.lr_lora, **common)
    elif method == "student_ce":
        model = build_model(cfg.student_model, "student", smoke)
        trainer = Trainer(model, train, dev, lr=cfg.lr_student, **common)
    elif method == "student_kd":
        model = build_model(cfg.student_model, "student", smoke)
        teacher = load_teacher(cfg, seed, smoke, log)
        if teacher.config.vocab_size != model.config.vocab_size:
            raise ValueError("teacher and student must share a vocabulary (same tokenised inputs)")
        trainer = DistillTrainer(model, train, dev, lr=cfg.lr_student, teacher=teacher,
                                 temperature=cfg.kd_temperature, alpha=cfg.kd_alpha, **common)
    else:
        raise ValueError(f"unknown method '{method}'")

    params = count_params(model)
    log(f"  trainable {params['trainable_params']:,} / {params['total_params']:,} "
        f"({params['trainable_pct']:.2f}%)")

    with timer() as t:
        history = trainer.fit()
    test_m = evaluate_model(model, test, device)
    example = next(iter(test))
    latency = measure_latency(model, example, device)

    if method == "full_ft" or method.startswith("student"):
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(run_dir, "model.pt"))
    if method.startswith("lora"):
        adapter_kb = save_adapter(model, os.path.join(run_dir, "adapter.pt")) / 1024

    extra = {}
    if is_own_lora:  # sanity check: merging the update into W must not change predictions
        merge_lora(model)
        merged = evaluate_model(model, test, device)
        extra = {"merged_test_acc": merged["acc"], "merged_latency_ms": measure_latency(model, example, device)}
        log(f"  merged-LoRA test acc {merged['acc']:.4f} (unmerged {test_m['acc']:.4f}), "
            f"latency {latency:.1f} -> {extra['merged_latency_ms']:.1f} ms")

    metrics = {
        "method": method, "seed": seed, "smoke": smoke,
        "test_acc": test_m["acc"], "test_f1": test_m["f1"], "test_loss": test_m["loss"],
        "dev_acc": history[-1]["dev_acc"], "train_time_s": t["seconds"],
        "latency_ms": latency, "adapter_size_kb": adapter_kb,
        **params, "model_size_mb": model_size_mb(model), **extra,
        "history": history, "config": cfg.to_dict(), "git": git_hash(), "env": env_info(),
    }
    save_json(metrics, meta_path)
    log(f"  test_acc={test_m['acc']:.4f}  test_f1={test_m['f1']:.4f}  time={t['seconds']:.0f}s")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["full_ft", "lora_r8", "student_ce", "student_kd"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--smoke", action="store_true", help="offline pipeline check with tiny random models")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--train_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--kd_temperature", type=float, default=None)
    ap.add_argument("--kd_alpha", type=float, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.epochs, cfg.lr_full, cfg.lr_lora, cfg.lr_student = 4, 2e-3, 2e-3, 2e-3
        cfg.out_dir = "results_smoke"
    for k in ("out_dir", "train_size", "epochs", "batch_size", "max_len", "kd_temperature", "kd_alpha"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    for seed in args.seeds:
        for method in args.methods:
            run_one(method, seed, cfg, smoke=args.smoke)

    print("\n=== summary ===")
    print(to_markdown(aggregate(load_runs(cfg.out_dir))))


if __name__ == "__main__":
    main()
