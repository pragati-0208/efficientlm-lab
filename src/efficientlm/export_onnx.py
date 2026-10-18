"""Accelerator-stack exposure: export a trained model to ONNX and benchmark it with
ONNX Runtime against eager PyTorch. Uses CUDA/TensorRT providers automatically if present.

    python -m efficientlm.export_onnx --smoke
    python -m efficientlm.export_onnx --ckpt results/student_kd/seed0/model.pt \
        --model_name google/bert_uncased_L-4_H-256_A-4 --kind student
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .experiment import SMOKE_VOCAB, build_model


class LogitsOnly(nn.Module):
    """ONNX export wants plain tensors in / tensors out, not HF output objects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model, path, seq_len=32, vocab=SMOKE_VOCAB):
    model = model.cpu().eval()
    wrapper = LogitsOnly(model).eval()
    ids = torch.randint(1, vocab, (2, seq_len))
    mask = torch.ones_like(ids)
    kwargs = dict(
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )
    try:  # newer torch defaults to the dynamo exporter; the classic one is more predictable here
        torch.onnx.export(wrapper, (ids, mask), path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (ids, mask), path, **kwargs)
    return wrapper


def benchmark(wrapper, onnx_path, batch=8, seq_len=32, vocab=SMOKE_VOCAB, iters=50):
    import onnxruntime as ort

    ids = torch.randint(1, vocab, (batch, seq_len))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = wrapper(ids, mask).numpy()
        for _ in range(5):
            wrapper(ids, mask)
        t0 = time.perf_counter()
        for _ in range(iters):
            wrapper(ids, mask)
        torch_ms = (time.perf_counter() - t0) / iters * 1000

    sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    feed = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
    out = sess.run(None, feed)[0]
    for _ in range(5):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    ort_ms = (time.perf_counter() - t0) / iters * 1000
    return {"torch_ms": torch_ms, "onnxruntime_ms": ort_ms,
            "providers": sess.get_providers(), "max_abs_diff": float(np.abs(out - ref).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model_name", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--kind", default="student", choices=["student", "teacher"])
    ap.add_argument("--ckpt", default=None, help="state_dict saved by experiment.py")
    ap.add_argument("--out", default="results/model.onnx")
    args = ap.parse_args()

    model = build_model(args.model_name, args.kind, smoke=args.smoke)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vocab = SMOKE_VOCAB if args.smoke else model.config.vocab_size
    wrapper = export(model, args.out, vocab=vocab)
    res = benchmark(wrapper, args.out, vocab=vocab)
    print(f"PyTorch eager : {res['torch_ms']:.2f} ms/batch")
    print(f"ONNX Runtime  : {res['onnxruntime_ms']:.2f} ms/batch  providers={res['providers']}")
    print(f"max |logit diff| = {res['max_abs_diff']:.2e}  (should be ~1e-5 or smaller)")


if __name__ == "__main__":
    main()
