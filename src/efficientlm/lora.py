"""LoRA implemented from scratch (Hu et al., 2021, https://arxiv.org/abs/2106.09685).

Idea: instead of updating a big weight matrix W (out x in), freeze it and learn a
low-rank update  dW = B @ A  with  A: (r x in),  B: (out x r),  r << in, out.

    y = W x + (alpha / r) * B (A x)

B starts at zero, so at step 0 the model behaves exactly like the pretrained one.
"""
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank r must be positive")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + update * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Fold the low-rank update into W so inference has zero extra cost."""
        merged = nn.Linear(
            self.base.in_features, self.base.out_features, bias=self.base.bias is not None
        )
        merged.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged.to(self.base.weight.device, self.base.weight.dtype)


def _get_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    model: nn.Module,
    target_suffixes: Sequence[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    modules_to_save: Iterable[str] = ("pre_classifier", "classifier"),
) -> nn.Module:
    """Freeze the whole model, wrap matching nn.Linear layers with LoRA, and keep
    the classification head trainable (it is randomly initialised, so it must learn)."""
    for p in model.parameters():
        p.requires_grad = False

    targets = [
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and name.split(".")[-1] in set(target_suffixes)
    ]
    if not targets:
        raise ValueError(f"No nn.Linear module matched {list(target_suffixes)}")
    for name in targets:
        parent, child = _get_parent(model, name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))

    keep = set(modules_to_save)
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top in keep or any(part in keep for part in name.split(".")):
            p.requires_grad = True
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear by a plain Linear with the update folded in."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent, child = _get_parent(model, name)
        setattr(parent, child, getattr(parent, child).merged_linear())
    return model


def trainable_state_dict(model: nn.Module) -> dict:
    """Only the parameters that were trained (LoRA matrices + head). This is the
    'adapter': a few hundred KB instead of the whole model."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable}


def save_adapter(model: nn.Module, path: str) -> int:
    torch.save(trainable_state_dict(model), path)
    import os

    return os.path.getsize(path)
