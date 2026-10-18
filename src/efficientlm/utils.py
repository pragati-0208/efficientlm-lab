import json
import os
import platform
import random
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """Seed every RNG we use so that a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def env_info() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(get_device()),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def model_size_mb(model: torch.nn.Module) -> float:
    n = sum(p.numel() * p.element_size() for p in model.parameters())
    n += sum(b.numel() * b.element_size() for b in model.buffers())
    return n / 1024 ** 2


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@contextmanager
def timer():
    t = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield t
    finally:
        t["seconds"] = time.perf_counter() - start
