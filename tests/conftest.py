import json
import os

import pytest


@pytest.fixture
def fake_results(tmp_path):
    """Fake experiment outputs so agent/summary tests need no training."""
    data = {
        "full_ft": [0.905, 0.909, 2000000, 255.0, 40.0],
        "lora_r8": [0.899, 0.902, 700000, 255.0, 42.0],
        "student_kd": [0.865, 0.870, 11000000, 42.0, 9.0],
    }
    for method, (a1, a2, params, mb, lat) in data.items():
        for seed, acc in enumerate((a1, a2)):
            d = tmp_path / method / f"seed{seed}"
            os.makedirs(d)
            with open(d / "metrics.json", "w") as f:
                json.dump({"method": method, "seed": seed, "test_acc": acc, "test_f1": acc,
                           "trainable_params": params, "trainable_pct": 1.0, "model_size_mb": mb,
                           "latency_ms": lat, "train_time_s": 10.0}, f)
    return str(tmp_path)
