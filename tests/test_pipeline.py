"""End-to-end smoke tests on tiny random models + synthetic data (no downloads)."""
import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from efficientlm.config import Config
from efficientlm.experiment import run_one


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.out_dir = str(tmp_path)
    c.epochs, c.lr_full, c.lr_lora, c.lr_student = 3, 2e-3, 2e-3, 2e-3
    return c


def test_all_methods_run_and_write_metrics(cfg):
    for method in ["full_ft", "lora_r4", "student_ce", "student_kd"]:
        m = run_one(method, 0, cfg, smoke=True, log=lambda *_: None)
        assert 0.0 <= m["test_acc"] <= 1.0
        assert os.path.exists(os.path.join(cfg.out_dir, method, "seed0", "metrics.json"))
    lora = json.load(open(os.path.join(cfg.out_dir, "lora_r4", "seed0", "metrics.json")))
    full = json.load(open(os.path.join(cfg.out_dir, "full_ft", "seed0", "metrics.json")))
    assert lora["trainable_params"] < 0.5 * full["trainable_params"]
    assert abs(lora["merged_test_acc"] - lora["test_acc"]) < 1e-6  # merge must not change predictions


def test_training_actually_learns(cfg):
    cfg.epochs = 6
    m = run_one("full_ft", 0, cfg, smoke=True, log=lambda *_: None)
    assert m["history"][-1]["train_loss"] < m["history"][0]["train_loss"]


def test_same_seed_same_result(cfg, tmp_path):
    a = run_one("lora_r4", 3, cfg, smoke=True, log=lambda *_: None)
    cfg2 = Config()
    cfg2.out_dir = str(tmp_path / "again")
    cfg2.epochs, cfg2.lr_lora = cfg.epochs, cfg.lr_lora
    b = run_one("lora_r4", 3, cfg2, smoke=True, log=lambda *_: None)
    assert a["test_acc"] == b["test_acc"]
    assert abs(a["history"][-1]["train_loss"] - b["history"][-1]["train_loss"]) < 1e-5
