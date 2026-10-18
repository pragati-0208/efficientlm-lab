import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from efficientlm.lora import LoRALinear, apply_lora, merge_lora, trainable_state_dict
from efficientlm.utils import count_params


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin = nn.Linear(16, 16)
        self.v_lin = nn.Linear(16, 16)
        self.other = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        return self.classifier(self.other(self.q_lin(x) + self.v_lin(x)))


def test_lora_is_identity_at_init():
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    x = torch.randn(4, 8)
    assert torch.allclose(LoRALinear(base, r=4, alpha=8)(x), base(x), atol=1e-6)  # B = 0


def test_only_lora_and_head_are_trainable():
    m = apply_lora(Tiny(), ["q_lin", "v_lin"], r=4, alpha=8, modules_to_save=["classifier"])
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == {"q_lin.lora_A", "q_lin.lora_B", "v_lin.lora_A", "v_lin.lora_B",
                         "classifier.weight", "classifier.bias"}
    assert count_params(m)["trainable_params"] < count_params(m)["total_params"]


def test_param_count_formula():
    m = apply_lora(Tiny(), ["q_lin"], r=4, alpha=8, modules_to_save=[])
    assert count_params(m)["trainable_params"] == 4 * 16 + 16 * 4  # A (r x in) + B (out x r)


def test_merge_matches_unmerged_output():
    torch.manual_seed(0)
    m = apply_lora(Tiny(), ["q_lin", "v_lin"], r=4, alpha=8, modules_to_save=["classifier"])
    for mod in m.modules():  # make B non-zero so the test is meaningful
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_B, std=0.1)
    m.eval()
    x = torch.randn(5, 16)
    before = m(x)
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    assert torch.allclose(before, m(x), atol=1e-5)


def test_gradients_reach_lora_not_base():
    m = apply_lora(Tiny(), ["q_lin"], r=2, alpha=4, modules_to_save=["classifier"])
    m(torch.randn(3, 16)).sum().backward()
    assert m.q_lin.base.weight.grad is None
    assert m.q_lin.lora_B.grad is not None


def test_adapter_is_small():
    m = apply_lora(Tiny(), ["q_lin", "v_lin"], r=2, alpha=4, modules_to_save=["classifier"])
    assert len(trainable_state_dict(m)) == 6


def test_no_match_raises():
    with pytest.raises(ValueError):
        apply_lora(Tiny(), ["does_not_exist"])
