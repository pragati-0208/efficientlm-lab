import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from efficientlm.distill import distillation_loss


def test_alpha_one_is_plain_cross_entropy():
    s, t, y = torch.randn(6, 2), torch.randn(6, 2), torch.randint(0, 2, (6,))
    assert torch.allclose(distillation_loss(s, t, y, T=3.0, alpha=1.0), F.cross_entropy(s, y))


def test_soft_term_zero_when_student_equals_teacher():
    s, y = torch.randn(6, 2), torch.randint(0, 2, (6,))
    assert distillation_loss(s, s.clone(), y, T=4.0, alpha=0.0).abs() < 1e-6


def test_soft_term_positive_when_different():
    s, t, y = torch.randn(6, 2), torch.randn(6, 2) * 3, torch.randint(0, 2, (6,))
    assert distillation_loss(s, t, y, T=4.0, alpha=0.0) > 0


def test_gradient_flows_to_student_only():
    s = torch.randn(4, 2, requires_grad=True)
    t = torch.randn(4, 2, requires_grad=True)
    distillation_loss(s, t.detach(), torch.tensor([0, 1, 0, 1])).backward()
    assert s.grad is not None and t.grad is None
