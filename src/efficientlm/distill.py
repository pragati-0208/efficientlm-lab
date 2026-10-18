"""Knowledge distillation (Hinton, Vinyals, Dean 2015, https://arxiv.org/abs/1503.02531).

The student learns from the teacher's *soft* probabilities (which carry "dark knowledge":
how similar the wrong classes are), plus the normal hard-label loss:

    L = alpha * CE(student, y)  +  (1 - alpha) * T^2 * KL( softmax(t/T) || softmax(s/T) )

The T^2 factor keeps gradient magnitudes comparable when T changes.
"""
import torch
import torch.nn.functional as F

from .trainer import Trainer


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1.0 - alpha) * soft


class DistillTrainer(Trainer):
    def __init__(self, *args, teacher, temperature=4.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.T = temperature
        self.alpha = alpha

    def compute_loss(self, batch):
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        with torch.no_grad():
            t_logits = self.teacher(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        loss = distillation_loss(out.logits, t_logits, batch["labels"], self.T, self.alpha)
        return loss, out.logits
