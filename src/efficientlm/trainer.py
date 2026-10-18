import math
import time

import torch
import torch.nn.functional as F


class Trainer:
    """Small, readable training loop: AdamW + linear warmup/decay + grad clipping + optional AMP."""

    def __init__(self, model, train_loader, dev_loader, device, lr, epochs,
                 weight_decay=0.01, warmup_ratio=0.1, grad_clip=1.0, amp=True, log=print):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log = log
        self.use_amp = bool(amp and device.type == "cuda")  # fp16 autocast on CUDA only

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        total = max(1, epochs * len(train_loader))
        warm = max(1, int(warmup_ratio * total))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            return max(0.0, (total - step) / max(1, total - warm))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # Subclasses (e.g. distillation) override this one method.
    def compute_loss(self, batch):
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(out.logits.float(), batch["labels"])
        return loss, out.logits

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def fit(self):
        history = []
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running, n, t0 = 0.0, 0, time.perf_counter()
            for batch in self.train_loader:
                batch = self._to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    loss, _ = self.compute_loss(batch)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                running += loss.item() * batch["labels"].size(0)
                n += batch["labels"].size(0)
            dev = self.evaluate(self.dev_loader)
            rec = {"epoch": epoch, "train_loss": running / n, "dev_acc": dev["acc"],
                   "dev_loss": dev["loss"], "seconds": time.perf_counter() - t0}
            history.append(rec)
            self.log(f"  epoch {epoch}/{self.epochs}  train_loss={rec['train_loss']:.4f}  "
                     f"dev_acc={rec['dev_acc']:.4f}  ({rec['seconds']:.0f}s)")
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        return evaluate_model(self.model, loader, self.device)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels, loss_sum, n = [], [], 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        loss_sum += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
        preds.append(logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
        n += batch["labels"].size(0)
    preds, labels = torch.cat(preds), torch.cat(labels)
    return {"acc": (preds == labels).float().mean().item(), "f1": binary_f1(preds, labels), "loss": loss_sum / n}


def binary_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


@torch.no_grad()
def measure_latency(model, batch, device, warmup=5, iters=30):
    """Average forward-pass time per batch in milliseconds (synchronises CUDA properly)."""
    model.eval()
    ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    for _ in range(warmup):
        model(input_ids=ids, attention_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(input_ids=ids, attention_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0
