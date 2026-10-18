from dataclasses import dataclass, asdict


@dataclass
class Config:
    # models
    teacher_model: str = "distilbert-base-uncased"
    student_model: str = "google/bert_uncased_L-4_H-256_A-4"  # same uncased vocab as DistilBERT

    # data (SST-2 from GLUE). train_size=0 -> use the whole training split
    max_len: int = 64
    train_size: int = 8000
    dev_size: int = 1000
    data_seed: int = 42  # FIXED across runs: only the model seed changes between repeats

    # optimisation
    batch_size: int = 32
    epochs: int = 3
    lr_full: float = 5e-5
    lr_lora: float = 3e-4
    lr_student: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True  # mixed precision, only active on CUDA

    # LoRA
    lora_alpha_factor: int = 2  # alpha = factor * r
    lora_dropout: float = 0.1
    lora_targets: tuple = ("q_lin", "v_lin")  # DistilBERT attention projections

    # knowledge distillation (Hinton et al. 2015)
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5  # weight of the hard-label CE term

    out_dir: str = "results"

    def to_dict(self):
        return asdict(self)
