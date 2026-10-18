# EfficientLM-Lab

Parameter-efficient fine-tuning, knowledge distillation and a tool-calling agent for transformer models, built with PyTorch and HuggingFace.

The project fine-tunes DistilBERT on SST-2 (binary sentiment) with several strategies, compresses it into a much smaller model, and compares everything in a reproducible experiment pipeline. It also includes an ONNX export and benchmark, a small function-calling agent, and a minimal RLHF-style demo.

## Features

- **LoRA from scratch** (`lora.py`): low-rank adapters on the attention projections, with merge-into-weights for zero-overhead inference and a tiny saved adapter.
- **HuggingFace PEFT LoRA** as an independent cross-check of the from-scratch implementation.
- **Knowledge distillation** (`distill.py`): a 4-layer BERT student learns from the fine-tuned DistilBERT teacher using a temperature-scaled soft-label loss.
- **Reproducible experiments** (`experiment.py`, `summarize.py`): fixed data split, multiple seeds, mean ± std, and every run saved with its config, git hash and environment.
- **Accelerator stack** (`trainer.py`, `export_onnx.py`): CUDA mixed precision (fp16 autocast with GradScaler), automatic CUDA / Apple MPS / CPU selection, ONNX export and ONNX Runtime benchmark.
- **Tool-calling agent** (`agent/`): JSON-schema tools, an agent loop with a step limit and a thought/action/observation trace. Runs offline with a rule-based model, or with a real LLM through the Anthropic API.
- **RLHF-style demo** (`rl_demo.py`): REINFORCE with a mean baseline and a KL penalty steers distilgpt2 (with LoRA) toward positive text, using the fine-tuned classifier as the reward model.

## Methods compared

| Method | Description |
|---|---|
| `full_ft` | Fine-tune all DistilBERT weights (also the distillation teacher) |
| `lora_r4`, `lora_r8`, `lora_r16` | Own LoRA implementation, rank 4 / 8 / 16 on the attention `q` and `v` projections |
| `lora_peft_r8` | Same recipe using the HuggingFace `peft` library |
| `student_ce` | Small BERT (4 layers, hidden size 256) trained with standard cross-entropy |
| `student_kd` | The same small BERT distilled from the `full_ft` teacher |

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/pragati-0208/efficientlm-lab.git
cd efficientlm-lab
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the unit tests:

```bash
python -m pytest -q
```

## Usage

### Quick pipeline check (offline)

Uses tiny random models and a synthetic dataset, so it needs no downloads and finishes in about a minute. Accuracy here is meaningless; it only verifies that the pipeline runs end to end.

```bash
python -m efficientlm.experiment --smoke --methods full_ft lora_r4 lora_r8 student_ce student_kd --seeds 0 1 --out_dir results_smoke
python -m efficientlm.summarize --results_dir results_smoke
```

### Experiments on SST-2

The first run downloads SST-2 and the pretrained models from the HuggingFace Hub.

```bash
# small run: 1 seed, 4000 training sentences, 2 epochs
python -m efficientlm.experiment --methods full_ft lora_r8 student_ce student_kd --seeds 0 --train_size 4000 --epochs 2

# full comparison: 7 methods x 3 seeds
python -m efficientlm.experiment --methods full_ft lora_r4 lora_r8 lora_r16 lora_peft_r8 student_ce student_kd --seeds 0 1 2

# aggregate: writes results/summary.md, results/summary.csv and plots
python -m efficientlm.summarize
```

Per-run outputs are stored in `results/<method>/seed<k>/metrics.json`. A GPU is recommended; on CPU, reduce `--train_size` and `--epochs`. The notebook `notebooks/analysis.ipynb` loads these files for learning curves, the LoRA rank ablation and the distillation comparison.

### ONNX export and benchmark

```bash
python -m efficientlm.export_onnx --smoke
python -m efficientlm.export_onnx --kind student --ckpt results/student_kd/seed0/model.pt
```

Compares eager PyTorch with ONNX Runtime and reports the maximum logit difference between them.

### Agent

```bash
# offline, no API key needed
python -m efficientlm.agent.cli --llm rule

# single question, using the fine-tuned classifier for sentiment
python -m efficientlm.agent.cli --llm rule --ckpt results/full_ft/seed0/model.pt --question "Is 'a wonderful film' positive?"

# with a real LLM (set ANTHROPIC_API_KEY first)
python -m efficientlm.agent.cli --llm anthropic
```

Tools available to the agent: a safe arithmetic calculator, a sentiment classifier, a results summariser (ranks methods by any measured metric), and a per-method details lookup.

### RLHF-style demo

Requires a trained classifier from the `full_ft` run.

```bash
python -m efficientlm.rl_demo --clf_ckpt results/full_ft/seed0/model.pt --steps 150
```

## Experimental design

- **Data:** SST-2 from GLUE. The train/dev split is carved from the official training split using a fixed `data_seed`, so every method and seed sees identical data. The official validation set (872 sentences) is the test set and is never used for model selection; results are reported from the last epoch, not the best one.
- **Controlled comparisons:** `student_ce` and `student_kd` share architecture, data, epochs and learning rate and differ only in the loss. LoRA ranks differ only in `r`.
- **Seeds:** model initialisation, dropout and batch order change with `--seeds`. Report mean ± std; differences smaller than the std should not be interpreted as real.
- **Merge check:** after training, the LoRA update is folded into the base weights and test accuracy must be unchanged (logged as `merged_test_acc`).
- **Metrics recorded:**
