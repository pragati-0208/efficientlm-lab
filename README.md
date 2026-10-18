# EfficientLM-Lab

**Parameter-efficient fine-tuning, knowledge distillation and a tool-calling agent for transformer models**
A B.Tech mini-project. Every skill in the job description maps to a concrete piece of this repo.

## What the project does

1. **Fine-tunes DistilBERT on SST-2 (sentiment)** three ways: full fine-tuning, our own **LoRA** (written from scratch), and LoRA through HuggingFace **PEFT**.
2. **Distills** the fine-tuned model into a much smaller 4-layer BERT (**knowledge distillation**) and compares it with the same small model trained normally.
3. Runs everything as **controlled, reproducible experiments** (fixed data split, several seeds, mean ± std, config + git hash + environment saved with every result).
4. Exports the small model to **ONNX** and benchmarks **ONNX Runtime vs eager PyTorch** (accelerator software stack).
5. Includes a **tool-calling agent** (calculator, sentiment classifier, "ask about my experiment results") with an offline rule-based brain and an optional real-LLM brain.
6. Includes a tiny **RLHF-style demo**: REINFORCE + KL penalty steers distilgpt2 towards positive text using our classifier as the reward model.

## Skill → where it lives

| Skill in the job description | Where |
|---|---|
| Python fundamentals | whole repo, typed dataclasses, tests |
| ML / DL understanding | `trainer.py` (training loop, schedulers, AMP), `metrics` in `trainer.py` |
| LLMs, GenAI, Transformers, NLP | DistilBERT/BERT classifiers, distilgpt2 generation (`rl_demo.py`) |
| VS Code, Jupyter, Git | `notebooks/analysis.ipynb`, `.gitignore`, git hash stored in every result |
| AI accelerator software stack | CUDA fp16 autocast + GradScaler (`trainer.py`), device selection incl. Apple MPS (`utils.py`), ONNX export + ONNX Runtime providers (`export_onnx.py`) |
| PyTorch | everything |
| AI agents, tool use, function calling, reasoning | `agent/` (JSON-schema tools, agent loop with step limit, Anthropic tool-use adapter, thought/action/observation trace) |
| LoRA / PEFT | `lora.py` (from scratch) + `lora_peft_r*` methods (library) |
| Knowledge distillation | `distill.py` |
| Reinforcement learning / RLHF | `rl_demo.py` (REINFORCE, baseline, KL to base model) |
| Read papers, implement ideas | LoRA (Hu 2021), Distillation (Hinton 2015); see "Papers reproduced" |
| Reproduce results, controlled experiments | `experiment.py`, `summarize.py`, seeds, fixed split, ablations, tests |

## Setup

```bash
git init && git add . && git commit -m "initial commit"     # keep your work under version control
python -m venv .venv && source .venv/bin/activate           # Windows: .venv\Scripts\activate
make install            # or: pip install -r requirements.txt && pip install -e .
make test               # unit tests (a few seconds)
make smoke              # whole pipeline on tiny random models + synthetic data, no downloads
```

On Windows without `make`, run the commands inside the `Makefile` directly (set `PYTHONPATH=src` or run `pip install -e .`).

## Run the real experiments

```bash
make quick   # 4 methods, 1 seed, 4000 training sentences, 2 epochs  (good first run)
make full    # 7 methods x 3 seeds, 8000 sentences, 3 epochs; writes results/summary.{md,csv} + plots
```

The first run downloads SST-2 and the models from the HuggingFace Hub (internet needed once).
Rough expectation: minutes on a GPU, a couple of hours on a laptop CPU for `make full`. Use `--train_size` / `--epochs` to shrink it.

Results land in `results/<method>/seed<k>/metrics.json`. Then:

```bash
python -m efficientlm.summarize           # table + plots
jupyter lab notebooks/analysis.ipynb      # analysis, ablations, your conclusions
```

### Other components

```bash
# Accelerator stack: export + benchmark
python -m efficientlm.export_onnx --smoke
python -m efficientlm.export_onnx --kind student --ckpt results/student_kd/seed0/model.pt

# Agent (offline demo, no API key)
python -m efficientlm.agent.cli --llm rule
python -m efficientlm.agent.cli --llm rule --ckpt results/full_ft/seed0/model.pt --question "Is 'a wonderful film' positive?"
# Agent with a real LLM:  export ANTHROPIC_API_KEY=...   then   --llm anthropic

# RLHF-style demo (needs a trained results/full_ft/seed0/model.pt)
python -m efficientlm.rl_demo --clf_ckpt results/full_ft/seed0/model.pt --steps 150
```

## Experimental design (what makes it "controlled")

- **Same data for every method**: the train/dev split depends only on `data_seed=42`. Test = the official SST-2 validation set (872 sentences), never used for model selection; we report the *last* epoch, not the best one.
- **Only one thing changes at a time**: `student_ce` vs `student_kd` share architecture, data, epochs and learning rate, only the loss differs. LoRA ranks 4/8/16 share everything except `r`.
- **Seeds**: model init, dropout and batch order change with `--seeds`; report mean ± std. A difference smaller than the std is not evidence of anything.
- **Own LoRA vs PEFT LoRA** is a correctness cross-check: they should land within noise of each other.
- **Merge check**: after training, our LoRA update is folded into the weights; test accuracy must be unchanged (it is asserted in the tests and logged as `merged_test_acc`).

## Results (fill in after your runs)

Paste `results/summary.md` here. Do not copy numbers from anywhere else; the point of the project is that *your* table is reproducible from *your* code.

| method | seeds | test acc (%) | trainable params | model MB | adapter KB | latency ms/batch |
|---|---|---|---|---|---|---|
| _run `make full`_ | | | | | | |

## Papers reproduced

- **LoRA** (Hu et al., 2021): claim to check, LoRA reaches close to full fine-tuning quality while training a small fraction of the parameters and storing a tiny adapter.
- **Distillation** (Hinton et al., 2015): claim to check, a student trained on the teacher's soft targets beats the same student trained on hard labels alone.

Either claim may *not* replicate at this small scale, and reporting that honestly is a perfectly good result.

## What has and has not been verified

- Verified by running: the agent, tool registry, safe calculator, result aggregation (`tests/test_agent.py`, `tests/test_summarize.py`, 16 tests).
- Written but **not executed by the author** (the authoring environment had no room to install PyTorch): `lora.py`, `distill.py`, `trainer.py`, `experiment.py`, `export_onnx.py`, `rl_demo.py` and their tests (`test_lora.py`, `test_distill.py`, `test_pipeline.py`, `test_rl.py`). Run `make test` and `make smoke` first. If anything fails, fix it before running the long experiments; the tests are designed to point at the broken piece.
- The Anthropic-backed agent needs an API key and was not run live; the message conversion is unit-tested.

## Known limitations (put these in your report)

- SST-2 is an easy binary task; gaps between methods are small, so you need multiple seeds to say anything.
- Only attention `q`/`v` projections get LoRA; other targets (FFN layers) are an easy extension.
- The RL demo is plain REINFORCE, not PPO, and the reward model can be gamed (the policy can learn to spam positive words).
- Latency is measured on whatever device you run on, with one batch shape; treat it as relative, not absolute.
- The rule-based agent is a router, not a language model; the Anthropic adapter is the real function-calling path.

## Extension ideas

LoRA on FFN layers or QLoRA (4-bit); layer-wise / hidden-state distillation; PPO via `trl`; a larger dataset (IMDB, MNLI); dynamic int8 quantisation of the student and re-benchmark; a Gradio demo.

## Viva questions to prepare

1. Why is `B` initialised to zero and `A` randomly? 2. Why the `alpha / r` scaling? 3. Why multiply the KD loss by `T²`? 4. Why can a merged LoRA model have zero inference overhead? 5. Why report mean ± std over seeds? 6. What is the KL penalty in the RL demo protecting against? 7. Why does the agent need a maximum step count? 8. What does the GradScaler do in fp16 training?

## Layout

```
src/efficientlm/
  lora.py  distill.py  trainer.py  data.py  config.py  utils.py
  experiment.py  summarize.py  export_onnx.py  rl_demo.py
  agent/ tools.py  llm.py  agent.py  cli.py
tests/        notebooks/analysis.ipynb        REPORT_TEMPLATE.md
```
