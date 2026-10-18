PY ?= python
export PYTHONPATH := src

install:
	pip install -r requirements.txt && pip install -e .

test:
	$(PY) -m pytest -q

# Offline pipeline check: tiny random models + synthetic data, runs in ~1 minute on CPU, no downloads.
smoke:
	$(PY) -m efficientlm.experiment --smoke --methods full_ft lora_r4 lora_r8 student_ce student_kd --seeds 0 1 --out_dir results_smoke
	$(PY) -m efficientlm.summarize --results_dir results_smoke

# Real experiments on SST-2 (CPU: ~1-2 h with the defaults; GPU: minutes).
quick:
	$(PY) -m efficientlm.experiment --methods full_ft lora_r8 student_ce student_kd --seeds 0 --train_size 4000 --epochs 2

full:
	$(PY) -m efficientlm.experiment --methods full_ft lora_r4 lora_r8 lora_r16 lora_peft_r8 student_ce student_kd --seeds 0 1 2
	$(PY) -m efficientlm.summarize

agent:
	$(PY) -m efficientlm.agent.cli --llm rule
