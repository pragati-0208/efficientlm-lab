# Project report template (B.Tech)

1. **Title & abstract** (150 words): problem, what you built, one headline result from *your* table.
2. **Introduction**: why fine-tuning large models is expensive; why small/efficient models matter.
3. **Background**: transformers & fine-tuning; LoRA (equation, rank r, scaling); knowledge distillation (loss with T and alpha); policy gradient / REINFORCE and the KL penalty; function calling for agents.
4. **System design**: block diagram (data → models → trainer → results → summary/agent); module table from the README.
5. **Experimental setup**: dataset and splits, models, hyper-parameters (copy from `config.py`), seeds, hardware, software versions (from `metrics.json -> env`).
6. **Results**: main table (`results/summary.md`), accuracy-vs-parameters plot, LoRA rank ablation, KD vs CE comparison, latency / ONNX Runtime table, RL reward curve.
7. **Discussion**: did LoRA match full fine-tuning? Did KD help? Are differences larger than the seed std? Failures and surprises.
8. **Agent demo**: 3-4 transcripts including one tool error and how it was handled.
9. **Limitations & future work**.
10. **References**: Hu et al. 2021 (LoRA); Hinton et al. 2015 (distillation); Sanh et al. 2019 (DistilBERT); Socher et al. 2013 (SST); Williams 1992 (REINFORCE); Turc et al. 2019 (compact BERTs).
