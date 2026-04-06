# Softmax Experiments: Truth Jailbreak & Attentional Hijacking

Mechanistic interpretability experiments investigating how factually true context can hijack a transformer's attention mechanism, causing catastrophic suppression of ground-truth knowledge — a phenomenon we call the **"Truth Jailbreak."**

## Core Finding

A 100% factually true "chaos message" can cause a **97.27% collapse** in ground-truth activation probability by starving attention to relevant tokens, without corrupting the model's internal representations (cosine similarity = 1.000).

## Experiments

| Script | What it tests |
|--------|---------------|
| `inspect_softmax.py` | Visualize the attention matrix and output softmax for a given prompt |
| `nirenberg_experiment.py` | Measure token salience shift under neutral vs. chaos framing |
| `nirenberg_v3_rigorous.py` | Integrated attention analysis across all layers/heads |
| `nirenberg_truth_jailbreak.ipynb` | Notebook: end-to-end truth jailbreak demonstration |
| `hijack_experiment.py` | Quantify attention hijacking from last token to target tokens |
| `activation_competition.py` | Measure head-level activation competition between competing facts |
| `logit_suppression.py` | Track next-token probability shifts under adversarial framing |
| `natural_completion.py` | Compare natural completions under neutral vs. chaos prompts |
| `suppression_study.py` | Confirm hidden state integrity despite output suppression |
| `research_harness.py` | General-purpose attention analysis harness |

## Key Results

- **Representational Integrity:** Hidden states remain perfectly intact (cosine sim = 1.000)
- **Attentional Starvation:** ~31% drop in attention to ground-truth tokens
- **Output Suppression:** 97.27% collapse in target token probability
- **Preference Inversion:** P/N ratio shifts from 2.98x to 0.77x

## Requirements

```
torch
transformers
```

Tested with GPT-2 (all experiments use `gpt2` 124M by default).

## Reports

- [`RESEARCH_SUMMARY_AI_SAFETY.md`](RESEARCH_SUMMARY_AI_SAFETY.md) — Summary for the AI safety community
- [`TRUTH_JAILBREAK_REPORT.md`](TRUTH_JAILBREAK_REPORT.md) — Detailed technical report

## License

MIT
