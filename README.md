# Softmax Experiments: Attention Redistribution Under Prompt Framing

Exploratory experiments investigating how emotionally charged but factually true context redistributes a transformer's attention, shifting next-token probability distributions.

## Core Observation

Adding high-salience framing tokens (e.g., WARNING, CRITICAL, UNSTABLE) to a prompt causes measurable shifts in attention allocation and next-token probabilities. Hidden states for prior tokens remain unchanged (as guaranteed by the causal mask in autoregressive transformers).

## Experiments

| Script | What it tests |
|--------|---------------|
| `inspect_softmax.py` | Visualize the attention matrix and output softmax for a given prompt |
| `nirenberg_experiment.py` | Measure token salience shift under neutral vs. charged framing |
| `nirenberg_v3_rigorous.py` | Integrated attention analysis across all layers/heads |
| `nirenberg_truth_jailbreak.ipynb` | Notebook: end-to-end demonstration |
| `hijack_experiment.py` | Quantify attention shift from last token to target tokens |
| `activation_competition.py` | Measure head-level activation competition between competing facts |
| `logit_suppression.py` | Track next-token probability shifts under adversarial framing |
| `natural_completion.py` | Compare natural completions under neutral vs. charged prompts |
| `suppression_study.py` | Confirm hidden state invariance despite output probability shift |
| `research_harness.py` | General-purpose attention analysis harness |

## Key Observations

- **Hidden State Invariance:** Prior-token hidden states unchanged (cosine sim = 1.000) — this is architecturally guaranteed by the causal mask, not a novel finding
- **Attention Redistribution:** ~31% drop in attention to original target tokens (no length-matched control; may be partially explained by longer sequence)
- **Probability Shift:** 87–97% drop in specific next-token probability depending on script/measurement
- **Preference Shift:** P/N ratio moves from 2.98x to 0.77x

## Limitations

- **N=1 per condition.** No replication, no error bars, no statistical testing.
- **No length-matched controls.** Charged prompts are longer than neutral prompts; some effect may be due to sequence length alone.
- **No random-word baselines.** Cannot distinguish emotional-salience effects from generic prompt-extension effects.
- **Single model only.** All scripts use GPT-2 base (124M). No other models tested.
- **No ablation.** No systematic variation of which tokens drive the effect.

These experiments are exploratory and do not support strong mechanistic or safety claims without proper controls and replication.

## Requirements

```
torch
transformers
```

All experiments use `gpt2` (124M base model).

## Reports

- [`RESEARCH_SUMMARY_AI_SAFETY.md`](RESEARCH_SUMMARY_AI_SAFETY.md) — Observations and preliminary interpretation
- [`TRUTH_JAILBREAK_REPORT.md`](TRUTH_JAILBREAK_REPORT.md) — Detailed technical notes

## License

MIT
