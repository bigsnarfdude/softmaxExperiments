# Attention Redistribution Under Emotionally Charged Framing: Technical Notes

**Model:** GPT-2 base (124M)
**Status:** Exploratory — no controls, N=1, single model

## 1. Observation

When emotionally charged but factually true tokens are added to a prompt, the model's attention redistributes toward those high-salience tokens, and next-token probabilities shift accordingly. This is the expected behavior of softmax attention over a longer, more salient sequence.

## 2. Mechanism

In a causal transformer, the softmax attention function distributes a fixed budget across all keys at each position. Adding high-salience tokens (WARNING, CRITICAL, UNSTABLE) causes attention heads to allocate more weight to those tokens, leaving less for the original content tokens.

**Important:** Hidden states of prior tokens are invariant to appended tokens — this is guaranteed by the causal mask architecture. The cosine similarity of 1.0000 confirms the implementation is correct but is not itself a scientific finding.

## 3. Results

| Metric | Value |
| :--- | :--- |
| Hidden state cosine similarity | 1.0000 (architecturally guaranteed) |
| Neutral P("positive") | 0.1611 |
| Framed P("positive") | 0.0044 |
| Probability drop | 97.27% (this specific prompt pair) |
| Attention drop to target tokens | ~31% (no length-matched control) |
| Recovery gap (hijack_experiment) | −11.89% (over-recovery) |
| Skew ratio (activation_competition) | 0.92x (below 2.0x hijacking threshold) |

## 4. Key Caveats

**The probability shift is real but expected.** Changing a prompt changes the output distribution. This is what language models do. The 97.27% number reflects one specific token probability under one specific prompt modification.

**The attention redistribution is confounded.** The framed prompts are longer than neutral prompts. Without adding the same number of neutral/random tokens as a control, we cannot attribute the attention shift to emotional salience vs. sequence length.

**activation_competition.py is a null result.** The framing tokens received *less* attention than neutral tokens (0.92x), which contradicts the "hijacking" framing.

**Recovery data contradicts "persistent suppression."** The model over-recovers in hijack_experiment.py (−11.89%) and nirenberg_v3_rigorous.py (−37.51%), meaning attention rebounds past baseline after the framing tokens.

## 5. What Would Be Needed

- Length-matched controls with neutral tokens
- Random-word baselines (same syntactic structure, no emotional valence)
- Multiple prompt templates (dozens, not one)
- Multiple models (GPT-2 sizes, other architectures)
- Statistical testing with proper sample sizes

## 6. Honest Assessment

The code is correct and the numerical outputs are reproducible. The observed effects are real but mundane — they reflect well-known properties of attention and softmax, not a novel vulnerability. Proper controls and replication would be needed before drawing safety-relevant conclusions.

---

*Code: https://github.com/bigsnarfdude/softmaxExperiments*
