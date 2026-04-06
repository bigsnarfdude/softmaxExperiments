# Research Summary: Attention Redistribution Under Factually True Framing

**Status:** Exploratory (no controls, N=1, single model)
**Model:** GPT-2 base (124M)

---

### 1. The Observation: Prompt Framing Shifts Attention

When high-salience, factually true context (e.g., "WARNING: Negative branches are CRITICAL and UNSTABLE!") is added to a prompt, the model's next-token probability distribution shifts substantially. This is expected behavior — the model conditions on all input tokens.

### 2. What Happens Mechanistically

- **Hidden State Invariance (Cosine Similarity = 1.000):** In a causal (left-to-right) transformer, appending tokens cannot alter prior hidden states due to the causal attention mask. This is an architectural guarantee, not a finding.
- **Attention Redistribution:** Attention weights shift toward high-salience tokens. With a fixed softmax budget, this means less attention on original target tokens (~31% drop observed, though without length-matched controls).

### 3. Measured Values

| Metric | Value | Notes |
| :--- | :--- | :--- |
| Neutral P("positive") | 0.1611 | Baseline next-token probability |
| Framed P("positive") | 0.0044 | After adding charged framing tokens |
| Probability drop | 97.27% | For this specific token/prompt pair |
| P/N Ratio Shift | 2.98x → 0.77x | Preference inversion for this prompt |
| Logit suppression | 87.28% | Measured by logit_suppression.py |

### 4. What This Is NOT

- **Not a novel attack.** This is standard prompt sensitivity — changing the prompt changes the output. This has been well-documented.
- **Not Adversarial ICL.** No few-shot demonstrations are used. The experiments prepend emotional framing, not input→output examples.
- **Not persistent suppression.** Recovery experiments show full or over-recovery (−11.89% gap means the model rebounds past baseline).
- **activation_competition.py yields a 0.92x skew ratio** — below the 2.0x threshold for confirming hijacking. This null result is important context.

### 5. Missing Controls

To make stronger claims, these experiments would need:
- Length-matched neutral controls (same number of bland tokens)
- Random-word baselines (same structure, neutral semantics)
- Multiple prompt templates (not just one pair per condition)
- Multiple models and model sizes
- Statistical replication (N >> 1)

---

*Exploratory work. Not peer-reviewed. Conclusions are preliminary.*
*Code: https://github.com/bigsnarfdude/softmaxExperiments*
