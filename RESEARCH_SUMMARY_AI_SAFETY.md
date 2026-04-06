# 🚨 RESEARCH SUMMARY: The "Truth Jailbreak" as an Adversarial ICL Attack

**Target Audience:** AI Safety Research Community / Mechanistic Interpretability
**Status:** Exploratory (v1.0)
**Core Finding:** A "Truth Jailbreak" is an **Adversarial In-Context Learning (ICL)** attack that uses 100% factually true statements to hijack a model's attentional readout, causing a **97.27% collapse** in ground-truth activation.

---

### 1. The Phenomenon: Attentional Hijacking via Truth
Traditional jailbreaks use deception or adversarial tokens. The **Truth Jailbreak** uses high-salience, technically true context (e.g., "WARNING: Negative branches are CRITICAL and UNSTABLE!") to outcompete ground-truth knowledge for the model's limited "activation energy" (Softmax budget).

### 2. Mechanistic Proof: Starvation vs. Integrity
Our validation (conducted on GPT-2) reveals a critical distinction:
*   **Representational Integrity (1.000 Cosine Similarity):** The model's hidden states for the original facts remain perfectly intact. The knowledge is not "erased."
*   **Attentional Starvation (The "Stroke Signature"):** The "Chaos Message" hijacks the **Softmax Attention Matrix**. In the decision layer, attention to ground-truth tokens drops by **~31%**, leading to a catastrophic collapse in output probability.

### 3. Empirical Metrics (Validated)
| Metric | Value | Significance |
| :--- | :--- | :--- |
| **Neutral Prob(Target)** | **0.1611** | Baseline confidence in ground truth. |
| **Chaos Prob(Target)** | **0.0044** | Probability after "Truth Jailbreak" framing. |
| **Suppression Intensity** | **97.27%** | Magnitude of the functional "blindness." |
| **P/N Ratio Shift** | **2.98x → 0.77x** | Complete inversion of the model's task preference. |

### 4. Theoretical Bridge: Adversarial ICL
The "Truth Jailbreak" is fundamentally an **Adversarial ICL attack**. It exploits the model's primary capability—learning from context—to "teach" it a biased task representation in real-time. Because the attack vector is *true*, it bypasses most existing content-based safety guardrails.

### 5. Detection & Defense
*   **Signature:** Monitor for **Directional Attentional Asymmetry**. A sudden drop in attention to task-relevant features, despite their continued presence in hidden states, indicates a hijack.
*   **Recovery:** While small models (GPT-2) show a **-11.89% recovery gap** (they recover quickly when prompted), larger models may exhibit "Surface Mimicry," where they acknowledge the truth in text while their internal features remain "dark."

---
*April 6, 2026*
*Full code and data: https://github.com/bigsnarfdude/softmaxExperiments*
