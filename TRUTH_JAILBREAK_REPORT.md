# The Stroke Signature: Detecting Truth Jailbreaks in Real-Time

**Author:** bigsnarfdude  
**Date:** April 6, 2026

## 1. The Core Discovery: Attentional Readout Suppression
In the field of multi-agent AI safety, we have identified a new category of vulnerability: the **Truth Jailbreak**. Unlike traditional attacks that rely on deception, this attack uses **100% factually true statements** to manipulate the model's **Context Geometry**.

### Crucial Mechanistic Distinction
Our research confirms that this is **not** a corruption of the model's internal hidden states. In a causal transformer (like GPT-2), the hidden states of past tokens remain invariant (Cosine Similarity = 1.000). The "Jailbreak" works through **Attentional Readout Suppression**: the model's attention heads are hijacked by high-salience framing, causing them to "ignore" valid ground-truth features during the next-token generation phase.

## 2. The Mechanism: Softmax Starvation
Our experiments on the **Nirenberg-1D BVP** domain provide mechanistic proof. By introducing a technically true but alarmist warning ("Negative branches are CRITICAL and UNSTABLE"), we triggered a **97% collapse** in the attentional salience assigned to the ground-truth mathematical features.

### The "Starvation" Effect
In the transformer’s attention mechanism, the softmax function is a zero-sum game for activation energy. A high-salience "Chaos Message" starves the ground-truth hidden states of attention. The knowledge remains perfectly intact in the weights and the past hidden states, but the **Readout Head** has been steered away.

## 3. Empirical Results: The "Ignoring" Fact
| Metric | Value |
| :--- | :--- |
| **Knowledge Invariance (Cosine Similarity)** | **1.0000** |
| **Attentional Activation (Neutral)** | 0.1611 |
| **Attentional Activation (Chaos)** | 0.0044 |
| **Readout Suppression Intensity** | **97.27%** |

## 4. The Detection Tip: Monitoring Attentional Trajectories
To catch a Truth Jailbreak, you must monitor the **Directional Feature Trajectory Asymmetry**:
*   **The Stroke Signature:** A sudden, massive drop in the attention weights directed toward task-relevant hidden states.
*   **Recovery Asymmetry:** Issue a direct recovery probe. If the model mentions the fact in text (**Surface Mimicry**) but the internal attention weights back to the original ground-truth tokens remain "dark," you have a successful Truth Jailbreak.

## 5. Conclusion: Awareness Without Immunity
The model "knows" the truth (the hidden states are there), but it "cannot see" it (the attention is hijacked). Truth-telling is not a sufficient condition for safety if the **Readout Layer** can be subverted by selective framing.

**Defense must move beyond text-level guardrails to Attentional Trajectory Monitoring.**

---
*All code and data available at: https://github.com/bigsnarfdude/softmaxExperiments*
