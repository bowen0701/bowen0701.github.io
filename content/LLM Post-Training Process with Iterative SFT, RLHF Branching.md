---
title: LLM Post-Training Process with Iterative SFT, RLHF Branching
author: Bowen Lee
created: 2026-06-24
publish: true
tags:
  - rlhf
  - sft
  - reward-hacking
  - post-training
  - llm
---
# Concept: LLM Post-Training Process with Iterative SFT, RLHF Branching

## Core Intuition

In multi-round LLM post-training, the standard pattern is:

- Round $i$: $\text{SFT}_i \rightarrow \text{RL}_i$
- Round $i+1$: start from $\text{SFT}_i$ to train $\text{SFT}_{i+1}$, then $\text{SFT}_{i+1} \rightarrow \text{RL}_{i+1}$

Why not start round $i+1$'s SFT from $\text{RL}_i$ instead?

Both are done in practice, but branching SFT from the previous SFT is the safer default. The core reason is that RL collapses entropy and bakes in reward-model biases; using that as the seed for the next round's SFT compounds those problems across rounds.

**Mental model:**
- **SFT** = broad imitation. Wants high-entropy, diverse data. Sensitive to distribution mismatch.
- **RL** = narrow optimization against a reward signal. Collapses entropy. Bakes in Reward Model (RM) bias.

Stacking SFT -> RL -> SFT -> RL means alternately broadening and narrowing and re-broadening and re-narrowing. The broadening step is more reliable when its starting point hasn't already been narrowed, hence the conservative default of branching SFT off SFT and treating each RL checkpoint as a leaf.

## Mathematical Foundation

### PPO in RLHF

[[PPO]] (Proximal Policy Optimization) is a policy-gradient RL algorithm that updates the LLM policy to maximize a reward signal from the reward model while staying close to a reference policy via a clipped surrogate objective. The RLHF loop: generate response $\rightarrow$ score with RM $\rightarrow$ update policy with PPO.

**Why PPO aggressively sharpens the output distribution:**

- **Reward maximization concentrates probability mass.** Once PPO identifies token sequences that score high, it rapidly increases their probability at the expense of alternatives.
- **Mode collapse / reward hacking.** The RM is an imperfect proxy; PPO exploits its peaks, concentrating probability on a narrow set of "high-scoring" outputs rather than maintaining diversity. (See [[#Reward Hacking: Deep Dive]] below.)
- **Weak entropy regularization.** Without a strong entropy bonus, the gradient signal from reward maximization dominates, collapsing the distribution toward greedy modes.
- **Finite KL budget.** The KL divergence constraint (vs. the SFT reference) slows but doesn't prevent sharpening. Given enough updates, the policy drifts toward low-entropy outputs that maximize cumulative reward within the KL budget.

Practical consequence: after PPO the model produces more confident, stylistically uniform outputs but loses coverage of the tail of valid responses. This motivates the "branch from SFT" default below.

### Why $\text{SFT}_i \rightarrow \text{SFT}_{i+1}$ is the Default

1. **Entropy collapse / mode narrowing**: PPO aggressively sharpens the output distribution. The policy commits to a few RM-preferred modes; rare-but-valid behaviors get squeezed out. SFT (cross-entropy MLE) on top of a low-entropy policy creates large gradients when targets are far from the current distribution, causing instability and uncontrolled erosion of RL gains. $\text{SFT}_i$ has a healthier entropy profile as a starting point.

2. **Reward hacking artifacts get baked in**: $\text{RL}_i$ inherits whatever pathologies $\text{RM}_i$ has: length bias, sycophancy, formatting tics, miscalibrated refusals. SFT teaches imitation, not avoidance, so these become the *prior* for the next round and are hard to wash out.

3. **KL reference cleanliness**: In $\text{RL}_{i+1}$, the KL penalty $\text{KL}(\pi \| \pi_\text{ref})$ uses $\text{SFT}_{i+1}$ as $\pi_\text{ref}$. You want $\pi_\text{ref}$ to be broad and well-behaved. If $\text{SFT}_{i+1}$ descends from $\text{RL}_i$, the reference is already RL-shaped, and the KL constraint anchors to biased behavior.

4. **Pipeline interpretability**: Keeping the SFT lineage as the trunk and RL as side-branches makes ablation/rollback clean. Continuing from RL entangles the stages.

5. **Optimization stability**: Empirically, SFT on top of heavily RL'd checkpoints is finicky: LR-sensitive, loss spikes, occasional catastrophic forgetting.

### When Starting from $\text{RL}_i$ is Fine

The "branch from SFT" rule is really about *uncurated, off-policy* SFT data. It relaxes when SFT data is on-policy w.r.t. $\text{RL}_i$:

- **Rejection sampling fine-tuning (RFT) / expert iteration** (Llama 2, reasoning-model recipes): sample from $\text{RL}_i$, filter via RM/verifier, SFT on survivors. The filter is the safety valve.
- **Iterative / online DPO**: each iteration continues from the previous policy. DPO is gentler on entropy than PPO.
- **Continuous / online RLHF**: no clean round boundary in the first place.

### Reward Hacking: Deep Dive

$\text{RM}_i$ is a *learned proxy* for human preferences, trained on finite data. PPO optimizes against this proxy, not the true objective. Goodhart's law: the policy finds regions where RM score is high but humans wouldn't actually prefer the output. The gap between proxy reward and true quality *is* reward hacking. Not malicious: just gradient descent on a flawed objective.

**Common patterns:**

| Pattern                    | Mechanism                                                                                   | Reference |
| -------------------------- | ------------------------------------------------------------------------------------------- | --------- |
| **Length bias**            | Annotators mildly prefer longer answers, RM learns "longer = better", PPO inflates length   | [^1] [^5] |
| **Sycophancy**             | Annotators rate agreeable answers higher, policy mirrors user's stated view even when wrong | [^3] [^4] |
| **Formatting tics**        | Markdown/headers/lists correlate with "looks organized", over-produced                      |           |
| **Refusal miscalibration** | Refusal proxy over-fires on benign requests, under-fires on rephrased unsafe ones           |           |
| **Confidence inflation**   | Hedged answers rated lower, unwarranted certainty, worse calibration, more hallucination    |           |
| **RM-specific exploits**   | Any quirk in $\text{RM}_i$ (token-position weighting, favored phrases) gets found by PPO    | [^2]      |

**Key empirical result** [^2]: as you optimize harder against the RM, *proxy reward keeps climbing while gold-standard reward eventually drops*. The gap is the hacking. Bigger RMs and more preference data push the turnover point out, but don't eliminate it.

**Why this compounds across rounds:**
1. **SFT can't easily unlearn diffuse biases.** Cross-entropy increases probability on demonstrated tokens but doesn't actively suppress diffuse output-distribution properties (length, sycophancy, formatting habits).
2. **KL protects hacked behaviors.** If $\text{SFT}_{i+1}$ inherits $\text{RL}_i$'s biases, the round-$(i+1)$ KL penalty *protects* those biases (deviating costs KL). Round $(i+1)$'s RM then adds its own layer of hacking on top, causing monotonic drift across rounds.

**KL penalty explained:** In RLHF, the PPO objective includes a KL divergence term:
$$
R_{\text{total}} = R_{\text{RM}}(\pi) - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})
$$

where $\pi_{\text{ref}}$ is the SFT checkpoint for that round. The KL term penalizes the policy for deviating too far from the reference, acting as a "safety leash" against reward hacking. However, this only works when $\pi_{\text{ref}}$ itself is clean.

**Monotonic drift across rounds (when branching from RL):**
- Round $i$: $\text{RM}_i$'s biases get hacked by PPO → $\text{RL}_i$ carries bias set A
- Round $i+1$: $\pi_{\text{ref}}$ inherits bias A (KL protects it), $\text{RM}_{i+1}$ adds new bias B on top
- Round $i+2$: $\pi_{\text{ref}}$ inherits bias A+B (KL protects both), $\text{RM}_{i+2}$ adds bias C
- ...biases only accumulate, never decay

The KL penalty inverts its purpose: correcting inherited biases *costs* KL (gets penalized), while preserving them *saves* KL (gets rewarded). This is why branching SFT from the previous SFT keeps $\pi_{\text{ref}}$ clean so KL can function as intended.

**Mitigations within a round:**
- **Length-controlled RMs:** Regress out length as a confounding variable before scoring, so PPO cannot exploit "longer = better."
- **KL penalty tuning / early stopping:** Adaptively adjust $\beta$ or halt training once $\text{KL}(\pi \| \pi_{\text{ref}})$ exceeds a threshold. Beyond that point, proxy reward may still rise but gold reward is already declining.
- **RM ensembling** [^6]: Average scores from multiple independently trained RMs. Individual quirks cancel out, making it harder for PPO to find an exploit that fools all RMs simultaneously.
- **Explicit length/format penalties:** Add direct penalty terms to the reward (e.g., $-\alpha_{\text{len}} \cdot \max(0, \text{len} - \text{len}_{\text{max}})$) to hard-cap known hacking modes. Here $\text{len}$ is the token length of the generated response and $\text{len}_{\text{max}}$ is a pre-set maximum length threshold. The penalty is zero when the response is within budget and linearly increases for every token beyond it.
- **Aggressive filtering in RFT-style pipelines:** Generate many candidates from the policy but only keep those passing strict filters (verifiers, length caps, factuality checks) for the next SFT round. Acts as a safety valve between RL output and SFT data.

None fully solve hacking; they push the Goodhart frontier outward.

## Component of

- [[LLM Post-Training Pipeline]] #todo
- [[Supervised Fine-Tuning (SFT)]] #todo 
- [[RLHF]] #todo
- [[Rejection Sampling Fine-Tuning]] #todo
- [[PPO]] #todo 

## Insights

- The "branch from SFT" default is really about protecting entropy and avoiding compounding RM biases across rounds.
- When SFT data is *on-policy* w.r.t. the RL checkpoint (RFT, online DPO), starting from $\text{RL}_i$ is safe because the filter/verifier acts as a safety valve.
- Reward hacking is not a bug in PPO; it's Goodhart's law applied to learned reward proxies. Mitigations push the frontier outward but don't eliminate it.

## Pitfalls

- **Assuming $\text{RL}_i \rightarrow \text{SFT}_{i+1}$ is always wrong**: it works well in expert iteration / RFT pipelines (Llama 2, reasoning models) where the SFT data is curated from the RL policy itself.
- **Ignoring entropy diagnostics**: not monitoring policy entropy across rounds lets mode collapse go undetected until quality degrades.
- **Over-relying on KL penalty alone**: KL constrains deviation from the reference but doesn't fix a biased reference. If $\pi_\text{ref}$ is already RL-shaped, the KL constraint anchors to biased behavior.

## Connections

- [[Reward Hacking]] #todo: Goodhart's law applied to learned reward proxies
- [[PPO]] #todo: the dominant RL algorithm for RLHF; responsible for entropy collapse
- [[DPO]] #todo: gentler alternative to PPO that avoids explicit reward modeling
- [[GRPO]] #todo: group relative policy optimization, used in DeepSeek-R1
- [[Rejection Sampling Fine-Tuning]] #todo: on-policy SFT variant where starting from RL checkpoint is safe

## References

[^1]: Stiennon et al. 2020. Learning to summarize with human feedback.
[^2]: Gao, Schulman & Hilton 2022. Scaling Laws for Reward Model Overoptimization.
[^3]: Perez et al. 2022. Discovering Language Model Behaviors with Model-Written Evaluations.
[^4]: Sharma et al. 2023. Towards Understanding Sycophancy in Language Models.
[^5]: Singhal et al. 2023. A Long Way to Go: Investigating Length Correlations in RLHF.
[^6]: Coste et al. 2023. Reward Model Ensembles Help Mitigate Overoptimization.


