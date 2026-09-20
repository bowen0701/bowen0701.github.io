---
author: Bowen Lee
date: '2026-06-24'
layout: post
tags:
- rlhf
- sft
- reward-hacking
- post-training
- llm
title: LLM Post-Training Iterative SFT, RLHF Branching
---
<details class="toc-details" markdown="1">
<summary><b>Table of Contents</b></summary>

* TOC
{:toc}

</details>


*[Updated on 2026-09-20: Add per-token KL divergence computation with derivation, pseudocode, and AV (MAGNIFIED) connection.]*

## Core Intuition

In multi-round LLM post-training, the standard pattern is:

- Round <span markdown="0">\(i\)</span>: <span markdown="0">\(\text{SFT}_i \rightarrow \text{RL}_i\)</span>
- Round <span markdown="0">\(i+1\)</span>: start from <span markdown="0">\(\text{SFT}_i\)</span> to train <span markdown="0">\(\text{SFT}_{i+1}\)</span>, then <span markdown="0">\(\text{SFT}_{i+1} \rightarrow \text{RL}_{i+1}\)</span>

Why not start round <span markdown="0">\(i+1\)</span>'s SFT from <span markdown="0">\(\text{RL}_i\)</span> instead?

Both are done in practice, but branching SFT from the previous SFT is the safer default. The core reason is that RL collapses entropy and bakes in reward-model biases; using that as the seed for the next round's SFT compounds those problems across rounds.

**Mental model:**
- **SFT** = broad imitation. Wants high-entropy, diverse data. Sensitive to distribution mismatch.
- **RL** = narrow optimization against a reward signal. Collapses entropy. Bakes in Reward Model (RM) bias.

Stacking SFT -> RL -> SFT -> RL means alternately broadening and narrowing and re-broadening and re-narrowing. The broadening step is more reliable when its starting point hasn't already been narrowed, hence the conservative default of branching SFT off SFT and treating each RL checkpoint as a leaf.

## Mathematical Foundation

### PPO in RLHF

PPO (Proximal Policy Optimization) is a policy-gradient RL algorithm that updates the LLM policy to maximize a reward signal from the reward model while staying close to a reference policy via a clipped surrogate objective. The RLHF loop: generate response <span markdown="0">\(\rightarrow\)</span> score with RM <span markdown="0">\(\rightarrow\)</span> update policy with PPO.

**Why PPO aggressively sharpens the output distribution:**

- **Reward maximization concentrates probability mass.** Once PPO identifies token sequences that score high, it rapidly increases their probability at the expense of alternatives.
- **Mode collapse / reward hacking.** The RM is an imperfect proxy; PPO exploits its peaks, concentrating probability on a narrow set of "high-scoring" outputs rather than maintaining diversity. (See #Reward Hacking: Deep Dive below.)
- **Weak entropy regularization.** Without a strong entropy bonus, the gradient signal from reward maximization dominates, collapsing the distribution toward greedy modes.
- **Finite KL budget.** The KL divergence constraint (vs. the SFT reference) slows but doesn't prevent sharpening. Given enough updates, the policy drifts toward low-entropy outputs that maximize cumulative reward within the KL budget.

Practical consequence: after PPO the model produces more confident, stylistically uniform outputs but loses coverage of the tail of valid responses. This motivates the "branch from SFT" default below.

### Why SFT_i → SFT_{i+1} is the Default

1. **Entropy collapse / mode narrowing**: PPO aggressively sharpens the output distribution. The policy commits to a few RM-preferred modes; rare-but-valid behaviors get squeezed out. SFT (cross-entropy MLE) on top of a low-entropy policy creates large gradients when targets are far from the current distribution, causing instability and uncontrolled erosion of RL gains. <span markdown="0">\(\text{SFT}_i\)</span> has a healthier entropy profile as a starting point.

2. **Reward hacking artifacts get baked in**: <span markdown="0">\(\text{RL}_i\)</span> inherits whatever pathologies <span markdown="0">\(\text{RM}_i\)</span> has: length bias, sycophancy, formatting tics, miscalibrated refusals. SFT teaches imitation, not avoidance, so these become the *prior* for the next round and are hard to wash out.

3. **KL reference cleanliness**: In <span markdown="0">\(\text{RL}_{i+1}\)</span>, the KL penalty <span markdown="0">\(\text{KL}(\pi \,\Vert\, \pi_\text{ref})\)</span> uses <span markdown="0">\(\text{SFT}_{i+1}\)</span> as <span markdown="0">\(\pi_\text{ref}\)</span>. You want <span markdown="0">\(\pi_\text{ref}\)</span> to be broad and well-behaved. If <span markdown="0">\(\text{SFT}_{i+1}\)</span> descends from <span markdown="0">\(\text{RL}_i\)</span>, the reference is already RL-shaped, and the KL constraint anchors to biased behavior.

4. **Pipeline interpretability**: Keeping the SFT lineage as the trunk and RL as side-branches makes ablation/rollback clean. Continuing from RL entangles the stages.

5. **Optimization stability**: Empirically, SFT on top of heavily RL'd checkpoints is finicky: LR-sensitive, loss spikes, occasional catastrophic forgetting.

### When Starting from RL_i is Fine

The "branch from SFT" rule is really about *uncurated, off-policy* SFT data. It relaxes when SFT data is on-policy w.r.t. <span markdown="0">\(\text{RL}_i\)</span>:

- **Rejection sampling fine-tuning (RFT) / expert iteration** (Llama 2, reasoning-model recipes): sample from <span markdown="0">\(\text{RL}_i\)</span>, filter via RM/verifier, SFT on survivors. The filter is the safety valve.
- **Iterative / online DPO**: each iteration continues from the previous policy. DPO is gentler on entropy than PPO.
- **Continuous / online RLHF**: no clean round boundary in the first place.

### Reward Hacking: Deep Dive

<span markdown="0">\(\text{RM}_i\)</span> is a *learned proxy* for human preferences, trained on finite data. PPO optimizes against this proxy, not the true objective. 
- **Goodhart's law:** the policy finds regions where RM score is high but humans wouldn't actually prefer the output. 
- **Reward hacking**: The gap between proxy reward and true quality *is* reward hacking. Not malicious: just gradient descent on a flawed objective.

**Common patterns:**

| Pattern                    | Mechanism                                                                                   | Reference                                     |
| -------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------- |
| **Length bias**            | Annotators mildly prefer longer answers, RM learns "longer = better", PPO inflates length   | Stiennon et al. 2020, Singhal et al. 2023 |
| **Sycophancy**             | Annotators rate agreeable answers higher, policy mirrors user's stated view even when wrong | Perez et al. 2022, Sharma et al. 2023     |
| **Formatting tics**        | Markdown/headers/lists correlate with "looks organized", over-produced                      |                                           |
| **Refusal miscalibration** | Refusal proxy over-fires on benign requests, under-fires on rephrased unsafe ones           |                                           |
| **Confidence inflation**   | Hedged answers rated lower, unwarranted certainty, worse calibration, more hallucination    |                                           |
| **RM-specific exploits**   | Any quirk in <span markdown="0">\(\text{RM}_i\)</span> (token-position weighting, favored phrases) gets found by PPO    | Gao et al. 2022                           |

**Key empirical result** (Gao et al. 2022): as you optimize harder against the RM, *proxy reward keeps climbing while gold-standard reward eventually drops*. The gap is the hacking. Bigger RMs and more preference data push the turnover point out, but don't eliminate it.

**Why this compounds across rounds:**
1. **SFT can't easily unlearn diffuse biases.** Cross-entropy increases probability on demonstrated tokens but doesn't actively suppress diffuse output-distribution properties (length, sycophancy, formatting habits).
2. **KL protects hacked behaviors.** If <span markdown="0">\(\text{SFT}_{i+1}\)</span> inherits <span markdown="0">\(\text{RL}_i\)</span>'s biases, the round-<span markdown="0">\((i+1)\)</span> KL penalty *protects* those biases (deviating costs KL). Round <span markdown="0">\((i+1)\)</span>'s RM then adds its own layer of hacking on top, causing monotonic drift across rounds.

**KL penalty explained:** In RLHF, the PPO objective includes a KL divergence term:
$$
R_{\text{total}} = R_{\text{RM}}(\pi) - \beta \cdot \text{KL}(\pi \,\Vert\, \pi_{\text{ref}})
$$

where <span markdown="0">\(\pi_{\text{ref}}\)</span> is the SFT checkpoint for that round. The KL term penalizes the policy for deviating too far from the reference, acting as a "safety leash" against reward hacking. However, this only works when <span markdown="0">\(\pi_{\text{ref}}\)</span> itself is clean.

**How KL is computed (per-token):** KL divergence is defined as <span markdown="0">\(D_{KL}(P \Vert Q) = \mathbb{E}_{x \sim P}[\log \frac{P(x)}{Q(x)}]\)</span>. The <span markdown="0">\(\pi_{ref}\)</span> in the denominator is not importance sampling: it's the definition of KL: "how much more likely is each token under <span markdown="0">\(\pi_\theta\)</span> than under <span markdown="0">\(\pi_{ref}\)</span>?" It's "on-policy" because the expectation samples from <span markdown="0">\(\pi_\theta\)</span> (the first argument of KL), so no importance weights are needed. The true KL:

$$
D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{y \sim \pi_\theta}\!\left[\sum_{t=1}^{T} \log \frac{\pi_\theta(a_t \mid x, a_{<t})}{\pi_{ref}(a_t \mid x, a_{<t})}\right]
$$

For a single sampled sequence <span markdown="0">\(y = (a_1, \ldots, a_T)\)</span> from <span markdown="0">\(\pi_\theta\)</span>, the **unbiased Monte Carlo estimate** is:

$$
\hat{D}_{KL} = \sum_{t=1}^{T} \left[\log \pi_\theta(a_t \mid x, a_{<t}) - \log \pi_{ref}(a_t \mid x, a_{<t})\right]
$$

PPO averages over the batch of rollouts to reduce variance. Critical: tokens must be sampled from <span markdown="0">\(\pi_\theta\)</span> **(on-policy)**. If from a dataset or <span markdown="0">\(\pi_{ref}\)</span>, you'd need **importance weights**. Implementation:

```python
# Forward pass through BOTH models on the same sampled tokens
logits_policy = policy.forward(prompt + tokens)      # (T, vocab)
logits_ref    = ref_model.forward(prompt + tokens)    # (T, vocab) frozen

# Per-token KL = difference of log-probs of the sampled token
log_p_policy = log_softmax(logits_policy).gather(tokens)  # (T,)
log_p_ref    = log_softmax(logits_ref).gather(tokens)      # (T,)
kl_per_token = log_p_policy - log_p_ref                    # (T,)

# Penalty in reward
reward_with_kl = reward - beta * kl_per_token.sum()
```

- <span markdown="0">\(\pi_{ref}\)</span> is **frozen**: doubles GPU memory (must hold both models)
- **Per-token** penalty, not per-sequence: prevents drift at any position
- For AV (MAGNIFIED): tokens = waypoints, <span markdown="0">\(\pi_{ref}\)</span> = frozen IL-pretrained model

**Monotonic drift across rounds (when branching from RL):**
- Round <span markdown="0">\(i\)</span>: <span markdown="0">\(\text{RM}_i\)</span>'s biases get hacked by PPO → <span markdown="0">\(\text{RL}_i\)</span> carries bias set A
- Round <span markdown="0">\(i+1\)</span>: <span markdown="0">\(\pi_{\text{ref}}\)</span> inherits bias A (KL protects it), <span markdown="0">\(\text{RM}_{i+1}\)</span> adds new bias B on top
- Round <span markdown="0">\(i+2\)</span>: <span markdown="0">\(\pi_{\text{ref}}\)</span> inherits bias A+B (KL protects both), <span markdown="0">\(\text{RM}_{i+2}\)</span> adds bias C
- ...biases only accumulate, never decay

The **KL penalty inverts its purpose: correcting inherited biases *costs* KL (gets penalized), while preserving them *saves* KL (gets rewarded)**. This is why branching SFT from the previous SFT keeps <span markdown="0">\(\pi_{\text{ref}}\)</span> clean so KL can function as intended.

**Mitigations within a round:**
- **Length-controlled RMs:** Regress out length as a confounding variable before scoring, so PPO cannot exploit "longer = better."
- **KL penalty tuning / early stopping:** Adaptively adjust <span markdown="0">\(\beta\)</span> or halt training once <span markdown="0">\(\text{KL}(\pi \,\Vert\, \pi_{\text{ref}})\)</span> exceeds a threshold. Beyond that point, proxy reward may still rise but gold reward is already declining.
- **RM ensembling** (Coste et al. 2023): Average scores from multiple independently trained RMs. Individual quirks cancel out, making it harder for PPO to find an exploit that fools all RMs simultaneously.
- **Explicit length/format penalties:** Add direct penalty terms to the reward (e.g., <span markdown="0">\(-\alpha_{\text{len}} \cdot \max(0, \text{len} - \text{len}_{\text{max}})\)</span>) to hard-cap known hacking modes. Here <span markdown="0">\(\text{len}\)</span> is the token length of the generated response and <span markdown="0">\(\text{len}_{\text{max}}\)</span> is a pre-set maximum length threshold. The penalty is zero when the response is within budget and linearly increases for every token beyond it.
- **Aggressive filtering in RFT-style pipelines:** Generate many candidates from the policy but only keep those passing strict filters (verifiers, length caps, factuality checks) for the next SFT round. Acts as a safety valve between RL output and SFT data.

None fully solve hacking; they push the Goodhart frontier outward.

## Component of

- LLM Post-Training Pipeline #todo
- Supervised Fine-Tuning (SFT) #todo 
- RLHF #todo
- Rejection Sampling Fine-Tuning #todo
- PPO #todo 

## Insights

- The "branch from SFT" default is really about protecting entropy and avoiding compounding RM biases across rounds.
- When SFT data is *on-policy* w.r.t. the RL checkpoint (RFT, online DPO), starting from <span markdown="0">\(\text{RL}_i\)</span> is safe because the filter/verifier acts as a safety valve.
- Reward hacking is not a bug in PPO; it's Goodhart's law applied to learned reward proxies. Mitigations push the frontier outward but don't eliminate it.

## Pitfalls

- **Assuming <span markdown="0">\(\text{RL}_i \rightarrow \text{SFT}_{i+1}\)</span> is always wrong**: it works well in expert iteration / RFT pipelines (Llama 2, reasoning models) where the SFT data is curated from the RL policy itself.
- **Ignoring entropy diagnostics**: not monitoring policy entropy across rounds lets mode collapse go undetected until quality degrades.
- **Over-relying on KL penalty alone**: KL constrains deviation from the reference but doesn't fix a biased reference. If <span markdown="0">\(\pi_\text{ref}\)</span> is already RL-shaped, the KL constraint anchors to biased behavior.

## Connections

- Reward Hacking #todo: Goodhart's law applied to learned reward proxies
- PPO #todo: the dominant RL algorithm for RLHF; responsible for entropy collapse
- DPO #todo: gentler alternative to PPO that avoids explicit reward modeling
- GRPO #todo: group relative policy optimization, used in DeepSeek-R1
- Rejection Sampling Fine-Tuning #todo: on-policy SFT variant where starting from RL checkpoint is safe

## References

- Stiennon et al. (2020). Learning to summarize with human feedback.
- Gao, Schulman & Hilton (2022). Scaling Laws for Reward Model Overoptimization.
- Perez et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations.
- Sharma et al. (2023). Towards Understanding Sycophancy in Language Models.
- Singhal et al. (2023). A Long Way to Go: Investigating Length Correlations in RLHF.
- Coste et al. (2023). Reward Model Ensembles Help Mitigate Overoptimization.


