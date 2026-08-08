---
title: "Xavier Initialization vs He Initialization"
author: Bowen Lee
created: 2026-05-13
publish: true
tags: [deep-learning, initialization, xavier, relu]
---
# Concept: Xavier Initialization vs He Initialization

[Updated on 2026-08-07: Extend to Xavier Initialization vs He Initialization.]

## Core Intuition

Both methods solve the same problem: **keep the variance of activations stable across layers** so gradients neither vanish nor explode. They differ only in what activation function they assume.

## Mathematical Foundation

Consider a single neuron with output $y = \sum_{i=1}^{n_{in}} w_i x_i$, where weights and inputs are independent and zero-mean:

$$
\text{Var}(y) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

For deep networks to learn, we need $\text{Var}(y) \approx \text{Var}(x)$ (variance preservation) so the signal doesn't shrink or blow up layer after layer.

### Xavier: linear regime

For variance preservation ($\text{Var}(y) = \text{Var}(x)$):

$$
n_{in} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{in}} \implies \text{std}(w) = \sqrt{\frac{1}{n_{in}}}
$$

$$
W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)
$$

This holds when the activation is approximately linear around zero (Tanh, Sigmoid in their linear regime).

```python
W = np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)
```

### He: compensating for ReLU

ReLU zeros out all negative values, halving the variance that passes through:

$$
\text{Var}(y_{\text{after ReLU}}) \approx \frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

To compensate for the factor of $\frac{1}{2}$:

$$
\frac{1}{2} \cdot n_{in} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{2}{n_{in}} \implies \text{std}(w) = \sqrt{\frac{2}{n_{in}}}
$$

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

```python
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
```

**Analogy:** Like adjusting a volume knob to be twice as loud because you know half of your speakers are muted, keeping the total sound level consistent across the room.

## Comparison

| | Xavier | He |
|---|---|---|
| **Variance** | $1 / n_{in}$ | $2 / n_{in}$ |
| **Std** | $\sqrt{1/n_{in}}$ | $\sqrt{2/n_{in}}$ |
| **Assumes** | Linear activation (Tanh, Sigmoid) | ReLU (half the distribution zeroed) |
| **Key insight** | Preserve variance as-is | Multiply by 2 to compensate for 50% signal loss |

## Practical Guidance

- **He** is the industry standard for ReLU, Leaky ReLU, and GELU activations.
- **Xavier** is preferred for Tanh and Sigmoid.
- Using He with Tanh/Sigmoid can lead to exploding gradients (variance too high for bounded activations).
- Proper scaling ensures enough neurons start in the "active" region, preventing dead neurons early in training.

## Implementation Notes

- **He Normal** uses a truncated normal distribution.
- **He Uniform** uses a range of $[-\sqrt{6/n_{in}}, \sqrt{6/n_{in}}]$.
- **Xavier Uniform** (Glorot) uses a range of $[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$, averaging fan-in and fan-out.

## Connections

- Deep Neural Networks #todo
- ReLU Activation #todo
- Gradient Vanishing #todo

## References

- Glorot & Bengio (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.
- He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv:1502.01852
