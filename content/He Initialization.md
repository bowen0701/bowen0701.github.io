---
title: "He Initialization"
author: Bowen Lee
created: 2026-05-13
publish: true
tags: [deep-learning, initialization, relu]
---
# Concept: He Initialization

## Core Intuition
Designed to solve the vanishing/exploding gradient problem in deep networks using **ReLU** activations. It ensures that the variance of activations stays roughly constant across layers to enable effective learning in deep architectures.

## Mathematical Foundation
1. **Variance Preservation:** For deep networks to learn, the variance of activations should stay roughly constant across layers.
2. **The ReLU Problem:** ReLU zeros out half of the input distribution (negative values). This halves the variance at every layer.
3. **The Compensation:** While Xavier initialization uses $\sqrt{1/n_{in}}$, He initialization doubles the variance (using a factor of 2 in the numerator) to compensate for the 50% signal loss caused by ReLU.

## Key Equation

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

Code:
```python
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
```

## Analogy
Like adjusting a volume knob to be twice as loud because you know half of your speakers are muted—keeping the total sound level consistent across the room.

## Component of
- [[Deep Neural Networks]] #todo
- [[ReLU Activation]] #todo

## Insights  
- **Xavier vs. He:** Xavier assumes symmetric activations (Tanh/Sigmoid), whereas He is tailored for asymmetric rectifiers.
- **Standard for ReLU:** He initialization is the industry standard for ReLU, Leaky ReLU, and GELU activations.
- **Dead Neurons:** Proper scaling ensures that in the early stages of training, enough neurons are in the "active" region of the ReLU.

## Pitfalls  
- Using He initialization with Tanh or Sigmoid can lead to exploding gradients as the variance will be too high for those bounded activations.

## Connections
- [[Xavier Initialization]] #todo
- [[Gradient Vanishing]] #todo

## Implementation Notes
- The "He Normal" version uses a truncated normal distribution.
- Can also be implemented as "He Uniform" with a range of $[-\sqrt{6/n_{in}}, \sqrt{6/n_{in}}]$.
