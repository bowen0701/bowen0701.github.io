---
title: "Manual Linear Layer"
created: 2026-05-13
publish: true
tags: [deep-learning, math, backpropagation, implementation]
---
# 🧠 Concept: Manual Linear Layer

## Core Intuition
A Linear Layer (or Fully Connected/Dense layer) performs an affine transformation on input data. It is the fundamental building block of neural networks, mapping an input vector to an output vector through weight multiplication and bias addition.

## Mathematical Foundation
For a layer with input $X \in \mathbb{R}^{B \times D_{in}}$, weights $W \in \mathbb{R}^{D_{out} \times D_{in}}$, and bias $b \in \mathbb{R}^{D_{out}}$:

### Forward Pass
The output $Z \in \mathbb{R}^{B \times D_{out}}$ is calculated as:
$$Z = XW^T + b$$

### Backward Pass (Derivation)
Given the gradient of the loss with respect to the output $\frac{\partial L}{\partial Z} \in \mathbb{R}^{B \times D_{out}}$, we need to calculate:

1. **Gradient w.r.t. Weights ($W$):**
   To find $\frac{\partial L}{\partial W}$, consider the element $W_{jk}$ (weight connecting input $k$ to output $j$).
   From $Z = XW^T + b$, an element of the output is: $z_{ij} = \sum_{k} x_{ik} W_{jk} + b_j$.
   The partial derivative of a single output $z_{ij}$ with respect to $W_{jk}$ is:
   $$\frac{\partial z_{ij}}{\partial W_{jk}} = x_{ik}$$
   Using the chain rule, we sum the contributions of $W_{jk}$ to all outputs it affected (which are only the $j$-th column of $Z$):
   $$\frac{\partial L}{\partial W_{jk}} = \sum_{i=1}^{B} \frac{\partial L}{\partial z_{ij}} \frac{\partial z_{ij}}{\partial W_{jk}} = \sum_{i=1}^{B} \frac{\partial L}{\partial z_{ij}} x_{ik}$$
   This summation corresponds to the dot product between the $j$-th column of $\frac{\partial L}{\partial Z}$ and the $k$-th column of $X$. In matrix form, this is:
   $$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^T X$$
   Resulting shape: $(D_{out} \times B) \times (B \times D_{in}) = (D_{out} \times D_{in})$.

2. **Gradient w.r.t. Bias ($b$):**
   The bias $b_j$ is added to every sample in the batch for the $j$-th output dimension ($z_{ij} = \text{linear\_term}_i + b_j$). Thus, $\frac{\partial z_{ij}}{\partial b_j} = 1$. By the chain rule:
   $$\frac{\partial L}{\partial b_j} = \sum_{i=1}^{B} \frac{\partial L}{\partial z_{ij}} \frac{\partial z_{ij}}{\partial b_j} = \sum_{i=1}^{B} \frac{\partial L}{\partial z_{ij}}$$
   In vector form, the gradient for the bias vector is the sum of gradients across the batch:
   $$\frac{\partial L}{\partial b} = \sum_{batch} \frac{\partial L}{\partial Z}$$
   Resulting shape: $(1 \times D_{out})$.
3. **Gradient w.r.t. Input ($X$):**
   Using the chain rule: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot \frac{\partial Z}{\partial X}$
   Since $Z = XW^T + b$, then $\frac{\partial Z}{\partial X} = W$.
   $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W = {\partial Z} \cdot W$$
   Resulting shape: $(B \times D_{out}) \times (D_{out} \times D_{in}) = (B \times D_{in})$.

## Key Equation
$$Z = XW^T + b$$
$$\nabla_W L = (\nabla_Z L)^T X$$
$$\nabla_X L = (\nabla_Z L) W$$
## Intuitive Gradient Rules
For quick derivation of $Z = XW^T + b$, use these mental shortcuts:
- **Dimension Matching:** If $\nabla_Z L$ is $(B \times D_{out})$ and you need $\nabla_W L$ as $(D_{out} \times D_{in})$, the only valid matrix product is $(\nabla_Z L)^T X$.
- **The "Mirror" Rule:** To find the gradient of a term, you multiply the incoming gradient ($\nabla_Z L$) by the *other* term in the product, transposed.
    - For $W$: multiply by $X^T$ (adjusted for the $W^T$ layout in $Z = XW^T + b$).
    - For $X$: multiply by $W$.
- **Batch Pooling (for Bias):** Since the bias $b$ is "shared" (added to every single sample), its gradient must aggregate the error signals from the entire batch. Summing across the batch is the natural way to "pool" this shared influence.

## Analogy
Think of a linear layer as a **projection screen**. The input is the object, the weights are the angle and properties of the lens that project it into a new space (dimension), and the bias is the translation (moving the projection on the screen).

## Component of
- [[Multi-Layer Perceptron (MLP)]]
- [[Transformer]] (Feed-forward blocks)
- [[Convolutional Neural Networks]] (Final classification layers)

## Insights  
- **Affine Transformation:** It's linear transformation + translation.
- **Dimensionality Change:** Used to expand or compress the feature space.
- **Weight Shape:** In PyTorch, weights are stored as `(out_features, in_features)` to optimize the matrix multiplication as $XW^T$.

## Pitfalls  
- **Vanishing Gradients:** Without non-linear activations, stacking linear layers is equivalent to a single linear layer.
- **Initialization:** Poor initialization (e.g., all zeros) leads to broken symmetry where all neurons learn the same features.

## Connections
- [[Backpropagation]]
- [[Chain Rule]]
- [[Xavier Initialization]]
- [[He Initialization]]

## Implementation Notes
```python
import numpy as np
from typing import Tuple, Dict, Any

class Linear:
	"""
	Linear Layer with manual backward pass implementation.
	
	Architecture Note: Stateless/Functional Pattern
	----------------------------------------------
	This implementation follows a stateless pattern where the layer does not 
	internally store the forward pass data (X). Instead, it returns a 'cache'.
	
	Why this pattern?
	1. Thread-Safety: Allows the same layer instance to be used in parallel.
	2. Weight Sharing: The same layer can be called multiple times in a single 
	   computational graph without overwriting internal state.
	3. Explicit Backprop: Clearly demonstrates exactly which tensors from the 
	   forward pass are required to compute gradients.
	
	Who 'catches' the cache? 
	In a full framework, a 'Model' or 'Sequential' container stores these 
	caches in a list during forward() and provides them back to the layers 
	in reverse order during backward().
	"""

    def __init__(self, in_features: int, out_features: int):
        # He initialization for ReLU networks
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        # Gradients stored after backward()
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Z = XW^T + b
        X shape: (batch_size, in_features)
        W shape: (out_features, in_features)
        b shape: (1, out_features)
        """
        Z = np.dot(X, self.W.T) + self.b
        cache = {"X": X}
        return Z, cache

    def backward(self, dZ: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """
        dZ shape: (batch_size, out_features)
        """
        X = cache["X"]
        batch_size = X.shape[0]

        # 1. Gradient wrt Weights: (out, batch) @ (batch, in) -> (out, in)
        self.dW = np.dot(dZ.T, X)

        # 2. Gradient wrt Bias: sum across batch
        self.db = np.sum(dZ, axis=0, keepdims=True)

        # 3. Gradient wrt Input (to pass to previous layer): (batch, out) @ (out, in) -> (batch, in)
        dX = np.dot(dZ, self.W)

        return dX
```
