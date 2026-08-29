---
author: Bowen Lee
date: '2026-05-15'
layout: post
tags:
- deep-learning
- math
- backpropagation
- implementation
title: Manual Layers
---
<details class="toc-details" markdown="1">
<summary><b>Table of Contents</b></summary>

* TOC
{:toc}

</details>


**Repo**: [dl_eng/models/manual_layers.py](https://github.com/bowen0701/dl-eng/blob/main/dl_eng/models/manual_layers.py)

---

## Linear Layer

### Core Intuition
A Linear Layer (or Fully Connected/Dense layer) performs an affine transformation on input data. It is the fundamental building block of neural networks, mapping an input vector to an output vector through weight multiplication and bias addition.

### Mathematical Foundation
For a layer with input $X \in \mathbb{R}^{B \times D_{in}}$, weights $W \in \mathbb{R}^{D_{out} \times D_{in}}$, and bias $b \in \mathbb{R}^{D_{out}}$:

#### Forward Pass
The output $Z \in \mathbb{R}^{B \times D_{out}}$ is calculated as:

$$Z = XW^T + b$$

#### Backward Pass (Derivation)
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

   $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W$$

   Resulting shape: $(B \times D_{out}) \times (D_{out} \times D_{in}) = (B \times D_{in})$.

### Key Equation

$$Z = XW^T + b$$

$$\nabla_W L = (\nabla_Z L)^T X$$

$$\nabla_X L = (\nabla_Z L) W$$

### Intuitive Gradient Rules
For quick derivation of $Z = XW^T + b$, use these mental shortcuts:
- **Dimension Matching:** If $\nabla_Z L$ is $(B \times D_{out})$ and you need $\nabla_W L$ as $(D_{out} \times D_{in})$, the only valid matrix product is $(\nabla_Z L)^T X$.
- **The "Mirror" Rule:** To find the gradient of a term, you multiply the incoming gradient ($\nabla_Z L$) by the *other* term in the product, transposed.
    - For $W$: multiply by $X^T$ (adjusted for the $W^T$ layout in $Z = XW^T + b$).
    - For $X$: multiply by $W$.
- **Batch Pooling (for Bias):** Since the bias $b$ is "shared" (added to every single sample), its gradient must aggregate the error signals from the entire batch. Summing across the batch is the natural way to "pool" this shared influence.

### Analogy
Think of a linear layer as a **projection screen**. The input is the object, the weights are the angle and properties of the lens that project it into a new space (dimension), and the bias is the translation (moving the projection on the screen).

### Component of
- Multi-Layer Perceptron (MLP) #todo
- Transformer (Feed-forward blocks)
- Convolutional Neural Networks (Final classification layers) #todo

### Insights
- **Affine Transformation:** It's linear transformation + translation.
- **Dimensionality Change:** Used to expand or compress the feature space.
- **Weight Shape:** In PyTorch, weights are stored as `(out_features, in_features)` to optimize the matrix multiplication as $XW^T$.

### Pitfalls
- **Vanishing Gradients:** Without non-linear activations, stacking linear layers is equivalent to a single linear layer.
- **Initialization:** Poor initialization (e.g., all zeros) leads to broken symmetry where all neurons learn the same features.

### Connections
- Backpropagation #todo
- Chain Rule #todo
- [Xavier Initialization vs He Initialization](/re-log/xavier-initialization-vs-he-initialization/)

### Implementation
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

---

## ReLU (Rectified Linear Unit)

### Core Intuition
Element-wise gating: pass the input through if positive, block it (zero) otherwise. Introduces non-linearity so stacked linear layers can learn non-linear functions. No learnable parameters.

### Mathematical Foundation

#### Forward Pass
$$A = \max(0, Z)$$

Element-wise: $a_i = \max(0, z_i)$.

#### Backward Pass (Derivation)
The derivative of ReLU is a step function:

$$\frac{\partial a_i}{\partial z_i} = \mathbf{1}(z_i > 0) = \begin{cases} 1 & \text{if } z_i > 0 \\ 0 & \text{if } z_i \le 0 \end{cases}$$

By the chain rule, the gradient flows through unchanged where the input was positive, and is blocked where it was non-positive:

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \mathbf{1}(Z > 0)$$

where $\odot$ is element-wise (Hadamard) multiplication. This is a **gradient gate**: the binary mask $\mathbf{1}(Z > 0)$ from the forward pass decides which gradients survive.

### Key Equation

$$A = \max(0, Z)$$

$$\nabla_Z L = \nabla_A L \odot \mathbf{1}(Z > 0)$$

### Pitfalls
- **Dying ReLU**: If a neuron's pre-activation $Z \le 0$ for all inputs in the dataset, its gradient is permanently zero and it stops learning. Common with large learning rates that push weights too negative.
- **Variants**: LeakyReLU ($\max(\alpha z, z)$, $\alpha = 0.01$) and GELU (used in Transformers) allow small gradients for negative inputs.

### Implementation
```python
class ReLU:
    """ReLU Activation: element-wise, no learnable parameters."""

    def forward(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward: A = max(0, Z).  Shapes: Z: (N, D), A: (N, D)."""
        A = np.maximum(0, Z)
        cache = {"Z": Z}
        return A, cache

    def backward(self, dA: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Backprop: dL/dZ = dA * 1(Z > 0).  Shapes: dA: (N, D), dZ: (N, D)."""
        Z = cache["Z"]
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ
```

---

## Sigmoid (Logistic)

### Core Intuition
Squashes any real number into $(0, 1)$, acting as a smooth "switch." Used as the output activation for binary classification and as a gating mechanism in LSTMs/GRUs. No learnable parameters.

### Mathematical Foundation

#### Forward Pass

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

#### Backward Pass (Derivation)
The sigmoid has an elegant derivative expressed purely in terms of its own output:

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**Derivation**: Let $\sigma = (1 + e^{-z})^{-1}$. By the chain rule:

$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma \cdot (1 - \sigma)$$

With $A = \sigma(Z)$, applying the chain rule:

$$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot A \odot (1 - A)$$

The gradient is computed entirely from the forward output $A$: no need to cache the input $Z$.

### Key Equation

$$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$

$$\nabla_Z L = \nabla_A L \odot A \odot (1 - A)$$

### Fused Sigmoid + Binary Cross-Entropy Gradient

The binary analogue of #Fused Softmax + Cross-Entropy Gradient. In practice, sigmoid is always fused with BCE loss, and the fused gradient has the same elegant form.

#### Setup

For a single sample with logit $z \in \mathbb{R}$, binary label $y \in \{0, 1\}$, and sigmoid output $a = \sigma(z)$:

$$L = -[y \log a + (1 - y) \log(1 - a)]$$

**Goal**: Find $\frac{\partial L}{\partial z}$ directly.

#### Derivation

**Step 1: Chain rule through sigmoid.**

$$\frac{\partial L}{\partial z} = -\left[y \cdot \frac{\sigma'(z)}{\sigma(z)} + (1 - y) \cdot \frac{-\sigma'(z)}{1 - \sigma(z)}\right]$$

**Step 2: Substitute $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.** Both denominators cancel:

$$= -\left[y(1 - \sigma(z)) - (1 - y)\sigma(z)\right]$$

**Step 3: Expand and simplify.**

$$= -\left[y - y\sigma(z) - \sigma(z) + y\sigma(z)\right] = -[y - \sigma(z)] = \sigma(z) - y$$

In vector form:

$$\boxed{\nabla_z L = \sigma(z) - y}$$

This is identical in structure to the softmax CE result $\nabla_z L = p - \text{onehot}(y)$: predicted minus truth.

#### Numerically stable forward

The naive computation $-[y \log \sigma(z) + (1-y) \log(1 - \sigma(z))]$ is unstable because $\sigma(z)$ saturates to 0 or 1, causing $\log(0)$. The stable form avoids computing $\sigma(z)$ explicitly:

$$L = \max(z, 0) - zy + \log(1 + e^{-\lvert z \rvert})$$

**Derivation**: Start from the BCE definition and eliminate $\sigma(z)$:

$$L = -[y \log \sigma(z) + (1 - y) \log(1 - \sigma(z))]$$

**Step 1**: Rewrite the log-sigmoid terms. Since $\sigma(z) = \frac{1}{1+e^{-z}}$ and $1 - \sigma(z) = \frac{1}{1+e^{z}}$:

$$\log \sigma(z) = -\log(1 + e^{-z}), \quad \log(1 - \sigma(z)) = -\log(1 + e^{z})$$

**Step 2**: Substitute into BCE:

$$L = y \log(1 + e^{-z}) + (1 - y) \log(1 + e^{z})$$

**Step 3**: Unify the two $\log$ terms. The key identity is $\log(1 + e^{-z}) = \log(1 + e^z) - z$, which follows from multiplying inside the log by $\frac{e^z}{e^z}$:

$$\log(1 + e^{-z}) = \log\frac{e^z + 1}{e^z} = \log(1 + e^z) - z$$

Substituting:

$$L = y[\log(1 + e^z) - z] + (1-y)\log(1 + e^z)$$

$$= y\log(1 + e^z) - yz + \log(1 + e^z) - y\log(1 + e^z)$$

$$= -zy + \log(1 + e^z)$$

**Step 4**: Make it overflow-safe. The term $e^z$ overflows for large positive $z$. Rewrite the key identity $\log(1 + e^z) = \max(z, 0) + \log(1 + e^{-\lvert z \rvert})$:
- If $z \ge 0$: $\log(1 + e^z) = z + \log(e^{-z} + 1) = z + \log(1 + e^{-z})$, and $e^{-z} \le 1$.
- If $z < 0$: $\log(1 + e^z) = \log(1 + e^z)$, and $e^z < 1$.

This gives the final stable form (what PyTorch's `F.binary_cross_entropy_with_logits` uses):

$$L = \max(z, 0) - zy + \log(1 + e^{-\lvert z \rvert})$$

#### Batched form

For $N$ samples with logits $z \in \mathbb{R}^{N}$ and labels $y \in \{0, 1\}^{N}$:

$$\nabla_z L = \frac{1}{N}(\sigma(z) - y)$$

### Pitfalls
- **Saturation**: For large $\lvert z \rvert$, $\sigma'(z) \approx 0$, causing vanishing gradients. This is why ReLU is preferred in hidden layers.
- **Not zero-centered**: Outputs are always positive, which can cause zig-zag gradient updates. BatchNorm or careful initialization helps.
- **Numerical stability**: For very negative $z$, $e^{-z}$ overflows. Branch: use $\frac{1}{1+e^{-z}}$ for $z \ge 0$, $\frac{e^z}{1+e^z}$ for $z < 0$ (implementation below uses `np.where` for this).

### Implementation
```python
def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: branch on sign of z."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class Sigmoid:
    """Sigmoid Activation: element-wise, no learnable parameters."""

    def forward(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward: A = sigmoid(Z), numerically stable.
        Shapes: Z: (N, D), A: (N, D)."""
        A = _stable_sigmoid(Z)
        cache = {"A": A}  # only need output, not input
        return A, cache

    def backward(self, dA: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Backprop: dL/dZ = dA * A * (1 - A).  Shapes: dA: (N, D), dZ: (N, D)."""
        A = cache["A"]
        dZ = dA * A * (1 - A)
        return dZ


class BinaryCrossEntropyLoss:
    """Fused sigmoid + binary cross-entropy loss. No learnable parameters.
    Operates on raw logits for numerical stability."""

    def forward(self, Z: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward: L = max(z,0) - zy + log(1 + exp(-|z|)).
        Z: (N,) or (N, 1) logits, y: (N,) or (N, 1) binary labels.
        Returns (scalar loss, cache)."""
        # Stable BCE: max(z,0) - zy + log(1 + exp(-|z|))
        loss_unreduced = np.maximum(Z, 0) - Z * y + np.log(1 + np.exp(-np.abs(Z)))
        loss = loss_unreduced.mean()
        A = _stable_sigmoid(Z)
        cache = {"A": A, "y": y, "N": Z.shape[0]}
        return loss, cache

    def backward(self, cache: Dict[str, Any]) -> np.ndarray:
        """Backprop: dL/dZ = (sigmoid(Z) - y) / N.
        Returns gradient w.r.t. logits Z."""
        A, y, N = cache["A"], cache["y"], cache["N"]
        grad = (A - y) / N
        return grad
```

---

## Softmax

### Core Intuition
Generalizes sigmoid to $K$ classes: converts a vector of logits into a probability distribution that sums to 1. Used as the output activation for multi-class classification (paired with cross-entropy loss). No learnable parameters.

### Mathematical Foundation

#### Forward Pass
For logits $z \in \mathbb{R}^K$:

$$p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Numerical stability**: Subtract the max before exponentiation (the result is mathematically identical because $\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$ for any constant $c$):

$$p_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}$$

#### Backward Pass (Derivation)
Unlike ReLU and sigmoid, softmax is **not element-wise**: each output $p_i$ depends on all inputs $z_j$ through the denominator. This makes the Jacobian a full (non-diagonal) matrix.

**Jacobian elements**: Consider two cases for $\frac{\partial p_i}{\partial z_j}$:

**Case $i = j$** (diagonal): Using the quotient rule on $p_i = \frac{e^{z_i}}{S}$ where $S = \sum_k e^{z_k}$:

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} S - e^{z_i} e^{z_i}}{S^2} = p_i - p_i^2 = p_i(1 - p_i)$$

**Case $i \neq j$** (off-diagonal): The numerator $e^{z_i}$ does not depend on $z_j$, so:

$$\frac{\partial p_i}{\partial z_j} = e^{z_i} \cdot \frac{-e^{z_j}}{S^2} = -p_i p_j$$

**Combined** using the Kronecker delta $\delta_{ij}$:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

In matrix form: $J = \text{diag}(p) - pp^T$.

**Chain rule**: For a single sample, let $g = \nabla_p L$ (the incoming gradient):

$$\frac{\partial L}{\partial z_j} = \sum_i g_i \cdot p_i(\delta_{ij} - p_j) = g_j p_j - p_j \sum_i g_i p_i$$

Let $s = \langle g, p \rangle = \sum_i g_i p_i$ (scalar dot product). Then:

$$\nabla_z L = p \odot (g - s)$$

For a batch $P \in \mathbb{R}^{N \times K}$ and $G = \nabla_P L \in \mathbb{R}^{N \times K}$: compute $s$ per sample as a row-wise dot product, then broadcast.

### Key Equation

$$P = \text{softmax}(Z)$$

$$s = \sum_j (\nabla_P L)_j \cdot P_j \quad \text{(per sample)}$$

$$\nabla_Z L = P \odot (\nabla_P L - s)$$

### Fused Softmax + Cross-Entropy Gradient

In practice, softmax is never backpropped through in isolation. It is always fused with cross-entropy loss, and the fused gradient has an elegant closed form. This section derives it from the Jacobian above.

#### Setup

For a single sample with logits $z \in \mathbb{R}^K$, true class label $y \in \{0, \dots, K-1\}$ (integer index), and softmax output $p = \text{softmax}(z) \in \mathbb{R}^K$:

$$L = -\log p_y$$

**Goal**: Find $\frac{\partial L}{\partial z_j}$ directly (end-to-end from loss to logits).

#### Derivation

**Step 1: Gradient of loss w.r.t. softmax output.** Only the $y$-th probability appears in the loss:

$$\frac{\partial L}{\partial p_i} = -\frac{\delta_{iy}}{p_y}$$

where $\delta_{iy} = 1$ if $i = y$, else $0$. So $\nabla_p L$ is a sparse vector: zero everywhere except $-\frac{1}{p_y}$ at position $y$.

**Step 2: Chain through the softmax Jacobian.** Using $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ from above:

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}$$

Since $\frac{\partial L}{\partial p_i}$ is nonzero only at $i = y$, the sum collapses to a single term:

$$\frac{\partial L}{\partial z_j} = \left(-\frac{1}{p_y}\right) \cdot \frac{\partial p_y}{\partial z_j} = -\frac{1}{p_y} \cdot p_y(\delta_{yj} - p_j)$$

**Step 3: Simplify.** The $p_y$ cancels:

$$\frac{\partial L}{\partial z_j} = -(\delta_{yj} - p_j) = p_j - \delta_{yj}$$

In vector form:

$$\boxed{\nabla_z L = p - \text{onehot}(y) = \text{softmax}(z) - \text{onehot}(y)}$$

where $p = \text{softmax}(z) \in \mathbb{R}^K$ and $\text{onehot}(y) \in \mathbb{R}^K$ is a vector of zeros with a 1 at index $y$: $[\text{onehot}(y)]_j = \delta_{yj}$. Both are $K$-dimensional (one entry per class), so the gradient $\nabla_z L \in \mathbb{R}^K$ has the same shape as the input logits $z$.

#### Why this matters
- **No Jacobian matrix needed**: The full $K \times K$ Jacobian $\text{diag}(p) - pp^T$ never needs to be materialized. The sparsity of $\nabla_p L$ (only one nonzero entry) collapses the matrix-vector product to a single row.
- **Numerically stable**: Computing $p - \text{onehot}(y)$ involves no division by small probabilities and no log of near-zero values (the log happens in the forward pass via `log_softmax`).
- **Intuitive**: The gradient at each logit is the model's predicted probability minus the truth. If $p_y = 0.9$ for the correct class, the gradient there is $0.9 - 1 = -0.1$ (small push up). If $p_j = 0.05$ for a wrong class, the gradient is $0.05 - 0 = 0.05$ (small push down).

#### Batched form
For a batch of $N$ samples with logits $Z \in \mathbb{R}^{N \times K}$ and labels $y \in \mathbb{Z}^N$:

$$\nabla_Z L = \frac{1}{N}(P - Y_{\text{onehot}})$$

where $P = \text{softmax}(Z) \in \mathbb{R}^{N \times K}$, $Y_{\text{onehot}} \in \mathbb{R}^{N \times K}$ has a 1 at column $y_n$ in each row $n$ (and 0 elsewhere), and the $\frac{1}{N}$ comes from mean reduction over the batch.

### Pitfalls
- **Never used alone in practice**: Softmax + cross-entropy is always fused into a single `log_softmax + NLLLoss` operation, because the fused gradient simplifies to $\nabla_z L = p - \text{onehot}(y)$ (derived above), which is simpler, faster, and numerically stable. See ML Coding#2. Numerical stability intuition.
- **Temperature scaling**: Dividing logits by temperature $\tau$ before softmax controls sharpness: $\tau < 1$ sharpens (more confident), $\tau > 1$ flattens (more uniform). Used in knowledge distillation and language model sampling.

### Implementation
This is exactly what `stable_cross_entropy` computes. The full implementation (also in ML Coding#2. Numerical stability intuition):

```python
class Softmax:
    """Softmax Activation: converts logits to probabilities. Not element-wise."""

    def forward(self, Z: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward: P = softmax(Z) with max-shift for stability.
        Shapes: Z: (N, K), P: (N, K)."""
        Z_shifted = Z - Z.max(axis=-1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        P = exp_Z / exp_Z.sum(axis=-1, keepdims=True)
        cache = {"P": P}  # only need output, not input
        return P, cache

    def backward(self, dP: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Backprop: dL/dZ = P * (dP - sum(dP * P, axis=-1, keepdims=True)).
        Shapes: dP: (N, K), dZ: (N, K)."""
        P = cache["P"]
        s = np.sum(dP * P, axis=-1, keepdims=True)  # (N, 1)
        dZ = P * (dP - s)
        return dZ


def _stable_logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z)))).
    z: (N, K) -> returns (N,).  The sum reduces axis=-1 (the K dim)."""
    z_max = z.max(axis=axis, keepdims=True)          # (N, 1)
    z_max = np.where(np.isfinite(z_max), z_max, 0.0) # guard fully-masked rows
    return z_max.squeeze(axis) + np.log(              # (N,)
        np.sum(np.exp(z - z_max), axis=axis))         # (N,): sum over K


def _stable_log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log softmax = z - logsumexp(z). Never computes softmax explicitly.
    z: (N, K) -> returns (N, K)."""
    lse = _stable_logsumexp(z, axis=axis)             # (N,)
    return z - np.expand_dims(lse, axis=axis)          # (N, K) - (N, 1) -> (N, K)


class SoftmaxCrossEntropyLoss:
    """Fused softmax + cross-entropy loss. No learnable parameters.
    Computes log_softmax internally for numerical stability."""

    def forward(self, Z: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Forward: L = -mean(log_softmax(Z)[y]).
        Z: (N, K) logits, y: (N,) integer labels. Returns (scalar loss, cache)."""
        N = Z.shape[0]
        logp = _stable_log_softmax(Z, axis=-1)        # (N, K)
        # logp[np.arange(N), y] picks one entry per row: the log-prob of the
        # correct class for each sample. np.arange(N) selects rows 0..N-1,
        # y selects the column (class index) per row. Result shape: (N,).
        loss = -logp[np.arange(N), y].mean()           # (N,) -> scalar
        cache = {"logp": logp, "y": y, "N": N}
        return loss, cache

    def backward(self, cache: Dict[str, Any]) -> np.ndarray:
        """Backprop: dL/dZ = (softmax(Z) - onehot(y)) / N.
        Returns (N, K) gradient w.r.t. logits Z."""
        logp, y, N = cache["logp"], cache["y"], cache["N"]
        grad = np.exp(logp)                            # (N, K): softmax probs
        # grad[np.arange(N), y] -= 1.0 is vectorized onehot subtraction:
        # for each row n, subtract 1 from column y[n].
        # Before: grad[n, :] = [p0, ..., p_y, ..., pK-1]
        # After:  grad[n, :] = [p0, ..., p_y - 1, ..., pK-1] = p - onehot(y)
        grad[np.arange(N), y] -= 1.0                   # (N, K)
        grad /= N                                      # (N, K): mean reduction
        return grad
```

**Line-by-line connection to the math**:
- `_stable_logsumexp(Z)`: computes $\text{logsumexp}(z) = \max(z) + \log\sum_j e^{z_j - \max(z)}$. Input $(N, K)$, output $(N{,})$: the sum reduces the class dimension $K$, leaving one scalar per sample.
- `_stable_log_softmax(Z)`: computes $\log p = z - \text{logsumexp}(z)$. Broadcasts $(N, K) - (N, 1) \to (N, K)$.
- `logp[np.arange(N), y]`: **advanced indexing** that picks one element per row. `np.arange(N)` = row indices $[0, 1, \dots, N{-}1]$, `y` = column indices $[y_0, y_1, \dots, y_{N-1}]$. Result: $(N{,})$ vector of $\log p_{y_n}$ values.
- `.mean()`: reduces $(N{,}) \to$ scalar, giving $L = -\frac{1}{N}\sum_n \log p_{y_n}$.
- `np.exp(logp)`: recovers $p = \text{softmax}(z) \in \mathbb{R}^{N \times K}$ from $\log p$ (no second forward pass).
- `grad[np.arange(N), y] -= 1.0`: same advanced indexing pattern. Subtracts 1 at position $(n, y_n)$ for each sample, giving $p - \text{onehot}(y)$. Avoids constructing the full $(N, K)$ onehot matrix.
- `grad /= N`: the $\frac{1}{N}$ from mean reduction.

**Note**: Unlike the activation layers, `SoftmaxCrossEntropyLoss.backward()` takes no incoming gradient `dL` argument: it is the terminal node of the computation graph, so $\frac{\partial L}{\partial L} = 1$ is implicit. It also takes `y` (labels) in `forward()` alongside the logits.
