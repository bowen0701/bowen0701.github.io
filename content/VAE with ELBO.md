---
title: VAE with ELBO
author: Bowen Lee
created: 2026-08-06
publish: true
tags:
  - deep-learning
  - math
  - generative-models
---
# Concept: VAE with ELBO

We want to maximize log-evidence $\log p(x)$ but integrating over all $z$ is intractable. The VAE framework (Kingma & Welling, 2014) introduces a variational posterior $q(z|x)$ and multiplies/divides inside the log:

$$
\begin{aligned}
\log p(x)
&= \log \int p_\theta(x|z)\,p(z)\,\frac{q(z|x)}{q(z|x)}\,dz \\
&= \log \mathbb{E}_{q(z|x)} \!\left[ \frac{p(x,z)}{q(z|x)} \right]
\end{aligned}
$$

Apply Jensen's inequality ($\log \mathbb{E}[f] \geq \mathbb{E}[\log f]$ for concave $\log$) and expand using $p(x,z) = p_\theta(x|z)\,p(z)$:

$$
\begin{aligned}
\log p(x)
&\geq \mathbb{E}_{q(z|x)} \!\left[ \log \frac{p(x,z)}{q(z|x)} \right] \\
&= \mathbb{E}_{q(z|x)} \!\left[ \log \frac{p(x|z)\,p(z)}{q(z|x)} \right] \\
&= \mathbb{E}_{q(z|x)} \!\left[ \log p(x|z) \right] - \text{KL}(q(z|x) \,\|\, p(z)) \\
&:= \text{ELBO}
\end{aligned}
$$

This lower bound is the Evidence Lower Bound (ELBO). The gap between log-evidence $\log p(x)$ and ELBO is $\log p(x) - \text{ELBO} = \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$ (for derivations see below), the posterior approximation error. Maximizing the ELBO simultaneously maximizes likelihood and minimizes the posterior gap.

## Interpreting the ELBO

The ELBO can be rewritten to show the gap between evidence and bound (see Kingma & Welling, 2019):

$$
\log p(x) = \text{ELBO}(\phi, \theta) + \text{KL}(q_\phi(z|x) \,||\, p_\theta(z|x))
$$

where $\text{ELBO}(\phi, \theta) = \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - \text{KL}( q_\phi(z|x) \,||\, p(z) )$

**Derivation of the gap:**

$$
\begin{aligned}
\log p(x) - \text{ELBO}
&= \mathbb{E}_q[\log p(x)] - \mathbb{E}_q\!\left[\log \frac{p(x,z)}{q(z|x)}\right] \\
&= \mathbb{E}_q\!\left[\log q(z|x) - \log p(x,z) + \log p(x)\right] \\
&= \mathbb{E}_q\!\left[\log q(z|x) - \log p(z|x) - \log p(x) + \log p(x)\right] \\
&= \mathbb{E}_q\!\left[\log \frac{q(z|x)}{p(z|x)}\right] \\
&= \text{KL}(q(z|x) \,\|\, p(z|x))
\end{aligned}
$$

**Key insights of ELBO:**

1. **The gap is always non-negative**: Since $\text{KL}(q_\phi(z|x) \,||\, p_\theta(z|x)) \geq 0$, this confirms ELBO $\leq \log p(x)$: the ELBO is always a lower bound on $\log p(x)$

2. **Maximizing ELBO "pushes up" the evidence**: 
   - When we maximize ELBO w.r.t. model parameters $\theta$, we increase $\log p(x)$ (the thing we actually care about)
   - When we maximize ELBO w.r.t. variational parameters $\phi$ (optimizing $q$), we tighten the bound by reducing $\text{KL}(q_\phi(z|x) \,||\, p_\theta(z|x))$. Specifically, from the identity $\log p(x) = \text{ELBO}(\phi, \theta) + \text{KL}(q_\phi(z|x) \,||\, p_\theta(z|x))$: when $\theta$ is held fixed, $\log p(x)$ is a constant, so maximizing ELBO w.r.t. $\phi$ is exactly equivalent to minimizing the KL gap. This is classical variational inference (Blei et al., 2017): finding the member of the family $\{q_\phi\}$ closest (in reverse KL) to the intractable true posterior $p_\theta(z|x)$.

3. **Two competing terms in ELBO**:
   - **Reconstruction term** $\mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ]$: encourages $q$ to find latent codes $z$ that explain the data well
   - **Prior matching term** $\text{KL}( q_\phi(z|x) \,||\, p(z) )$: regularizes $q$ to stay close to the prior, preventing overfitting

4. **Perfect bound when $q(z|x) = p(z|x)$**: The KL gap becomes zero, and ELBO equals the true log-evidence

## Maximizing the ELBO: The Reparameterization Trick

We have the ELBO:

$$
\mathcal{L}(\phi, \theta) = \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - \text{KL}( q_\phi(z|x) \,||\, p(z) )
$$

Two neural networks, two parameter sets:

| | Network | Parameters | Role |
|---|---|---|---|
| **Encoder** | $q_\phi(z\|x)$ | $\phi$ | Compresses $x$ into a distribution over $z$ (approximate posterior) |
| **Decoder** | $p_\theta(x\|z)$ | $\theta$ | Reconstructs $x$ from a latent sample $z$ (generative model) |

We want to take gradients w.r.t. both $\phi$ and $\theta$ to jointly train them:

- $\nabla_\theta \mathcal{L}$: straightforward: $p_\theta(x|z)$ is a standard neural network forward pass from $z$ to $x$, so standard backprop works. No trick needed.
- $\nabla_\phi \mathcal{L}$: the hard part: sampling $z \sim q_\phi(z|x)$ blocks gradients, which is why the reparameterization trick is needed (see below).

**The Problem with $\nabla_\phi$ (Naive Gradients)**

The naive gradient is:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - \nabla_\phi \text{KL}( q_\phi(z|x) \,||\, p(z) )
$$

**Problem**: We can't push $\nabla_\phi$ inside the expectation because the distribution $q_\phi$ itself depends on $\phi$. If we sample $z \sim q_\phi(z|x)$, the gradient $\nabla_\phi \log p_\theta(x|z)$ is zero.

**Solution: The Reparameterization Trick (Kingma & Welling, 2014)**

Instead of sampling $z$ directly from $q_\phi(z|x)$, express $z$ as a deterministic function of $\phi$ and noise $\epsilon$:

$$
z = g(\phi, x, \epsilon), \quad \epsilon \sim p(\epsilon)
$$

For example, if $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

Now the expectation is over $\epsilon$ (which doesn't depend on $\phi$):

$$
\mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] = \mathbb{E}_{p(\epsilon)} [ \log p_\theta(x | g(\phi, x, \epsilon)) ]
$$

**Now we can push the gradient inside**:

$$
\nabla_\phi \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] = \mathbb{E}_{p(\epsilon)} [ \nabla_\phi \log p_\theta(x | g(\phi, x, \epsilon)) ]
$$

**Chain rule through $z$**

With $s_\phi = \log\sigma^2_\phi$ (so $\sigma_\phi = e^{s_\phi/2}$), the gradient flows from the decoder ($p_\theta(x|z)$) back through $z = \mu_\phi + \sigma_\phi \cdot \epsilon$ to the encoder ($q_\phi(z|x)$) outputs:

$$
\nabla_{\mu_\phi} \log p_\theta(x|z) = \frac{\partial \log p_\theta}{\partial z} \cdot \underbrace{\frac{\partial z}{\partial \mu_\phi}}_{=\,1}
$$

$$
\nabla_{s_\phi} \log p_\theta(x|z) = \frac{\partial \log p_\theta}{\partial z} \cdot \underbrace{\frac{\partial z}{\partial s_\phi}}_{=\,\epsilon \cdot \sigma_\phi / 2}
$$

Both gradients route through $\partial \log p_\theta / \partial z$: the decoder's ($p_\theta(x|z)$) gradient w.r.t. its input. In practice, autograd handles this automatically: `z = mu + sigma * eps` is a tensor op, so backprop flows from the reconstruction loss through $z$ to $\mu_\phi$ and $\sigma_\phi$, then to encoder ($q_\phi(z|x)$) weights. The reparameterization trick is the design choice that makes this computational graph connected.

**Complete Gradient Estimator**

1. **Sample** $\epsilon \sim \mathcal{N}(0, 1)$
2. **Compute** $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon$
3. **Estimate gradient**:

$$
\nabla_\phi \mathcal{L} \approx \nabla_\phi \log p_\theta(x|z) - \nabla_\phi \text{KL}(q_\phi(z|x) \,||\, p(z))
$$

The KL term often has a closed form. 

For example, with encoder $q_\phi(z|x) = \mathcal{N}(\mu_\phi, \sigma^2_\phi)$ and prior $p(z) = \mathcal{N}(0, 1)$, start from the definition:

$$
\text{KL}(q_\phi(z|x) \,\|\, p(z)) = \mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z|x)}{p(z)}\right] = \mathbb{E}_{q_\phi}[\log q_\phi(z|x)] - \mathbb{E}_{q_\phi}[\log p(z)]
$$

Substituting the Gaussian log-densities:

$$
\begin{aligned}
\log q_\phi(z|x) &= -\tfrac{1}{2}\log(2\pi\sigma^2_\phi) - \tfrac{(z-\mu_\phi)^2}{2\sigma^2_\phi} \\
\log p(z) &= -\tfrac{1}{2}\log(2\pi) - \tfrac{z^2}{2}
\end{aligned}
$$

Taking expectations under $z \sim q_\phi(z|x) = \mathcal{N}(\mu_\phi, \sigma^2_\phi)$, using $\mathbb{E}[(z-\mu_\phi)^2] = \sigma^2_\phi$ and $\mathbb{E}[z^2] = \mu^2_\phi + \sigma^2_\phi$:

$$
\begin{aligned}
\text{KL}(q_\phi(z|x) \,\|\, p(z))
&= \left(-\tfrac{1}{2}\log\sigma^2_\phi - \tfrac{1}{2}\right) - \left(-\tfrac{\mu^2_\phi + \sigma^2_\phi}{2}\right) \\
&= \frac{1}{2}\left(\mu^2_\phi + \sigma^2_\phi - \log\sigma^2_\phi - 1\right)
\end{aligned}
$$

**Gradient w.r.t. $\phi$:** In practice the encoder outputs $\mu_\phi(x)$ and $s_\phi(x) := \log\sigma^2_\phi(x)$ (log-variance, for numerical stability), so $\sigma^2 = e^s$:

$$
\text{KL} = \frac{1}{2}\!\left(\mu^2 + e^s - s - 1\right)
$$

$$
\nabla_{\mu_\phi}\,\text{KL} = \mu_\phi, \qquad \nabla_{s_\phi}\,\text{KL} = \frac{1}{2}\!\left(e^{s_\phi} - 1\right)
$$

These are exact, zero-variance gradients: no sampling required for the KL term.

## VAE Training Algorithm

Pseudo codes:
```python
for each minibatch (x₁, ..., xₙ):
    # Encoder (q_φ(z|x)): outputs parameters of approximate posterior
    μ, log_σ² = encoder(x)

    # Reparameterization: makes sampling differentiable w.r.t. φ
    ε ~ N(0, I)
    z = μ + exp(log_σ²/2) · ε

    # Decoder (p_θ(x|z)): outputs parameters of reconstruction distribution
    x_recon = decoder(z)

    # Compute ELBO loss
    recon_loss = -log p_θ(x | z)  # see "Reconstruction Loss" below
    kl_loss = KL(q_φ(z|x) || p(z))  # closed-form for Gaussians

    loss = recon_loss + kl_loss

    # Backprop through the entire computational graph
    loss.backward()
    optimizer.step()
```

**Reconstruction Loss: $-\log p_\theta(x|z)$**

The decoder outputs parameters of a distribution over $x$, not $x$ directly. The reconstruction loss is the negative log-likelihood of the true $x$ under that distribution. The choice of distribution determines the loss function:

| Decoder assumption | $p_\theta(x\|z)$                           | $-\log p_\theta(x\|z)$ becomes                                 | When to use                                                |
| ------------------ | ------------------------------------------ | -------------------------------------------------------------- | ---------------------------------------------------------- |
| Gaussian           | $\mathcal{N}(\mu_\theta(z),\, \sigma^2 I)$ | MSE: $\|\|x - \mu_\theta(z)\|\|^2$ (up to constants)           | Continuous data (images with pixel values in $\mathbb{R}$) |
| Bernoulli          | $\text{Bernoulli}(\sigma(f_\theta(z)))$    | BCE: $-\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$ | Binary/bounded data (e.g., binarized MNIST)                |

**Derivation: Gaussian decoder → MSE**

Assume the decoder $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z),\, \sigma^2 I)$. The PDF is:

$$
p_\theta(x|z) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\!\left(-\frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2}\right)
$$

Taking $-\log$:

$$
-\log p_\theta(x|z) = \frac{\|x - \mu_\theta(z)\|^2}{2\sigma^2} + \underbrace{\frac{d}{2}\log(2\pi\sigma^2)}_{\text{constant w.r.t. } \theta}
$$

The constant doesn't affect optimization, so minimizing $-\log p_\theta(x|z)$ w.r.t. $\theta$ is equivalent to minimizing $\|x - \mu_\theta(z)\|^2$ (MSE).

**Derivation: Bernoulli decoder → BCE**

Assume each dimension $x_i \in \{0, 1\}$ and the decoder $p_\theta(x|z)$ outputs $\hat{x}_i = \sigma(f_\theta(z))_i$:

$$
p_\theta(x|z) = \prod_i \hat{x}_i^{\,x_i}(1-\hat{x}_i)^{1-x_i}
$$

Taking $-\log$:

$$
-\log p_\theta(x|z) = -\sum_i \left[x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)\right]
$$

This is exactly binary cross-entropy (BCE).

In both cases, `x_recon = decoder(z)` outputs the distributional parameters ($\mu_\theta$ or logits), and the loss evaluates how well the true $x$ is explained.

**Practical Architectures**

No special architecture required. They are standard NNs with one VAE-specific constraint: the encoder $q_\phi(z|x)$ must output **two vectors** ($\mu_\phi$ and $\log\sigma^2_\phi$), not a single embedding.

| Data type | Encoder | Decoder |
|---|---|---|
| Images | Conv → Conv → FC → ($\mu_\phi$, $\log\sigma^2_\phi$) | FC → DeConv → DeConv → output |
| Sequences | Transformer/RNN → FC → ($\mu_\phi$, $\log\sigma^2_\phi$) | Transformer/RNN from $z$ |
| Tabular | FC → FC → ($\mu_\phi$, $\log\sigma^2_\phi$) | FC → FC → output |

The decoder roughly mirrors the encoder. Between them sits the reparameterization layer: $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon$.

**Key Insight of Reparameterization Trick**

The reparameterization trick converts sampling (which blocks gradients) into deterministic computation with external noise (which allows gradients to flow). This makes the ELBO differentiable end-to-end, enabling standard backpropagation.

## Summary: Overall Optimization

A single `loss.backward()` call jointly optimizes both $\phi$ and $\theta$. Here is what each gradient does and why:

| Gradient | What it optimizes | Mechanism | Effect on $\log p(x) = \text{ELBO} + \text{KL gap}$ |
|---|---|---|---|
| $\nabla_\theta \mathcal{L}$ | Decoder ($p_\theta(x\|z)$) | Standard backprop through decoder | Increases $\log p(x)$ (better generative model) |
| $\nabla_\phi \mathcal{L}$: reconstruction | Encoder ($q_\phi(z\|x)$) | Reparameterization trick, chain rule through $z$ | Finds latent codes that decode well |
| $\nabla_\phi \mathcal{L}$: KL | Encoder ($q_\phi(z\|x)$) | Closed-form gradient (exact, no sampling) | Tightens bound by reducing $\text{KL}(q_\phi(z\|x) \,\|\, p_\theta(z\|x))$ |

The **computational graph in one forward pass:**

$$
x \xrightarrow{\text{encoder } q_\phi} (\mu_\phi, \sigma^2_\phi) \xrightarrow{z = \mu_\phi + \sigma_\phi \cdot \epsilon} z \xrightarrow{\text{decoder } p_\theta} \hat{x}
$$

Backprop flows from loss through the entire graph. The reparameterization $z = \mu_\phi + \sigma_\phi \cdot \epsilon$ is what makes the encoder-to-decoder connection differentiable: without it, the sampling step $z \sim q_\phi(z|x)$ would sever the gradient path.

## VAE Training Algorithm in PyTorch (Gaussian Decoder)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class VAE(nn.Module):
    """Conv VAE with Gaussian decoder for MNIST dataset of 1x28x28 images: 1 channel (grayscale), 28x28 pixels."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (q_φ(z|x)): x -> (μ_φ, log σ²_φ)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # -> 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64x7x7
            nn.ReLU(),
            nn.Flatten(),                                # -> 3136
        )
        self.fc_mu = nn.Linear(3136, latent_dim)
        self.fc_logvar = nn.Linear(3136, latent_dim)

        # Decoder (p_θ(x|z)): z -> μ_θ(z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # constrain μ_θ to [0,1] matching normalized pixels
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ_φ + σ_φ · ε,  ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)  # σ = exp(log_σ² / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def elbo_loss(x, x_recon, mu, logvar):
    """Negative ELBO = recon_loss + kl_loss, averaged over batch.

    Recon (Gaussian decoder): -E_q[log p_θ(x|z)] ∝ ||x - μ_θ(z)||²
    KL (closed-form):  0.5 * Σ_j (μ²_j + exp(s_j) - s_j - 1)
    """
    batch_size = x.size(0)
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch_size
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / batch_size
    return recon + kl, recon, kl


# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True,
)

for epoch in range(1, 21):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        batch_size = x.size(0)
        
        x_recon, mu, logvar = model(x)
        loss, recon, kl = elbo_loss(x, x_recon, mu, logvar)

        optimizer.zero_grad()
        loss.backward()   # single backward: jointly optimizes φ and θ
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size

    n = len(train_loader.dataset)
    print(f"Epoch {epoch:2d} | loss={total_loss/n:.4f} "
          f"recon={total_recon/n:.4f} kl={total_kl/n:.4f}")

# Generate from prior: z ~ N(0, I) -> decode
with torch.no_grad():
    z = torch.randn(16, model.latent_dim, device=device)
    samples = model.decode(z)  # shape: (16, 1, 28, 28)
```

## References

- Kingma & Welling (2014). Auto-Encoding Variational Bayes. arXiv:1312.6114
- Kingma & Welling (2019). An Introduction to Variational Autoencoders. arXiv:1906.02691
- Blei, Kucukelbir & McAuliffe (2017). Variational Inference: A Review for Statisticians. arXiv:1601.00670

