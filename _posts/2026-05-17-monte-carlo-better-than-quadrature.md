---
author: Bowen Lee
date: '2026-05-17'
layout: post
tags:
- monte-carlo
- numerical-methods
- bayesian
- statistics
- math
title: Monte Carlo Better than Quadrature
---
<details class="toc-details" markdown="1">
<summary><b>Table of Contents</b></summary>

* TOC
{:toc}

</details>


## Core Intuition

Q: Why **Monte Carlo method** is often the default method for high-dimensional integral approximation, **instead of Quadrature methods?** I got this question while reading Fearnhead et al. (2025, p. 6).

This comes directly from the following:
- Suppose we apply a cubature rule based on a grid of $m+1$ equally spaced points in each dimension.
- The spacing of these points will be $\delta = 1/m$
- Then there will be $n=(m+1)^d$ points in total
- If we have a **cubature whose error decays in $\delta^r$, for some power $r$,** then the **error decays at a rate of $m^{-r} \simeq n^{-r/d}$**
- Note for **Monte Carlo method: its error decays at a rate of $n^{1/2}$**, by the **central limit theorem**

## Mathematical Foundation

Q: Why cubature whose error decays in $\delta^r$? **Taylor's theorem with integral remainder.** 

### Taylor's Theorem with Remainder

For a function $f$ with $r$ continuous derivatives on an interval containing $x_0$ and $x$:

$$
f(x) = \sum_{k=0}^{r-1} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k + R_r(x)
$$

where the remainder can be written as:

$$
R_r(x) = \frac{1}{(r-1)!} \int_{x_0}^{x} (x-t)^{r-1} f^{(r)}(t) \, dt
$$

### Proof of the Error Rate Bound in $\delta^r$

**Step 1:** Take absolute value of the remainder:

$$
|R_r(x)| = \left| \frac{1}{(r-1)!} \int_{x_0}^{x} (x-t)^{r-1} f^{(r)}(t) \, dt \right|
$$

**Step 2:** Apply triangle inequality:

$$
|R_r(x)| \leq \frac{1}{(r-1)!} \int_{x_0}^{x} |x-t|^{r-1} |f^{(r)}(t)| \, dt
$$

**Step 3:** Use $|f^{(r)}(t)| \leq |f^{(r)}|_{\infty}$ (definition of sup norm):

$$
|R_r(x)| \leq \frac{|f^{(r)}|_{\infty}}{(r-1)!} \int_{x_0}^{x} |x-t|^{r-1} \, dt
$$

**Step 4:** Evaluate the integral. Let $h = |x - x_0|$:

$$
\int_{x_0}^{x} |x-t|^{r-1} \, dt = \int_{0}^{h} s^{r-1} \, ds = \frac{h^r}{r}
$$

**Step 5:** Combine:

$$
|R_r(x)| \leq \frac{|f^{(r)}|_{\infty}}{(r-1)!} \cdot \frac{h^r}{r} = \frac{|f^{(r)}|_{\infty}}{r!} h^r
$$

**Step 6:** Set $\delta = h = |x - x_0|$ and $C = \frac{1}{r!}$:

$$
\boxed{|R_r(x)| \leq C \cdot \delta^r \cdot |f^{(r)}|_{\infty}}
$$

### Derivation of Taylor's Theorem with Remainder Formula

The key idea is to **repeatedly apply integration by parts** to express the error in terms of the highest derivative.

#### Starting Point: Fundamental Theorem of Calculus

$$
f(x) = f(x_0) + \int_{x_0}^{x} f'(t) \, dt
$$

This is just: $f(x) - f(x_0) = \int_{x_0}^x f'(t) \, dt$

#### Step 1: Integration by Parts (once)

Apply integration by parts to $\int_{x_0}^{x} f'(t) \, dt$ with:
- $u = f'(t)$, so $du = f''(t) \, dt$
- $dv = dt$, so $v = t - x$

$$
\int_{x_0}^{x} f'(t) \, dt = \left[f'(t)(t-x)\right]_{x_0}^{x} - \int_{x_0}^{x} (t-x) f''(t) \, dt
$$

$$
= - f'(x_0)(x_0 - x) + \int_{x_0}^{x} (x-t) f''(t) \, dt
$$

$$
= f'(x_0)(x - x_0) + \int_{x_0}^{x} (x-t) f''(t) \, dt
$$

So:

$$
f(x) = f(x_0) + f'(x_0)(x-x_0) + \int_{x_0}^{x} (x-t) f''(t) \, dt
$$

#### Step 2: Integration by Parts (again)

Apply integration by parts to $\int_{x_0}^{x} (x-t) f''(t) \, dt$ with:
- $u = f''(t)$, so $du = f'''(t) \, dt$
- $dv = (x-t) \, dt$, so $v = -\frac{(x-t)^2}{2}$

$$
\int_{x_0}^{x} (x-t) f''(t) \, dt = \left[-\frac{(x-t)^2}{2} f''(t)\right]_{x_0}^{x} + \int_{x_0}^{x} \frac{(x-t)^2}{2} f'''(t) \, dt
$$

$$
= \frac{(x-x_0)^2}{2} f''(x_0) + \int_{x_0}^{x} \frac{(x-t)^2}{2} f'''(t) \, dt
$$

So:

$$
f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \int_{x_0}^{x} \frac{(x-t)^2}{2!} f'''(t) \, dt
$$

#### Pattern Recognition

After k integration by parts, we get:

$$
f(x) = \sum_{j=0}^{k-1} \frac{f^{(j)}(x_0)}{j!}(x-x_0)^j + \int_{x_0}^{x} \frac{(x-t)^{k-1}}{(k-1)!} f^{(k)}(t) \, dt
$$

Setting k = r:

$$
\boxed{R_r(x) = \frac{1}{(r-1)!} \int_{x_0}^{x} (x-t)^{r-1} f^{(r)}(t) \, dt}
$$

#### Intuition

The integral form shows that the remainder is a **weighted average of $f^{(r)}(t)$** over the interval $[x_0, x]$, with weights $(x-t)^{r-1}$ that give more importance to points closer to $x_0$.

## Key Equation

For curvature: If we have a cubature whose **error decays in $\delta^r$**. for some power $r$, then the error decays at a rate of $m^{-r} \simeq n^{-r/d}$.

$$
f(x) = \sum_{k=0}^{r-1} \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k + R_r(x),
$$

$$
R_r(x) = \frac{1}{(r-1)!} \int_{x_0}^{x} (x-t)^{r-1} f^{(r)}(t) \, dt,
$$

$$
|R_r(x)| \leq C \cdot \delta^r \cdot |f^{(r)}|_{\infty}
$$

As $\delta = 1/m$, $n=(m+1)^d$, so $m^{-r} \simeq n^{-r/d}$.

For Monte Carlo: The error decays at a rate of $n^{1/2}$**, by the central limit theorem**.

## Component of

- Numerical Integration #todo
- Bayesian Computation #todo
- Scalable Machine Learning #todo

## Insights

- **Curse of dimensionality breaks quadrature:** cubature error scales as $n^{-r/d}$, so for fixed $n$ and $r$, doubling $d$ halves the exponent. Monte Carlo error $n^{-1/2}$ is completely dimension-free.
- **Crossover dimension:** Monte Carlo beats an order-$r$ cubature rule when $d > 2r$. For a second-order rule ($r=2$), the crossover is $d > 4$; for most practical integrands in ML/Bayesian settings, $d$ is in the hundreds or thousands.
- **Higher-order rules don't rescue quadrature at scale:** increasing $r$ shifts the crossover dimension, but collecting enough smooth derivatives of $f$ in high $d$ is itself intractable.
- **Monte Carlo's slow 1D rate is a feature in high $d$:** $n^{-1/2}$ looks poor compared to $n^{-r}$ in 1D, but its constant does not grow exponentially with $d$.

## Pitfalls

- **Monte Carlo is not always preferable:** in $d \leq 2$ or $d \leq 4$ (for $r \geq 2$), a well-chosen quadrature rule is faster to converge; defaulting to Monte Carlo in low dimensions wastes samples.
- **Rate assumes i.i.d. samples:** the $n^{-1/2}$ CLT rate breaks down for correlated samples (e.g., MCMC chains), which introduce an effective-sample-size penalty.
- **Ignoring variance:** the CLT bound hides the integrand variance $\sigma^2$; a high-variance integrand can make Monte Carlo impractical even when the dimension favors it. Variance reduction (importance sampling, control variates) is then essential.
- **Confusing $n$ (total points) with $m$ (points per dimension):** the grid has $n = (m+1)^d$ points, so comparing quadrature and Monte Carlo at the same $n$ (not the same $m$) is critical.

## Connections

- Quasi-Monte Carlo #todo: low-discrepancy sequences beat i.i.d. draws, achieving $O((\log n)^d/n)$ error for moderate $d$.
- MCMC #todo: Markov chains replace i.i.d. draws when the target is known up to a constant; $n^{-1/2}$ rate survives asymptotically via ergodic theory.
- Importance Sampling #todo: reweights a surrogate distribution to reduce variance; $n^{-1/2}$ rate holds with a smaller constant.
- Curse of Dimensionality #todo: $n^{-r/d}$ cubature rate is the canonical example of exponential cost growth with $d$.
- Central Limit Theorem #todo: source of Monte Carlo's $n^{-1/2}$ guarantee for i.i.d. estimators.

## References

- Fearnhead et al. (2025). Scalable Monte Carlo for Bayesian Learning. https://arxiv.org/abs/2407.12751

