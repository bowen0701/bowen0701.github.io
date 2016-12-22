---
layout: post
title: "ML Notes: Chi-squared Test in 2x2 Table"
date: 2016-04-30
---

## Observed data

We summarize data in the following $$2 \times 2$$ table:

|                        |  Event: $$D$$   | No Event: $$\bar D$$ |         |
|------------------------|-----------------|----------------------|---------|
| Exposure: $$E$$        |      $$a$$      |         $$b$$        | $$n_B$$ |
| No Exposure: $$\bar E$$|      $$c$$      |         $$d$$        | $$n_A$$ |
|                        |     $$a+c$$     |        $$b+d$$       |  $$n$$  |

**The interested hypothesis: the exposure $$E$$ and the event $$D$$ are independent.**

## Pearson (1900)'s chi-squared test

Let $$p_B = Pr(D\vert E)$$ be the conditional probability of event, $$D$$, given the exposure group, B, and $$p_A = Pr(D\vert\bar E)$$ the conditional probability of event, $$D$$, given the non-exposure group, A.

The interested hypothesis can be represented as

$$
H_0: p_B = p_A vs. H_1: p_B \neq p_A
$$

**Chi-squared test statistics under $$H_0$$:**

$$
X_p = \frac{(\widehat {p}_B - \widehat {p}_A)^2}{\widehat{p} (1 - \widehat{p}) \left ( \frac{1}{n_B} + \frac{1}{n_A} \right )} = \frac{n (ad - bc)^2}{(a + b) (a + c) (b + d) (c + d)} \\
= \sum_{i=1}^2 \sum_{j=1}^2 \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \rightarrow^d \chi^2_{(1)}
\quad \text{(1)}
$$

where

- $$\widehat{p}_B = a / n_B$$ and $$\widehat{p}_A = c / n_A$$
- $$p = Pr(\text{D})$$, with estimate $$\widehat{p} = (a + c) / n$$
- $$O_{ij}$$, $$i,j = 1,2$$: observed entries in 2x2 table
- $$E_{ij} = E(O_{ij})$$, $$i,j = 1,2$$: expected entries in 2x2 table

By applying (1) we can obtain the corresponding testing procedure or confidence interval.

## Derivations of (1)

Let $$p_B = \Pr(D | E)$$, and $$p_A = \Pr(D | \bar E)$$,

$$
\widehat p_B = \frac{a}{a+b} = \frac{a}{n_B}, \\
\widehat p_A = \frac{c}{c+d} = \frac{c}{n_A}
$$

Under $$H_0: p_B = p_A = p$$,

$$
E(\widehat p_B - \widehat p_A) = 0, \\
Var(\widehat p_B - \widehat p_A) = \frac{1}{n_B} p_B (1 - p_B) + \frac{1}{n_A} p_A (1 - p_A) = p (1 - p) \left [ \frac{1}{n_B} + \frac{1}{n_A} \right ] := \widehat V
$$

From the **Central Limit Theorem,**

$$
\frac{\widehat p_B - \widehat p_A}{\sqrt{\widehat V}} = \frac{\widehat p_B - \widehat p_A)}{\sqrt{\widehat p (1 - \widehat p) \left [ \frac{1}{n_B} + \frac{1}{n_A} \right ]}} \rightarrow^d N(0, 1)
$$

Thus,

$$
\frac{(\widehat p_B - \widehat p_A)^2}{\widehat p (1 - \widehat p) \left [ \frac{1}{n_B} + \frac{1}{n_A} \right ]} \rightarrow^d \chi^2_{(1)}
$$

Numerator:

$$
(\widehat p_B - \widehat p_A)^2 = \left ( \frac{a}{n_B} - \frac{c}{n_A} \right )^2 = \frac{(ad - bc)^2}{(n_B n_A)^2}
$$

Denominator:

$$
\widehat p (1 - \widehat p) \left [ \frac{1}{n_B} + \frac{1}{n_A} \right ] = \frac{a+c}{n} \frac{b+d}{n} \frac{n_B + n_A}{n_B n_A} = \frac{(a+c) (b+d)}{n n_B n_A}
$$

Hence the first identity in equation (1) follows. Alternatively, first focus on cell $$O_{11} = a$$. Under $$H_0:$$ exposure and event are independent,

$$
E_{11} = E(O_{11}) = E(a) = n \Pr(E \cdot D) = n \Pr(E) \Pr(D) = n \frac{a+c}{n} \frac{a+b}{n}
$$

Similar results can be derived for cells $$b$$, $$c$$ and $$d$$. Hence the second identity in equation (1) follows.

## References

- Jewell (2004). Statistics for Epidemiology.
