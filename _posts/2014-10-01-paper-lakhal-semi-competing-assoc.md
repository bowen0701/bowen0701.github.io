---
layout: post
comments: true
title: "Paper Notes: Estimating Survival and Association in a Semicompeting Risks Model"
date: 2014-10-01
---

## [Paper: Lakhal, Rivest & Abdous (Biometrics, 2008)](https://www.jstor.org/stable/25502035?seq=1#page_scan_tab_contents)

**Semi-competing risks data** occurs often in biomedical research, in which if the terminal event occurs first, it would censor the non-terminal event, but not vice versa. The goals are to study the **association between the non-terminal and terminal event times** and their **marginal behaviors**. This study note is to provide some details of the seminal work of Lakhal, Rivest & Abdous (Biometrics, 2008) which extends the work of [Fine, Jiang & Chappell (Bmka, 2001)](http://www.jstor.org/stable/2673691) from a **gamma frailty model**, for the joint survival function of the terminal and non-terminal event times over the observable region, to a general class of [Archemedean copulas](http://en.wikipedia.org/wiki/Copula_(probability_theory)).

## Key Points

The main idea of the work is to utilize smartly the relationships between the Archemedean copulas and [Kendall's tau](http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient) and the **cross-ratio** for establishing the estimating equations for the association parameter.

## Summary

**Example: Bone marrow transplant study.** Let $$X$$ be the time of relapse of leukemia, and $$Y$$ the time of death. It is intuitive that if death occurs prior to relapse, $$X$$ can not be observed anymore while if a patient relapses at time $$X$$, his death time $$Y$$ can still be observed. That is, only a pair of lifetime pair $$(X,Y)$$ such that $$X\le Y$$, with a censoring indicator, is then recorded for each patient.

**Semi-competing risks data:**

- $$X$$: time to non-terminal event
- $$C$$: independent censoring time
- $$R=Y\wedge C$$
- $${{\delta }_{R}}=I(Y<C)$$
- $$S=X\wedge Y\wedge C=X\wedge R$$
- $${{\delta }_{S}}=I(X<R)$$

Observed data: $$\{({{R}_{i}},{{S}_{i}},{{\delta }_{Ri}},{{\delta }_{Si}}),i=1,...,n\}$$. Note that $$C$$ may be the time of dropout or the end of study which is unrelated to the failure process.

