---
layout: post
comments: true
title: "Paper Notes: Estimating Survival and Association in a Semicompeting Risks Model"
date: 2014-10-01
---

## [Paper: Lakhal, Rivest & Abdous (Biometrics, 2008)](https://www.jstor.org/stable/25502035?seq=1#page_scan_tab_contents)

**Semi-competing risks data** occurs often in biomedical research, in which if the terminal event occurs first, it would censor the non-terminal event, but not vice versa. The goals are to study the **association between the non-terminal and terminal event times** and their **marginal behaviors**. This study note is to provide some details of the seminal work of Lakhal, Rivest & Abdous (Biometrics, 2008) which extends the work of [Fine, Jiang & Chappell (Bmka, 2001)](http://www.jstor.org/stable/2673691) from a **gamma frailty model**, for the joint survival function of the terminal and non-terminal event times over the observable region, to a general class of **Archimedean copulas.**

## Key Points

The main idea of the work is to utilize smartly the relationships between the Archimedean copulas and [Kendall's tau](http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient) and the **cross-ratio** for establishing the estimating equations for the association parameter.

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

**Estimation Procedure:**

Suppose the Archimedean copulas for the joint survival function of $$X$$ and $$Y$$ in the upper wedge, for $$0\le x\le y$$,

$$
F(x,y):=Pr(X>x,Y>y)= \phi _{\alpha }^{-1}\{{{\phi }_{\alpha }}({{F}_{X}}(x))+{{\phi }_{\alpha }}({{F}_{Y}}(y))\}, \ \ \ \ \ (1)
$$

where the generator $${{\phi }_{\alpha }}$$ is a nondecreasing convex function defined on $$(0,1]$$ with $${{\phi }_{\alpha }}(1)=0,$$ $$\alpha \ge 1$$ , for example

- $${{\phi }_{\alpha }}(t)={{t}^{1-\alpha }}-1$$: Clayton copula;
- $${{\phi }_{\alpha }}(t)={{(-\log t)}^{\alpha }}$$: Gumbel copula,

$$\alpha$$ is the association parameter of main interest, and $$F_{X}(x)$$ and $$F_{Y}(y)$$ are marginal survival functions of $$X$$ and $$Y$$ respectively. For Archimedean copulas, there is a relationship between them and the Kendall's tau,

$$
\tau:=Pr\{(X_i-X_j)(Y_i-Y_j)>0\}-Pr\{(X_i-X_j)(Y_i-Y_j)<0\},
$$

which is an association measure to describe the correlation between two random variables $$X$$ and $$Y$$. (We do not have to bother about what the relationship really is since it is not utilized in the estimation; for details please refer to Lakhal, Rivest & Abdous (2008).) As long as $$x\le y$$, the cross-ratio proposed by Oakes (1989),

$$
\theta_{\alpha}^{*}(x,y) \\ :=\frac{F(x,y)\:{{D}_{1}}{{D}_{2}}F(x,y)}{{{D}_{1}}F(x,y)\:{{D}_{2}}F(x,y)}\\ =\frac{Pr\{(X_i-X_j)(Y_i-Y_j)>0\ |\ \tilde{X_{ij}}=x,\tilde{Y_{ij}}=y\}}{Pr\{(X_i-X_j)(Y_i-Y_j)<0\ |\ \tilde{X_{ij}}=x,\tilde{Y_{ij}}=y\}}, \ \ \ \ \ (2)
$$

remains valid, where $${{\tilde{X}}_{ij}}={{X}_{i}}\wedge {{X}_{j}}$$, $${{\tilde{Y}}_{ij}}={{Y}_{i}}\wedge {{Y}_{j}}$$, and the operators $${{D}_{1}}=-\partial /(\partial x)$$ and $${{D}_{2}}=-\partial /(\partial y)$$. For proof of (2) please see the Appendix. Further, $$\theta _{\alpha }^{*}(x,y)$$ depends on $$x$$ and $$y$$ only through $$v=F(x,y)$$,

$$
\theta_{\alpha }^{*}(x,y)={{\theta }_{\alpha }}(v)=-v\frac{{{{{\phi }''}}_{\alpha }}(v)}{{{{{\phi }'}}_{\alpha }}(v)}. \ \ \ \ \ (3)
$$

For proof of (3) please see the Appendix.

Note that the pairs $$({{X}_{i}},{{Y}_{i}})$$ and $$({{X}_{j}},{{Y}_{j}})$$ are comparable iff $${{\tilde{X}}_{ij}}\le {{\tilde{Y}}_{ij}}\le {{\tilde{C}}_{ij}}$$, where $${{\tilde{C}}_{ij}}={{C}_{i}}\wedge {{C}_{j}}$$.

Proof: From the diagram below we can observe that 1) for $${{\tilde{X}}_{ij}}=x\le {{\tilde{Y}}_{ij}}=y$$, the restriction in the upper wedge; 2) for $${{\tilde{X}}_{ij}}\le {{\tilde{C}}_{ij}}$$, then $${{X}_{i}},{{X}_{j}}$$ and $${{C}_{i}},{{C}_{j}}$$ will satisfy the following three relationships. Thus $${{X}_{i}}$$ and $${{X}_{j}}$$ are comparable; and 3) For $${{\tilde{Y}}_{ij}}\le {{\tilde{C}}_{ij}}$$: similarly, $${{Y}_{i}}$$ and $${{Y}_{j}}$$ are comparable. $$\Box$$

<div style="text-align:center">
<img src="/images/semicr.png" alt="Drawing" style="width: 450px;"/>
</div>

To restrict the Kendall’s tau only on the comparable pairs, define the conditional Kendall’s tau as

$$
\tau_{\alpha}:=Pr\{(X_i-X_j)(Y_i-Y_j)>0\:|\:Z_{ij}\}-Pr\{(X_i-X_j)(Y_i-Y_j)<0\:|\:Z_{ij}\}\\ =E\{ sgn[(X_i-X_j)(Y_i-Y_j)]\:|\:Z_{ij}\}, \ \ \ \ \ (4)
$$

where $${{Z}_{ij}}=I({{\tilde{X}}_{ij}}\le {{\tilde{Y}}_{ij}}\le {{\tilde{C}}_{ij}})$$ and $$sgn(x)=1$$ or $$-1$$ if $$x>0$$ or $$x<0$$. Further note that when $$\tilde{X}_{ij} \le \tilde{Y}_{ij} \le \tilde{C}_{ij}$$, $$\tilde{X}_{ij}=\tilde{S}_{ij},\tilde{Y}_{ij}=\tilde{R}_{ij}$$, where $$\tilde{S}_{ij}$$ and $$\tilde{R}_{ij}$$ are defined similarly with $$\tilde{X}_{ij}$$ and $$\tilde{Y}_{ij}$$. Thus $${{Z}_{ij}}=I({{\tilde{S}}_{ij}}\le {{\tilde{R}}_{ij}}\le {{\tilde{C}}_{ij}})$$. Now from (2) and (3) we can observe that

$$
Pr\{(X_i-X_j)(Y_i-Y_j)>0\:|\:\tilde{X_{ij}}=x,\tilde{Y_{ij}}=y\}\\ =E\{I\{(X_i-X_j)(Y_i-Y_j)>0\:|\:\tilde{X_{ij}}=x,\tilde{Y_{ij}}=y\}=\dfrac{{{\theta}_{\alpha}}\{F(x,y)\}}{{{\theta}_{\alpha}}\{F(x,y)\}+1}.
$$

Then by double expectation we obtain

$$
\tau_{\alpha}=E\{E[I\{(X_i-X_j)(Y_i-Y_j)>0\:|\:\tilde{X_{ij}},\tilde{Y_{ij}}]\vert Z_{ij}\}=E\left\{ \frac{{{\theta}_{\alpha}}\{F(X_{ij},Y_{ij})\}}{{{\theta}_{\alpha}}\{F(X_{ij},Y_{ij})\}+1} \biggr\rvert Z_{ij}\right\}.
$$

Finally we can construct the estimating equation for $$\alpha$$ as

$$
\Psi_{n}(\alpha)=\dbinom{n}{2}^{-1} \sum_{i \le j}{W({{{\tilde{S}}}_{ij}},{{{\tilde{R}}}_{ij}}){{Z}_{ij}}\left\{ {{\Delta }_{ij}}-\frac{{{\theta }_{\alpha }}\{\hat{F}({{{\tilde{S}}}_{ij}},{{{\tilde{R}}}_{ij}})\}}{{{\theta }_{\alpha }}\{\hat{F}({{{\tilde{S}}}_{ij}},{{{\tilde{R}}}_{ij}})\}+1} \right\}}=0, \ \ \ \ \ (5)
$$

where $${{\Delta}_{ij}}=I\{({{S}_{i}}-{{S}_{j}})({{R}_{i}}-{{R}_{j}})>0\}$$ is the concordant indicator, $$W(x,y)$$ is a suitable random weight associated with the pair $$(i,j)$$, $$\hat{F}(x,y)$$ is the estimator of $$F(x,y)$$ under univariate censoring with

$$
\hat{F}(x,y)=\frac{{{n}^{-1}}\sum\nolimits_{i=1}^{n}{I({{S}_{i}}>x,{{R}_{i}}>y)}}{\hat{G}(y)}
$$,

and $$\hat{G}(t)$$ is the Kaplan-Meier estimator for $$G(t)=Pr(C>t)$$ based on $$\{({{R}_{i}},1-{{\delta }_{Ri}}), i=1,...,n\}$$.

The above arguments only focus on estimation of association parameter $$\alpha$$; those for estimation of marginal survival functions of $$X$$ and $$Y$$, respectively, are interesting but not so technical as the above, please refer to Lakhal, Rivest & Abdous (2008).

## Appendix: Proof of (2)

We start from the definition of $$\theta_{\alpha}^{*}(x,y)$$,

$$
theta _{\alpha }^{*}(x,y) \\ :=\frac{F(x,y)\ {{D}_{1}}{{D}_{2}}F(x,y)}{{{D}_{1}}F(x,y)\ {{D}_{2}}F(x,y)}\\ =\frac{2\Pr (X>x,Y>y)\Pr (X=x,Y=y)}{2\Pr (X=x,Y>y)\Pr (X>x,Y=y)} \\
=\frac{\Pr ({{X}_{i}}>{{X}_{j}},{{Y}_{i}}>{{Y}_{j}},{{X}_{j}}=x,{{Y}_{j}}=y)+\Pr ({{X}_{j}}>{{X}_{i}},{{Y}_{j}}>{{Y}_{i}},{{X}_{i}}=x,{{Y}_{i}}=y)}{\Pr ({{X}_{i}}=x,{{Y}_{i}}>{{Y}_{j}},{{X}_{j}}>{{X}_{i}},{{Y}_{j}}=y)+\Pr ({{X}_{j}}=x,{{Y}_{j}}>{{Y}_{i}},{{X}_{i}}>{{X}_{j}},{{Y}_{i}}=y)} \\
=\frac{\Pr \{({{X}_{i}}-{{X}_{j}})({{Y}_{i}}-{{Y}_{j}})>0|{{{\tilde{X}}}_{ij}}=x,{{{\tilde{Y}}}_{ij}}=y\}}{\Pr \{({{X}_{i}}-{{X}_{j}})({{Y}_{i}}-{{Y}_{j}})<0|{{{\tilde{X}}}_{ij}}=x,{{{\tilde{Y}}}_{ij}}=y\}}.
$$

Hence we obtain (2).

## Appendix: Proof of (3)

We would like to show

$$
\theta _{\alpha }^{*}(x,y):=\frac{F(x,y)\ {{D}_{1}}{{D}_{2}}F(x,y)}{{{D}_{1}}F(x,y)\ {{D}_{2}}F(x,y)}={{\theta }_{\alpha }}(v)=-v\frac{{{{{\phi }''}}_{\alpha }}(v)}{{{{{\phi }'}}_{\alpha }}(v)}.
$$

Recall first $$F(x,y)=\phi _{\alpha }^{-1}\{{{\phi }_{\alpha }}({{F}_{X}}(x))+{{\phi }_{\alpha }}({{F}_{Y}}(y))\}$$. Let $$\phi _{\alpha }^{-1}(u)=v=F(x,y)$$, then $${{\phi }_{\alpha }}(v)=u$$. By chain rule we have $${{D}_{1}}F(x,y)=-\frac{\partial }{\partial x}F(x,y)=-\frac{\partial }{\partial u}\phi _{\alpha }^{-1}(u)\cdot \frac{\partial u}{\partial x}$$. Since $$\frac{\partial }{\partial u}\phi _{\alpha }^{-1}(u)=\frac{\partial }{\partial u}v=\frac{\partial v}{\partial {{\phi }_{\alpha }}(v)}=\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}$$, $${{D}_{1}}F(x,y)=-\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}\frac{\partial u}{\partial x}$$. Similarly we can obtain $${{D}_{2}}F(x,y)=-\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}\frac{\partial u}{\partial y}$$. Further, note that

$$
D}_{1}}{{D}_{2}}F(x,y) \\ =+\frac{\partial }{\partial x}\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}\frac{\partial u}{\partial y} \\
=\frac{\partial }{\partial u}\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}\frac{\partial u}{\partial x}\frac{\partial u}{\partial y} \\
=\frac{\partial }{\partial {{\phi }_{\alpha }}(v)}\frac{1}{{{{{\phi }'}}_{\alpha }}(v)}\frac{\partial u}{\partial x}\frac{\partial u}{\partial y} \\
=\frac{\partial v}{\partial {{\phi }_{\alpha }}(v)}\frac{\partial {{{{\phi }'}}_{\alpha }}(v)}{\partial v}\left[ \frac{\partial }{\partial {{{{\phi }'}}_{\alpha }}(v)}\frac{1}{{{{{\phi }'}}_{\alpha }}(v)} \right]\frac{\partial u}{\partial x}\frac{\partial u}{\partial y} \\
=-\frac{{{{{\phi }''}}_{\alpha }}(v)}{{{{{\phi }'}}_{\alpha }}(v)}\frac{1}{{{[{{{{\phi }'}}_{\alpha }}(v)]}^{2}}}\frac{\partial u}{\partial x}\frac{\partial u}{\partial y}=-\frac{{{{{\phi }''}}_{\alpha }}(v)}{{{[{{{{\phi }'}}_{\alpha }}(v)]}^{3}}}\frac{\partial u}{\partial x}\frac{\partial u}{\partial y}.
$$

Plugging the four components of $$\theta _{\alpha }^{*}(x,y)$$ the result follows.

## Comments

