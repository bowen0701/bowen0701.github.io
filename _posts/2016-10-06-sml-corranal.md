---
layout: post
title: "SML Notes: Correspondence Analysis"
date: 2016-10-06
---

## Outline

- What is Correspondence Analysis?
- Why Correspondence Analysis?
- Methodology Summary
- Visualization Maps

## 1. What is Correspondence Analysis?

- **Aggregate data based:** more useful and convenient, compared with raw data-based.
- **Dimension reduction:** represent associations in a table of nonnegative **counts.**
- **Data visualization for association**: the positions of the points reflect associations.

### Numerous Applications

- Segments vs. Genders
- Segments vs. 24 hours
- Segments vs. 7 weekdays
- Segments vs. Locations
- Segments vs. App detection
- Segments vs. {only-impressions, with-clicks and further-with-actions}
- Campaigns vs. {only-impressions, with-clicks and further-with-actions}
- Other groups vs. characteristics, etc...

### Correspondence analysis of archaeological data: sites vs. types

<div style="text-align:center">
<img src="/images/contingency_table.png" alt="Drawing" style="width: 450px;"/>
</div>

Common data visualizations:

<div style="text-align:center">
<img src="/images/site_type_profile.png" alt="Drawing" style="width: 700px;"/>
</div>

However, common data visualizations of type by site (left) and site by type (right) **cannot quantify associations.**

**Correspondence analysis vidualization:**

<div style="text-align:center">
<img src="/images/ca_demo.png" alt="Drawing" style="width: 550px;"/>
</div>

- **Sites association:** 
  * *P1* and *P2* are close together, and thus have similar **type profiles**
  * *P0* ad *P6* are far apart, and thus have different type profiles
- **Types association:**
  * *A*, *B*, *C* and *D* have different **site profiles**
- **Site and type association:** (rough, see later)
  * Site *P0* is associated almost exclusively with type *D*
  * Site *P6* is similarly associated with type *C*
  * Sites *P1*, *P2* and *P3* (to lesser degrees) are associated with type *A*
- **Measure of retained information:**
  * **Inertia:** amount of retained information with
    - 1st dimension: $$\lambda^2_1 = 0.28 (55\%)$$
    - 2nd dimension: $$\lambda^2_2 = 0.17 (28\%)$$
  * The two dimensions account for $$55\% + 28\% = 88\%$$ of the total inertia
  * The representations fits the data well

## Why Correspondence Analysis?

Reference: Abdi & Williams (2010).

```python
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp 
import pandas as pd
```

```python
%load_ext rpy2.ipython
```

```python
%%R
library(ggplot2)
library(scales)
library(dplyr)
library(data.table)
library(grid)
library(gridExtra)

library(ca)
```

**French authors dataset:**

- Goal: Derive a map that reveals the **similarities in punctuation style between authors.**
- Note: Zola, who wrote a *small* novel under a pseudonym of Aloz.

```python
data = {'period': [7836, 53655, 115615, 161926, 38177, 46371, 2699],
        'comma': [13112, 102383, 184541, 340479, 105101, 58367, 5675],
        'others': [6026, 42413, 59226, 62754, 12670, 14299, 1046]}
data = pd.DataFrame(data, columns=['period', 'comma', 'others'],
                    index=['Rousseau', 'Chateaubriand', 'Hugo', \
                           'Zola', 'Proust', 'Giraudoux', 'Aloz'])
data
```

<div style="text-align:center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period</th>
      <th>comma</th>
      <th>others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rousseau</th>
      <td>7836</td>
      <td>13112</td>
      <td>6026</td>
    </tr>
    <tr>
      <th>Chateaubriand</th>
      <td>53655</td>
      <td>102383</td>
      <td>42413</td>
    </tr>
    <tr>
      <th>Hugo</th>
      <td>115615</td>
      <td>184541</td>
      <td>59226</td>
    </tr>
    <tr>
      <th>Zola</th>
      <td>161926</td>
      <td>340479</td>
      <td>62754</td>
    </tr>
    <tr>
      <th>Proust</th>
      <td>38177</td>
      <td>105101</td>
      <td>12670</td>
    </tr>
    <tr>
      <th>Giraudoux</th>
      <td>46371</td>
      <td>58367</td>
      <td>14299</td>
    </tr>
    <tr>
      <th>Aloz</th>
      <td>2699</td>
      <td>5675</td>
      <td>1046</td>
    </tr>
  </tbody>
</table>
</div>

**First (bad) idea: PCA (sometimes)**

- Aloz punctuates the style similarity as Zola, but is farther away from Zola than any authors.
- That is because **PCA is mainly sensitive to the number** of produced punctuation marks. 

<div style="text-align:center">
<img src="/images/bad_pca.png" alt="Drawing" style="width: 700px;"/>
</div>

**Correspondence analysis is successful:**

- From correspondence analysis results, Aloz and Zola are close together.
- It successfully **reveals profile (style) similarity.**

```python
%%R -i data
plot(ca(data))
```

<div style="text-align:center">
<img src="/images/ca_demo_ggplot.png" alt="Drawing" style="width: 700px;"/>
</div>

## 2. Methodology Summary

Correspondence Analysis is based on **generalized singular value decomposition (SVD),** which is equivalent to **principal component analysis (PCA);** for introduction see [post](https://bowen0701.github.io/blog/2016/10/05/sml-pca-svd).

### Correspondence analysis methodology

Let the observed data be a **contigency table** of *unscaled* **counts** which is summarized data,

$$
\underset{(I \times J)}X = \{x_{ij}\} =
\begin{bmatrix}
x_{11} & \cdots & x_{1J} \\
\vdots &        & \vdots \\ 
x_{I1} & \cdots & x_{IJ}
\end{bmatrix}
$$

The rows and columns of $$X$$ correspond to different **categories (groups)** of different **characteristics.**

Define:

- **Correspondence matrix**: divide $$x_{ij}$$ by **total count,** $$n = \textstyle \sum_{i=1}^I \sum_{j=1}^J x_{ij}$$, to obtain

$$
p_{ij} = \frac{1}{n} x_{ij}, or 
\underset{I \times J}P = \frac{1}{n} X
$$

- **Row and column profiles:**

$$
r_i = \sum_{j=1}^J p_{ij} = \frac{1}{n} \sum_{j=1}^J x_{ij}, or \underset{I \times 1} r = P \mathbf{1}_J
$$
$$
c_j = \sum_{i=1}^I p_{ij} = \frac{1}{n} \sum_{i=1}^I x_{ij}, or \underset{J \times 1} c = P^T \mathbf{1}_I
$$

- Diagonal matrices with elements of $$r$$ and $$c$$:

$$
D_r = diag(r_1,...,r_I)
$$
$$
D_c = diag(c_1,...,c_J)
$$

**Correspondence analysis:**

It can be formulated as the **reduced rank-K "weighted" least squares approximation** to select $$\widehat{P} = \{p_{ij}\}$$ which **minimizes**

$$
\sum_{i=1}^I \sum_{j=1}^J \frac{\left( p_{ij} - \widehat{p}_{ij} \right)^2}{r_i c_j}
= tr \left[ \left( D_r^{-1/2} (P - \widehat{P}) D_c^{-1/2} \right) \left( D_r^{-1/2} (P - \widehat{P}) D_c^{-1/2} \right)^T \right]
$$

From the **result of Johnson & Wichern (2002), p. 72:** The term $$r c^T$$ is common to the approximation $$\widehat{P}$$ whatever the correspondence matrix $$P$$. Thus, it is equivalent to minimize

$$
tr \left[ \left( D_r^{-1/2} (P - r c^T - \widehat{P}) D_c^{-1/2} \right) \left( D_r^{-1/2} (P - r c^T - \widehat{P}) D_c^{-1/2} \right)^T \right]
$$

Similarly with [SVD](https://bowen0701.github.io/blog/2016/10/05/sml-pca-svd), compute the **SVD of $$D_r^{-1/2} (P - r c^T) D_c^{-1/2}$$**:

$$
D_r^{-1/2} (P - r c^T) D_c^{-1/2} = U \Sigma V^T
$$

where $$U$$ and $$V$$ are *orthogonal* matrices with $$U^T U = V^T V = I$$, and $$\Sigma$$ is a rank-K *diagonal* matrix. Thus,

$$
P - r c^T = D_r^{1/2} \left( U \Sigma V^T \right) D_c^{1/2}
          = A \Sigma B
$$

where $$A = D_r^{1/2} U$$ and $$B = D_c^{1/2} V$$.

The above decomposition is often called **generalized SVD:**

$$
P - r c^T = A \Sigma B, \mbox{ with } A^T D_r^{-1} A =  B^T D_c^{-1} B = I
$$

- **Row profile matrix:** divide each row by its sum,

$$
R = D_r^{-1} P = 
\begin{bmatrix}
\frac{p_{11}}{r_1} & \frac{p_{12}}{r_1} & \cdots & \frac{p_{1J}}{r_1} \\
\vdots &        & \vdots \\ 
\frac{p_{I1}}{r_I} & \frac{p_{I2}}{r_I} & \cdots & \frac{p_{IJ}}{r_I} \\
\end{bmatrix}
$$

- **Column profile matrix:** divide each column by its sum,

$$
C = P D_c^{-1} = 
\begin{bmatrix}
\frac{p_{11}}{c_1} & \frac{p_{12}}{c_2} & \cdots & \frac{p_{1J}}{c_J} \\
\vdots &        & \vdots \\ 
\frac{p_{I1}}{c_1} & \frac{p_{I2}}{c_2} & \cdots & \frac{p_{IJ}}{c_J} \\
\end{bmatrix}
$$

- **Row deviations from average row profile:**

$$
R - \mathbf{1}_I c^T = D_r^{-1} P - \mathbf{1}_I c^T
         = D_r^{-1} (P - rc^T)
         = D_r^{-1} A \Sigma B^T \\
         = D_r^{-1} \left( D_r^{1/2} U \right) \Sigma B^T
         = D_r^{-1/2} U \Sigma B^T
$$

- **Column deviations from average column profile:** similarly,

$$
(C - r\mathbf{1}_J^T)^T = D_c^{-1} P - r\mathbf{1}_J^T
                        =  D_c^{-1/2} V \Sigma A^T 
$$

- **Principal coordinates of rows:** The coordinates for $$(R - \mathbf{1}_I c^T)$$ w.r.t. the axes of $$b_1,...,b_k$$ are given by the columns of

$$
F = D_r^{-1/2} U \Sigma
$$

- **Principal coordinates of columns:** The coordinates for $$(C - r\mathbf{1}_J^T)^T$$ w.r.t. the axes of $$b_1,...,b_k$$ are given by the columns of

$$
G = D_c^{-1/2} V \Sigma
$$

- **Standard coordinates of rows:**

$$
\Phi = D_r^{-1/2} U
$$

- **Standard coordinates of columns:**

$$
\Gamma = D_c^{-1/2} V
$$

Relationship:

$$
F^T D_r F = G^T D_c G = \Sigma^2
$$
$$
\Phi^T D_r \Phi = \Gamma^T D_c \Gamma = I
$$

**Inertia:** total variance of the correspondence matrix $$P$$, which resembles a chi-square statistic,

$$
Inertia = \sum_{i=1}^I \sum_{j=1}^J \frac{\left( p_{ij} - r_i c_j \right)^2}{r_i c_j} 
        = \sum_{k=1}^K \lambda_k^2
$$


**Evaluation of 2D graphical display:**

- **Inertia associated with dimension $$k$$, for $$k = 1,2$$:** $$\lambda_k^2$$.
- **Proportion of total inertia:** explained total variance, total the larger, the better.

$$
\frac{\left(\lambda_1^2 + \lambda_2^2 \right)}{\sum_{k=1}^K \lambda_k^2}
$$

## Visualization Maps

- **(1) Symmetric map:** $$(F, G)$$, rows and columns in principal coordinates.
- **(2) Asymmetric map with row principal:** $$(F, \Gamma)$$, rows (of more interest) in principal and columns in standard coordinates.
- **(3) Asymmetric map with column principal:** $$(\Phi, G)$$, rows in standard and columns (of more interest) in principal corordinates. 

**Symmetric map (1):**

- Since principal coordinates of rows and columns, $$(F, G)$$, are scaled similarly, **joint display of two separate maps** finds some justification.
- Distances between row points (and between column points) are meaningful.
- However, there is a **danger in interpreting row-to-column distances directly:** not possible to deduce from the closeness of a row and column point the fact that the corresponding row and column necessarily have a high association.

**Asymmetric maps (2) and (3):**

- Notice that the **row and column points in biplot lie in the same space**, with the column points defining the most extreme profiles possible, and for example, row profiles are at weighted averages of the column points, the weights being the profile elements.
- Hence, closeness of a row and column point indicates a high association; we can calculate **row-to-column distances for one column at a time.**

Figures 9.4 and 9.5 in Greenacre (2007):

<div style="text-align:center">
<img src="/images/symmetric_map.png" alt="Drawing" style="width: 550px;"/>
</div>

<div style="text-align:center">
<img src="/images/asymmetric_map.png" alt="Drawing" style="width: 550px;"/>
</div>

## References

- Johnson & Wichern (2002). Applied Multivariate Statistical Analysis.
- Rencher & Christensen (2012). Methods of Multivariate Analysis.
- Greenacre (2007). Correspondence Analysis in Practice.
- Abdi & Williams (Ency. Res. Design, 2010). Correspondence Analysis.
