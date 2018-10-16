---
layout: post
comments: true
title: "ML Notes: Machine Learning Metrics"
excerpt: "Machine learning metrics for regression and classification problems."
date: 2018-01-26
mathjax: true
---

## Introduction

How to measure our machine learning model's performance? Different problems with different purposes need different metics for us to fairly and effectively make decision. This notebook is to introduce useful metrics for machine learning evaluation.

### Outline

- R-Square: To measure model fitness for continuous response
- Mean Squared Errors (MSE)
- Mean Absolute Errors (MAE)
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)
- Variance Inflation Factor (VIF): To measure multi-collineairity for features
- Cook Distance: To detect influential instance points
- Confusion Matrix
- Accurary
- Precission / Recall
- Specificity
- False Positive Rate (FPR)
- Area under Curve (AUC)
- Gain / Lift Charts
- Kolmogorov-Smirnov Chart
- Gini Coefficient
- Kendall's tau
- $$F_{\beta}$$-Measures
- Macro / Micro / Weighted Precision and Recall
- Precision@K
- Average Precision (AP)
- Mean Average Precision (MAP)
- Mean Recipocal Rank (MRR)
- Discounted Cumulative Gain (DCG)
- Normalized DCG (NDCG)
- Expected Reciprocal Rank (ERR)

## R-Square

For regression problems with continuous response $$y$$, R-square is to measure goodness of fit for regression model, defined as a ratio: **Explained Variation / Total Variation.**

$$
R^2 = 1- \sum_{i=1}^n (\hat y_i - y_i)^2 / \sum (y_i - \bar y)^2
$$

where $$y_i$$ is the continuous response for instance $$i$$, with sample mean $$\bar y$$, $$\hat y_i$$ is the predicted value of $$y_i$$, $$\sum_{i=1}^n (\hat y_i - y_i)^2$$ is the **residual sum of squares,** and $$\sum (y_i - \bar y)^2$$ is the **total sum of squares.**

## Mean Squared Errors (MSE)

For regression problems, MSE is a common metric for model evaluation and use **squared errors** to measure the discrepancy between $$\hat y_i$$ and $$y_i$$.

$$
MSE = \sum_{i=1}^n (\hat y_i - y_i)^2
$$

## Mean Absolute Errors (MAE)

For regression problems, MAE, instead of using squared errors, uses **absolute errors** to measure prediction discrepancy.

$$
MAE = \sum_{i=1}^n |\hat y_i - y_i|
$$

## Akaike Information Criterion (AIC)

For model selection based on likelihood inference, we would like to penalize higher likelihood (which is good) by using more features (which is bad). A classic metric is the AIC:

$$
AIC = -2 \times \text{log-likelihood} + 2k
$$

where $$k$$ is the number of features.

## Bayesian Information Criterion

For model selection in Bayesian framework, we can use the BIC:

$$
BIC = -2 \times \text{log-likelihood} + k \log(n)
$$

where $$n$$ is the number of instances, and $$k$$ is the number of features.

Now we shift geer a little to talk about useful metrics to measure features's multi-collinearity and to detect influential instance.

## Variance Inflation Factor (VIF)

To measure multi-collineairity for multiple features, the VIF is a classic metric. 

- First, we calculate $$R^2_j$$, which is the R-square by fitting $$X_j$$ by $$X_k$$, for all $$k \neq j$$, 
- Then, the VIF is to measure other features's interpretation ability to one specific feature, say $$X_j$$,

$$
VIF_j = 1 / (1 - R^2_j)
$$

## Cook Distance

To detect influential instance:

- $$\hat \beta$$: usual regression weights
- $$\hat \beta_{(-i)}$$: regression weights obtained by excluding instance $$i$$
- The Cook distance is defined by

$$
(\hat \beta - \hat \beta_{(-i)})^T X^T X (\hat \beta - \hat \beta_{(-i)}) \big / [(p + 1) \sigma^2]
$$

where $$X$$ is the feature matrix.

The higher Cook distance for instance $$i$$, the higher likely instance $$i$$ is a influential point.

Here we start to introduce lots of metrics basically focusing on classification problems with categorical response; for example binary response $$y = 0, 1$$.

## Confusion Matrix

|   True       | Predicted Positive | Predicted Negative |
|-----------------------------------|--------------------|
|Postitive (P) |         TP         |          FN        |
|Negative  (N) |         FP         |          TN        |

where $$n = Total = TP + FP + FN + TN$$.

## Accurary

Accuracy is the proportion of true positive and negative cases are correctly identified.

$$
Accurary = (TP + TN) / n
$$

## Precision

Precision is the proportion of predicted positive cases are true positive.

- Also called **Positive Predictive Value (PPV)**
- **Precision-Recall Curve's y-axis** (see later)

$$
Precision = TP / (\textit{Predicted Positive}) = TP / (TP + FP)
$$

## False Discovery Rate (FDR)

$$
FDR =  1 - Precision = FP / (\textit{Predicted Positive}) = FP / (TP + FP)
$$

## Recall

Recall is the proportion of true positive cases are identified correctly as predictive positive.

- Also called **Sensitivity,** or **True Positive Rate (TPR)**
- **ROC Curve's y-axis** and **Preciasion-Recall Curve's x-axis** (see later)

$$
Recall = Sensitivity = TP / Positive = TP / P = TP / (TP + FN)
$$

## Specificity

Specificity is the proportion of predicted negative cases among true negative.

- Also called **True Negative Rate (TNR)**
- $$1 - Specificity$$: **ROC Curve's x-axis** (see later)

$$
Specificity = TN / N = TN / (FP + TN)
$$

## False Positive Rate (FPR)

FPR is the proportion of predicted positive cases among true negative.

- **ROC Curve's x-axis** (see later)

$$
FPR = 1 - Specificity = FP / N = FP / (FP + TN)
$$

## Area under Curve (AUC)

To summarize a single number for model selection. Note that the following two ROC and PR Curves are almost independent of the response rate, compared with lift chart (see later).

### AUC under Receiver Operating Characteristic (ROC) Curve

**ROC Curve:** 

- y-axis: $$Recall = Sensitivity = TP / P = TP / (TP + FN)$$
- x-axis: $$FPR = 1 - Specificity = FP / N = FP / (FP + TN)$$

**AUC-ROC:** area under the ROC Curve.

Performance guideline:

- 0.90-1.00: excellent
- 0.80-0.90: good
- 0.70-0.80: fair
- 0.60-0.70: poor
- 0.50-0.60: fail

### AUC under Precision-Recall (PR) Curve

**PR Curve:**

- y-axis: $$Precision = TP / (\textit{Predicted Positive}) = TP / (TP + FP)$$
- x-axis: $$Recall = Sensitivity = TP / (\textit{True Positive}) = TP / P = TP / (TP + FN)$$

**AUC-PR:** area under the PR Curve.

## Gain / Lift Charts

To check the **rank ordering of the probabilities,** we can use the Gain Chart or Lift Chart.

- Step 1 : Calculate probability for each observation
- Step 2 : Rank these probabilities in decreasing order.
- Step 3 : Build deciles with each group having almost 10% of the observations; group i is located in the (i - 1)th and ith deciles.
- Step 4 : Calculate the response rate at each deciles for Good (positive), Bad (negative) and total.

Then we calculate the following information in a table:

- Decile ID: 1,...,10
- Count for true negative cases
- Count for true positive cases
- Grand total count for one decile
- %Right
- %Wrong
- %Population
- Cum %Right
- Cum %Population
- Lift @decile
- Total lift

[Insert Table here??]

### Cumulative Gain Chart

- y-axis: Cum %Right
- x-axis: Cum %Population

**Lift:** Cum %Right / Cum %Population.

For example, the first decile with 10% of the population has 14% of positive cases. This means we have a 14% / 10% = 140% lift at first decile.

[Insert Figure here??]

How is this result? We can evaluate the result compared with the **maximum lift at first decile.** For example,

- Total number of positive cases are 3850. 
- Also the first decile contains 543 observations. 
- So the maximum lift at first decile could have been 543/3850 ~ 14.1%. 
- Hence, we are quite close to perfection with this model.

### Lift Charts

**Total lift chart:**

- y-axis: Total lift
- x-axis: Cum %Population (decile) 

[Insert Figure here??]

**Decile-wise lift chart:**

- y-axis: Lift @decile
- x-axis: Cum %Population (decile)

[Insert Figure here??]

The purpose for the decile-wise lift chart? For example,

- Our model does well till the 7th decile. Post which every decile will be skewed towards negative cases. 
- Any model with lift @ decile above 100% till minimum 3rd decile and maximum 7th decile is a good model.


### Lift based on conditional probability

$$
Lift = \Pr(action | feature) / \Pr(action)
     = \Pr(action, feature) / [\Pr(action) * \Pr(feature)]
$$

**Issue for lift:** Lift is dependent on total response rate of the population. 

## Kolmogorov-Smirnov Chart

We first calculate the following information in a table:

- Decile ID: 1,...,10
- Count for true negative cases
- Count for true positive cases
- Grand total count for one decile
- %Right
- %Wrong
- %Population
- Cum %Right
- Cum %Wrong
- Cum %Population
- K-S: Cum %Right - Cum %Wrong

[Insert Table here??]

### Kolmogorov-Smirnov Chart

Two curves for Cum %Right & Cum %Wrong

- y-axis: percent
- x-axis: decile

[Insert Figure here??]

**K-S Statistics:** *Maximum separation* between Cum %Right and Cum %Wrong.

## Gini Coefficient

Suppose:

- $$A$$: Area between the ROC Curve and the diagnol line
- $$B$$: Area between the ROC Curve and the y-axis
- $$A+B$$: Area between the diagonal line and the y-axis, which is 0.5.

$$
\textit{Gini} = \frac{ A }{ A + B } = \frac{ A }{ 0.5 } = 2 A = 1 - 2 B
$$

Gini above 60% is good.

## Kendall's tau

For two variable vectors of $$\{(X_i, Y_i), i = 1,...,n\}$$, Kendall's tau is defined by **probability of concordant pairs - probability of discordant pairs.** For two vectors $$(X_1, Y_1), (X_2, Y_2)$$, 

$$
\tau
= \Pr[(X_j - X_i)(Y_j - Y_i) > 0] - \Pr[(X_j - X_i)(Y_j - Y_i) < 0]
$$

- **Concordant pairs:** $$(X_2 - X_1)(Y_2 - Y_1) > 0$$, e.g., $$X_2 > X_1$$ and $$Y_2 > Y_1$$
- **Discordant pairs:** $$(X_2 - X_1)(Y_2 - Y_1) < 0$$, e.g., $$X_2 > X_1$$ and $$Y_2 < Y_1$$

$$
\hat \tau
= \frac{\sum_{i < j} I[(X_j - X_i)(Y_j - Y_i) > 0] - \sum_{i < j} I[(X_j - X_i)(Y_j - Y_i) < 0]}{n(n-1)/2}
$$

The following are variations of Precision or Recall or their combinations.

## $$F_{\beta}$$-Measures

$$F_{\beta}$$-Measures is a **harmonic mean of Precision & Recall:** for $$\beta > 0$$,

$$
F_{\beta} 
= \frac{1}{\frac{\beta^2}{\beta^2 + 1} \cdot \frac{1}{Precision} + \frac{1}{\beta^2 + 1} \cdot \frac{1}{Recall}}
= \frac{ (\beta^2 + 1) Precision \cdot Recall }{ \beta^2 \cdot Precision + Recall}
$$

## Macro / Micro / Weighted Precision and Recall

**Macro:** Mean of the binary metrics, giving equal weight to each class. 

- In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance. 
- On the other hand, the assumption that all classes are equally important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.

**Micro:** Give each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). 

- Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient.
- Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored.
- In multi-class problem, micro Precision & Recall will be same value

**Weighted:** accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.

### Example: Micro / Macro-Precision / Recall

For a set of data:

- True positive (TP1) = 12
- False positive (FP1) = 9
- False negative (FN1) = 3
- Precision (P1) = TP1 / (TP1 + FP1) = 57.14% 
- Recall (R1) = TP1 / (TP1 + FN1) = 80%

For a different set of data:

- True positive (TP2) = 50
- False positive (FP2) = 23
- False negative (FN2) = 9
- Precision (P2) = TP2 / (TP2 + FP2) = 68.49
- Recall (R2) = TP2 / (TP2 + FN2) = 84.75

Thus,

- Micro-Average Precision = (TP1 + TP2) / (TP1 + TP2 + FP1 + FP2) = (12 + 50) / (12 + 50 + 9 + 23) = 65.96.
- Micro-Average Recall = (TP1 + TP2) / (TP1 + TP2 + FN1 + FN2) = (12 + 50) / (12 + 50 + 3 + 9) = 83.78.
- Macro-Average Precision = (P1 + P2) / 2 = (57.14 + 68.49) / 2 = 62.82.
- Macro-average Recall = (R1 + R2) / 2 = (80 + 84.75) / 2 = 82.25.

The following are the well known **Information Retrieval Metrics.**

## Precision@K

Precision@K is the **number of relevant results on the first k results.**

- Nevertheless, it fails to take into account the positions of the relevant documents among the top k.
- More useful than Recall@k since few users will be interested in reading all of their interested articles.

## Average Precision (AP)

Average Precision (AP) is defined by **averaging the precision over a set of equally spaced recall levels {0, 0.1, 0.2,..., 1.0}:**

$$
AP
={\frac {1}{11}}\sum_{recall \in \{0, 0.1, \ldots, 1.0\}} precision_{\text{inter}}(recall)
$$

where $$precision_{\text{inter}}(recall)$$ is an interpolated precision that takes the **maximum precision over all recalls greater than $$recall$$:**

$$
precision_{\text{inter}}(r)= \max_{\tilde r: \tilde {r} \geq r} precision(\tilde r)
$$

[Insert Figure here??]

## Mean Average Precision (MAP)

Mean Average Precision (MAP) is for a set of queries $$Q$$, defined by the **mean of the average precision scores for each query.**

$$
MAP(Q) 
= \frac{1}{\vert Q\vert} \sum_{j=1}^{\vert Q\vert} AP_j
= \frac{1}{\vert Q\vert} \sum_{j=1}^{\vert Q\vert} \frac{1}{m_j}
\sum_{k=1}^{m_j} precision(recall_{jk})
$$

where $$\vert Q\vert$$ is the number of queries, and $$AP_j$$ is the average precision for query $$j$$ with $$m_j$$ number of recall levels.

[Insert Figure here??]

## Mean Reciprocal Rank (MRR)

Mean Reciprocal Rank (MRR) is the average of the reciprocal ranks of the 1st relevent document for a set of queries $$Q$$:

$$
MRR(Q) = \frac {1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_{i}}.
$$

where $$rank_{i}$$ refers to the rank position of the 1st relevant document for the i-th query.

## Cumulative Gain (CG)

Cumulative Gain (CG) allows us to consider, instead of binary level for document relevance, **relative relevance levels.** CG at a particular rank position $$k$$ is defined as:

$$
CG_k = \sum_{i=1}^{k} l_{i}
$$

where $$r$$ is the rank, $$l_{r}$$ is the **relevance level at rank $$r$$,** and $$k$$ is the number of documents. We typically use **five levels of relevance:** $$l_{r} = 0, 1, 2, 3, 4$$.

## Discounted Cumulative Gain (DCG)

The advantage of DCG over CG is that it **penalizes highly relevant documents appearing lower in a search result list** as the **discounted relevance value** is reduced logarithmically proportional to the position of the result.

$$
DCG_{k} 
= \sum_{r=1}^{k} \frac{l_{r}}{log_{2}(r + 1)} 
$$

An alternative form of DCG is

$$
DCG_{k} 
= \sum_{r=1}^{k} \frac{2^{l_{r}} - 1}{log_{2}(r + 1)} 
$$

where $$l_{r}$$ is the relative degree of relevance at rank $$r$$.

## Normalized DCG (NDCG)

Since DCG may vary significantly for different queries or systems, to fairly compare performances the **normalised version of DCG uses a maximun DCG,** which sorts documents by their relevance, generating an maximum DCG, $$maxDCG_{k}$$, at rank $$k$$:

$$
NDCG_{k} = \frac{DCG_{k}}{maxDCG_{k}}
$$

Note that in a perfect ranking system, the $$DCG_{k}$$ will be the same as the $$DCG_{k}$$, thus this results in $$maxDCG_k = 1.0$$. For different queries or systems, all of NDCG $$\in$$ [0, 1] and are comparable.

## Expected Reciprocal Rank (ERR)

$$
ERR 
= \sum_{r=1}^n \frac{1}{r} P(\text{user stops at position } r)
= \sum_{r=1}^n \frac{1}{r} R_r \prod_{i=1}^{r-1} (1 - R_i)
$$

where

$$
R_i = \frac{2^{l_i} - 1}{2^{l_m}}
$$

and $$l_m$$ is the maximum relevance level value.

Note tht Precision@k, AP, MAP and MRR are applicable for **binary relevance levels.** Whereas NDCG and ERR are for **multiple relevance levels.** Further, TBD.

## References

- Burges (2010). From RankNet to LambdaRank to LambdaMART: An Overview.


```python

```
