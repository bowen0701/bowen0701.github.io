---
layout: post
comments: true
title: "Review: Amazon.com Recommendations Item-to-Item Collaborative Filtering"
date: 2016-07-22
---

## [Paper: Linden, Smith & York (IEEE Internet Computing, 2003)](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)

- Company: Amazon
- Test of Time Award from The International World Wide Web Conference committee: "this outstanding paper has had a considerable real-world impact".


## Summary

This seminal paper introduced what is recommender system (RecSys) and the corresponding challenges. Futher, it reviews two types of RecSys and summarizes the tradictional user-to-user collaborative filtering (CF), cluster models, content-based methods and the proposed item-to-item CF.


## Introduction

**Recommendation system:** Use input about a user's interest to generate a list of a recommendated items. user's interest can be items that users purchase, items that users explicitly rate, items viewed, demographic data, suject interests, and favorite artists, etc.

**Challenges:**

- Huge amount of data: ten of millions of users and millions of distinct catalog items.
- Require the recommendation results set to be returned in realtime.
- Cold start: New users typically have extremely limited information.
- Information glut: Old users can have a glut of information.
- Each interaction provides valuable user data, and algorithm must respond immediately to new information.

**Literature review:** Common RecSys approaches include

- Traditional collaborative filtering (CF)
- Cluster models
- Content-based (or search-based) methods.

This work focuses on the item-to-item CF.

**Two types of RecSys:**

- Finding similar "users" whose purchased and rated items overlap the user's purchased and rated items.
  * Traditional CF / cluster models.
- Finding similar "items", not similar users.
  * Content-based methods / item-to-item CF.


## Traditional CF

- Represent a user as an N-dimensional vector of items (N: # of distinct catalog items).
- Typically multiply user vector components by the inverse frequency to make the less well-known items much more relevant.
- User vectors are generally sparse.
- Measure user vectors's similarity by cosine similarity measure.
- Collect similar users's items.
- Rank each item according to how many similar users purchased it.

**Remarks:**

- Scaling issue due to expensive computational cost.
  * Reduce user size by (1) random sampling or (2) discarding users with fewer purchases.
  * Reduce item size by (1) partitioning item space.
  * Dimensionality reduction techniques: (1) Clustering, and (2) PCA.
- The above methods may reduce recommendation quality.


## Cluster models

- Divide the users into many segments and treat recommendation as a classification problem.
  * Applying clustering or other unsupervised learning algortihms via similarity measures.
  * Manually determined segments.
- Assign the user to the segment containing the most similar users.
- Generate recommendations by using the purchased and ratings of the users in the segment.

**Remarks:**

- Better scalability than tranditional CF: complex and expensive clustering is run offline, improved by
  * Sampling
  * Dimensionality reduction.
- The recommendation may be less relevant: similar users that the cluster models find are not the most similar users.


## Content-based methods

Treat the recommendation as a "search query" to find other popular items: **information retrieval.**

- same author / director
- similar keywords or subjects.

**Remarks:**

- For old users with thousands of purchases, it is impractical.
- Resolve by random sampling subset / summary of the data.
- Recommendation quality is relatively poor.


## Proposed item-to-item CF

- Match each of the user's purchased and rated "items" to similar "items".
- Combine those similar items into a recommendation list.
- Since many item pairs have no common users, all pairs computation of similar-items table by building product-to-product matrix is inefficient.
- Proposed algorithm: 
  * Represent a "item" as an M-dimensional vector of users (M: # of users who have purchase that item)
  * Calculate the (cosine) similarity between a "single" item and "all related" items

<div style="text-align:center">
<img src="/images/amazon_item2itemCF_algo.png"/>
</div>

- Specifically, for item $$I_1$$, there are users, $$(C_1,...,C_M)$$, who purchased or rated this item. For each user, $$C_m, m = 1,...,M$$, collect all of her other purchased or rated items, say item $$I_2$$. After going through all of users's purchased or rated items, compute the similarity between all pairs of $$(I_1, I_2)$$, where $$I_2$$ are all related items to $$I_1$$.

**Remarks:**

- The offline computation of the similar-items table: computationally expensive.
- The online recommendation: very quick, depending only on the number of items the user purchased or rated.
  * Find similar items to the user's purchases and ratings.
  * Aggregate those items.
  * Recommend the most popular or correlated items.


## Comments

- One of the best and valuable and production-efficient recommendation algorithm, massively applied by Amazon.
- I think the literature review for RecSys types may be not clear enough and be missing one important type: (1) the traditional CF shall be clarified as the user-to-user CF, and (2) there shall exist the user-to-item CF which forms an user-to-item feedback matrix and compute the latent vectors of users and items by applying the stochastic gradient descent (SGD) or alternative least squares (ALS).
