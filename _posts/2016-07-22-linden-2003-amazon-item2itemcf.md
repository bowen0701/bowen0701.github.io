---
layout: post
title: "Paper Notes on 'Amazon.com Recommendations Item-to-Item Collaborative Filtering'"
date: 2016-07-22
---

## Paper

- Link: https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf
- Authors: Linden, Smith & York (IEEE Internet Computing, 2003)
- Company: Amazon
- Citations: 3819
- Test of Time Award from The International World Wide Web Conference committee: "this outstanding paper has had a considerable real-world impact".

## Summary

This seminal paper introduced what is recommender system (RecSys) and the corresponding challenges. Futher, it reviews two types of RecSys and summarizes the tradictional user-to-user collaborative filtering (CF), cluster models, content-based methods and the proposed item-to-item CF.

## Key Points

**Recommendation system:** Use input about a customer's interest to generate a list of a recommendated items. Customer's interest can be
- items that customers purchase and 
- items that customers explicitly rate
- items viewed
- demographic data
- suject interests
- favorite artists, etc.

**Challenges:**
- Huge amount of data: ten of millions of customers and millions of distinct catalog items.
- Require the recommendation results set to be returned in realtime.
- Cold start: New customers typically have extremely limited information.
- Information glut: Old customers can have a glut of information.
- Each interaction provides valuable customer data, and algorithm must respond immediately to new information.

**Literature Review:**
- Common recsys approaches (different from the work):
  * Traditional collaborative filtering (CF)
  * Cluster models
  * Content-based (or search-based) methods
- This work: Item-to-item CF.

**Two types of RecSys:**
- Finding similar "customers" whose purchased and rated items overlap the user's purchased and rated items.
  * Traditional CF / cluster models.
- Finding similar "items", not similar customers.
  * Content-based methods / item-to-item CF.

**Traditional CF:**
- Represent a customer as an N-dimensional vector of items (N: # of distinct catalog items).
- Typically multiply user vector components by the inverse frequency to make the less well-known items much more relevant.
- User vectors are generally sparse.
- Measure user vectors's similarity by cosine similarity measure.
- Collect similar customers's items.
- Rank each item according to how many similar customers purchased it.
- Remarks:
  * Scaling issue due to expensive computational cost.
    - Reduce customer size by (1) random sampling or (2) discarding customers with fewer purchases.
    - Reduce item size by (1) partitioning item space.
    - Dimensionality reduction techniques: (1) Clustering, and (2) PCA.
  * The above methods may reduce recommendation quality.

**Cluster models:**
- Divide the customers into many segments and treat recommendation as a classification problem.
  * Applying clustering or other unsupervised learning algortihms via similarity measures.
  * Manually determined segments.
- Assign the user to the segment containing the most similar customers.
- Generate recommendations by using the purchased and ratings of the customers in the segment.
- Remarks:
  * Better scalability than tranditional CF: complex and expensive clustering is run offline, improved by
    - Sampling
    - Dimensionality reduction.
  * The recommendation may be less relevant: similar customers that the cluster models find are not the most similar customers.

**Content-based methods:**
- Treat the recommendation as a "search query" to find other popular items: Information retrieval.
  * same author / director
  * similar keywords or subjects.
- Remarks:
  * For old users with thousands of purchases, it is impractical.
  * Resolve by random sampling subset / summary of the data.
  * Recommendation quality is relatively poor.

**Proposed item-to-item CF:**
- Match each of the user's purchased and rated "items" to similar "items".
- Combine those similar items into a recommendation list.
- Since many item pairs have no common customers, all pairs computation of similar-items table by building product-to-product matrix is inefficient.
- Proposed algorithm: 
  * Represent a "item" as an M-dimensional vector of users (M: # of customers who have purchase that item)
  * Calculate the (cosine) similarity between a "single" item and "all related" items
- The offline computation of the similar-items table: computationally expensive.
- The online recommendation: very quick, depending only on the number of items the user purchased or rated.
  * Find similar items to the user's purchases and ratings.
  * Aggregate those items.
  * Recommend the most popular or correlated items.

## Comments

- One of the best and valuable and production-efficient recommendation algorithm, massively applied by Amazon.
- I think the literature review for RecSys types may be not clear enough and be missing one important type: (1) the traditional CF shall be clarified as the user-to-user CF, and (2) there shall exist the user-to-item CF which forms an user-to-item feedback matrix and compute the latent vectors of users and items by applying the stochastic gradient descent (SGD) or alternative least squares (ALS).
