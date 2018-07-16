---
layout: post
comments: true
title: "Netflix's Personalized Recommendation"
excerpt: "Netflix's two old but classic articles about personalization and recommendation."
date: 2018-07-13
mathjax: true
---

## Introduction

I read these two famous blog posts from [Netflix's tech blog](https://www.netflix.com/) many years ago, and I really like them:

- [Netflix Recommendations: Beyond the 5 stars (Part 1)](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429)
- [Netflix Recommendations: Beyond the 5 stars (Part 2)](https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5)

Since I recently joined [Zalando](http://www.zalando.com/) as a Data Scientist working on *machine learning algorithms/engineering in Personalization Engine,* these two articles occurred to me and I would like to seriously review them. Here are my notes (maybe I will revise them and add some thoughts later).

## Personalization and Recommendation

- Of course, when we say “you”, we really mean everyone in your household. It is important to keep in mind that Netflix’ personalization is intended to handle a household that is likely to have different people with different tastes... Even for a single person household we want to appeal to your range of interests and moods. To achieve this, in many parts of our system we are not only optimizing for accuracy, but also for diversity.
- Our business objective is to maximize member satisfaction and month-to-month subscription retention, which correlates well with maximizing consumption of video content. We therefore optimize our algorithms to give the highest scores to titles that a member is most likely to "play" and "enjoy".
- If you are looking for a ranking function that optimizes consumption, an obvious baseline is item popularity. The reason is clear: on average, a member is most likely to watch what most others are watching. However, popularity is the opposite of personalization: it will produce the same ordering of items for every member. Thus, the goal becomes to find a personalized ranking function that is better than item popularity, so we can better satisfy members with varying tastes.

### Metrics for optimization

- Popularity
- Predicted Ratings
- Similarity: in a very broad sense: between movies or between members, and can be in multiple dimensions such as metadata, ratings, or viewing data.
- Accuracy, Precision, Recall, or F-score.
- Ranking: 
  * Normalized Discounted Cumulative Gain (NDCG), 
  * Mean Reciprocal Rank (MRR), or 
  * Fraction of Concordant Pairs (Kendall's tau)
- Freshness
- Diversity

## Importance of Both Data and Models

Importance of both data and models in creating an optimal personalized experience for our members.

### Some of the data sources
  
- everal billion item ratings from members. And we receive millions of new ratings a day.
- We already mentioned item popularity as a baseline. But, there are many ways to compute popularity. We can compute it over various time ranges, for instance hourly, daily, or weekly. Or, we can group members by region or other similarity metrics and compute popularity within that group.
- We receive several million stream plays each day, which include context such as duration, time of day and device type.
- Our members add millions of items to their queues each day.
- Each item in our catalog has rich metadata: actors, director, genre, parental rating, and reviews.
- Presentations: We know what items we have recommended and where we have shown them, and can look at how that decision has affected the member’s actions. We can also observe the member’s interactions with the recommendations: scrolls, mouse-overs, clicks, or the time spent on a given page.
- Social data has become our latest source of personalization features; we can process what connected friends have watched or rated.
- Our members directly enter millions of search terms in the Netflix service each day.
- All the data we have mentioned above comes from internal sources. We can also tap into external data to improve our features. For example, we can add external item data features such as box office performance or critic reviews.
- Of course, that is not all: there are many other features such as demographics, location, language, or temporal data that can be used in our predictive models.

### Features from tons of data sources

- Your explicit taste preferences and ratings
- Your viewing history
- Your implicit genre preferences — recent plays, ratings, and other interactions
- Your explicit feedback provided through our taste preferences survey
- Your friends’ recommendations (social media, i.e. Facebook)

### List of personanalization models

Incomplete list of models you should probably know about if you are working in machine learning for personalization:

- Linear Regression
- Logistic Regression
- Elastic Nets
- Singular Value Decomposition (SVD)
- Restricted Boltzmann Machines (RBM)
- Markov Chains
- Latent Dirichlet Allocation (LDA)
- Association Rules
- Gradient Boosted Decision Trees (GBDT)
- Random Forests
- Clustering techniques from the simple k-means to novel graphical approaches such as Affinity Propagation
- Matrix factorization
- Learning to Rank

## Offline-online Testing Process

- Thomas Watson Sr, founder of IBM, put it: “If you want to increase your success rate, double your failure rate.”
- **Offline-online testing process:** tries to combine the best of both worlds. 
  * The offline testing cycle is a step where we test and optimize our algorithms prior to performing online A/B testing. To measure model performance offline we track multiple metrics used in the machine learning community: from ranking measures such as normalized discounted cumulative gain, mean reciprocal rank, or fraction of concordant pairs, to classification metrics such as accuracy, precision, recall, or F-score. We also use the famous RMSE from the Netflix Prize or other more exotic metrics to track different aspects like diversity. We keep track of how well those metrics correlate to measurable online gains in our A/B tests. However, since the mapping is not perfect, offline performance is used only as an indication to make informed decisions on follow up tests.
  * The online A/B testing: Once offline testing has validated a hypothesis, we are ready to design and launch the A/B test that will prove the new feature valid from a member perspective.

<div style="text-align:center">
<img src="/images/netflix_personalized_recsys.png" alt="Drawing" style="width: 900px;"/>
</div>

### A/B testing

1. Start with a hypothesis: 
  * Algorithm/feature/design X will increase member engagement with our service and ultimately member retention
2. Design a test: 
  * Develop a solution or prototype. Ideal execution can be 2X as effective as a prototype, but not 10X.
  * Think about dependent & independent variables, control, significance…
3. Execute the test
4. Let data speak for itself

When we execute A/B tests, we track many different metrics. But we ultimately trust member engagement (e.g. hours of play) and retention... We typically have scores of A/B tests running in parallel.
