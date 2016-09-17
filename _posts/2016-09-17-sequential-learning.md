---
layout: post
title: "Notes: Sequential Learning"
date: 2016-09-17
---

I just (on 2015/05/07) read a clear and insightful article on what is the **Sequential Learning problem**, so-called the [Multi-arm Bandit problem](http://www.quora.com/What-is-the-multi-arm-bandit-problem-What-are-some-of-its-implications), on Quora, which was provided by Professor [Yisong Yue](http://www.yisongyue.com/) at Caltech, and finally got a basic understanding about the problem and its potential applications. This topic is interesting to me and will be one of my research projects. Thus I will begin to wrap up some notes on what I've learned during the journey. In this preliminary note, I will quickly summarize, from that article, the basic notions of sequential learning problem and give some of their application examples.

## 1. Multi-arm bandit problem

The multi-arm bandit problem is a **partial-information sequential decision making problem:**

- There are slot machines, where each slot machine is also called an one-armed bandit.
- The setting proceeds over $T$ iterations; hence **sequential**.
- During each iteration, the decision maker chooses an arm to "pull", and receives reward from that action.
- Each arm pull gives a **random** reward, with an unknown but fixed expected value. Of course rewards from pulling different arms are random variables with different expected values.

The goal of the decision maker is to maximize the total reward over all $T$ arm pulls. The decision maker only learns about arms by pulling them; hence **partial information**.

## 2. The central question for multi-arm bandit problem

How to **balance the trade-off between**

- **Exploitation:** exploit our past experience to pull arms that appear to have high average reward,
- **Exploration:** explore by pulling other seemingly poorer arms to gather their information.

## 3. Interesting applications for multi-arm bandit problem

- Online advertising,
- Music or movie recommender system,
- Clinical trial.

### 3.1 Example: Music recommender system

- Exploitation: recommend a song which we are confident is good from our past experience to our user,
- Exploration: try out recommending a new song to see if it may be better and gather more information.

### 3.2 Example: Online advertising

- Exploitation: run an ad that we are confident is good from our past experience,
- Exploration: try out a new ad to see if it may be better and gether more information.

Thus the sequential learning problem is also known as the **problem of sequential experimental design**.

I definitely will exploit and explore this topic further in the near future. :-)
