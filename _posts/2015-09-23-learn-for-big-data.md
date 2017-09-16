---
layout: post
comments: true
title: "Learning for Big Data"
date: 2015-09-23
---

I recently attended an insightful talk, given by [Hsuan-Tien Lin](http://www.csie.ntu.edu.tw/~htlin/) (Professor of Computer Science at National Taiwan University). The topic was about (human) learning for big data. (Yes, not learning *from* big data.) I learned a lots from the talk and would like to take some notes here.

## 1. How to find big data questions?

He first commented that for statistical modeling and machine learning, the must-learn for *big* data is almost equivalent to that for *small* data, but the former is with bigger seriousness.

Then he noted that finding (big) data questions equals to finding research topics:

- **Motivation:** what are you interested in?
    1. Something publishable?
    2. Something that improves xyz performance?
    3. Something that inspired deeper study?
- **Feasibility:** What can or cannot be done?
    1. Modeling feasibility
    2. Computational feasibility
    3. Budget feasibility
    4. Timeline feasibility

Motivation helps generate questions; feasibility helps filter questions. Further, **4Vs in big data**: **variety**, **velocity**, **volume**, **veracity** have different impacts on big data questions: (1) variety and velocity on motivations; that is, the former makes us dream more in big data age, the latter evolves data and questions. And (2) volume and veracity on feasibility; that is, the former is key to computational bottleneck, the latter forces us to model with non-textbook data. Note that we brainstorm from motivation and retionalize from feasibility. We never find right questions in our first study. Good questions come **interactively**.

## 2. What is the best machine learning model for (big) data?

The answer: the best model is **data-dependent!**

- Let's talk about our *data* first
- Perhaps we can start by thinking about *simple* models.

He argued that now there is a *myth*: big data work best with the most sophisticated model; this is partially true: deep learning for image recognition at Google. (I think this is because the correlation structure is too complex to model for image data, and deep learning can embeddedly handle this hidden structure well.) However, quotes from Karl Popper: "Science must begin with myths, and with the criticism of myths." Lin then made some criticism of sophisticated model and thought that should not be the first-choice for big data:

- Time-consumming to train and predict: often mismatch to big data,
- Difficult to tune or modify: often exhausting to use,
- Point of no return: often cannot *simplify* or *analyze*.

I totally agree with his points, but think that these criticism shall not prevent us from understanding and applying the sophisticated models, for example neural networks and deep learning; we just need to understand when to use them, at least not at the beginning.

## 3. What is the first machine learning model for (big) data?

**Linear model** first: Because

- Easy to train and predict,
- Easy to tune and modify,
- Somewhat analyzable,
- Little risk,
- If linear model is good enough, we live happily thereafter. :-)
- Otherwise, try something more complicated, with theoretically nothing lost except *wasted little* computation.

**KISS priciple: Keep it simple, safe.** Keep it in mind.

The must-learn models include

- Linear models,
- Simple models with frequency-based probability estimates, such as Naive Bayes,
- Decision Tree (or perhaps even better, Random Forest) as a KISS non-linear model.

## 4. How should we improve ML performance with (big) data? Feature engineering

Do we have domain knowledge about our problem? Perhaps we can start by encoding our human intelligence and knowledge. In Lin's opinion, data construction is more important for big data than machine learning is. But what is data construction?

**Feature engineering:** make our feature data concrete by embedding domain knowledge:

- If available, great!
- If not, start by **analyzing data** first, not learning from data.

Common feature engineering techniques include encoding, combination, importance estimation, etc. (I will take notes on feature engineering later.) Note that we may ask interactive questions that allow encoding human intelligence into feature construction.

## 5. How to improve unsatisfactory test performance on (big) data?

**Step by step diagonosis:**

- If (training performance is ok, say 90% of the time)
- Combat **overfitting**
- Correct **training/testing mismatch**
- Check for misuse
- Else construct better features by asking more questions
- Now we can try more sophisticated models

## 5.1 Combating overfitting

*Myth:* Our big data is so big that overfitting is impossible. No, big data still require carefull treatment of overfitting, because

- Big data usually are high-dimensional
- Big data usually are heterogeneous
- Big data usually are redundant
- Big data usually are noisy

Hence we need the following two techniques

- **Regularization:** Put brake for models - important to know where the brake is.
- **Validation:** Monitor dashboard - important to ensure correctness.

## 5.2 Correcting training/testing mismatch

**Practical rule of thumb:** Let training scenario match test scenario as much as possible.

## 5.3 Avoid data snooping

Data snooping is equal to human overfitting. We all know honesty matters, however, it is very hard to avoid data snooping.

**Guidlines:**

- Be blind: avoid making modeling decision completely by data,
- Be suspicious: interpret findings by proper feeling of contamination. Keep our data fresh if possible.

## 6. Lin's last secret to winning KDDCups

Carefully balance between

- Data-driven modeling (snooping) and
- Validation (no snooping).

