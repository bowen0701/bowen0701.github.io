---
layout: post
title: "Notes on 'At Airbnb, Data Science Belongs Everywhere'"
date: 2016-09-24
---

Today (actually it was on 2015/07/12) I was so fortunate to read this excellent Airbnb Engineering Blog article: ["At Airbnb, Data Science Belongs Everywhere"](http://nerds.airbnb.com/scaling-data-science/), which was written by Riley Newman, head of data science at Airbnb. Here I took some notes consisting of lots of quotes and a little rewriting.

## 1. The higher-level issues data science teams encounter

As companies grow and how data scientists at Airbnb have responded, will mostly center around how to connect data science with other business functions:

- How we characterize data science,
- How it’s involved in decision-making, and
- How we’ve scaled it to reach all sides of Airbnb.

The foundation on which a data science team rests is the culture and perception of data elsewhere in the organization. So defining **how we think about data** has been a prerequisite to ingraining data science in business functions.

In the past, data was often constructed purely as a measurement tool. ... Interactions with us data scientists would therefore tend to come in the form of a request for a fact: how many listings do we have in Paris? What are the top 10 destinations in Italy?

While answering questions and measuring things is certainly part of the job, at Airbnb we characterize data in a more human light: it’s the voice of our customers. ... If you can recreate the sequence of events leading up to that decision, you can learn from it; it’s an indirect way of the person telling you what they like and don’t like. ... This sort of feedback can be a goldmine for decisions about community growth, product development, and resource prioritization.

## 2. How we characterize data science?

Data science is an act of interpretation -- we translate the customer’s “voice” into a language more suitable for decision-making.

A good data scientist is therefore able to get in the mind of people who use our product and understand their needs.

Our distinction between good and great is impact -- using insights to influence decisions and ensuring that the decisions had the intended effect.

## 3. Connecting data scientists with decision-makers

When decision-makers don’t understand the ramifications of an insight, they don’t act on it. When they don’t act on it, the value of the insight is lost.

**How data science is involved in decision-making process:**

**1. Learn:** Begin by learning about the context of the problem, putting together a full synopsis of past research and efforts toward addressing the opportunity. This is more of an exploratory process aimed at sizing opportunities, and generating hypotheses that lead to actionable insights.
**2. Plan:** That synopsis translates to a plan, which encompasses prioritizing the lever we intend to utilize and forming a hypothesis for the effect of our efforts. Predictive analytics is more relevant in this stage, as we have to make a decision about what path to follow, which is based on where they expect to have the largest impact.
**3. Test:** Design a controlled experiment through which to roll the plan out. A/B testing is very common now, but our collaboration with all sides of the business opens up opportunities to use experimentation in a broader sense -- operational market-based tests, as well as more traditional online environments.
** 4. Measure:** Finally, we measure the results of the experiment, identifying the causal impact of our efforts. If successful, we launch to the whole community; if not, we cycle back to learning why it wasn’t successful and repeat the process.

## 4. Hybrid of centralized & embedded data science teams

Centralized model (we began with it): offering of opportunities to learn from each other and stay aligned on metrics, methodologies, and knowledge of past work. However, ... we [found that although we’re ultimately in the business of decision-making, we] couldn’t do this successfully when siloed:

- Partner teams didn’t fully understand how to interact with us, and
- The data scientists on our team didn’t have the full context of what they were meant to solve or how to make it actionable. Over time we became viewed as a resource and, as a result, our work became reactive -- responding to requests for statistics rather than being able to think proactively about future opportunities.

Hybrid centralized/embedded structure (now): still follow the centralized model, in that we have a singular data science team where our careers unfold, but we have broken this into sub-teams that partner more directly with engineers, designers, product managers, marketers, and others. ... Doing so we can

- Accelerate the adoption of data throughout the company, and has elevated data scientists from reactive stats-gatherers to proactive partners.
- Be able to maintain a vantage point over every piece of the business, allowing us to form a neural core that can help all sides of the company learn from one another.

Being completely data-driven can lead to optimizing toward a local maximum; finding a global maximum requires shocking the system from time to time. But they reflect different points where data can be leveraged in a project’s lifecycle.

## 5. How we’ve scaled data science?

Scaling a data science team is an essential part of the company, but doing it in hypergrowth isn’t easy. But it is possible. Especially if everyone agrees that it’s not just a nice part of the company, it’s an essential part.

How we’ve scaled data science to reach all sides of Airbnb? By democratizing it. The reality of a hypergrowth startup is that the scale and speed at which decisions need to be made will inevitably outpace the growth of the data science team. ... [and] it was now also impossible to meet and work with every employee. We needed to find a way to improve

**1. Individual interactions:** Investment in tools to empower data scientists.  Those individual interactions become more efficient as data scientists are empowered to move more quickly. Investing in data infrastructure is the biggest lever here -- adopting faster and more reliable technologies for querying an ever-growing volume of data.
** 2. Empowering other teams:** Democratizing ability to access information. It is about removing the burden of reporting and basic data exploration from the shoulders of data scientists so they can focus on more impactful work. Dashboards are a common example of a solution. We’ve also developed a tool to help people author queries ... against a robust and intuitive data warehouse.
** 3. Informing the company:** Education programs and exploratory tools. ... We think about the culture of data in the company as a whole. Educating people on how we think about Airbnb’s ecosystem, as well as how to use tools like [Airpal](http://nerds.airbnb.com/airpal/), removes barriers to entry and inspires curiosity about how everyone can better leverage data. Similar to empowering teams, this has helped liberate us from ad-hoc requests for stats.
** 4. Connecting our users:** Building **data products**. The broadest example of scaling data science is enabling guests and hosts to learn from each other directly. This mostly happens through data products, where machine-learning models interpret signals from one set of community-members to help guide others.

## 6. Measuring the impact of a data science team

Measuring the impact of a data science team is ironically difficult, but good signals are

- There’s now a unanimous desire to consult data for decisions that need to be made by technical and non-technical people alike, and
- Our team members are seen as partners in the decision-making process, not just reactive stats-gatherers.

## 7. Future plans

When we are ready to take on exciting new problems?

- Our data infrastructure is stable,
- Our query tools are sophisticated, and
- Our warehouse is clean and reliable.

On the immediate horizon, we look forward to

- Shifting from batch to real-time processing;
- Developing a more robust anomaly-detection system;
- Deepening our understanding of network effects; and
- Increasing our sophistication around matching and personalization.
