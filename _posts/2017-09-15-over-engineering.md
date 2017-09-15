---
layout: post
comments: true
title: "Notes: Please Don't Do Over-Engineering"
date: 2017-09-15
---

I recently read an insightful article ["You are not Google"](https://blog.bradfieldcs.com/you-are-not-google-84912cf44afb) by Ozan Onay, which I think really resonates me: "if you’re using a technology that originated at a large company, but your use case is very different, it’s unlikely that you arrived there deliberately; no, it’s more likely you got there through a ritualistic belief that imitating the giants would bring the same riches."

The following are my notes on this article:

- "Next time you find yourself Googling some cool new technology to (re)build your architecture around, I urge you to stop and follow UNPHAT instead:"
  * "Don’t even start considering solutions until you Understand the problem. Your goal should be to “solve” the problem mostly within the problem domain, not the solution domain."
  * "eNumerate multiple candidate solutions. Don’t just start prodding at your favorite!
Consider a candidate solution, then read the Paper if there is one."
  * "Determine the Historical context in which the candidate solution was designed or developed."
  * "Weigh Advantages against disadvantages. Determine what was de-prioritized to achieve what was prioritized."
  * "Think! Soberly and humbly ponder how well this solution fits your problem. What fact would need to be different for you to change your mind? For instance, how much smaller would the data need to be before you’d elect not to use Hadoop?"
- "Is it the best solution to your problem? What is your problem exactly, and what are other ways you could solve it?"
- "Maybe you expect to scale. But have you done the math? Are you likely to accumulate data faster than the rate at which SSD prices will go down? How much would your business need to grow before all your data would no longer fit on one machine? As of 2016, [Stack Exchange served 200 million requests per day, backed by just four SQL servers:](https://nickcraver.com/blog/2016/02/17/stack-overflow-the-architecture-2016-edition/) a primary for Stack Overflow, a primary for everything else, and two replicas."
- "What’s important is that you actually use the right tool for the job. Google knows this well: once they decided that MapReduce wasn’t the right tool for building the index, they stopped using it."
- "What we’re all imploring you to do is to think! And to actually understand the problem you are trying to solve."
- "In Polya’s galvanic words: It is foolish to answer a question that you do not understand. It is sad to work for an end that you do not desire."

Speed is the most important factor for company / team / product successes; the shorter time we need for deploying experiments / development to production, the more we can perform prototyping and iterations, and thus the more likely we can achieve our goals. Note that we are data scientists and data engineers in 2017 using "great enough" software stack, such as Spark on AWS, we should first understand the problem we are trying to solve and weigh advantages against disadvantages for all of multiple candidate tools or solutions, but not pursue the "most advanced" ones for us to be "professional" and "scale in the long-term future." The more time we spend on migrating to those "over-engineered infra", the less likely we can get better and better by letting our products interacting with the real-world users.

Finally, let me recall my blog article: ["Precious Lecture by my Data Science Advisor"](https://bowen0701.github.io/blog/2016/02/13/data-science-advisor), 

- "Why did you do it that way? Why not simpler? Why not more complicated?" (Note that "simpler" goes before "more complicated.")
- "Why that metric to evaluate yourself?"
- "What is the purpose of this? Is there a better way to achieve that purpose? And so on."
