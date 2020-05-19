---
layout: post
comments: true
title: "Collected Wisdom of Scientists"
excerpt: "Computer scientists, statisticians, mathematicians on research and life."
date: 2020-02-20
mathjax: true
---

<span style="color: #286ee0;">[Updated on 2020/05/17: add Claude Shannon.]</span>
<br/>
<span style="color: #286ee0;">[Updated on 2020/05/16: add Bertrand Russell.]</span>

## Outline

- [Claude Shannon (Computer Scientist)](#claude-shanno-computer-scientist)
- [Bertrand Russell (Mathematician)](#bertrand-russell-mathematician)
- [Thomas Cover (Computer Scientist, Statistician)](#thomas-cover-computer-scientist-statistician)
- [Maryam Mirzakhani (Mathematician, First Female Fields Medalist)](#maryam-mirzakhani-mathematician-first-female-fields-medalist)
- [John Tukey (Mathematician, Statistician)](#john-tukey-mathematician-statistician)
- [George Box (Statistician)](#george-box-statistician)
- [Bradley Efron (Statistician)](#bradley-efron-statistician)
- [Jerzy Neyman (Statistician)](#jerzy-neyman-statistician)
- [Fred Mosteller (Statistician)](#fred-mosteller-statistician)
- [David Blackwell (Statistician)](#david-blackwell-statistician)
- [Sara van de Geer (Mathematician, Statistician)](#sara-van-de-geer-mathematician-statistician)
- [Nilanjan Chatterjee (Statistician)](#nilanjan-chatterjee-statistician)
- [David Dunson (Statistician)](#david-dunson-statistician)

## Claude Shannon (Computer Scientist)

Claude Shannon is well known as the father of information theory. Recently I read a fantastic article ["Claude Shannon's Problem Solving Process"](http://www.businessinsider.com/engineer-claude-shannon-problem-solving-process-2017-7), on Business Insider, which introduced to me a Shannon's unpublished paper for lecture to his colleagues at Bell Labs: [Creative Thinking](http://www1.ece.neu.edu/~naderi/Claude%20Shannon.html). I think this paper is a great reference for data scientists about how to solve problem.

The following are my notes on the classic paper:

"I think we could set down three things that are fairly necessary for scientific research or for any sort of inventing or mathematics or physics or anything along that line. I don’t think a person can get along without any one of these three. The first one is obvious - training and experience. The second thing is a certain amount of intelligence or talent. In other words, you have to have an IQ that is fairly high to do good research work. [The third thing is] motivation. In other words, you have to have some kind of a drive, some kind of a desire to find out the answer, a desire to find out what makes things tick. If you don’t have that, you may have all the training and intelligence in the world, you don’t have questions and you won’t just find answers. ... A good scientist has a great deal of what we can call curiosity. I won’t go any deeper into it than that. He wants to know the answers. He’s just curious how things tick and he wants to know the answers to questions; and if he sees thinks, he wants to raise questions and he wants to know the answers to those."

"[I]dea of [constructive] dissatisfaction.... we don’t like the way things are. ... The idea could be expressed in the words: This is OK, but I think things could be done better. I think there is a neater way to do this. I think things could be improved a little. In other words, there is continually a slight irritation when things don’t look quite right; and I think that dissatisfaction in present days is a key driving force in good scientists."

"Pleasure in seeing net results or methods of arriving at results needed, designs of engineers, equipment, and so on. ... I get a big bang myself out of providing a theorem. If I’ve been trying to prove a mathematical theorem for a week or so and I finally find the solution, I get a big bang out of it. And I get a big kick out of seeing a clever way of doing some engineering problem, a clever design for a circuit which uses a very small amount of equipment and gets apparently a great deal of result out of it."

"I think that good research workers apply these things unconsciously; that is, they do these things automatically."

"The first one that I might speak of is the idea of simplification. ... Probably a very powerful approach to this is to attempt to eliminate everything from the problem except the essentials; that is, cut it down to size. ... Almost every problem that you come across is befuddled with all kinds of extraneous data of one sort or another; and if you can bring this problem down into the main issues, you can see more clearly what you're trying to do."
- "You may have simplified it to a point that it doesn't even resemble the problem that you started with; but very often if you can solve this simple problem, you can add refinements to the solution of this until you get back to the solution of the one you started with."

"A very similar device is seeking similar known problems. ... You have a problem P here and there is a solution S which you do not know yet perhaps over here. If you have experience in the field represented, that you are working in, you may perhaps know of a somewhat similar problem, call it P', which has already been solved and which has a solution, S', all you need to do - all you may have to do is find the analogy from P' here to P and the same analogy from S' to S in order to get back to the solution of the given problem. ... This is the reason why experience in a field is so important that if you are experienced in a field, you will know thousands of problems that have been solved. Your mental matrix will be filled with P's and S's unconnected here and you can find one which is tolerably close to the P that you are trying to solve and go over to the corresponding S' in order to go back to the S you’re after. ... It seems to be much easier to make two small jumps than the one big jump in any kind of mental thinking."

"Another approach for a given problem is to try to restate it in just as many different forms as you can. ... Change the words. Change the viewpoint. Look at it from every possible angle. After you’ve done that, you can try to look at it from several angles at the same time and perhaps you can get an insight into the real basic issues of the problem, so that you can correlate the important factors and come out with the solution. ... It’s difficult really to do this, but it is important that you do. If you don’t, it is very easy to get into ruts of mental thinking. ... That is the reason why very frequently someone who is quite green to a problem will sometimes come in and look at it and find the solution like that, while you have been laboring for months over it."

"Another mental gimmick for aid in research work, I think, is the idea of generalization. ... Can I apply the same principle in more general ways? Can I use this same clever idea represented here to solve a larger class of problems? Is there any place else that I can use this particular thing? ... Next one I might mention is the idea of structural analysis of a problem. ... Suppose you have your problem here and a solution here. You may have two big a jump to take. What you can try to do is to break down that jump into a large number of small jumps. ... Many proofs in mathematics have been actually found by extremely roundabout processes. A man starts to prove this theorem and he finds that he wanders all over the map. He starts off and prove a good many results which don’t seem to be leading anywhere and then eventually ends up by the back door on the solution of the given problem; and very often when that’s done, when you’ve found your solution, it may be very easy to simplify; that is, to see at one stage that you may have short-cutted across here and you could see that you might have short-cutted across there. ... The same thing is true in design work. If you can design a way of doing something which is obviously clumsy and cumbersome, uses too much equipment; but after you’ve really got something you can get a grip on, something you can hang on to, you can start cutting out components and seeing some parts were really superfluous. You really didn’t need them in the first place."

"Now one other thing I would like to bring out which I run across quite frequently in mathematical work is the idea of inversion of the problem. ... You are trying to obtain the solution S on the basis of the premises P and then you can’t do it. Well, turn the problem over supposing that S were the given proposition, the given axioms, or the given numbers in the problem and what you are trying to obtain is P. Just imagine that that were the case. Then you will find that it is relatively easy to solve the problem in that direction. You find a fairly direct route. If so, it’s often possible to invent it in small batches. ... Now I think the same thing can happen in design work. Sometimes I have had the experience of designing computing machines of various sorts in which I wanted to compute certain numbers out of certain given quantities.... But then I got the idea that if I inverted the problem, it would have been very easy to do - if the given and required results had been interchanged; and that idea led to a way of doing it which was far simpler than the first design. The way of doing it was doing it by feedback; that is, you start with the required result and run it back until - run it through its value until it matches the given input. So the machine itself was worked backward putting range S over the numbers until it had the number that you actually had and, at that point, until it reached the number such that P shows you the correct way."

## Bertrand Russell (Mathematician)

Bertrand Russell's 10 commandments, I saw this from Yann LeCun's Facebook post. 

"Do not feel absolutely certain of anything."

"Do not think it worthwhile to produce belief by concealing evidence, for the evidence is sure to come to light."

"Never try to discourage thinking, for you are sure to succeed."

"When you meet with opposition, even if it should be from your spouse or your children, endeavour to overcome it by argument and not by authority, for a victory dependent upon authority is unreal and illusory."

"Have no respect for the authority of others, for there are always contrary authorities to be found."

"Do not use power to suppress opinions you think pernicious, for if you do the opinions will suppress you."

"Do not fear to be eccentric in opinion, for every opinion now accepted was once eccentric."

"Find more pleasure in intelligent dissent than in passive agreement, for, if you value intelligence as you should, the former implies a deeper agreement than the latter."

"Be scrupulously truthful, even when truth is inconvenient, for it is more inconvenient when you try to conceal it."

"Do not feel envious of the happiness of those who live in a fool’s paradise, for only a fool will think that it is happiness."

## Thomas Cover (Computer Scientist, Statistician)

"With such interests one should go take as many probability and statistics classes as possible from the stat department, plus his classes in advanced information theory, pattern recognition, (etc)."

"The best research occurs when there is a simple surprise in the understanding of what might be thought of as a deep result. After all there are not too many simple things to go around (Tom’s favorite Kolmogorov theory quantifies this). So if something is new, important, and simply explained it is rare indeed."

"Good research results should be like good jokes, crisp and surprising, you couldn’t see it coming and then it became obvious once you saw it."

"On paper wrote with a dull pencil, on the blackboard with oversize chalk. (N)ever let details get in the way of the big picture."

"Always made it clear that work and life were only worth the effort if they were both joyful and rewarding."

"Vigorous thinking is so much fun, and you can always laugh when you’re doing research."

"Your goal should be to change the way people think.

## Maryam Mirzakhani (Mathematician, First Female Fields Medalist)

"It is invaluable to have a friend who shares your interests, and helps you stay motivated."

"The more I spent time on mathematics, the more excited I became."

"I started attending the informal seminar organized by Curt McMullen. Well, most of the time I couldn't understand a word of what the speaker was saying. But I could appreciate some of the comments by Curt. I was fascinated by how he could make things simple and elegant. So I started regularly asking him questions, and thinking about problems that came out of these illuminating discussions. His encouragement was invaluable. Working with Curt had a great influence on me, though now I wish I had learned more from him. By the time I graduated I had a long list of raw ideas that I wanted to explore."

"I find it fascinating that you can look at the same problem from different perspectives, and approach it using different methods."

"It’s hard to predict [what research problems and areas I am likely to explore in the future]. But I would prefer to follow the problems I start with wherever they lead me."

"I find collaboration quite exciting. I am grateful to my collaborators for all I have learned from them. But in some ways I would prefer to do both; I usually have some problems to think about on my own."

"Of course, the most rewarding part is the "Aha" moment, the excitement of discovery and enjoyment of understanding something new – the feeling of being on top of a hill and having a clear view. But most of the time, doing mathematics for me is like being on a long hike with no trail and no end in sight!"

"I find discussing mathematics with colleagues of different backgrounds one of the most productive ways of making progress."
 
"I am a slow thinker, and have to spend a lot of time before I can clean up my ideas and make progress. So I really appreciate that I didn’t have to write up my work in a rush."
 
"I am really not in a position to give advice; I usually use the career advice on Terry Tao’s web page for myself. Also, everyone has a different style, and something that works for one person might not be so great for others."
 
"I don't think that everyone should become a mathematician, but I do believe that many students don't give mathematics a real chance. I did poorly in math for a couple of years in middle school; I was just not interested in thinking about it. I can see that without being excited mathematics can look pointless and cold. The beauty of mathematics only shows itself to more patient followers."

## John Tukey (Mathematician, Statistician)

"This experience [working on problems related to ordnance] provided him with many examples of statistical problems that he would investigate in future years. It also gave him a great appreciation for the nature of practical problems." -- In David Salsburg's words.

"The best thing about being a statistician is that you get to play in everyone's backyard."
    
"It's better to have an approximate answer to the right question than an exact answer to the wrong one."

"He proposed a method that seemed to satisfy the engineer (recall Tukey's aphorism about the usefulness of an approximate answer to the right question). Tukey himself was not satified, however, and continued thinking about the problem. The result was the Fast Fourier Transform." -- In David Salsburg's words.

"Nothing has been too mundane for Tukey to attack with original insight, and nothing is too sacrosanct for him to question." -- In David Salsburg's words.

"If we need a short suggestion of what exploratory data analysis is, I would suggest that it is an attitude and a flexibility and some graph paper (or transparencies, or both)."

"In a world in which the price of calculation continues to decrease rapidly, but the price of theorem proving continues to hold steady or increase, elementary economics indicates that we ought to spend a larger and larger fraction of our time on calculation."

## George Box (Statistician)

"I said [to the colonel] one day, 'You know, we really need to have a statistician look at these data because they are very variable.'"

"Essentially, all models are wrong, but some are useful."

"Scientific research," he noted, "consisted of more than a single experiment. The scientist arrives at the experiment with a large body of prior knowledge or at least with a prior expectation of what might be the result. The study is designed to refine that knowledge, and the design depends upon what type of refinement is sought....This one experiment is part of a stream of experiments, The previous knowledge is then reconsidered in terms of both the new experiment and new analyses of the old experiments. The scientists never cease to return to older studies to refine their interpretation of them in terms of the newer studies....The procedure continues forever - there is no final *correct* solution....The sequence of scientific experiments followed by examination and reexamination of data has no end - there is no final scientific truth." -- In David Salsburg's word.

## Bradley Efron (Statistician)

"You just can't work in fields like statistics by yourself. You have to have the kind of stimulation that comes from being around smart people challenging you."

"I'm motivated by real problems. Some people, more honorably, want to do applications for the application's sake, but I've always been interested in it for statistics' sake.

"Statistics is a backward thinking. You think from the specific case, back to the general case, rather than vice versa."

"I've have many papers turned down, I usually work really hard on revisions. I try hard to rewrite and take referee seriously, but I'm never discouraged by referees not like something because sometimes it's because you may have a new idea."

"I'm a bad reader ab initio, but once I've started working on something, I want to read everything in that area."

"When you're writing, tell someone why you're doing something, not what you're doing."

"Astronomers have stars, geologists have rocks, we have science - it's the raw material statisticians work from."

"I always wished I'd made more of an attempt to communicate. Since then I've been careful about trying to write communications to my clients or collaborators in reasonable language, or at least reasonably like what they're used to think in."

## Jerzy Neyman (Statistician)

"Statistical theory needed to be motivated by its importance and usefulness for *applications*."

"We didn't have to know and match all the cases and understand all the relationships to find answers to the most important questions." -- In Morris Hansen's words.

## Fred Mosteller (Statistician)

"The only way to find out what will happen when a complex system is disturbed is to disturbed the system, not merely to observe it passively."

## David Blackwell (Statistician)

"Basically, I’m not interested in doing research and I never have been. I’m interested in understanding, which is quite a different thing. And often to understand something you have to work it out yourself because no one else has done it."

## Sara van de Geer (Mathematician, Statistician)

"It's the faith of the statisticians that you have to know lots of about everything. You can use all kinds of math."

## Nilanjan Chatterjee (Statistician)

"I am engaged in important scientific problems."

"I have the freedom and support to pursue those problems that intellectually interest me most."

"I sit with a pen and paper and become immersed in formula: statistically formulate a new scientific problem or develop asymptotic theory for a new estimator."

"For PhD students,learn as much statistical theory and computation as you can."

"For assistant professors, develop a vision so you can identify important problems on your own: take a real interest in scientific applications and develop subject-matter expertise."

"Make you a better applied statistician and lead you to identify new methodological and theoretical topics of current interest."

"Avoid posing complex solution of applied problems, more productive to come up with simple intuitive solutions that can be grounded with some theoretical foundations."

"I have always been very independent researcher."

## David Dunson (Statistician)

"Enjoy what you are doing, be creative, and take risks in choosing what to work on."

"Enjoy the process - if you love what you are doing, then productivity and success will come naturally."

"Don't choose to work on topic A because of a bandwagon effect."

"Read the literature, go to talks on a wide variety of subjects."

"Have a skeptical eye and thoughts toward limitations of what people are currently doing and how to do something better fully motivated by applications."

"Don't get bogged down in the details of papers and seminars, but figure out a better 'big-picture' way of doing things."

*Continue to update...*
