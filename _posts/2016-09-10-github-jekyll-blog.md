---
layout: post
comments: true
title: "Moving my Blog from Wordpress to Jekyll"
date: 2016-09-10
---

## Thank you and Byebye Wordpress!

Inspired by Andrej Karpathy's blog post ["Switching Blog from Wordpress to Jekyll"](http://karpathy.github.io/2014/07/01/switching-to-jekyll/), I decide to move my blog from Wordpress to Jekyll as well. After some practices, similarly as Karpathy, I can confidently say that I could not be happier about this decision.

## So What's Wrong with Wordpress?

You may ask the above question. "Let's see, everything," quotes from Karpathy, and he clarified his conclusion in the following bullets:

- Wordpress blogs are clunky, slow and bloated.
- Wordpress is dynamically rendered with .php. There are really only few niche applications where this is necessary. Dynamic code execution exposes your blog to hackers and exploits: zero-day attacks, viruses, etc. My own blog was hacked ~2 months ago and all my posts had been infected with spammy content that kept re-inserting itself magically when I removed it.
- Wordpress is popular among the masses of people who don't know any better, and therefore attracts the largest amount of spammers.
- Your posts are stuck forever in an ugly, Wordpress-specific SQL database (ew). You can't easily import/export posts. You do not really own your content in raw and nimble form.
- Wordpress is blocked in China.

## Jekyll on Github <3

[Jekyll](http://jekyllrb.com/) introduces itself as a tool for building "Simple, blog-aware, static sites", and was originally written by one of the Github co-founders, Tom Preston-Werner. In Karpathy's words: "It is flat and transparent: Your blog workspace is a single folder with a config file, and a few folders for CSS and HTML templates." With Jekyll you can write your blog posts in simple [markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet), which is easy to use with script and mathjax and image syntax. Further, as you might expect, Karpathy added, "Jekyll is tightly integrated with [Github](https://github.com): create a repository that looks like `username.github.io` and add your files to the repo. Github will automatically compile your files with Jekyll." For example, mine lives on [bowen0701.github.io](bowen0701.github.io). Thus, Github makes sure that your blog is backed up forever in markdown, and also hosts your content!

However, for me it is not trivial to build Jekyll blog on `github.io`. As the first step I followed Joshua Lande's blog post (which is of course built by Jekyll hosted on github.io :-)), ["How I Created a Beautiful and Minimal Blog Using Jekyll, Github Pages, and Poole"](http://joshualande.com/jekyll-github-pages-poole), to give Jekyll a shot. Following the post you can create the HTML templates which get composed to your final site. It's not-so-trivial for me to tweak the [CSS](http://www.w3schools.com/css/css_list.asp) or any of the HTML templates, but after lots of trials and errors I finally succeeded. :-) For example, I followed 
Lewis Gavin's post, ["Adding Google Analytics and Google AdSense to a Jekyll Website"](http://www.lewisgavin.co.uk/Google-Analytics-Adsense/), to add [Google Analytics](https://analytics.google.com/) tracking code to all my pages by tweaking the html template, and also Webjeda's post, ["3 Ways to Get Feedback on Your Jekyll Blog through Comments"](https://blog.webjeda.com/jekyll-comments/), to add [Disqus](https://disqus.com/) comments to all my posts by tweaking the posts template with the Disqus Javascript code. 

Now I prefer to write my blog post on [Jupyter Notebook](http://jupyter.org/) with script, mathjax and image syntax, then convert it to jekylly markdown `ipython nbconvert --to markdown my_notebook.ipynb`, and finally upload the resulting converted files to `github.io`. That's a lot of fun. You shall try Jekyll on `github.io` as well! :-)

In summary, Jekyll plus Github strike the balance: Their combination is packed with just the right amount of features.
