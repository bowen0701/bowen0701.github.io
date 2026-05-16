---
layout: default
title: Blog
permalink: /blog/
---

# Blog

<ul class="posts">
{% for post in site.posts %}
  <li>
    <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
    &nbsp; <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
