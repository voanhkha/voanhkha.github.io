---
layout: page
title: "Posts"
permalink: /posts/
main_nav: true
---

  <ul class="posts-list">
  {% for post in site.posts %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
  

<br>
