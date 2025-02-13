---
layout: post
comments: true
title: LLM Offline Policy
excerpt: "How to embed an offline policy knowledge to an LLM AI chatbot that plays the game of 20 Questions, and won rank 13/832 in a Kaggle simulation competition."
mathjax: true
date:   2024-08-29 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---


[[Code]](https://www.kaggle.com/code/khahuras/offline-policy-questioner-agent)

Hi all!

I would like to share (part of) my approach to this competition, which will work in both public and private LB.
I named it as "offline-policy-questioner agent".

You can find the notebook here.

The idea is that public open-weight LLMs are not so skillful in 20 Questions game. Asking broad questions and narrowing down gradually is the only way to find out the secret keyword, but many LLMs seem not to be great in that area.

In my approach, we will prepare an offline dataset of hundreds of possible questions and answers to every question-keyword pair. Then we will try to follow the policy of asking based on this offline dataset.

The advantages of offline-policy approach are: 1) it can partly overcome the issue of LLM not being able to play the game properly, 2) it can resolve the issue of word-limit in asking questions, and 3) it can replace the LLM's questioning responsibility, which is the hardest part in [questioning, answering, guessing].

Another good point is that you can submit it right away, as it can work in private leaderboard without no further changes.

You can see an example episode below where this bot excellently correctly guessed the keyword in 10 moves.

The successful rate of this bot is about 10%, which means you will win 1 game in about 10 games played! This is considered a high percentage given the difficulty of the game.

In my own version of this bot, I have many more features that I want for my own. My version can reach >900 points on the LB recently (but it got deactivated when I submitted some new bots).

<div class="imgcap">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1829450%2Fd9c51f05c4df55dee4a0e58624bf5fa9%2Fkhavo_episode_example.gif?generation=1722868290679729&alt=media" width="400">
<div class="thecap"> A game of 20 questions conducted by 4 bots in a 2-vs-2 match-up </div>
</div>

