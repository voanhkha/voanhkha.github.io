---
layout: post
comments: true
title: My LLM plays a coop-competitive game
excerpt: "How to embed an offline policy knowledge to an LLM AI chatbot that plays the game of 20 Questions, and won rank 13/832 in a Kaggle simulation competition."
mathjax: true
date:   2024-08-29 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

I've just earned my top solo silver medal (rank 13) in a Kaggle simulation competition, where we need to develop an LLM to play the game of 20 questions. <br>


<div class="imgcap">
<img src="/images/llm20q_kaggleLB.png" width="500">
<div class="thecap"> Final leaderboard </div>
</div>
<br>

The great thing about this challenge is that in each match, there are 4 players involved, instead of 2. It is a 2 versus 2 match-up. A team will comprise of 2 players, one acts as the questioner and the other as the answerer. <br>

At the start of the game, a common secret keyword is given to the answerers of the 2 teams. Then the questioner of each team will start to ask its teammate answerer a binary (yes or no) question, receive an answer, and produce a keyword guess. <br>

If the guess of the questioner of any team is correct, 2 players of that team will win the match and rise on the global leaderboard (if both correct in the same turn, so a tie). If both teams are wrong, comes the next turn. This will repeat for 20 questions (20 turns). <br>

A key thing about this competition is that a fixed list of keywords will be embeded to the referee (Kaggle server). This list will change and inaccessible by any means in the final evaluation period (final 2 weeks). <br>

During the competition, I have publicly shared to the Kaggle forum my approach which worked really well in both public and private LB.
I named it as "offline-policy-questioner agent". It got >50 upvotes and >250 forks!  <br>

You can find the code in my [[Kaggle notebook]](https://www.kaggle.com/code/khahuras/offline-policy-questioner-agent). Enjoy! <br>

The idea is that public open-weight LLMs are not so skillful in 20 Questions game. Asking broad questions and narrowing down gradually is the only way to find out the secret keyword, but many LLMs seem not to be great in that area. <br>

In my approach, we will prepare an offline dataset of hundreds of possible questions and answers to every question-keyword pair. Then we will try to follow the policy of asking based on this offline dataset. <br>

The advantages of offline-policy approach are: <br>

1) it can partly overcome the issue of LLM not being able to play the game properly, <br>
2) it can resolve the issue of word-limit in asking questions, and 3) it can replace the LLM's questioning responsibility, which is the hardest part in [questioning, answering, guessing]. <br>

Another good point is that you can submit it right away, as it can work in private leaderboard without no further changes.<br>

You can see an example episode below where this bot excellently correctly guessed the keyword in 10 moves.<br>

<div class="imgcap">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1829450%2Fd9c51f05c4df55dee4a0e58624bf5fa9%2Fkhavo_episode_example.gif?generation=1722868290679729&alt=media" width="400">
<div class="thecap"> A game of 20 questions conducted by 4 bots in a 2-vs-2 match-up </div>
</div>

<br>
The successful rate of this bot is about 10%, which means you will win 1 game in about 10 games played! This is considered a high percentage given the difficulty of the game.<br>

In my own version of this bot, I have many more features that I want for my own. My version can reach >900 points on the LB recently (but it got deactivated when I submitted some new bots). <br>

I aimed for my first solo gold medal and missed only 2 ranks (13 instead of 11) finally. It was a little bit disappointed. However I also realized many of the top bots (in gold regions) integrated my solution into theirs! And that made me really happy! <br>

Thanks for reading and see you in another post!
