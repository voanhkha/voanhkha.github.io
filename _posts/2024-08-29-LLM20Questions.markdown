---
layout: post
comments: true
title: Finetune an LLM to play the game of '20 questions'
excerpt: "...and won rank 13/832 in a Kaggle simulation competition."
mathjax: true
date:   2024-08-29 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

I've just earned my top solo silver medal (rank 13) in a Kaggle simulation competition ["LLM 20 Questions"](https://www.kaggle.com/competitions/llm-20-questions), where we need to develop an LLM to play the game of 20 questions. <br>

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

The idea is that, public open-weight LLMs are not so skillful in 20 Questions game. Asking broad questions and narrowing down gradually is the only way to find out the secret keyword, but many LLMs seem not to be great in that area. <br>

In my approach, I prepare an offline dataset of hundreds of possible questions and answers to every question-keyword pair. Then we will try to follow the policy of asking based on this offline dataset. <br>

Why it works? That's because the organizer has confirmed that the current public keyword list is representative to the unseen private list which we don't have. So the principles of this approach is: <br>

1) Collect hundreds of diverse binary questions that are both broad or narrow that can help us determine the keyword. <br>
2) Use an LLM to find out the answer (yes or no or unsure) to each question-keyword pair. The keyword is drawn from the public keyword list. <br>
3) Use the pre-built knowledge to ask the next question given the history of asked questions and corresponding answers. <br>
4) After each round, with all the question-answer pair, remove the unqualified keywords. <br>
5) Select the next question that has the most entropy split as possible (i.e., 50% yes 50% no) based on the remaining qualified keywords. <br>
6) Unofficially, guess the keyword from the public list, that means use the keywords in the public list just as the proxy to guide the asking policy. That means in each round, we memorize these "unofficial guesses" to prepare for the next best question only. <br>
Officially, use the asked questions and answers to make the LLM predict the real private keyword by itself. <br>

<div class="imgcap">
<img src="/images/llm_20q_questions_df.png" width="400">
<div class="thecap"> I prepared a dataframe of questions bank using OpenAI's GPT4</div>
</div>
<br>

The advantages of offline-policy approach are: <br>

1) it can partly overcome the issue of LLM not being able to play the game properly, <br>
2) it can resolve the issue of word-limit in asking questions, and  <br>
3) it can replace the LLM's questioning responsibility, which is the hardest part in [questioning, answering, guessing]. <br><br>

Another good point is that anybody can submit it right away, as it can work in private leaderboard without no further changes. You can see an example episode below where this bot excellently correctly guessed the keyword in 10 moves.<br>

<div class="imgcap">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1829450%2Fd9c51f05c4df55dee4a0e58624bf5fa9%2Fkhavo_episode_example.gif?generation=1722868290679729&alt=media" width="400">
<div class="thecap"> A game of 20 questions conducted by 4 bots in a 2-vs-2 match-up </div>
</div>

<br>
The successful rate of this bot is about 10%, which means you will win 1 game in about 10 games played! This is considered a high percentage given the difficulty of the game.<br>

In my own version of this bot, I have many more features that I want for my own. My version can reach >900 points on the LB recently (but it got deactivated when I submitted some new bots). <br>

I aimed for my first solo gold medal and missed only 2 ranks (13 instead of 11) finally. It was a little bit disappointed. However I also realized many of the top bots (in gold regions) integrated my solution into theirs! And that made me really happy! <br>

There are a few key takeaways from this competition that I found really valuable: <br>
- Using vllm efficiently <br>
- Stabilize LLM outputs using xml tags <br>
- Finetuning LLAMA3 experience <br>
- Realize the big gap improvement from LLAMA3 to LLAMA3.1 <br>
- A critical trick on the game ([alpha trick](https://www.kaggle.com/code/cdeotte/make-animation-of-winning-teams)) that disrupted the LB, and got understood by about 20 top teams, including mine.
- A brilliant [1st place solution](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106) from a new Kaggle superstar [c-number](https://www.kaggle.com/cnumber). I like this solution so much! <br>

<div class="imgcap">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6624777%2Fb3e468f4bc8f9da52f5d7ff1c5d7a52a%2Fscatter.png?generation=1725032484048482&alt=media" width="400">
<div class="thecap"> The scatter plot demonstrates the impact of the probability calculation. Public keywords tend to have a lower thing probability rank (meaning high probability) and a low frequency rank (meaning high frequency). The ranks and values are inverted because I ranked the keywords from high value to low value. </div>
</div>
<br>

<div class="imgcap">
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F6624777%2F798d60bf764599a824e1af8e8f39a2f0%2Fheatmap.png?generation=1725033082865389&alt=media" width="400">
<div class="thecap"> Using this heatmap, we can calculate the probability of each keyword being included in the private keyword list, based on the observation that the public keyword list closely resembles the private one. </div>
</div>

<br>

That's it for a painstaking but wonderful Kaggle simulation competition. Thanks for reading and see you in another post!
