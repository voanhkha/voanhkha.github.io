---
layout: post
comments: true
title: Unsloth AI - the best python library to finetune LLMs!
excerpt: "... that helped me finish 6/1400 in ARC Prize 2024 and meet intimately with Francois Chollet!"
mathjax: true
date:   2024-11-18 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

[ARC Prize](https://www.kaggle.com/competitions/llm-20-questions) (Abstract Reasoning Challenge) on Kaggle has always been one of the most (if not the most) difficult AI challenge nowadays that even OpenAI o3 pro struggles! <br>

For the 2024 edition, luckily, I've found a holy gem that helped me finish at SIXTH place out of roughly 1400 teams! That gem is [UNSLOTH AI](unsloth.ai), a free and (recent) shining LLM finetuning library with SOTA performance on small and medium models! <br>

No one has ever thought of using LLMs in this competition, especially given the context that the previous edition (2019) of this Kaggle challenge has seen ALL of the top 10 solutions being program synthesis or brute-force style approaches. <br>

My idea of using LLMs to predict the output grid DIRECTLY from the input grid is, IMHO, really BOLD. But... I am not the only person who thought of this! The top 3 solutions also used this approach. <br>

So,... that's why we all occupied top places in the LB. <br>

After this feat, I was contacted by and then privately met with Francois Chollet, a TIME AI's top 100 most influential persons in 2024!to share about my solution and approach! We also talked about a possibility that whether I can work for his new founded start-up (but for some undisclosed reason I didn't end up working for him).

<div class="imgcap">
<img src="/images/chollet_kha.png" width="500">
<div class="thecap"> Final leaderboard </div>
</div>
<br>

Now let's get to the main part: how unsloth AI had helped me? There are 3 main reasons: <br>

1) It makes BOTH training and inference much faster. Unsloth's ability to significantly speed up (3x) the inference process made a huge difference when we want to run LLM inferencing inside a Kaggle notebook with limited resource (in this case: only 2xP100 GPUs are given to each participant within a 12-hour runtime). <br>
2) It is really easy to use. The most difficult part is to installing and adjust the configurations. But when it comes to the training / inferencing, it is not so much different than a normal deep learning pytorch library. Everything went smoothly. <br>
3) It supports a wide range of models, including LLAMA, Deepseek, Qwen, Mixtral, Phi, Gemma... This allows me to play and find the best suitable model (which is Qwen2.5-0.5B-Instruct). <br>

However, unsloth alone cannot get me to the top 6 position, as usual, for an extremely competitive platform like Kaggle. I need other techniques, such as: <br>

+ Test-time finetuning: when the trained model faces a hidden test task in the inference mode, it will perform a quick re-training JUST FOR THAT TASK. This helps specifically only in this unique competition, because we have various ways to augment the on-hand task (flip, rotate, color swap...).
+ Ensembling with past-solutions: some heuristic-based solutions from the 2019 edition were still very strong and helped the ensemble of predictions.
+ Candidate selection optimization: I used a unique self-made selection elimination technique that helps to reduce the irrelevant predictions for a single task, by observing various heuristic-based factors of that task: background detection, object count, task type classification... <br>

However, if you look at the competition LB by now, you don't see me there because of the Kaggle buggy procedure when then hand out money prizes. It was a really long and frustrating story. If you want to understand more about this unfortunate incident, that also happened to 3 other teams, please read this Kaggle [discussion topic](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442) and all of its comments there. You'll also see me ranting about it :) <br>

Anyway, that doesn't reduce the validity of my solution, but as how it turned out, I would like to hold off the sharing of my full code solution for now, to gain an advantage on the ongoing 2025 edition (which is now running from Mar to Nov 2025). <br>

I will post my full solution once ARC Prize 2025 is concluded. <br>

Thank you for reading.


