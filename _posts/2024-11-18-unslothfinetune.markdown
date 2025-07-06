---
layout: post
comments: true
title: Finetune LLMs to solve the famous abstract reasoning challenge with Unsloth AI...
excerpt: "... and finished 6/1400 in ARC Prize 2024, then met intimately with Francois Chollet!"
mathjax: true
date:   2024-11-18 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

[ARC Prize](https://arcprize.org/) (Abstract Reasoning Challenge) on Kaggle has always been one of the most (if not the most) difficult AI challenge nowadays that even OpenAI o3 pro struggles! <br>

For the ARC Prize 2024 edition, which lasted for 6 MONTHS (yes you read it right), I original finished at SIXTH place out of roughly 1400 teams! <br>

Because of this painstaking achievement, I was contacted by and then privately met with Francois Chollet, one of TIME's top 100 most influential persons in AI in 2024!, to share about my solution and approach! We also talked about a possibility that whether I can work for his start-up (but for some undisclosed reason I didn't end up working for him). For some reason, I was finally removed from the Kaggle leaderboard (read the last part of this post to understand why). <br>

<div class="imgcap">
<img src="/images/chollet_kha.png" width="500">
<div class="thecap"> My honour to have a 1-on-1 discussion with Francois Chollet, one of the top celebrities in AI </div>
</div>
<br>

No one has ever thought of using LLMs in this competition, especially given the context that the previous edition (2019) of this Kaggle challenge has seen ALL of the top 10 solutions being program synthesis or brute-force style approaches. <br>

My idea of using LLMs to predict the output grid DIRECTLY from the input grid is, IMHO, really BOLD. But... I am not the only person who thought of this! The top 3 solutions also used this approach. <br>

So,... that's why we all occupied top places in the LB. <br>

Before going into my solution, let's take a brief look at the problem statement for this challenge. <br>

For this competition, we are given a train dataset of 800 'tasks'. For each task, we will have a few (from 2 to 5) pairs of 'train' input-output images, and a few (from 1 to 3) pairs of 'test' input images. It's hard to explain so let me plot a few tasks. <br>

<div class="imgcap">
<img src="/images/arc25_train_1.png" width="500">
<div class="thecap"> A training task (1) </div>
</div>

<div class="imgcap">
<img src="/images/arc25_train_2.png" width="500">
<div class="thecap"> A training task (2) </div>
</div>

<div class="imgcap">
<img src="/images/arc25_train_3.png" width="500">
<div class="thecap"> A training task (3) </div>
</div>

<div class="imgcap">
<img src="/images/arc25_train_4.png" width="500">
<div class="thecap"> A training task (4) </div>
</div>

<div class="imgcap">
<img src="/images/arc25_train_5.png" width="500">
<div class="thecap"> A training task (5) </div>
</div>
<br>

As you can see, for each task, there is an 'abstract reasoning' rule that transform an input into an output. For normal humans, we can quickly grasp how the inputs are transformed into the outputs (for each task). However for the machine, it is of near impossibility to conceive these simple but non-textual abstraction. <br>

So, given these 800 training tasks, our main goal is to develop an AI model/system, that can perform on these types of tasks. The most difficult part about this contest is that, the model will need to perform on UNSEEN HIDDEN tasks that it has never seen before! And the types / natures of these hidden tasks are of complete difference than the training tasks! Well, how can you tell the AI to understand a new abstract movements such as "fill in the hole" or "combine some objects to make a square", if they have never seen them before. So that's why we have this competition. Let's have a look at some 'hidden' test tasks (that were actually kept as secrets for the whole ARC 2024 competition and were only recently released in 2025 as the additional training data for ARC 2025 competition which is ongoing). <br>


<div class="imgcap">
<img src="/images/arc-agi-2-unsolved-1.png" width="500">
<div class="thecap"> A hidden task (that we couldn't see) related to symbolic interpretation during the competition (out of a total of 100 hidden test tasks) </div>
</div>

<div class="imgcap">
<img src="/images/arc-agi-2-unsolved-2.png" width="500">
<div class="thecap"> Another hidden task (that we couldn't see) related to compositional reasoning during the competition (out of a total of 100 hidden test tasks) </div>
</div>

<div class="imgcap">
<img src="/images/arc-agi-2-unsolved-3.png" width="500">
<div class="thecap"> Another hidden task (that we couldn't see) related to contextual rule application during the competition (out of a total of 100 hidden test tasks) </div>
</div>

<br>

For viewing the full training tasks for ARC Prize 2025, I have compiled a nice-looking public [Kaggle notebook here](https://www.kaggle.com/code/khahuras/visualizing-all-1120-tasks). <br>

Even SOTA reasoning models like OpenAI o3-pro, Claude, or anything (up until the point of this writing), the performance is under 5%, which is crazy, but understandable. We can't expect LLMs models which were trained on text data to perform on vision data. However, the previous statement is not completely true. For SOTA vision models, they still can't perform on these tasks! The ARC Prize Foundation continually updates their benchmark tests on new SOTA models anytime they're released. And this is the result: <br>

<div class="imgcap">
<img src="/images/arc_chart_202507.png" width="500">
<div class="thecap"> Performance of all of the world's SOTA models on ARC Prize 2024 </div>
</div>
<br>

So, with this kind of really challenging contest, how did I approach the problem? <br>

Let's get to the main part: my solution. There are a few main parts: <br>

## 1) Formulate the problem into a form that allows LLMs to solve <br>

## 2) Finetune LLMs with unsloth.ai <br>
<br>
I happened to find [unsloth.ai](unsloth.ai) as the SOTA tool for finetuning LLMs! Previously I used some no-code GUI approaches like H2O LLM Studio, or I finetuned models directly by LORA with original transformers and pytorch support. But after trying unsloth, I have to admit that it is indeed the BEST library to finetune LLM models, because: <br>

+ It makes BOTH training and inference much faster. Unsloth's ability to significantly speed up (3x) the inference process made a huge difference when we want to run LLM inferencing inside a Kaggle notebook with limited resource (in this case: only 2xP100 GPUs are given to each participant within a 12-hour runtime). <br>

+ It is really easy to use. The most difficult part is to installing and adjust the configurations. But when it comes to the training / inferencing, it is not so much different than a normal deep learning pytorch library. Everything went smoothly. <br>

+ It supports a wide range of models, including LLAMA, Deepseek, Qwen, Mixtral, Phi, Gemma... This allows me to play and find the best suitable model (which is Qwen2.5-0.5B-Instruct). <br>



## 3) Test-time finetuning during inference 
<br>
When the trained model faces a hidden test task in the inference mode, it will perform a quick re-training JUST FOR THAT TASK. This helps specifically only in this unique competition, because we have various ways to augment the on-hand task (flip, rotate, color swap...). <br>

## 4)  Ensembling with past-solutions 
<br>
Some heuristic-based solutions from the 2019 edition were still very strong and helped the ensemble of predictions. <br>

## 5) Candidate selection optimization
<br>
I used a unique self-made selection elimination technique that helps to reduce the irrelevant predictions for a single task, by observing various heuristic-based factors of that task: background detection, object count, task type classification... <br>



However, if you look at the competition LB by now, you don't see me there because of the Kaggle buggy procedure when they hand out money prizes. It was a really long and frustrating story. If you want to understand more about this unfortunate incident, that also happened to 3 other teams, please read this Kaggle [discussion topic](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442) and all of its comments there. You'll also see me ranting about it there :) <br>

Anyway, that doesn't reduce the validity of my solution, but as how it turned out, I would like to hold off the sharing of my full code solution for now, to gain an advantage on the ongoing 2025 edition (which is now running from Mar to Nov 2025). <br>

I will post my full solution once ARC Prize 2025 is concluded. <br>

Thank you for reading.


