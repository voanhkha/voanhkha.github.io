---
layout: post
comments: true
title: How to finetune an LLM to classify responses from other LLMs
excerpt: "...and won a SECOND PLACE in the 'H2O.ai Predict the LLM', met with H2O.ai CEO, and recruited by them!"
mathjax: true
date:   2023-11-08 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

Finetuning an LLM an optimizing its predictions is really an interesting task! <br>

It's so interesting that my team (me and @phanisrikanth) have been working together tirelessly to win a 2nd place in the Kaggle competition [H2O.ai Predicting the LLM](https://www.kaggle.com/competitions/h2oai-predict-the-llm/leaderboard)

<br>

<div class="imgcap">
<img src="/images/h2o_predictllm_lb.png" width="500">
<div class="thecap"> Final leaderboard </div>
</div>
<br>

For this result, I was invited to Singapore for a conference about Large Language Model Framework h2oGPT (held by H2O.ai team), and especially had an interesting private discussion with the founder and CEO of H2O.ai, Mr Sri Ambati! <br>

A few weeks later he also came to Sydney (where I live) for a business trip and again personally meet me 1-on-1 to talk about genAI for very long!. You can see some of our photos at the end of this post.<br>

Let's get back to the main sauce: our solution. It's a seamless combination of various methods stacking on each other: 

1) Finetune an LLM decoder with a classification head with the competition’s data. <br>
2) Assignment optimization: optimize the prediction (binarizing it using linear assignment optimization method) <br>
3) Customized post-processing. <br>

We will go through each of the above 3 sections as belows. <br>

**1. THE LLM MODELS** <br>

On Kaggle, it’s established that encoders, such as DeBERTa, have ruled NLP competitions on Kaggle for years. We realized that simple DeBERTa models scored in the range of 1.6, 1.4 on mlogloss metric.

**1.1 Model Selection** <br>
Given that it’s 2023 and new decoders coming out almost each day, we decided to use a decoder model out of sheer curiosity. Now, which models do we begin with? We went with the most popular Mistral model for 2 reasons: 1) Performance. It outperforms Llama2-13b on some benchmarks, and 2) Apache 2.0 License.
<br>

**1.2 Experiments** <br>

Each of the experiments were done on H2O-LLMStudio. It’s a fantastic no-code tool to quickly run experiments with many LLMs. Highly recommend having this tool in your data science toolbox. 🙂 <br>

Coming to experiments, with Mistral-Instruct-v0.1, our local validation loss dramatically reduced and the leaderboard score fell from 1.6 to 1.063. <br>

This seemed like the right direction as psi’s benchmark was 0.88 and we're getting there. A little bit of tuning with epochs and we got 0.925 on LB. We tried modifying int4/int8/bfloat16 finetuning, LoRA parameters and always saw an adverse effect. One of the key parameters that improved our score was prompt length, answer length. The more they are, the better the logloss. This was easy to explain as some of the sentences (specifically answers) were longer than 2000 in length. <br>

One of the interesting parameters is the seed. Fixing the seed led to deterministic results but a random seed resulted in a different logloss and LB score. The variance is 0.05 which is extremely high in this competition. This led to the conclusion that averaging multiple models was the way to go to fight variance. <br>

Along the way, we clustered the data to visually explore it using cluestar library and found that GroupKFold was the way to go. This was a moment of personal learning that "explore the data first before modeling". <br>

**1.3 Open Models** <br>

With huggingface hosting so many models, the open_llm_leaderboard shows top models based on benchmark scores. Motivated to further improve the score, we tried multiple models finetuned on top of Mistral and while some worked, some did not. Ablation studies are in the logbook. <br>

**1.4 Averaging** <br>
Lastly, these models were not perfectly correlated and that led to nice performance on averaging. So, a straight average of predictions (not oof predictions) from the 9 models below, with different seeds, led to 0.63 on the leaderboard. Here is the architecture. <br>

<div class="imgcap">
<img src="/images/predictllm_1.png" width="500">
<div class="thecap"> The LLM architecture of our solution </div>
</div>
<br>

Five days prior to the competition end, we decided to team up and the optimization work (details in the next section) was a big part of this solution going from 0.63 to 0.50 on the public leaderboard. <br>

**1.5 Infrastructure** <br>
All the decoder models were trained on a machine with 8 x V100 GPUs. This meant that each finetuning run approximately took around 60 minutes along with inference on test data. <br> <br>


**2. OPTIMIZATION AND POST-PROCESS** <br>

This competition’s data has a very special trait, that for each question group, we should expect exactly 1 response for each of 7 label targets. During the normal modeling process, we may not exploit any of this extra information about relation between data samples. <br>

This sparks an idea of using an optimization method to find the best assignment of targets for a group of 7 responses to a specific question. The optimization method takes as input a 7x7 matrix, where each row is the normalized probability (sum to 1) of 1 data sample. We then try to maximize the selected probabilities given the constraint that each target must be assigned exactly one time. <br>

This optimization method alone, however, cannot outperform the original predictions due to its extreme nature (of hard 0/1 prediction) that can't drive the log-loss down. So I need to perform some more extra steps of combining this binary prediction (also a 7x7 matrix) with the original matrix. <br>

The combining steps and optimization coefficients were achieved manually by a systematic search to minimize the out-of-fold predictions. The single steps of post-processing are shown my published notebook code at Kaggle (link at the end of this post). In general, they are just the results of the following ideas: <br>

+ The highest prediction within 1 data sample (a row) should come to 1.
+ All the other 6 predictions within that data sample should come to 0.<br>

<div class="imgcap">
<img src="/images/predictllm_2.png" width="500">
<div class="thecap"> Customized assignment optimization post-processing </div>
</div>
<br>
  
The above 2 ideas can only be implemented after the assignment optimization step. <br>

The optimization function is highly customized and straightforward, that can be applied to any submission! Fellow Kagglers can directly try to use our function optim(df1, pred) at my published code to see how your score will be improved significantly in an instant! <br> <br>


**3. THE OTHER IDEAS THAT ALSO WORKED**  <br>

Sidelining the main 2 major factors presented, we also experimented with some other classical approaches and found out that they worked nicely, but we couldn’t find the time to combine that with LLM's out-of-fold predictions on time. <br>

My methods highly depend on the train out-of-fold predictions. Even the binary classification idea and post-process was based on my out-of-fold deberta-v3-large predictions, instead of the best LLM model. Had the optimization been tuned on the latter, we would have a better final score I'm pretty sure. <br>

**3.1 Using a tree-based method (LightGBM) to learn a stacking model on top of the NLP predictions.** <br>

The crafted features that helped are: <br>

+ Gpt_rating: asking another LLM to rate the response within a fixed scale.
  
+ Statistics (mean, median, min, max, std) of some main features (text length, gpt_rating) aggregated by each question’s group
  
+ Category of each question: asking another LLM to categorize each question into a few categories such as: biology, history, math, idea, recommendation... <br>
  
This modeling process boosted my raw deberta-v3-large score of 1.1 to about 0.950. <br>

**3.2 Pseudo-labeling with the test prediction retrieved after the optimization method** <br>

Using the test prediction as the additional training data isn’t a new idea. But in this competition it won’t give any benefits given the log-loss metric (binarize the prediction hurts the score, hence additional data is indeed detrimental).<br>

However if the binarized version is retrieved (simply by using an argmax function) after being optimized and post-processed (as described in section 2), pseudo-label will work. However (again), the training process with pseudo-labels is not straightforward, because if we train on full test data in any session, the knowledge cannot improve and the test prediction itself finally is biased to its original version, giving no benefit.<br>

So we have to train a k-fold on the question (suppose k=5), do the split on the train data, and also do the split on the pseudo data. In a single fold training, 4/5 parts of train data plus the 4/5 parts of pseudo data will serve as training data, the 1/5 remaining part of train data will serve as validation data, and the ⅕ remaining part of pseudo data will serve as test data (that needs prediction). This strategy can ensure that the test prediction on the ⅕ test part will not have bias towards its existing target knowledge.<br>
<br>

<div class="imgcap">
<img src="/images/predictllm_3.png" width="400">
<div class="thecap"> Pseudo-labeling the hidden competition test data (label not known!) based solely on train data </div>
</div>
<br>

With pseudo training and LightGBM modeling, we observed a nice-size score boost of about 0.1 to 0.15 on the raw deberta-v3-large submission. But as Phani didn’t generate out-of-fold LLM predictions so we couldn’t implement these two methods although we have the code.<br><br>

**3.3 Conclusive Remarks** <br>

We may have a significantly better score that could break 0.499 if we combine all of the ideas together, like in these steps:<br>

+ Step 1: Train an decoder LLM, predict the out-of-fold along with test predictions.
  
+ Step 2: Train a LightGBM model on top of step 1, with the hand-crafted features.
  
+ Step 3: Optimize and post-process the predictions from step 2.
  
+ Step 4: Use the test predictions from step 3 as the additional training data, repeat from step 1. <br>
  
Thank you for reading. Our full code is publicly available [here](https://www.kaggle.com/khahuras/2nd-place-solution). <br>

<div class="imgcap">
<img src="/images/meetsri_1.jpg" width="300">
<div class="thecap"> Invited and met with Sri Ambati, the CEO of H2O.ai in Singapore </div>
</div>
<br>

<div class="imgcap">
<img src="/images/meetsri_2.jpg" width="300">
<div class="thecap"> The H2O.ai conference on Large Language Models </div>
</div>
<br>

<div class="imgcap">
<img src="/images/meetsri_3.JPG" width="300">
<div class="thecap"> Private (long) discussion with Sri when he came to Sydney where I live </div>
</div>
<br>
