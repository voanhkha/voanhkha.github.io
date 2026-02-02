---
layout: post
comments: true
title: "Hidden Markov model (alone!) can almost win me a Kaggle competition" 
excerpt: "...instead I finished 3rd due to an unfortunate data leak that 2 other teams exploited in the Liverpool Ion Switching Competition"
mathjax: true
date:   2020-05-27 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
preview_media: /assets/previews/ion.webm
preview_type: video   # gif | video
---

I have just lost my first win (rank 1) of a Kaggle competition! It's due to an expected data leak at the end and two other teams exploited it! So finally we ranked 3rd. <br>

It is a 3-month contest about Deep Learning/Hidden Markove modelling to predict the number of open channels during the duration of time-series human electrophysiological signal data. 
It requires some finesse in time-series analysis, spotting the pivoting distinctive features from signal by systematic approaches (mainly unsupervised modelling like kNN and tSNE),
then use an ensemble of methods (RNNs, decision trees, random forest) to avoid overfitting. <br>

![leaderboard](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*V10hsLxhpun-sNbcF0dqgQ.png) <br>

It was a really fun and helpful experience in the whole journey of this competition with my teammates. Each of us wrote their part about various aspects of the competition: <br> <br>

- I wrote about the forward-backward algorithm that we used [here](https://www.kaggle.com/competitions/liverpool-ion-switching/discussion/153734)  <br><br>

- Gilles published our complete solution on Medium [here](https://medium.com/towards-data-science/identifying-the-number-of-open-ion-channels-with-hidden-markov-models-334fab86fc85)<br><br>

- Zidmie wrote about the data leak [here](https://www.kaggle.com/c/liverpool-ion-switching/discussion/153824)<br><br>

- My code about the best non-leak solution [here](https://www.kaggle.com/code/khahuras/1st-place-non-leak-solution?scriptVersionId=34894967)<br><br>

- Our full code [here](https://github.com/GillesVandewiele/Liverpool-Ion-Switching)<br><br>

Have fun!



