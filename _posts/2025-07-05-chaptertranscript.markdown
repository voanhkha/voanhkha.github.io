---
layout: post
comments: true
title: Chaptering Clipchamp Video Transcripts with LLMs and considerations for auto-continual improvement
excerpt: "(with optimization for both server-side and on-device inferencing)"
mathjax: true
date:   2025-07-05 00:00:00
author: Kha Vo
categories: Kaggle
tags:	AI
cover:  "/assets/instacode.png"
---

In this post, I would like to propose a system design for finetuning LLMs to perform an interesting task of chaptering (segmenting) transcripts. In short, this design will perform the following: <br>

1) Given a user video's transcript in Clipchamp app, the inferencing process will segment it into a few chapters, each with a short title. <br>
2) The LLM model is open-source, quantized, and continually finetuned by our own finetuning procedure, that takes as input all of the historical external plus internal transcripts, plus user feedbacks on past predictions. <br>
3) Applying SOTA advanced finetuning techniques (unsloth, bitsandbytes, lora), with careful data curation and augmentation. <br>
4) Optimized for both on-device and server-side inferencing performances. <br>
5) Code samples accompanied with each essential process.


<div class="imgcap">
<img src="/images/chapter_transcript_white.png" style="width: 1000px !important;">
<div class="thecap"> Proposed system design </div>
</div>
<br>

```python
print("Works perfectly in Jekyll")
```

llm_service.py <br>
```python
# llm_service.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from flask import Flask, request, jsonify

# === Load model ===
model_name = "Qwen/Qwen1.5-0.5B-Chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Thank you for reading.


