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

5) Code samples accompanied with each essential process. <br>


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

<br>
inference.py <br>
```python

# If on-server with default packages, will need to re-install these
!pip install fastapi 
!pip install bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from fastapi import FastAPI
from pydantic import BaseModel

model_name = "Qwen/Qwen1.5-0.5B-Chat"
# model_name = "Qwen2.5-0.5B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                                )

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             # quantization_config=bnb_config, 
                                             device_map="auto")


#=== Preprocessing ===

def split_transcript(transcript, max_words=500):
    words = transcript.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

#=== Generate chapters using Qwen model ===

def generate_chapters(transcript_chunk):
    prompt = f"""
    You are an expert video editor AI. Given the transcript below, segment it into 3-7 chapters.
    For each chapter, provide a short title summarizing the content in less than 10 words.
    
    Transcript:
    {transcript_chunk}
    
    Output format (JSON):
    [
    {{"chapter_title": "...", "start_text_snippet": "..."}},
    ...
    ]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500)
    content = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        chapters = json.loads(content[json_start:json_end])
    except json.JSONDecodeError:
        chapters = []
    return chapters


# === Process transcript ===

def process_transcript(transcript):
    chunks = split_transcript(transcript)
    all_chapters = []
    for chunk in chunks:
        chapters = generate_chapters(chunk)
        all_chapters.extend(chapters)
    return all_chapters

# === FastAPI app ===

app = FastAPI()

class TranscriptRequest(BaseModel):
    text: str

@app.post("/process_transcript")
async def process(req: TranscriptRequest):
    result = process_transcript(req.text)
    return {"chapters": result}


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": (
        """
Segment this paragraph into a few chapters.
You MUST return a list of ordered json objects like this, by the appearance order of chapters (and say nothing else).
Where 'title' is the chapter title, and 'start_text' is the exact starting sentence of that chapter.

[{'title': 'some title', 'start_text': 'some_text'}, # this is for chapter 1
 {'title': 'some title', 'start_text': 'some_text'}, # this is for chapter 2
 ...
]
        """
        f"Paragraph: {paragraph}"
    )}
]

# Build prompt manually
prompt = ""
for message in messages:
    if message["role"] == "system":
        prompt += f"System: {message['content']}\n"
    elif message["role"] == "user":
        prompt += f"User: {message['content']}\n"
prompt += "Assistant:"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.7
)

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_assistant_reply(full_text: str, tag="Assistant:"):
    if tag in full_text:
        return full_text.split(tag)[-1].strip()
    return full_text.strip()

answer = extract_assistant_reply(generated_text)

print('Question:\n', prompt)
print('\n\nAnswer:\n', answer)

"""
This is the printed result, tested successfully on Kaggle notebook

Question:
 System: You are a helpful assistant.
User: 
Segment this paragraph into a few chapters.
You MUST return a list of ordered json objects like this, by the appearance order of chapters (and say nothing else).
Where 'title' is the chapter title, and 'start_text' is the exact starting sentence of that chapter.

[{'title': 'some title', 'start_text': 'some_text'}, # this is for chapter 1
 {'title': 'some title', 'start_text': 'some_text'}, # this is for chapter 2
 ...
]
        Paragraph: 


- This is the biggest and the most expensive home currently on the market in the United States, with an asking price of $295 million. It is also the most requested episode on our channel, so we're really excited for this tour. Now, this home features 105,000 square feet of interior space, 21 bedrooms, 49 bathrooms, private nightclub, breathtaking views, situated on this incredible promontory.
And let's get started. (eerie bass tone) (upbeat bass music) - The One is a property that has been legendary to all Angelinos for the last 12 years, and worldwide in the last decade, has become this iconic sort of figure. It's kind of bigger than life, it's more than a home.
It's an entity. And the reason why Angelinos have followed this property is because there are so many changes that have taken place now in Los Angeles, which don't allow you to ever recreate this. For one thing, there are nearly 30,000 cubic yards of dirt, 30,000 cubic yards of dirt that were removed, excavated to create this, okay.
And you are on four acres, almost four acres, up the longest driveway you've ever seen, and landing on top of the hill. One of the highest elevations in all of Bel Air. So for all of those reasons, and then add the icing on the cake, which is a 360 degree view.
You are in one of the most sought after and iconic properties in all of Los Angeles. (relaxed music) - We're gonna start our tour on the motor court. Long driveway brings you up here, and halfway through the driveway, you actually have an entrance point to your subterranean garage.
And behind the property, we also have an amazing detached guest home that we're gonna tour later in the video. Now, coming here, we have this paved floors, palm trees, amazing scale. And coming to this section, we have the marble walkway taking you to the front door.
I love the moat that they have off of the entry. You have a art installation on your right, two massive skylights embedded into the pool, allowing natural light to the lower section. And before we get in here, I also want to thank the listing agents, Rayni Williams, Branden Williams, and Aaron Kirman for allowing us to tour this amazing listing.
And we will be actually talking to Rayni and Brandon later in the tour. Now, coming back here, the front facade is also clad with marble. I love the grid details on the right hand side.
This overhead section has built in LED lighting to light up the entry, and coming here, we have a fingerprint scanner, and your marble door actually recesses into the wall. Now, coming here, this is where we have the spacious foyer. Oversized marble on the floors.
We have this art installation right in the center, that rotates. Beautiful chandelier above, and throughout the property, I'm gonna talk about the scale because everything in this house is so oversized. So grand, and this is just your entry here.
You have seating on each side, suede walls, and sliding glass doors seamlessly takes you to the outdoors. Now, coming back here to the entry, we have this long hallway taking us to two special rooms. First, we're gonna check out the office.
Now, before we get there, on our right hand side we have one of the staircases going up to the second floor. And coming here, we have the office. 25 foot high ceilings, skylight right in the center, walls of glass facing the views.
We have water surrounding the entire room, and this corner glass looks towards your motor court. Scale is amazing. We have the desk set up right in the center.
And the entire room is complimented with Italian lacquer finished cabinetry, with LED lighting, that looks really sleek. And on top of that, right above us, we actually have the balcony looking down on this room, and we're gonna check that space out later in the tour. Now, bringing everybody back to the entry.
Again, we have the staircase on our left, going up to the second floor. This opening takes us to a beautiful powder room with a marble pedestal sink, and marble finished wall with a mere accent. And coming here, we have the main lounge.
- This is the king's lounge, where he entertains. Walls of glass, explosive views, two king palms. You have this beautiful art wall, with this 20 foot fireplace, you have this beautiful lounge.
- I mean, this is amazing. And Branden, furniture looks small here. Room is so big, it's insane.
How many square foot you said this room was? - I think this is about 3000 square feet. - That's incredible, I mean, these are good sized couches and sofa setup.
And it's insane, and you have another one over there. - This is the secondary lounge. This is my favorite lounge because this is where you have the most explosive views in Los Angeles.


Assistant:


Answer:
 [
    {
        "title": "Introduction",
        "start_text": "This is the biggest and the most expensive home currently on the market in the United States, with an asking price of $295 million. It is also the most requested episode on our channel, so we’re really excited for this tour. "
    },
    {
        "title": "Overview",
        "start_text": "This home features 105,000 square feet of interior space, 21 bedrooms, 49 bathrooms, private nightclub, breathtaking views, and sits on this incredible promontory."
    },
    {
        "title": "Tour Details",
        "start_text": "And let’s get started. (eerie bass tone) (upbeat bass music)"
    },
    {
        "title": "Front Entry",
        "start_text": "The One is a property that has been legendary to all Angelinos for the last 12 years, and worldwide in the last decade, has become this iconic sort of figure. It’s kind of bigger than life, it’s more than a home."
    },
    {
        "title": "Property Tour",
        "start_text": "It’s an entity. And the reason why Angelinos have followed this property is because there are so many changes that have taken place now in Los Angeles, which don’t allow you to ever recreate this. For one thing, there are nearly 30,000 cubic yards of dirt, 30,000 cubic yards of dirt that were removed, excavated to create this, okay. "
    },
]
    

"""

```

<br>
augment_training_data.py
```python
codes
```

<br>
prompt_construct.py
```python
codes
```

Thank you for reading.


