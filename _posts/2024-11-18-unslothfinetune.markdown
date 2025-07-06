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

Basically, I want to DIRECTLY ASK the LLM the following question:

```python
question = """
    
Let's see if you can solve this simple Abstraction and Reasoning Challenge (ARC) task.
Below there are some input-output grid examples that define the task.
Your job is to understand the transformation between the input and the output and apply it to the test input grid.
The transformations are always based on the following priors: objectness, goal-directed, numbers & counting, and basic geometry & topology.

### Input:
[insert train input-output pairs]

### Output
[insert test input image]

Your task is to create a new grid that follows the same distribution as the input grids from the Abstraction and Reasoning Challenge (ARC).
Below there are some grid examples, please create a new and different grid that follows the same distribution.
"""
```

The tricky part is that how can it understand the spacial abstract meaning of objects inside the grids which are also texts in the question? <br>

The answer is that we can modify the tokenizer to let it understand pixels! If we use the default tokenizer as is, there's no way it can understand that '[[1, 1, 1], [2, 2, 2]]' is an object with three pixels of color '1' and three pixels of color '2'. As a result, '1', '2', or even '[', ']' should represent as independent tokens without their original meaning, and must not be merged with any other tokens or with each other. We can do by modifying the tokenizers and the related model objects as belows.

```python
import json
import os, sys
import bz2
import pickle
import numpy as np
from tqdm import tqdm

def indices_required_for_merges(keep_indices, vocab, merges):
    merges_lookup = {}
    for m in merges:
        a, b = m.split(' ') if isinstance(m, str) else m
        key = vocab[f'{a}{b}']
        if key not in merges_lookup: merges_lookup[key] = set()
        merges_lookup[key].add(vocab[a])
        merges_lookup[key].add(vocab[b])
    to_process = list(keep_indices)
    while len(to_process):
        for w in merges_lookup.get(to_process.pop(), []):
            if w not in keep_indices:
                keep_indices[w] = None
                to_process.append(w)
    return keep_indices

def remove_unused_merges(merges, vocab):
    return [f'{a} {b}' for a, b in [m.split(' ') if isinstance(m, str) else m for m in merges] if all(w in vocab for w in [a, b, a + b])]

def map_special_tokens(data, mapping=None):
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:
            for v in special.values():
                tokens.update(v['ids'])
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
    for v in (data.values() if isinstance(data, dict) else data if isinstance(data, list) else []):
        tokens.update(map_special_tokens(v, mapping))
    return tokens

def remove_tokenizer_normalizer(tokenizer):
    from tokenizers import Tokenizer
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order):
    from tokenizers import Tokenizer
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    assert tokenizer_json['model']['type'] == "BPE"
    if keep_special_tokens:
        keep_indices.update({k: None for k in tokenizer.all_special_ids})
        keep_indices.update({k: None for k in map_special_tokens(tokenizer_json.get('post_processor'))})
    keep_indices = indices_required_for_merges(keep_indices, tokenizer_json['model']['vocab'], tokenizer_json['model']['merges'])
    if keep_token_order: keep_indices = sorted(keep_indices)
    mapping = {old: new for new, old in enumerate(keep_indices)}
    tokenizer_json['model']['vocab'] = {k: mapping[v] for k, v in tokenizer_json['model']['vocab'].items() if v in mapping}
    tokenizer_json['model']['merges'] = remove_unused_merges(tokenizer_json['model']['merges'], tokenizer_json['model']['vocab'])
    special_tokens_order = [t['id'] for t in tokenizer_json['added_tokens']]
    assert special_tokens_order==sorted(special_tokens_order)
    tokenizer_json['added_tokens'] = sorted([{**t, 'id': mapping[t['id']]} for t in tokenizer_json['added_tokens'] if t['id'] in mapping], key=lambda t: t['id'])
    map_special_tokens(tokenizer_json.get('post_processor'), mapping)
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    return mapping, keep_indices

def shrink_model_embeddings(model, keep_indices, mapping):
    import torch
    with torch.no_grad():
        row_select = torch.tensor(list(keep_indices))
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select.to(model.get_input_embeddings().weight.data.device))
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select.to(model.get_output_embeddings().weight.data.device))
        model.resize_token_embeddings(len(keep_indices))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def shrink_embeddings(model, tokenizer, corpus=None, keep_token_ids=[], keep_tokens=[], remove_token_ids=[], keep_model_tokens=True, keep_special_tokens=True, keep_normalizer=False, keep_token_order=True):
    if not keep_normalizer: remove_tokenizer_normalizer(tokenizer)
    from collections import OrderedDict  # use as OrderedSet
    keep_indices = OrderedDict()
    keep_indices.update({k: None for k in keep_token_ids})
    keep_indices.update({tokenizer.vocab[t]: None for t in keep_tokens})
    if corpus is not None: keep_indices.update({k: None for k in tokenizer(corpus)['input_ids']})
    if keep_model_tokens:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update({k: None for k in (v if isinstance(v, list) else [v])})
    keep_indices.pop(None, None)
    for idx in remove_token_ids: keep_indices.pop(idx, None)
    mapping, keep_indices = shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order)
    shrink_model_embeddings(model, keep_indices, mapping=mapping)
    return mapping

def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    import torch
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                if fix_weights and weight.dtype!=model.dtype:
                    module.to(model.dtype)
            else:
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype!=model.dtype:
                        qs.dtype = model.dtype
    return model

def merge_peft_into_base(model):
    print('*** Merge peft model into base model...')
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())

def save_model(store_path, model=None, tokenizer=None, merge=False):
    if merge: model = merge_peft_into_base(model)
    if store_path is not None:
        assert model is not None or tokenizer is not None
        print(f"*** Saving{' merged' if merge else ''} model/tokenizer to '{store_path}'...")
        if model is not None: model.save_pretrained(store_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(store_path)
            to_delete = os.path.join(store_path, 'tokenizer.model')
            if os.path.isfile(to_delete): os.remove(to_delete)
    return model

def is_unsloth_model(model):
    return model.model_tags is not None and 'unsloth' in model.model_tags

def is_peft_model(model):
    return hasattr(model, 'peft_type')

def download_model(repo_id, store_path, get_name=lambda n: os.path.join(n.replace('/', '--'), 'transformers', 'default', '1')):
    import os
    if os.path.exists(repo_id): return repo_id
    model_path = os.path.join(store_path, get_name(repo_id))
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        download_path = snapshot_download(repo_id=repo_id)
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        os.symlink(download_path, model_path, target_is_directory=True)
    return model_path

def get_and_fix_peft_weights(store):
    print(f"*** Load peft state_dict from '{store}'...")
    from peft import load_peft_weights
    state_dict = load_peft_weights(store)
    for k in list(state_dict.keys()):
        if 'modules_to_save' in k:
            del state_dict[k]
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict: del state_dict[original_module_key]
            assert k.replace('.modules_to_save.', '.') in state_dict
    return state_dict

def set_peft_weights(model, state_dict):
    print(f"*** Set model state_dict...")
    from peft import set_peft_model_state_dict
    res = set_peft_model_state_dict(model, state_dict)
    assert not res.unexpected_keys

def load_peft_state(model, store):
    set_peft_weights(model, get_and_fix_peft_weights(store))
```


## 2) Finetune LLMs with unsloth.ai <br>
<br>
I happened to find [unsloth.ai](unsloth.ai) as the SOTA tool for finetuning LLMs! Previously I used some no-code GUI approaches like H2O LLM Studio, or I finetuned models directly by LORA with original transformers and pytorch support. But after trying unsloth, I have to admit that it is indeed the BEST library to finetune LLM models, because: <br>

+ It makes BOTH training and inference much faster. Unsloth's ability to significantly speed up (3x) the inference process made a huge difference when we want to run LLM inferencing inside a Kaggle notebook with limited resource (in this case: only 2xP100 GPUs are given to each participant within a 12-hour runtime). <br>

+ It is really easy to use. The most difficult part is to installing and adjust the configurations. But when it comes to the training / inferencing, it is not so much different than a normal deep learning pytorch library. Everything went smoothly. <br>

+ It supports a wide range of models, including LLAMA, Deepseek, Qwen, Mixtral, Phi, Gemma... This allows me to play and find the best suitable model (which is Qwen2.5-0.5B-Instruct). <br>

The dataset for training is not simply the orignal training tasks transformed into questions. LLMs can't generalize with just about 1000 data samples, which is impossible. So I need to use all of the synthetic datasets (of new tasks) that were released by other competitors during the 2019 edition (as well as new datasets during 2024). But that's still not sufficient. We need to augment EACH TASK with variants of itself, such as:

```python
flip horizontally / vertically
rotate 0 / 90 / 180 / 270
random color swap
```

Combining the above operations on each task, we can easily have up to 32 variants of input-output pairs for a single task. That significantly increases the training population and also helps the LLMs generalize, since now for each task we have variants, and they can focus more on diffirentiate the spacial abstract meaning of the objects / movements, rather than just memorizing shapes or colors. <br>

Finetuning LLMs is the HARDEST part of everything else. There were countless hours of errors / debuggings, and frustrating and disappointed nights of model not learning. However, with the nature of trial-and-error of Kaggle competitions that I experienced in the last roughly 10 years, it usually came to a fruitful result. Here are some (dirty) training procedures: <br>

```python
def prepare_model(model, mode, tokenizer=None, formatter=None, shrink_embedding=False, dequantize=False, peft=[], local_files_only=False, add_special_tokens={}, set_pad_token=None, keep_tokens=[], keep_normalizer=None, peft_trainable=True, device_map=None, tf_grad_cp=True, tf_use_fa2=True, **kwargs):
    if isinstance(model, str):
        assert tokenizer is None
        print(f"*** Load base model and tokenizer from '{model}'...")
        if mode=='unsloth_4bit':
            assert device_map is None, 'unsupported'
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=model, dtype=None, load_in_4bit=True, local_files_only=local_files_only, **kwargs)
        elif mode in ['transformers', 'transformers_bf16', 'transformers_4bit', 'transformers_bf16_4bit', 'tokenizer_only']:
            import torch
            model_load_args = {}
            if device_map is not None: model_load_args['device_map'] = device_map
            if tf_use_fa2: model_load_args['attn_implementation'] = 'flash_attention_2'
            if mode in ['transformers_bf16', 'transformers_bf16_4bit']: model_load_args['torch_dtype'] = torch.bfloat16
            elif mode in ['transformers_4bit', 'transformers_bf16_4bit']:
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
                model_load_args['quantization_config'] = nf4_config
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(model, **model_load_args) if mode!='tokenizer_only' else None
            if tf_grad_cp and model is not None: model.gradient_checkpointing_enable()
        else: raise NotImplementedError('Unknown mode.')
    if add_special_tokens: tokenizer.add_special_tokens(add_special_tokens)
    if set_pad_token is not None: tokenizer.pad_token = set_pad_token
    if formatter is not None and not hasattr(formatter, 'corpus'):
        formatter = formatter(tokenizer=tokenizer)
    if (shrink_embedding<len(tokenizer.vocab) if type(shrink_embedding)==int else shrink_embedding) or keep_normalizer is False:
        print('*** Shrink embedding...')
        embedding_size_before_shrink = len(tokenizer.vocab)
        mapping = shrink_embeddings(model, tokenizer, formatter.get_corpus(), keep_tokens=keep_tokens, keep_normalizer=keep_normalizer)
        print(f'*** -> Reduced embedding size from {embedding_size_before_shrink} to {len(mapping)} words.')
    if dequantize:
        print(f'*** Dequantize model...')
        model = model.dequantize()
    if len(peft):
        peft_trained = True if is_peft_model(model) else None
        for i, m in enumerate(peft):
            if peft_trained is True: model, peft_trained = merge_peft_into_base(model), None
            if isinstance(m, str):
                if peft_trained is False:
                    _, peft_trained = load_peft_state(model, m), True
                else:
                    print(f"*** Load peft model from '{m}'...")
                    # be careful when using unsloth - using PeftModel to load the model will not apply unsloth optimizations
                    from peft import PeftModel
                    model, peft_trained = PeftModel.from_pretrained(model, m, trainable=peft_trainable), True
            else:
                assert peft_trained is None
                if isinstance(m, dict):
                    print('*** Create new peft model...')
                    if is_unsloth_model(model):
                        from unsloth import FastLanguageModel
                        my_get_peft_model = FastLanguageModel.get_peft_model
                    else:
                        from peft import LoraConfig, get_peft_model
                        my_get_peft_model = lambda model, **kwargs: get_peft_model(model, LoraConfig(**kwargs))
                    model, peft_trained = my_get_peft_model(model, **m), False
                else: assert m is None
    return model, tokenizer, formatter

def training_run(model, formatter, dataset, train_args, max_seq_length, merge=False, store=None, packing=False, grad_acc_fix=False, optimizers=None):
    assert merge is False, "merge after training does not seen to work (at least with unsloth, saved merged model will cointain the untrained weights!)"
    import torch
    from datasets import Dataset
    add_train_args = {}
    if is_unsloth_model(model):
        from unsloth import FastLanguageModel
        from unsloth import UnslothTrainer as Trainer
        from unsloth import UnslothTrainingArguments as TrainingArguments
        from unsloth import is_bfloat16_supported
        FastLanguageModel.for_training(model)
        add_train_args.update(fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported())
    else:
        from trl import SFTConfig as TrainingArguments
        from trl import SFTTrainer as Trainer
        model.train()
        add_train_args.update(bf16=True)

    formatter.tokenizer.padding_side = 'right'
    if is_unsloth_model(model):
        for convert_to_float in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if convert_to_float.weight.dtype!=torch.float32: convert_to_float.to(torch.float32)

    add_args = {}
    if optimizers is not None: add_args['optimizers'] = optimizers

    trainer = Trainer(
        model=model,
        tokenizer=formatter.tokenizer,
        data_collator=formatter.get_data_collator(),
        train_dataset=Dataset.from_list(dataset.as_list(formatter)),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=None,
        packing=packing,  # Can make training 5x faster for short sequences.
        **add_args,
        args=TrainingArguments(
            **add_train_args,
            **train_args
        ),
    )

    print('*** Start training run...')
    if grad_acc_fix and is_unsloth_model(model):
        from unsloth import unsloth_train
        trainer_stats = unsloth_train(trainer)
    else:
        if is_unsloth_model(model) and train_args['gradient_accumulation_steps']>1: print('*** WARNING: using faulty unsloth gradient accumulation')
        trainer_stats = trainer.train()
    try: print(f'*** -> Training took {trainer_stats.metrics["train_runtime"]} seconds.')
    except: pass
    if store is not None: save_model(store, model, formatter.tokenizer, merge=merge)
    return model, trainer_stats
```


## 3) Test-time finetuning during inference 
<br>
When the trained model faces a hidden test task in the inference mode, it will perform a quick re-training JUST FOR THAT TASK. This helps specifically only in this unique competition, because we have various ways to augment the on-hand task (flip, rotate, color swap...). <br>

The tricky part for test-time finetuning is to have all model re-training sessions (100 in total) fit within 12 hours of Kaggle runtime and with limited resources (1xP100 or 2xT4 GPUs). That can be done with asynchronous concurrent processes design:

```python
%%python --bg --proc train_proc0
start_training(gpu=0)

%%python --bg --proc train_proc1
start_training(gpu=1)

%%python --bg --proc infer_proc0
start_inference(gpu=0)

%%python --bg --proc infer_proc1
start_inference(gpu=1)

proc_exit_codes = await wait_for_subprocesses(train_proc0, train_proc1, infer_proc0, infer_proc1)
```
The asyncio-style process design allows me to better log outputs, but it has way more unknown errors (mostly training got freezed without any log produced) than normal multiprocessing Python package. <br>

Below are some inferencing procedures:
```python
def inference_load(store, keys=True, result_dict=None, always_read_from_file=False):
    if result_dict is None: result_dict = {}
    if store is not None:
        if keys is True: keys = os.listdir(store)
        for key in keys:
            if always_read_from_file or key not in result_dict:
                try:
                    with bz2.BZ2File(os.path.join(store, key)) as f: result_dict[key] = pickle.load(f)
                except: continue
    return result_dict

def inference_save(store, key, outputs):
    if store is not None:
        os.makedirs(store, exist_ok=True)
        with bz2.BZ2File(os.path.join(store, key), 'w') as f: pickle.dump(outputs, f)

class Decoder(object):
    def __init__(self, formatter, dataset, n_guesses, max_outputs=None, frac_score=False, quiet=False, name='', additional_decoders=None, prob_baseline=None):
        self.formatter = formatter
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        self.correct_solutions = {}
        self.keys_lim = set()
        self.keys_all = set()
        self.mult_cnt = {}
        self.keys_cnt = {}
        self.frac_score = frac_score
        self.max_outputs = max_outputs
        self.quiet = quiet
        self.input_len = [{} if formatter is not None and formatter.tokenizer is None else ds.get_lengths(formatter, name='input') for ds in [dataset, dataset.mod(np.transpose, keep_key=True)]]
        self.reply_len = [{} if formatter is not None and formatter.tokenizer is None else ds.get_lengths(formatter, name='reply') for ds in [dataset, dataset.mod(np.transpose, keep_key=True)]]
        self.additional_decoders = additional_decoders
        self.name = name
        self.prob_tracker = {}
        self.prob_tracker_best = {}
        self.prob_baseline = prob_baseline

    def score(self, *to_score):
        scores = [(sum(1/self.mult_cnt[k.split('_')[0]] for k in s) if self.frac_score else len(s)) for s in to_score]
        score_cnt = len(self.mult_cnt if self.frac_score else self.keys_cnt)
        return scores, score_cnt

    def from_store(self, store, **kwargs):
        for key, outputs in inference_load(store).items():
            self.process(key, outputs, **kwargs)
        return self

    def score_fmt(self, v):
        return f'{v:5.1f}' if self.frac_score else f'{v:3}'

    def process_single_output(self, key, output_len, decoded, print_func=print, len_info=None, device_info=None):
        import numpy as np
        inv_mod = {k: v if k.endswith('val') else self.dataset.invert_mod(v, key, inv_perm=(k.startswith('output') or k.startswith('score_all'))) for k, v in decoded.items()}
        base_key = key.split('.')[0]
        self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
        self.decoded_results[base_key][key] = inv_mod
        output = inv_mod.get('output')
        score = inv_mod.get('score')

        # quick scoring
        self.keys_cnt[base_key] = self.keys_cnt.get(base_key, 0) + 1
        mult_key, mult_sub = (base_key.split('_') + ['0'])[:2]
        self.mult_cnt[mult_key] = max(self.mult_cnt.get(mult_key, 0), int(mult_sub) + 1)
        if len(self.dataset.replies):
            correct_solution = self.dataset.replies.get(base_key)
            if correct_solution is not None:
                correct_solution = correct_solution[0]
                self.correct_solutions[base_key] = correct_solution
                is_correct = correct_solution is not None and np.array_equal(correct_solution, output)
                if is_correct:
                    self.keys_all.add(base_key)
                    if self.keys_cnt[base_key] <= self.n_guesses: self.keys_lim.add(base_key)
            corr_str = 'cant_decode' if output is None else 'sol_unknown' if correct_solution is None else 'ALL_CORRECT' if is_correct else 'bad_xy_size' if np.shape(correct_solution)!=np.shape(output) else 'bad_content'
            (score_lim, score_all), score_cnt = self.score(self.keys_lim, self.keys_all)

            tp_arr = (key.count('transpose') + key.count('rot90')) % 2
            msc = None if score is None else np.sum(score)
            fsc = inv_mod.get('score_val')
            if output is not None and fsc is not None:
                pt = self.prob_tracker[base_key] = self.prob_tracker.get(base_key, {})
                hash = tuple(map(tuple, output))
                prob = pt[hash] = pt.get(hash, 0) + (np.exp(fsc) if self.prob_baseline is None else fsc - np.log(self.prob_baseline))
                current_best = self.prob_tracker_best.get(base_key)
                if current_best is None or current_best[0]<prob:
                    self.prob_tracker_best[base_key] = (prob, output)
            fmt_name = f'{self.name}: ' if self.name else ''
            msc_print = f'{min(-msc, 9.99999):7.5f}' if msc is not None else 'unknown'
            fsc_print = f'{min(-fsc, 9.99999):7.5f}' if fsc is not None else 'unknown'
            if not self.quiet: print_func(f" {fmt_name}acc: {self.score_fmt(score_lim)}/{score_cnt:3}={min(score_lim/score_cnt, 0.999):5.1%} (2-guess), {self.score_fmt(score_all)}/{score_cnt:3}={min(score_all/score_cnt, 0.999):5.1%} (any);{f' {device_info}' if device_info else ''} tok:{self.input_len[tp_arr].get(base_key, '?'):>4}+{self.reply_len[tp_arr].get(base_key, '?'):>3}>{'n/a' if output_len is None else output_len:>3} {corr_str}:{msc_print}|{fsc_print} [{key}]")

    def get_current_best(self, base_key):
        current_best = self.prob_tracker_best.get(base_key)
        return None if current_best is None else current_best[1]

    def process_single_decode(self, key, de_tokenized, print_func=print, **kwargs):
        if len(de_tokenized)==3 and not isinstance(de_tokenized[1], float):  # for backwards compatibility
            output_len, *data = de_tokenized
            score_val = None
        else: output_len, score_val, *data = de_tokenized
        if self.formatter is None:
            assert len(data) == 1
            decoded = [data[0]]
        else: decoded = self.formatter.decode_to_array(*data)
        #if len(decoded)==2:
        #    same = np.array_equal(decoded[0].get('output'), decoded[1].get('output'))
        #    print_func(f"is_identical: {same}")
        #    if not same: for i in range(2): print_func(str(decoded[i].get('output')))
        for d in decoded: d['score_val'] = score_val
        for i, dec in enumerate(decoded):
            if i==0: self.process_single_output(key, output_len, dec, print_func=print_func, **kwargs)
            elif self.additional_decoders:
                if i-1<len(self.additional_decoders): self.additional_decoders[i-1].process_single_output(key, output_len, dec, print_func=print_func, **kwargs)
                else: print_func(f'{key} no decoder available for output #{i}')
            else: self.process_single_output(f'{key}.fix{i}', output_len, dec, print_func=print_func, **kwargs)

    def process(self, key, de_tokenized, **kwargs):
        for i, d in enumerate(de_tokenized):
            if self.max_outputs is None or i<=self.max_outputs:
                self.process_single_decode(f'{key}.out{i}', d, **kwargs)

    def get_unsolved_keys(self):
        unsolved = []
        for base_key, reply in self.dataset.replies.items():
            if not any(np.array_equal(reply[0], s.get('output')) for s in self.decoded_results.get(base_key, {}).values()):
                unsolved.append(base_key)
        return unsolved

    def run_selection_algo(self, selection_algorithm):
        return {bk: (selection_algorithm({k: g for k, g in v.items() if g.get('output') is not None}) if any(g.get('output') is not None for g in v.values()) else []) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self, selection_algorithms, skip_failed=True):
        import numpy as np
        results = {}
        print('*** Benchmark selection algorithms...')
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            try:
                selected = self.run_selection_algo(selection_algorithm)
                if self.formatter is not None:
                    for sols in selected.values():
                        for s in sols:
                            assert self.formatter.is_valid_solution(s), f'found invalid solutions {s}'
                correct_keys = {k for k, v in selected.items() if self.correct_solutions.get(k) is not None and any(np.array_equal(guess, self.correct_solutions[k]) for guess in v[:self.n_guesses])}
                (score,), score_cnt = self.score(correct_keys)
                results[name] = score
                print(f" acc: {score:5.1f}/{score_cnt:3}={score/score_cnt:6.2%} ('{name}')")
            except:
                print(f" {'execution failed':>21} ('{name}')")
                if not skip_failed: raise
        return results

    def calc_augmented_scores(self, model, base_keys=None, store=None, seed=0, max_len=None, make_unique=True, quiet=False, **kwargs):
        if base_keys is None: base_keys = list(self.decoded_results.keys())
        if store is not None: store = f'{store}_new'  # new format is not backwards compatible, so use new folder
        for bk in (base_keys if quiet else tqdm(base_keys, desc='calculate augmented scores', file=sys.stdout)):
            res = self.decoded_results.get(bk, {})
            known_scores = {}
            for k, v in sorted(res.items()):
                if 'output' in v:
                    k_store = None if store is None else os.path.join(store, k)
                    id = tuple(map(tuple, v['output']))
                    if not (make_unique and id in known_scores):
                        try:
                            assert k_store is not None
                            with bz2.BZ2File(k_store) as f: known_scores[id] = pickle.load(f)
                            if isinstance(known_scores[id], list): known_scores[id] = dict(score_multi=known_scores[id])  # for backwards compatibility
                            k_store = None
                        except:
                            temp_dataset = self.dataset.__class__(
                                keys=[bk],
                                queries={bk: self.dataset.queries.get(bk)},
                                replies={bk: [v['output'].tolist()]},
                            )
                            temp_decoder = self.__class__(self.formatter, temp_dataset, n_guesses=self.n_guesses, quiet=True)
                            temp_dataset = temp_dataset.augment(**kwargs, seed=(seed+hash(k)+hash(id)) % 1024**2, quiet=True)
                            if max_len is not None: temp_dataset = temp_dataset.cut_to_len(formatter=self.formatter, name='input', max_len=max_len, quiet=True)
                            for x in temp_dataset.as_list(self.formatter): calc_score(**x, formatter=self.formatter, model=model, decoder=temp_decoder)
                            known_scores[id] = dict(
                                score_multi=[np.sum(x['score']) for x in temp_decoder.decoded_results[bk].values()],
                                score_multi_nl=[x['score_val'] for x in temp_decoder.decoded_results[bk].values()],
                                score_multi_array=np.array([x['score'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_cum=np.array([x['score_cum'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_all=np.array([x['score_all'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_all_cum=np.array([x['score_all_cum'] for x in temp_decoder.decoded_results[bk].values()]),
                            )
                            if k_store is not None:
                                os.makedirs(store, exist_ok=True)
                                with bz2.BZ2File(k_store, 'w') as f: pickle.dump(known_scores[id], f)
                    v.update(known_scores[id])

def turbo_dfs(model, logits, path, eos_token_id, max_new_tokens, max_score, max_score_greedy, temperature, suppress_tokens, torch, score=0.0, pos=0, cache=None):
    logits, next_logits = logits[0], (logits[1:] if len(logits)>1 else None)
    nll = -(logits / temperature).detach().float().log_softmax(-1).cpu().numpy()
    greedy_index = nll.argmin(-1).item()
    nll = list(enumerate(nll))
    if path: nll[0], nll[path[0]], path = nll[path[0]], nll[0], path[1:]  # follow precomputed path first
    suffixes = []
    for i, s in nll:
        next_score = score + s
        allowed_max_score = max_score_greedy if i==greedy_index else max_score
        if next_score < allowed_max_score:
            if i==eos_token_id: next_suffixes = [(next_score, [], [])]
            elif max_new_tokens>1:
                if next_logits is None:
                    if pos<cache[0][0][0].shape[2]: cache[0] = tuple(tuple(c[:, :, :pos] for c in l) for l in cache[0])
                    next_logits, cache[0] = model(
                        input_ids= torch.full((1,1), i, device=model.device),
                        position_ids=torch.full((1,1), pos, device=model.device),
                        past_key_values=cache[0],
                    )[:2]
                    next_logits = next_logits[0]  # unbatch
                next_suffixes = turbo_dfs(model, logits=next_logits, path=path, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens-1, max_score=max_score, max_score_greedy=allowed_max_score, temperature=temperature, suppress_tokens=suppress_tokens, torch=torch, score=next_score, pos=pos+1, cache=cache)
            else: next_suffixes = []
            for suffix in next_suffixes:
                suffix[1].append(i)
                suffix[2].append(logits)
            suffixes.extend(next_suffixes)
        next_logits = None
    return suffixes

def inference_turbo_dfs(model, input_ids, eos_token_id, max_new_tokens, min_prob, min_prob_greedy=1, temperature=0.9, suppress_tokens=[], path=[], attention_mask=None):
    import torch
    with torch.no_grad():
        assert attention_mask is None or attention_mask.all(), 'not implemented'
        input_ids = torch.as_tensor(input_ids, device=model.device, dtype=int)
        if input_ids.ndim==2: input_ids = input_ids.squeeze(0)
        assert input_ids.ndim==1, 'batching not supported'
        max_score = -np.log(min_prob)
        max_score_greedy = (-np.log(min_prob_greedy)) if min_prob_greedy>0 else float('inf')  # avoid throwing numpy error
        max_score_greedy = max(max_score, max_score_greedy)
        if path is None: path = []
        if len(path) and path[-1]==eos_token_id: path = path[:-1]
        with torch.no_grad():
            full_path = input_ids
            if len(path): full_path = torch.cat([full_path, torch.as_tensor(path, device=model.device)])
            logits, cache = model(input_ids=full_path[np.newaxis])[:2]
            logits = logits[0, len(input_ids)-1:]
        result = turbo_dfs(model, logits=logits, path=path, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, max_score=max_score, max_score_greedy=max_score_greedy, temperature=temperature, suppress_tokens=suppress_tokens, torch=torch, score=0.0, pos=len(input_ids), cache=[cache])
        return sorted([(score_val, np.array(suffix[::-1]), torch.stack(score_arr[::-1]).float().cpu().numpy()) for score_val, suffix, score_arr in result], key=lambda x:x[0])

def inference_step(tokenized, model, remove_token_type_ids=True, num_beams=1, formatter=None, min_prob=None, current_best=None, **kwargs):
    import torch
    if remove_token_type_ids: tokenized.pop('token_type_ids', None)
    if min_prob is not None:
        assert num_beams==1
        gen = inference_turbo_dfs(model, **tokenized.to(model.device), path=current_best, min_prob=min_prob, eos_token_id=formatter.tokenizer.eos_token_id, **kwargs)
        tokens_out = [[g[1] for g in gen]]
        scores_out = [[g[2] for g in gen]]
    elif is_unsloth_model(model) and num_beams > 1:
        assert False, 'unsloth does not support beam search'
    else:
        gen = model.generate(**tokenized.to(model.device), return_dict_in_generate=True, output_logits=True, use_cache=True, **kwargs)
        tokens_out = gen['sequences'][:, torch.newaxis, tokenized['input_ids'].shape[-1]:].cpu().numpy().copy()
        scores_out = torch.stack(gen['logits'], axis=-2)[:, torch.newaxis].float().cpu().numpy().copy()
    return tokens_out, scores_out

def process_inference_output(key, outputs, formatter, store=None, decoder=None, decoder_args={}):
    de_tokenized = [formatter.de_tokenize(*output) for output in zip(*outputs)]
    inference_save(store, key, de_tokenized)
    if decoder is not None: decoder.process(key, de_tokenized, **decoder_args)
    return de_tokenized

```

## 4)  Ensembling with past-solutions and Candidate selection optimization

Some heuristic-based solutions from the 2019 edition were still very strong and helped the ensemble of predictions. <br>

To ensemble different predictions of a single task, I used a unique self-made selection elimination technique that helps to reduce the irrelevant predictions for a single task, by observing various heuristic-based factors of that task: background detection, object count, task type classification. For instance: <br>

+ Detect background color, eliminate predictions with wrong background color. <br>

+ Detect whether the task is of a specific type (such as only_recolor, only_move, always_num_input_objs_equal_num_output_objs, ...), then invalidate predictions that are not of that task type. <br>

+ Role swapping: use the prediction (A) as 1 input-output pair for training, and swap it with 1 real train input-output pair (B). Then with a new collection of train input-output pairs, try to predict the output of B and compare it with the real output of B. If they don't match then the prediction (A) is eliminated. <br>

### Remarks

Getting a solo gold in a Kaggle competition is no joke. This competition has consumed all my life and soul in 6 months. <br>

However, if you look at the competition LB by now, you don't see me there because of the Kaggle buggy procedure when they hand out money prizes. It was a really long and frustrating story. If you want to understand more about this unfortunate incident, that also happened to 3 other teams, please read this Kaggle [discussion topic](https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442) and all of its comments there. You'll also see me ranting about it there :) <br>

Anyway, that doesn't reduce the validity of my solution, but as how it turned out, I would like to hold off the sharing of my full code solution for now, to gain an advantage on the ongoing 2025 edition (which is now running from Mar to Nov 2025). <br>

I will post my full solution once ARC Prize 2025 is concluded (Nov 2025). <br>

Thank you for reading.


