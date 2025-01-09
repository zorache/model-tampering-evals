import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import random
from typing import List, Optional
import tempfile
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

random.seed(0)



class CustomDataset(Dataset):
    def __init__(self, forget_corpora, data_dir, min_len=50, max_len=2000):
        self.data = self.get_corpus_data(forget_corpora[0], data_dir, min_len, max_len)

    def get_corpus_data(self, name, data_dir, min_len, max_len):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if min_len < len(x['text']) <= max_len:
                    data.append(str(x['text']))
        else:
            with open(f"{data_dir}/data/{name}.jsonl", "r") as f:
                for line in f:
                    if "bio-forget-corpus" in name or "bio-remove-dataset" in name:
                        raw_text = json.loads(line)['text']
                    else:
                        raw_text = line
                    if min_len < len(raw_text) <= max_len:
                        data.append(str(raw_text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]   #, self.labels[idx]





def truncate_and_pad_two_batches(batch1, batch2, pad_token_id=0):
    # Find the minimum batch size between the two batches
    min_batch_size = min(len(batch1['input_ids']), len(batch2['input_ids']))
    
    # Truncate batches to the minimum batch size
    batch1 = {k: v[:min_batch_size] for k, v in batch1.items()}
    batch2 = {k: v[:min_batch_size] for k, v in batch2.items()}
    
    # Find the maximum sequence length across both batches
    max_len = max(
        max(seq.size(0) for seq in batch1['input_ids']),
        max(seq.size(0) for seq in batch2['input_ids'])
    )
    
    def pad_batch(batch, max_len):
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(batch['input_ids'], batch['attention_mask']):
            pad_len = max_len - ids.size(0)
            
            padded_ids = F.pad(ids, (0, pad_len), value=pad_token_id)
            padded_input_ids.append(padded_ids)
            
            padded_mask = F.pad(mask, (0, pad_len), value=0)
            padded_attention_mask.append(padded_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask)
        }
    
    # Pad both batches
    padded_batch1 = pad_batch(batch1, max_len)
    padded_batch2 = pad_batch(batch2, max_len)
    
    return padded_batch1, padded_batch2



def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]
    


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto"
        )
    except Exception as e:
        base_model= AutoModelForCausalLM.from_pretrained(
             "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(base_model, model_name_or_path)
        model = lora_model.merge_and_unload()
        del base_model
        del lora_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True, 
            use_fast=False

        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", 
            trust_remote_code=True, 
            use_fast=False

        )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        elif name =="bio-retain-corpus":
            from datasets import load_dataset
            raw_data = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        elif name == "imdb":
            from datasets import load_dataset
            raw_data = load_dataset("stanfordnlp/imdb", split="train")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(f"{data_dir}/data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                elif "bio-remove-dataset" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )



def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss