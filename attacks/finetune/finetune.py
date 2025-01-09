import os
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import torch
from transformers import AdamW
import tqdm
from peft import PeftModel, LoraModel, LoraConfig,get_peft_model

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from utils.jailbreak_utils import format_text, load_jb_dataset, evaluate_jailbreak_robustness, load_ultrachat
from utils.utils import load_model, get_params, forward_with_cache, get_data
import csv
import argparse

def finetune(
    model,
    tokenizer,
    data_list_train,
    data_list_val,
    args,
):
    print(args.lora)
    print("########")
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("using lora")
        params = model.parameters()
    else:
        model.train()
        if args.layer_ids == [-1]:
            params = model.parameters()
            for param in model.parameters():
                param.requires_grad = True
        else:
            params = get_params(model, args.layer_ids, args.param_ids)
            model.print_trainable_parameters()
    optimizer = AdamW(params, lr=args.lr)

    num_batches = min( args.max_num_batches, min([len(f) for f in data_list_train]),)


    for epoch in range(args.num_epoch):
        
        save_name=args.model_name_or_path.split("/")[-1]
        finetune_setting = get_finetune_setting(args, epoch+1)
        # fine_tune_setting = f"{args.data_list_train[0]}_{f'lora-{args.lora_r}-{args.lora_alpha}' if args.lora else 'full'}_lr-{args.lr:.0e}_batch-{args.batch_size*args.grad_acc}_num-{args.max_num_batches*args.batch_size}_epoch-{epoch+1}"
        print("finetune_setting",finetune_setting)


        print(f"======= Epoch {epoch + 1 } =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(data_list_train)   # index for the topic/datast
                batch_idx = idx // len(data_list_train)  # index for the batch for each topic/dataset
                batch = data_list_train[topic_idx][batch_idx]

                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len
                ).to(model.device)

                outputs = model(inputs.input_ids, labels = inputs.input_ids, attention_mask=inputs.attention_mask)

                # Update model
                loss = outputs.loss  
                loss = loss / args.grad_acc
                loss.backward()
                if (idx + 1) % args.grad_acc == 0 or idx == num_batches - 1:
                    optimizer.step()
                    optimizer.zero_grad()

                if args.layer_ids == [-1]:
                    print(f"loss: {loss.item():.4g}")
                else:
                    print(f"loss: {loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                pbar.update(1)

                

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.num_epoch - 1:
            output_dir = f"{args.output_dir}/{finetune_setting}"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(f"{output_dir}/{save_name}.txt"):
                print(f"Skipping eval - Finetune result file for {save_name} epoch {epoch +1} already exists.")
                continue
            else:
                if "unlearning" in output_dir:
                    eval_model = HFLM(model,tokenizer = tokenizer)
                    task = 'wmdp_bio'
                    wmdp_eval_results = evaluator.simple_evaluate(model=eval_model,
                                                                tasks=[task],
                                                                device='cuda',
                                                                verbosity='ERROR',
                                                                )
                    eval_result = wmdp_eval_results['results'][task]

                    print("Eval result is:")
                    print(task, eval_result)

                    with open(f"{output_dir}/{save_name}.txt", "a") as file:
                        file.write(f"{task}: {eval_result}\n")
                elif "refusal" in output_dir:
                    print(len(data_list_val))
                    jb_result = evaluate_jailbreak_robustness(model,tokenizer,data_list_val)
                    print(np.mean(jb_result['score']))
                    with open(f"{output_dir}/{save_name}.txt", "a") as file:
                            file.write(f"LAT-harmful seed={args.seed}, n={len(data_list_val)}: {np.mean(jb_result['score'])}\n")
                    jb_result.to_csv(f"{output_dir}/{save_name}_complete_results.csv")
                else:
                    raise ValueError("Error: Not implemented")

            if args.save_model:
                path = f"../../models/{save_name}/{finetune_setting}"
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                print(f"Saved model to {path}")
            



def get_model_paths(args):
    """Get list of model paths to process"""
    if args.model_name_or_path:
        return [args.model_name_or_path]
    model_paths = []
    csv_path = f'data/{args.type}_model_paths.csv'
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            model_paths.append(row[0])
    return model_paths


def should_skip_model(model_path, args, finetune_setting):
    """Determine if model should be completely skipped (all epochs done)"""
    save_name = model_path.split("/")[-1]
    result_path = f"{args.output_dir}/{finetune_setting}/{save_name}.txt"
    
    # Skip if results already exist
    if os.path.exists(result_path):
        print(f"Skipping - Finetune result file for {save_name} already exists.")
        return True
        
    # Skip if model matches skip patterns
    if args.skip:
        for skip_pattern in args.skip.split(","):
            if skip_pattern in save_name:
                print(f"Skipping - manually skipping {skip_pattern}")
                return True
                
    return False

def get_finetune_setting(args, epoch):
    return f"{args.data_list_train[0]}_{f'lora-{args.lora_r}-{args.lora_alpha}' if args.lora else 'full'}_lr-{args.lr:.0e}_batch-{args.batch_size*args.grad_acc}_num-{args.max_num_batches*args.batch_size}_epoch-{epoch}"
    


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--type", type=str, default="", choices=["unlearning","refusal"])


    # Data arguments
    parser.add_argument("--data_list_train", type=str, default="", help="Load comma separated train splits",)
    parser.add_argument("--data_list_val", type=str, default="", help="Load comma separated val splits",)
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)

    
    # Finetune configurations
    parser.add_argument("--layer_ids", type=str, default="-1", help="Comma separated layer to finetune. -1 will finetune all layers and parameters")
    parser.add_argument("--param_ids", type=str, default="6", help="Param to update for each layer")
    parser.add_argument("--lora", action="store_true", help="Using LoRA")
    parser.add_argument("--lora_r", type=int, default=64, help="")
    parser.add_argument("--lora_alpha", type=int, default=16, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--grad_acc", type=int, default=1, help="Grad accumulation steps")
    parser.add_argument("--num_epoch", type=int, default=1, help="")
    parser.add_argument("--eval_freq", type=int, default=1, help="")
    parser.add_argument("--save_model", action="store_true", help="Save finetuned model upon each eval")

    parser.add_argument("--skip", type=str, default="",help="Comma separated skip patterns to skip models")


    args = parser.parse_args()
    args.data_list_train = args.data_list_train.split(",")
    args.data_list_val = args.data_list_val.split(",")
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args 






if __name__ == "__main__":
    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)


    model_paths = get_model_paths(args)
    if "harmful" in args.data_list_train[0]:
        data_list_train, data_list_val = load_jb_dataset(seed=args.seed,include_test_responses=True, batch_size=args.batch_size,rejected_pair=True)
    elif "ultra" in args.data_list_train[0]:
        data_list_train = load_ultrachat(seed=args.seed,batch_size=args.batch_size)
        _, data_list_val = load_jb_dataset(seed=args.seed,include_test_responses=True, batch_size=args.batch_size,rejected_pair=True)
    else:
        data_list_train, data_list_val = get_data(
            args.data_list_train,
            args.data_list_val,
            args.min_len,
            args.max_len,
            args.batch_size,
        )

    for model_path in model_paths:
        args.model_name_or_path = model_path
        finetune_setting = get_finetune_setting(args, args.num_epoch)
        if should_skip_model(args.model_name_or_path, args, finetune_setting):
            continue
        model, tokenizer = load_model(args.model_name_or_path)
        finetune(
            model,
            tokenizer,
            data_list_train,
            data_list_val,
            args,
        )
