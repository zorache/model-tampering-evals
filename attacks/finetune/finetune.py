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

    if args.load_model_path:
        import re
        epoch_match = re.search(r'epoch-(\d+)', args.load_model_path)
        start_epoch = int(epoch_match.group(1)) if epoch_match else 0
    else:
        start_epoch = 0
    epochs_to_train = max(0, args.num_epoch - start_epoch)


    for epoch in range(epochs_to_train):
        current_epoch = start_epoch + epoch + 1
        
        save_name=args.model_name_or_path.split("/")[-1]
        fine_tune_setting = f"{args.data_list_train[0]}_{f'lora-{args.lora_r}-{args.lora_alpha}' if args.lora else 'full'}_lr-{args.lr:.0e}_batch-{args.batch_size*args.grad_acc}_num-{args.max_num_batches*args.batch_size}_epoch-{epoch+1}"
        print("fine_tune_setting",fine_tune_setting)


        print(f"======= Epoch {current_epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(data_list_train)   # index for the topic/datast
                batch_idx = idx // len(data_list_train)  # index for the batch for each topic/dataset
                batch = data_list_train[topic_idx][batch_idx]

                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length
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

                

        if (epoch + 1) % args.save_freq == 0 or epoch == args.num_epoch - 1:
            output_dir = f"{args.output_dir}/{fine_tune_setting}"
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(f"{output_dir}/{save_name}.txt"):
                print(f"Skipping eval - Finetune result file for {save_name} already exists.")
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
                path = f"../../models/{save_name}/{fine_tune_setting}"
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                print(f"Saved model to {path}")
            



def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default=""
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )

    ### Data arguments
    parser.add_argument(
        "--data_list_train",
        type=str,
        default="",
        help="Load comma separated train splits",
    )
    parser.add_argument(
        "--data_list_val",
        type=str,
        default="",
        help="Load comma separated val splits",
    )

    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000, help ="Max length of number of batches")
    parser.add_argument("--max_length", type=int, default=512, help ="Max length of batch")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--grad_acc", type=int, default=1, help="Grad accumulation steps")


    parser.add_argument("--layer_ids", type=str, default="-1", help="Comma separated layer to finetune. -1 will finetune all layers and parameters")
    parser.add_argument("--param_ids", type=str, default="6", help="Param to update for each layer")

    parser.add_argument("--lora", action="store_true", help="Using LoRA")
    parser.add_argument("--lora_r", type=int, default=64, help="")
    parser.add_argument("--lora_alpha", type=int, default=16, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="")

    parser.add_argument("--num_epoch", type=int, default=1, help="")

    parser.add_argument("--save_freq", type=int, default=1, help="")
    parser.add_argument("--save_model", action="store_true", help="Save finetuned model")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to the saved model to continue training")


    args = parser.parse_args()
    args.data_list_val = args.data_list_val.split(",")
    args.data_list_train = args.data_list_train.split(",")
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

    if args.load_model_path:
        # For continual training
        print(f"Loading model from {args.load_model_path}")
        if args.lora:
            base_model, tokenizer = load_model(args.model_name_or_path)
            model = PeftModel.from_pretrained(base_model, args.load_model_path)
        else:
            model, tokenizer = load_model(args.load_model_path)
    else:
        model, tokenizer = load_model(args.model_name_or_path)

    # We load the train and val splits
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
        
    finetune(
        model,
        tokenizer,
        data_list_train,
        data_list_val,
        args,
    )
