import os
import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch
import tqdm as tqdm
from transformers import AdamW
from peft import LoraModel, LoraConfig, get_peft_model
from utils.utils import load_model, get_params, forward_with_cache, get_data, get_loss
from config.model_paths import GAModelPath



def run_ga(
    updated_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    ga_config = vars(args)
    print("====GA Config====")
    print("\n".join(f"{k}={v}" for k,v in ga_config.items()))
    print("=====")


    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "gate_proj",
                "down_proj",
            ],
        )
        updated_model = get_peft_model(updated_model, lora_config)

    updated_model = updated_model.train()
    if args.layer_ids == [-1]:
        params = updated_model.parameters()
    else:
        params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side="right"

    for epoch in range(args.num_epoch):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            mid_idx = int(num_batches/2)
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                # Unlearning loss
                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(updated_model.device)


                outputs = updated_model(unlearn_inputs.input_ids, labels = unlearn_inputs.input_ids, attention_mask=unlearn_inputs.attention_mask)
                unlearn_loss  = outputs.loss * -1
                loss = args.beta * unlearn_loss


                if args.loss_type == "grad-diff":
                    retain_inputs = tokenizer(retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                    ).to(updated_model.device)
                    outputs = updated_model(retain_inputs.input_ids,labels=retain_inputs.input_ids, attention_mask=retain_inputs.attention_mask)
                    loss += outputs.loss

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if args.layer_ids==[-1]:
                    print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g}")
                else:
                    print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

                pbar.update(1)

        path = GAModelPath.create_path(args,epoch+1)
        tokenizer.truncation_side = truncation_side
        updated_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"Saved model to {path}")


# PLACEHOLDER: need to clean up the args a bit
def get_args():

    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default=""
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    ### Data arguments
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-remove-dataset",
        help="comma-separated list of corpora to forget",
    )
    ### hyperparameters

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    parser.add_argument("--lora", action="store_true", help="Using LoRA")
    parser.add_argument("--lora_r", type=int, default=64, help="")
    parser.add_argument("--lora_alpha", type=int, default=16, help="")
    parser.add_argument("--lora_dropout", type=int, default=0.05, help="")

    parser.add_argument("--loss_type", type=str, default="", help="")
    parser.add_argument("--beta", type=float, default=1, help="")

    parser.add_argument("--num_epoch", type=int, default=1, help="")





    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
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

    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_ga(
        updated_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
