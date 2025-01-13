"""
Adapted from Representation Noising Codebase: https://github.com/domenicrosati/representation-noising
"""

import os
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import torch
from torch import nn
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from transformers import AdamW
import tqdm as tqdm

from utils.utils import load_model, get_params, forward_with_cache, get_data, truncate_and_pad_two_batches
from config.model_paths import RepNoiseModelPath


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def register_activation_hook(model):
    activations = {}
    for name, param in model.named_modules():
        param.name = name
        def _hook(module, __, val):
            activations[module.name] = val
        param.register_forward_hook(_hook)
    return activations


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, xy_only=False):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



def masked_token_ce_loss(
    logits,
    labels,
    mask
):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # apply mask
    # shifted masks
    shift_logit_mask = mask[..., :-1].contiguous()
    expanded_mask = shift_logit_mask.unsqueeze(-1).expand(-1, -1, shift_logits.size(-1))
    shift_label_mask = mask[..., 1:].contiguous()
    shift_logits = shift_logits * expanded_mask
    shift_labels = shift_labels * shift_label_mask
    shift_labels[shift_labels == 0] = -100
    shift_labels = shift_labels.type(torch.LongTensor)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(DEVICE), shift_labels.view(-1).to(DEVICE))
    return loss


def rep_noise_loss(
    model, harmful_batch, harmless_batch,
    beta=0.001, alpha=1
):
    """ Calculate the representation noise loss

    Args:
        model (Pytorch Model): The model to calculate the loss, i.e. an LLM loaded with Huggingface Transformers
        harmful_batch (Dataloader): the paired harmful batch
        harmless_batch (Dataloader): the paired harmless batch
        beta (float, optional): _description_. Defaults to 0.001.
        alpha (int, optional): _description_. Defaults to 1.
    """
    mmd_loss = MMD_loss()
    activations = register_activation_hook(model)




    harmful_batch, harmless_batch = truncate_and_pad_two_batches(harmful_batch, harmless_batch, pad_token_id = tokenizer.pad_token_id)


    harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
    harmful_activations = []
    for i in range(len(model.base_model.layers)):
        harmful_activations.append(activations[f'model.layers.{i}.mlp'])
    mask = ~torch.eq(harmful_batch['input_ids'], harmless_batch['input_ids'])
    noise_loss = 0
    for i, hidden in enumerate(harmful_activations):
        hiddens_mask = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.device)
        hiddens = hidden * hiddens_mask
        gaussian = torch.randn_like(hiddens).to(hidden.device) * hiddens_mask
        noise_loss += mmd_loss(hiddens.view(hiddens.size(0), -1), gaussian.view(gaussian.size(0), -1)).to(DEVICE)
    noise_loss /= len(harmful_activations) # len(layer_idxs)

    mask = mask.float().to(DEVICE)
    harmful_outputs_loss = masked_token_ce_loss(
        harmful_outputs.logits,
        harmful_batch['input_ids'],
        mask
    )
    harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
    harmless_outputs_loss = masked_token_ce_loss(
        harmless_outputs.logits,
        harmless_batch['input_ids'],
        mask
    )

    harmless_losses = harmless_outputs_loss

    output_embeddings = model.get_output_embeddings()
    norm = model.base_model.norm

    harmful_losses = harmful_outputs_loss
    for i, h in enumerate(harmful_outputs.hidden_states):
        out = output_embeddings(norm(h))
        loss = masked_token_ce_loss(
            out.to(DEVICE), harmful_batch['input_ids'].to(DEVICE),
            mask
        )
        harmful_losses += loss
    harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1
    loss = harmless_losses + beta * noise_loss - alpha * torch.log(harmful_losses)
    return loss





def run_repnoise(
    model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    ga_config = vars(args)
    print("====Rep Noising Config====")
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
        model = get_peft_model(model, lora_config)

    model = model.train()
    if args.layer_ids == [-1]:
        params = model.parameters()
        print("sanity check, we should see this")
    else:
        params = get_params(model, args.layer_ids, args.param_ids)
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
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len
                ).to(model.device)


                # Retain loss
                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len
                ).to(model.device)

                loss = rep_noise_loss(model, unlearn_inputs, retain_inputs, beta = args.beta, alpha = args.alpha)

                # loss = loss / accumulation_steps # Normalize loss to account for batch accumulation

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (idx + 1) % accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                if args.layer_ids==[-1]:
                    print(f"loss: {loss.item():.4g}")
                else:
                    print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

                pbar.update(1)
        tokenizer.truncation_side = truncation_side
        path = RepNoiseModelPath.create_path(args, epoch+1)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"Saved model to {path}")



def get_args():

    import argparse

    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
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
    parser.add_argument("--layer_ids", type=str, default="-1", help="update layers")
    parser.add_argument("--param_ids", type=str, default="-1", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--verbose", action="store_true", help="Logging the activations norms and cosine at each step")

    parser.add_argument("--beta", type=float, default=0.001, help="")
    parser.add_argument("--alpha", type=float, default=1, help="")

    parser.add_argument("--lora", action="store_true", help="Using LoRA")
    parser.add_argument("--lora_r", type=int, default=64, help="")
    parser.add_argument("--lora_alpha", type=int, default=16, help="")
    parser.add_argument("--lora_dropout", type=int, default=0.05, help="")

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

    model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_repnoise(
        model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
