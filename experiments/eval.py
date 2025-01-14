import json
import csv
import os
import sys
import argparse
from typing import Optional
import numpy as np
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from utils.utils import load_model
from utils.jailbreak_utils import load_jb_dataset, evaluate_jailbreak_robustness
from config.eval_config import EvalConfig
from config.prefixes import PREFIX_DICT

sys.path.append(".")
sys.path.append("..")


def get_suffix_or_prefix(model_path: str, root_path: str) -> Optional[str]:
    """
    Retrieve the best performing suffix or prefix for a given model from saved results.
    """
    model_name = model_path.split("/")[-1]
    file_path = None
    for root, _, files in os.walk(root_path):
        for filename in files:
            if filename.startswith(model_name) and filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                break

    if file_path is None:
        raise ValueError(f"No suffix data found in {root_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    controls = data.get('controls', [])
    losses = data.get('losses', [])

    if not controls or not losses or len(controls) != len(losses):
        raise ValueError(f"Invalid data in {file_path}")

    return min(zip(losses, controls), key=lambda x: x[0])[1]


def evaluate_model(model_path: str, args, config: EvalConfig) -> None:
    """
    Evaluate a single model and save results.
    """

    output_dir = config.get_output_path(args.type, args.attack, args.attack_setting)

    os.makedirs(output_dir, exist_ok=True)

    save_name = model_path.split("/")[-1]
    result_file = f"{output_dir}/{save_name}.txt"

    if os.path.exists(result_file):
        print(f"Result {args.type} file for {save_name} already exists.")
        return

    if any(skip in save_name for skip in args.skip if args.skip != [""]):
        print(f"Skipping - manually skipping {save_name}")
        return

    print(f"Evaluating {args.type} for {save_name}")
    model, tokenizer = load_model(model_path)

    suffix_or_prefix = None
    if args.attack_setting:
        if args.type in PREFIX_DICT and args.attack_setting in PREFIX_DICT[args.type]:
            suffix_or_prefix = PREFIX_DICT[args.type][args.attack_setting]["prefix"]
        else:
            root_path = f"{config.attack_results_dirs[args.attack]}/{args.attack_setting}"
            suffix_or_prefix = get_suffix_or_prefix(model_path, root_path)
    else:
        root_path = f"{config.attack_results_dirs[args.attack]}/{args.attack}_1hr"
        suffix_or_prefix = get_suffix_or_prefix(model_path, root_path)

    if suffix_or_prefix is None:
        return

    if args.type == "unlearning":
        _evaluate_unlearning(model, tokenizer, suffix_or_prefix, save_name, output_dir, config)
    elif args.type == "refusal":
        _evaluate_refusal(model, tokenizer, suffix_or_prefix, args, save_name, output_dir)


def _evaluate_unlearning(model, tokenizer, suffix_or_prefix: str,
                        save_name: str, output_dir: str, config: EvalConfig) -> None:
    eval_model = HFLM(model, tokenizer=tokenizer)
    task = (config.eval_types['unlearning']['task_with_prefix']
            if suffix_or_prefix else config.eval_types['unlearning']['task'])

    wmdp_eval_results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=[task],
        device='cuda',
        verbosity='ERROR',
        prefix=suffix_or_prefix,
        num_fewshot=0
    )

    eval_result = wmdp_eval_results['results'][task]
    with open(f"{output_dir}/{save_name}.txt", "a", encoding='utf-8') as file:
        file.write(f"{task}: {eval_result}\n")


def _evaluate_refusal(model, tokenizer, suffix_or_prefix: str,
                     args, save_name: str, output_dir: str) -> None:
    _, eval_jb_dataset = load_jb_dataset(
        eval_size=args.refusal_eval_size,
        seed=args.seed,
        suffix=suffix_or_prefix
    )

    print(f"Loaded refusal jailbreak dataset (size={args.refusal_eval_size}, suffix={suffix_or_prefix})")
    print("Strong reject jailbreak evaluation for refusal")

    jb_result = evaluate_jailbreak_robustness(model, tokenizer, eval_jb_dataset)
    mean_score = np.mean(jb_result['score'])

    with open(f"{output_dir}/{save_name}.txt", "a", encoding='utf-8') as file:
        file.write(f"LAT-harmful seed={args.seed}, n={args.refusal_eval_size}: {mean_score}\n")

    jb_result.to_csv(f"{output_dir}/{save_name}_jb_result.csv", index=False)


def main(args):
    config = EvalConfig()

    folders = []
    if not args.model:
        with open(f'data/{args.type}_model_paths.csv', 'r', encoding='utf-8') as f:
            folders.extend(row[0] for row in csv.reader(f))
    else:
        folders.append(args.model)
    print(f"{len(folders)} models to evaluate")
    for model_path in folders:
        evaluate_model(model_path, args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="",
                       choices=["unlearning", "refusal"],
                       help='type of evaluation')
    parser.add_argument('--attack', type=str, default="",
                       choices=["gcg", "autoprompt, beast"],
                       help='attack of attack')
    parser.add_argument('--attack_setting', type=str, default="",
                       help='type of attack (e.g., gcg_1, gcg_500steps )')
    parser.add_argument('--model', type=str, default="",
                       help='model path for eval')
    parser.add_argument('--skip', type=str, default="",
                       help='models to skip')
    parser.add_argument('--refusal_eval_size', type=int, default=1000,
                       help='number of refusal jailbreak eval examples')
    parser.add_argument('--seed', type=int, default=42,
                       help='seed')
    args = parser.parse_args()
    args.skip = args.skip.split(",")
    main(args)