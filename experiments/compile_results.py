import os
import csv
import ast
import argparse
import pandas as pd
from config.results_config import EVALUATION_MAPPINGS



def process_file(file_path, directory_path):
    with open(os.path.join(directory_path,file_path), 'r') as file:
        lines = file.readlines()
    wmdp_bio_results = []
    mmlu_results = []
    lat_harmful_results = []

    for line in lines:
        try:
            if 'LAT' in line:
                key, value = line.strip().split(': ', 1)
                accuracy = float(value)
                lat_harmful_results.append(accuracy)
            else:
                key, value = line.strip().split(': ', 1)
                data = ast.literal_eval(value)
                if isinstance(data, dict):
                    if data.get('alias') == 'wmdp_bio':
                        wmdp_bio_results.append(data['acc,none'])
                    elif data.get('alias') == 'mmlu':
                        mmlu_results.append(data['acc,none'])
        except (ValueError, SyntaxError, KeyError):
            continue

    return {
        'file_name': os.path.basename(file_path),
        'wmdp_bio': wmdp_bio_results[-1] if wmdp_bio_results else None,
        'mmlu': mmlu_results[-1] if mmlu_results else None,
        'LAT-harmful seed=42, n=1000': lat_harmful_results[-1] if lat_harmful_results else None
    }

def create_summary(directory_path, output_path):
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    if not subdirectories:
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        results = [process_file(file, directory_path) for file in files]
        df = pd.DataFrame(results)
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f'{output_path}/summary.csv', index=False)
        return df
    else:
        all_results = []
        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(directory_path, subdirectory)
            files = [f for f in os.listdir(subdirectory_path) if f.endswith('.txt')]
            results = [process_file(file, subdirectory_path) for file in files]
            df = pd.DataFrame(results)
            os.makedirs(os.path.join(output_path, subdirectory), exist_ok=True)
            df.to_csv(f'{output_path}/{subdirectory}/summary.csv', index=False)
            all_results.extend(results)
        return pd.DataFrame(all_results)


def extract_and_combine_summaries(root_dir, mapping_dict, model_prefix, output_file):
    # If strings in model_prefix have '/', split and take the last element
    if "/" in model_prefix[0]:
        model_prefix = [prefix.split('/')[-1] for prefix in model_prefix]

    # Initialize DataFrame with model_prefix as index to maintain order
    combined_df = pd.DataFrame(index=model_prefix)

    for column_name, sub_dir in mapping_dict.items():
        # Construct the path to the summary.csv file
        if "baseline" in column_name:
            summary_path = os.path.join(sub_dir,'summary.csv')
        else:
            summary_path = os.path.join(root_dir, sub_dir, 'summary.csv')
        try:
            df = pd.read_csv(summary_path, index_col=0)
        except FileNotFoundError:
            print(f"Summary file not found at {summary_path}, creating it...")
            results_dir = os.path.dirname(summary_path)
            df = create_summary(results_dir, results_dir)
            df.to_csv(summary_path,index=False)
            print(f"Created summary file at {summary_path}")
            df = pd.read_csv(summary_path, index_col=0)


        filtered_df = df[df.index.to_series().str.startswith(tuple(model_prefix))]


        # Extract values while maintaining the order from model_prefix
        values = []
        for prefix in model_prefix:
            matching_rows = filtered_df[filtered_df.index.str.startswith(prefix)]

            values.append(matching_rows.iloc[0] if not matching_rows.empty else None)

        # Add the column with preserved order
        if "unlearning" in column_name:
            if "finetune" in sub_dir:
                combined_df[column_name] = [v['wmdp_bio'] if v is not None else None for v in values]
            else:
                combined_df[column_name] = [v['wmdp_fewshot_bio_balanced_results'] if v is not None else None for v in values]
        elif "refusal" in column_name:
            combined_df[column_name] = [v['LAT-harmful seed=42, n=1000'] if v is not None else None for v in values]


    print(combined_df)
    combined_df.to_csv(output_file)



def main(args):
    model_prefix = []
    if not args.model:
        with open(f'../data/{args.type}_model_paths.csv', 'r', encoding='utf-8') as f:
            model_prefix.extend(row[0] for row in csv.reader(f))
    else:
        model_prefix= args.model.split(",")
    print(f"{len(model_prefix)} models to create table for")


    extract_and_combine_summaries(args.root_dir, EVALUATION_MAPPINGS, model_prefix, args.output_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="refusal",
                       choices=["unlearning", "refusal"],
                       help='type of evaluation')
    parser.add_argument('--model', type=str, default="",
                       help='model path for table')
    parser.add_argument('--root_dir', type=str, default="../attack_results",
                       help='root directory path')
    parser.add_argument('--output_file', type=str, default="output.csv",
                       help='output file name')
    args = parser.parse_args()
    main(args)

