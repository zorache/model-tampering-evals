from datasets import load_dataset
import torch
import os
from strong_reject.evaluate import evaluate_dataset
EVALUATOR = "strongreject_rubric"
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")


def format_text(prompt, response=None):
    # Uses the llama3-8b-it template
    # https://pytorch.org/torchtune/0.2/tutorials/chat.html
    system_prompt = "You are a helpful, respectful, and honest assistant."
    formatted_text =  (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}"
        "<|eot_id|>\n"

        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|eot_id|>\n"

        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if response is not None:
        formatted_text += f"{response}<|eot_id|>"
    return formatted_text

def load_jb_dataset(eval_size=1000, seed=42, include_test_responses=False, batch_size=None, rejected_pair=False, suffix=""):
    # Load the dataset from Hugging Face
    dataset = load_dataset("LLM-LAT/harmful-dataset")

    # Set random seed for reproducibility
    dataset = dataset.shuffle(seed=seed)

    # Split the dataset
    train_dataset = dataset["train"].select(range(64))
    if eval_size is None:
        eval_dataset = dataset["train"].select(range(64, len(dataset["train"])))
    else:
        eval_dataset = dataset["train"].select(range(64, eval_size + 64))

    def add_suffix(example, sfx=""):
        example["prompt"] = example["prompt"] + " "+ sfx
        return example

    eval_dataset_with_suffix = eval_dataset.map(lambda x: add_suffix(x, sfx=suffix))

    def format_example(example, response=True):
        rejected_response = example['rejected']
        chosen_response = example['chosen']
        if response:
            return {
                "formatted_rejected": format_text(example['prompt'], rejected_response),
                "formatted_chosen": format_text(example['prompt'], chosen_response),
                "formatted_prompt": format_text(example['prompt'])
            }
        else:
            return {"formatted_prompt": format_text(example['prompt'])}

    # Create train and eval datasets
    train_dset = train_dataset.map(lambda x: format_example(x, response=include_test_responses))
    eval_dset = eval_dataset_with_suffix.map(lambda x: format_example(x))

    # Option to return batched dataset
    if batch_size is not None:
        if rejected_pair:
            train_dset = [train_dset['formatted_rejected'][i:i + batch_size] for i in range(0, len(train_dset), batch_size)]
        else:
            train_dset = [train_dset['formatted_prompt'][i:i + batch_size] for i in range(0, len(train_dset), batch_size)]

    return train_dset, eval_dset



def load_ultrachat(seed=42,train_size=64,batch_size=None):

    # Load the dataset from Hugging Face
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

    # Set random seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset["train_sft"].select(range(train_size))
    

    # function to format examples into templated prompt+response format
    def format_example(example):
        assert example['messages'][0]['role'] == 'user'
        assert example['messages'][1]['role'] == 'assistant'
        return {"formatted_prompt_response": format_text(example['messages'][0]['content'],
                                                         example['messages'][1]['content'])}

    # Create formatted dataset
    dataset = dataset.map(lambda x: format_example(x))
    if batch_size is not None:
        dataset = [dataset['formatted_prompt_response'][i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    

    return dataset

def evaluate_jailbreak_robustness(model, tokenizer, dataset, batch_size=16):
    responses = []

    # dataset = dataset.select(range(num_examples))
    for i in range(0, len(dataset), batch_size):
        batch_prompts = dataset['formatted_prompt'][i:i + batch_size]
        # Tokenize prompts in the batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=300)  # Adjust max_length as needed
        # Decode responses and add them to the list
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        responses.extend(batch_responses)
    
    responses = [extract_assistant_response(r) for r in responses]

    dataset = dataset.add_column("response", responses)
    dataset = dataset.add_column("forbidden_prompt", dataset['prompt'])

    eval_dataset = evaluate_dataset(dataset, [EVALUATOR])
    return eval_dataset  # np.mean(eval_dataset['score']) will give the average compliance: 0 is robust, 1 is compliant

def extract_assistant_response(text):
        try:
            # Split by the assistant marker and take the last part
            parts = text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            if len(parts) < 2:
                return ""
            # Split by EOT marker and take the content
            response = parts[-1].split("<|eot_id|>")[0].strip()
            return response
        except Exception:
            return ""  # Fallback in case of parsing errors
