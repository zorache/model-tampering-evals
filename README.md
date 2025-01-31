# Model Tampering Attacks Enables More Rigorous Evaluations of LLM Capabilities

[workshop paper](https://openreview.net/pdf?id=XmvgWEjkhG) appeared at the NeurIPS Safe GenAI workshop

We release a suite of unlearned models on [Hugging Face](https://huggingface.co/collections/LLM-GAT/) across 8 checkpoints for the 10 unlearning methods we benchmarked.


NOTE: this codebase is currently being updated.




### Setup
```
conda create -n tamper-evals python=3.11
conda activate tamper-evals

git clone https://github.com/zorache/model-tampering-evals.git
cd model-tampering-evals

git submodule init
git submodule update


conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia  # change this line based on your cuda requirements for your compute set up

pip install -e .
```

Note that for jailbreak evaluation, we use the StrongREJECT rule-based autograder that uses an OpenAI API key. For evaluations with open source model, please [see](https://github.com/alexandrasouly/strongreject/tree/main?tab=readme-ov-file).

To add your API key
```
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```


### Unlearning Models
We release a suite of Llama-3-8B-Instruct models unlearned on WMDP Bio Remove data on [Hugging Face](https://huggingface.co/collections/LLM-GAT/) across 8 checkpoints for the 10 unlearning methods we benchmarked. We include training scripts for some of the models in `unlearn/methods`.

### Refusal Models
We attack 8 public refusal trained Llama-3-8B-Instruct models.

### Attacking Models and evaluating attack success
Before attack, please update or populate `data/refusal_model_paths.csv` and `data/unlearning_model_paths.csv` respectively. For all attacks except fine-tuning, we find universal strings either as prefix or suffix as attacks. We use a set of 64 examples to find the best prefix and suffix. We include the refusal inputs used in `data/refusal_data_input.csv`.

Each input-space attack can be run within its subfolder in `attacks`. After adversarial strings have been found, evaluate the efficacy of the strings by calling `experiments/scripts` for the respective attacks.

Finetuning attack can be run and evaluated in `experiments/scripts/eval_finetune.sh`

Results for models acrossed different attacks can be compiled by running `experiments/compile_results.py`. Choose the columns to include in the results table in `config/results_config.py`
