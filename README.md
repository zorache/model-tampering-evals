# Code for "Model Tampering Attacks Enables More Rigorous Evaluations of LLM Capabilities"

[workshop paper](https://openreview.net/pdf?id=XmvgWEjkhG) appeared at the NeurIPS Safe GenAI workshop

We release a suite of unlearned models on [Hugging Face](https://huggingface.co/collections/LLM-GAT/) across 8 checkpoints for the 10 unlearning methods we benchmarked.


NOTE: this codebase is currently being updated.


### Setup
```
conda create -n tamper-evals python=3.11
conda activate tamper-evals
pip install -r requirements.txt
```

Note that for jailbreak evaluation, we use the StrongREJECT rule-based autograder that uses an OpenAI API key. For evaluations with open source model, please [see](https://github.com/alexandrasouly/strongreject/tree/main?tab=readme-ov-file).


