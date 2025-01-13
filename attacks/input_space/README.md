We adapted the LLM-Attacks Codebase for the GCG and AutoPrompt attacks in our experiments


In addition to the original repo, this fork has the addition of:

- An implementation of multi-prompt AutoPrompt attack
- Support for Llama-3 Chat Template
- Support for no chat template
- Support for attacks with the same target for multi-prompt


For unlearning robustness experiments, we conducted prefix finding attacks with no chat formatting
For refusal robustness experiments, we conducted suffix finding attacks with Llama-3 chat formatting

To run attacks, follow the set up of the repo, then

For 1-hr refusal attack
```
cd experiments/launch_scripts
bash run_gcg_time_bound_individual_refusal.sh "MODEL_PATH"
```

For 1-hr unlearning attack
```
cd experiments/launch_scripts
bash run_gcg_time_bound_individual_unlearning.sh "MODEL_PATH"
```

