from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalConfig:
    """Configuration for evaluation settings."""

    # Base paths
    base_output_dir: str = "attack_results"
    #attack_results_dir: str = "/data/zora_che/llm-attacks/experiments/results"


    attack_results_dirs= {
        "gcg": "attacks/input_space/llm_attacks/experiments/results/gcg/",
        "autoprompt": "attacks/input_space/llm_attacks/experiments/results/autoprompt",
        "beast":""
    }

    # Evaluation types
    eval_types = {
        "unlearning": {
            "base_path": "baseline_balanced_remaining_split",
            "task": "wmdp_bio",
            "task_with_prefix": "wmdp_fewshot_bio_balanced"
        },
        "refusal": {
            "base_path": "baseline",
            "result_prefix": "LAT-harmful"
        }
    }

    def get_output_path(self, eval_type: str, attack_method: Optional[str] = None,
                       attack_setting: Optional[str] = None) -> str:
        """
        Generate output directory path based on configuration.
        """
        base = f"{self.base_output_dir}/{eval_type}"

        if not attack_method:
            return f"{base}/{self.eval_types[eval_type]['base_path']}"

        attack_path = f"{base}/input_space/{attack_method}"

        if attack_setting:
            return f"{attack_path}/{attack_setting}"

        return f"{attack_path}/{attack_method}_1hr"