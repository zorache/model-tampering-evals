#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=single
#SBATCH --gpus-per-node=1
#SBATCH --time=3:30:00
#SBATCH --job-name eval-s


python eval.py --type "refusal" --attack "gcg" --attack_setting "gcg_100steps"