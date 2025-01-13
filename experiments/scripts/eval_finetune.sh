#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=single
#SBATCH --gpus-per-node=1
#SBATCH --time=3:30:00
#SBATCH --job-name eval-f


python attacks/tampering/finetune.py \
    --type unlearning \
    --data_list_1 bio-remove-dataset \
    --num_examples 400 \
    --batch_size 8 \
    --grad_acc 1 \
    --layer_ids -1 \
    --lr 5e-5 \
    --lora  \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_epoch 1