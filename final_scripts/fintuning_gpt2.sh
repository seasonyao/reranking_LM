#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./final_log/can_num20_1/fintuning_gpt2_lr1e4_bs64_rp20_%j.out
#SBATCH --error=./final_log/can_num20_1/fintuning_gpt2_lr1e4_bs64_rp20_%j.error

python -u final_scripts/fintuning_gpt2.py

