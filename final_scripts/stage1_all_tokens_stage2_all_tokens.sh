#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./final_log/can_num20_2/lr1e4_bs32_rp20_replace_%j.out
#SBATCH --error=./final_log/can_num20_2/lr1e4_bs32_rp20_replace_%j.error

python -u final_scripts/stage1_all_tokens_stage2_all_tokens.py
