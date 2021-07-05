#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./final_log/can_num20/lr1e4_bs32_rp20_no_stage2_%j.out
#SBATCH --error=./final_log/can_num20/lr1e4_bs32_rp20_no_stage2_%j.error

python -u final_scripts/stage1_all_tokens_no_stage2.py

