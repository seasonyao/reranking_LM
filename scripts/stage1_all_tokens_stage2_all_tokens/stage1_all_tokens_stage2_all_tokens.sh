#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/stage1_all_tokens_stage2_all_tokens/lr1e5_bs8_replace_%j.out
#SBATCH --error=./log/stage1_all_tokens_stage2_all_tokens/lr1e5_bs8_replace_%j.error

python -u scripts/stage1_all_tokens_stage2_all_tokens/stage1_all_tokens_stage2_all_tokens.py
