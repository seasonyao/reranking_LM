#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=1080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_all_tokens/rerank_linear_head_0.0002_scrach_lr5e4_%j.out
#SBATCH --error=./log/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_all_tokens/rerank_linear_head_0.0002_scrach_lr5e4_%j.error

python -u scripts/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_all_tokens.py
