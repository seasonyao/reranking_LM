#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_sample_tokens_prior_probability_sharing_head/%j.out
#SBATCH --error=./log/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_sample_tokens_prior_probability_sharing_head/%j.error

python -u scripts/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_stage2_sample_tokens_prior_probability_sharing_head.py
