#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:2
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/stage1_all_tokens_start_after_finetuning/randomlySamplingNwordsIntoTheCandidate/%j.out
#SBATCH --error=./log/stage1_all_tokens_start_after_finetuning/randomlySamplingNwordsIntoTheCandidate/%j.error

python -u scripts/stage1_all_tokens_start_after_finetuning/randomlySamplingNwordsIntoTheCandidate.py
