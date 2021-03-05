#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx8000-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/not_repeat/canNUM20_not_repeat_candidate/%j.out
#SBATCH --error=./log/not_repeat/canNUM20_not_repeat_candidate/%j.error

python -u scripts/reranking_baseline_wiki2021_not_repeat_candidate.py
