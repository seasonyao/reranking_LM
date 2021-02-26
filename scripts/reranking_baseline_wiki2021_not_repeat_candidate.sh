#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:2
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/canNUM20_not_repeat_candidate/%j.out
#SBATCH --error=./log/canNUM20_not_repeat_candidate/%j.error

python -u scripts/reranking_baseline_wiki2021_not_repeat_candidate.py
