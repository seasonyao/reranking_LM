#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx8000-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/not_repeat/canNUM20_randomize_candidates_order/%j.out
#SBATCH --error=./log/not_repeat/canNUM20_randomize_candidates_order/%j.error

python -u scripts/reranking_baseline_wiki2021_randomize_candidates_order.py
