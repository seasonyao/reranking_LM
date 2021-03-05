#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:2
#SBATCH --partition=rtx8000-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/reranking_baseline_wiki2021_add_candidates_predict_next_word/%j.out
#SBATCH --error=./log/reranking_baseline_wiki2021_add_candidates_predict_next_word/%j.error

python -u scripts/reranking_baseline_wiki2021_add_candidates_predict_next_word.py