#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=2080ti-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/reranking_baseline_wiki2021_exclude_cases_label_not_in_candidates/%j.out
#SBATCH --error=./log/reranking_baseline_wiki2021_exclude_cases_label_not_in_candidates/%j.error

python -u scripts/reranking_baseline_wiki2021_exclude_cases_label_not_in_candidates.py
