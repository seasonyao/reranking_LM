#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/reranking_baseline_wiki2021_prior_probability_sharing_head/%j.out
#SBATCH --error=./log/reranking_baseline_wiki2021_prior_probability_sharing_head/%j.error

python -u scripts/reranking_baseline_wiki2021_prior_probability_sharing_head.py
