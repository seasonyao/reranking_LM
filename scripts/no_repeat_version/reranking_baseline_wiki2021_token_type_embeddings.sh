#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/not_repeat/canNUM100_token_type_embeddings/%j.out
#SBATCH --error=./log/not_repeat/canNUM100_token_type_embeddings/%j.error

python -u scripts/no_repeat_version/reranking_baseline_wiki2021_token_type_embeddings.py
