#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/not_repeat/canNUM100_no_position_embeddings/keep_position_embedding_%j.out
#SBATCH --error=./log/not_repeat/canNUM100_no_position_embeddings/keep_position_embedding_%j.error

python -u scripts/no_repeat_version/reranking_baseline_wiki2021_token_type_embeddings-no_position_embeddings.py
