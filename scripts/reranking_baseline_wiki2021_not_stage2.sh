#!/bin/bash

#SBATCH --job-name=rerank2021
#SBATCH --gres=gpu:4
#SBATCH --partition=titanx-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/canNUM20_not_stage2/%j.out
#SBATCH --error=./log/canNUM20_not_stage2/%j.error

python -u scripts/reranking_baseline_wiki2021_not_stage2.py
