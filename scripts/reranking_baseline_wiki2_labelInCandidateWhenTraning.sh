#!/bin/bash

#SBATCH --job-name=rerank2
#SBATCH --gres=gpu:2
#SBATCH --partition=m40-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --output=./log/reranking_baseline_wiki2_labelInCandidateWhenTraning/%j.out
#SBATCH --error=./log/reranking_baseline_wiki2_labelInCandidateWhenTraning/%j.error

python -u scripts/reranking_baseline_wiki2_labelInCandidateWhenTraning.py