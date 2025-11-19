#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=output.out

source /vol/tensusers/vbolzonella/txmm/LanguageBiasAnalysis/.venv/bin/activate

cd /vol/tensusers/vbolzonella/txmm/LanguageBiasAnalysis

python3 data/crawler.py 