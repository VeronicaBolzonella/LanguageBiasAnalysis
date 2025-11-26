#!/bin/bash


source /vol/tensusers/vbolzonella/txmm/LanguageBiasAnalysis/.venv/bin/activate

cd /vol/tensusers/vbolzonella/txmm/LanguageBiasAnalysis

python3 bias_analysis/bias_identification.py embeddings_local.kv gender adj
python3 bias_analysis/bias_identification.py embeddings_local.kv gender professions
python3 bias_analysis/bias_identification.py embeddings_local.kv gender animals

python3 bias_analysis/bias_identification.py embeddings_local.kv age adj
python3 bias_analysis/bias_identification.py embeddings_local.kv age professions
python3 bias_analysis/bias_identification.py embeddings_local.kv age animals

python3 bias_analysis/bias_identification.py embeddings_local.kv class adj
python3 bias_analysis/bias_identification.py embeddings_local.kv class professions
python3 bias_analysis/bias_identification.py embeddings_local.kv class animals