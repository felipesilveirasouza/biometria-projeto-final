#!/usr/bin/env bash
# Reproduz todos os resultados do artigo, na ordem. Ver README.md.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Protocolos oficiais"
python make_protocols.py

echo "== Pré-processamento (alinhado por landmarks)"
python preprocess.py --dataset cfp_fp
python preprocess.py --dataset lfw

echo "== Embeddings (+ eficiência medida na base principal)"
python extract_embeddings.py --dataset cfp_fp --timing
python extract_embeddings.py --dataset lfw

echo "== Experimento A: base principal (CFP-FP, 10 folds)"
python evaluate.py --dataset cfp_fp

echo "== Experimento B: segunda base com protocolo próprio (LFW, 10 folds)"
python evaluate.py --dataset lfw

echo "== Experimento C: cross-dataset (threshold congelado do CFP-FP aplicado ao LFW)"
python evaluate.py --dataset lfw --frozen-from cfp_fp_aligned --tag cross_cfp_fp_to_lfw

echo "== Ablação: sem alinhamento (só bbox)"
python preprocess.py --dataset cfp_fp --no-align
python extract_embeddings.py --dataset cfp_fp --variant bbox
python evaluate.py --dataset cfp_fp --variant bbox

echo "== Experimento D: robustez (blur e oclusão, 4 níveis)"
python robustness.py

echo "== Análise de erros e figuras"
python errors.py --tag cfp_fp_aligned
python errors.py --tag cross_cfp_fp_to_lfw
python figures.py --dataset cfp_fp
python figures.py --dataset lfw

echo "== Tabela padronizada"
python report.py
