#!/usr/bin/env bash
set -euo pipefail

# SAE FS experiments (LR only)
FEATURE_ROOT="data/doc_features/sae"
OUT_ROOT="results/predictions/sae_fs"
VARIANTS=("sae_2b" "sae_9b_131k")

for var in "${VARIANTS[@]}"; do
  in_dir="$FEATURE_ROOT/$var"
  out_dir="$OUT_ROOT/$var"
  echo "→ SAE FS LR for $var"
  python classifiers/train_sae_fs.py \
    --feature-dir "$in_dir" \
    --out-dir "$out_dir" \
    --test-size 0.2
done

# Baseline experiments
FEATURE_ROOT="data/doc_features"
OUT_ROOT="results/predictions/baseline"

# SAE baseline (MLP + XGBoost)
for var in sae_2b sae_9b_131k; do
  in_dir="$FEATURE_ROOT/sae/$var"
  out_dir="$OUT_ROOT/sae_baseline/$var"
  echo "→ SAE baseline for $var"
  python classifiers/train_baselines.py \
    --feature-dir "$in_dir" \
    --out-dir "$out_dir" \
    --test-size 0.2
done

# CLS baseline (MLP + LR)
for var in cls_gemma_2b cls_gemma_9b cls_qwen_4b cls_llama_3b; do
  in_dir="$FEATURE_ROOT/cls/$var"
  out_dir="$OUT_ROOT/cls_baseline/$var"
  echo "→ CLS baseline for $var"
  python classifiers/train_baselines.py \
    --feature-dir "$in_dir" \
    --out-dir "$out_dir" \
    --test-size 0.2
done