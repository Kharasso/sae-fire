#!/usr/bin/env bash
set -euo pipefail

FEATURE_ROOT="data/doc_features"
OUT_ROOT="results/predictions"

SAE_VARIANTS=("sae_2b" "sae_9b_16k" "sae_9b_131k")
CLS_VARIANTS=("cls_gemma_2b" "cls_gemma_9b" "cls_qwen_4b" "cls_llama_3b")
METHODS=("none" "anova" "tree")
MODELS=("rf" "gb" "lr")
TEST_SIZE=0.2

# SAE experiments
for sae in "${SAE_VARIANTS[@]}"; do
  for method in "${METHODS[@]}"; do
    for model in "${MODELS[@]}"; do
      in_dir="$FEATURE_ROOT/sae/$sae"
      out_dir="$OUT_ROOT/sae/${sae}/${method}/${model}"
      echo "→ $model with $method on $sae"
      python classifiers/train_classifier.py \
        --feature-dir "$in_dir" \
        --out-dir "$out_dir" \
        --selection "$method" \
        --k 1000 \
        --threshold median \
        --model "$model" \
        --test-size $TEST_SIZE
    done
  done
done

# CLS experiments
for cls in "${CLS_VARIANTS[@]}"; do
  for method in "${METHODS[@]}"; do
    for model in "${MODELS[@]}"; do
      in_dir="$FEATURE_ROOT/cls/$cls"
      out_dir="$OUT_ROOT/cls/${cls}/${method}/${model}"
      echo "→ $model with $method on $cls"
      python classifiers/train_classifier.py \
        --feature-dir "$in_dir" \
        --out-dir "$out_dir" \
        --selection "$method" \
        --k 1000 \
        --threshold median \
        --model "$model" \
        --test-size $TEST_SIZE
    done
  done
done