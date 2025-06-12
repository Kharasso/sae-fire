#!/usr/bin/env bash
set -euo pipefail

YEARS=(2012 2013 2014)
ORDERS=(1 2)
CLS_IDS=(cls_gemma_2b cls_gemma_9b cls_qwen_4b cls_llama_3b)

for cls in "${CLS_IDS[@]}"; do
  for year in "${YEARS[@]}"; do
    for order in "${ORDERS[@]}"; do
      echo "→ CLS features for $cls ${year}_${order}"
      python -m runners.run_cls \
        --jsonl data/train_test_data/transcript_componenttext_${year}_${order}.jsonl \
        --meta  data/train_test_data/transcript_metadata_${year}_${order}.csv \
        --out   data/doc_features/cls/${cls}/${year}_${order} \
        --cls-id "$cls" \
        --flush 100
    done
  done
done