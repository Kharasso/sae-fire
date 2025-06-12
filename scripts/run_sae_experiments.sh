#!/usr/bin/env bash
set -euo pipefail

YEARS=(2012 2013 2014)
ORDERS=(1 2)
SAE_IDS=(sae_2b sae_9b_16k sae_9b_131k)

for sae in "${SAE_IDS[@]}"; do
  for year in "${YEARS[@]}"; do
    for order in "${ORDERS[@]}"; do
      echo "→ SAE features for $sae ${year}_${order}"
      python -m runners.run_sae \
        --jsonl data/train_test_data/transcript_componenttext_${year}_${order}.jsonl \
        --meta  data/train_test_data/transcript_metadata_${year}_${order}.csv \
        --out   data/doc_features/sae/${sae}/${year}_${order} \
        --sae-id "$sae" \
        --flush 100
    done
  done

done