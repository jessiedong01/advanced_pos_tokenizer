#!/usr/bin/env bash
# Vocabulary sweep: 2,000 and 32,000 types, seed 0, for the compared methods.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for lang in en es da ta tr; do
  for v in 2000 32000; do
    for m in bpe pos pos_content l2w hf_bpe unigram; do
      echo "python experiments/run.py --lang $lang --method $m --seed 0 --vocab $v --n-train 3000 > logs/runs/${lang}_${m}_v${v}_s0.log 2>&1"
    done
  done
done | xargs -P 12 -I{} sh -c '{}'
echo "SWEEP DONE"
