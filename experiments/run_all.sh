#!/usr/bin/env bash
# Main grid: five languages x eight methods x three seeds at an 8,000-type budget.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs/runs
for lang in en es da ta tr; do
  for seed in 0 1 2; do
    for m in bpe pos pos_trans pos_pen pos_content l2w hf_bpe unigram; do
      echo "python experiments/run.py --lang $lang --method $m --seed $seed --vocab 8000 --n-train 3000 > logs/runs/${lang}_${m}_v8000_s${seed}.log 2>&1"
    done
  done
done | xargs -P 12 -I{} sh -c '{}'
echo "MAIN GRID DONE"
