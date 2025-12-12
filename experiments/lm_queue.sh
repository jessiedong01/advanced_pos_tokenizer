#!/usr/bin/env bash
# Language-model probes for every main-grid run, three workers (one per seed) sharing the GPU.
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs/lm
for seed in 0 1 2; do
  (
    for lang in en es da ta tr; do
      for m in bpe pos pos_content l2w hf_bpe unigram pos_trans pos_pen; do
        run="results/$lang/${m}_v8000_s${seed}"
        [ -f "$run/lm.json" ] && grep -q '"epochs": 6.0' "$run/lm.json" && continue
        python experiments/lm_eval.py --run "$run" --epochs 6 > "logs/lm/${lang}_${m}_s${seed}.log" 2>&1
      done
    done
  ) &
done
wait
echo "LM QUEUE DONE"
