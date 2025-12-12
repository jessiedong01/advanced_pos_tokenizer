#!/usr/bin/env bash
# After the main grid: the vocabulary sweep (parallel, CPU) and the language-model
# evaluations (sequential, GPU) for every main-grid run.
cd "$(dirname "$0")/.."
source .venv/bin/activate
until grep -q "MAIN GRID DONE" logs/run_all.log; do sleep 30; done
experiments/run_sweep.sh > logs/run_sweep.log 2>&1 &
mkdir -p logs/lm
for seed in 0 1 2; do
  for lang in en es da ta tr; do
    for m in bpe pos pos_content l2w hf_bpe unigram pos_trans pos_pen; do
      run="results/$lang/${m}_v8000_s${seed}"
      [ -f "$run/lm.json" ] && grep -q '"epochs": 6.0' "$run/lm.json" && continue
      python experiments/lm_eval.py --run "$run" --epochs 6 > "logs/lm/${lang}_${m}_s${seed}.log" 2>&1
    done
  done
done
wait
echo "AFTER GRID DONE"
