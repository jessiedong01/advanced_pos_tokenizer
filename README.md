# POS-Aware Tokenizer

We built a tokenizer that uses morphology and basic syntax. It adds POS-aware costs to the merge schedule so the model prefers boundaries that match grammar, not only frequency.

## What this repo has
- `src/pos_preprocess.py` — tag a corpus with UPOS using Stanza (multilingual).
- `src/pos_transition.py` — compute POS transition probabilities and convert to reward factors.
- `src/tokenizer_models.py` — a compact BPE-style tokenizer with an option to use POS signals at word boundaries. Includes ablations.
- `src/train_tokenizers.py` — trains baseline and POS-aware tokenizers with the same settings.
- `src/evaluate_tokenizers.py` — metrics that compare vocab statistics and syntactic alignment.
- `src/run_experiments.py` — batch runner for English, Spanish, and Turkish. Saves CSVs to `results/`.
- `src/utils/metrics.py` and `src/utils/visualize.py` — helper code.
- `notebooks/pos_analysis.ipynb` — optional plots.

## Quick start
```bash
pip install -r requirements.txt

# 1) Tag corpus (downloads Stanza models on first use)
python src/pos_preprocess.py data/train_en.txt data/en_pos.json en

# 2) Transitions
python src/pos_transition.py data/en_pos.json data/en_transitions.json

# 3) Train both models
python src/train_tokenizers.py   --corpus data/train_en.txt   --pos_json data/en_pos.json   --pos_transitions data/en_transitions.json   --vocab_size 32000   --outdir results/en_models

# 4) Evaluate
python src/evaluate_tokenizers.py   --corpus data/train_en.txt   --pos_json data/en_pos.json   --models_dir results/en_models   --report results/en_report.csv
```

## Experiments at a glance
- Same corpus, same vocab size, same loop. The only change is the POS factor.
- We report: vocab overlap, avg token length, token length entropy, POS-alignment rate, and a small qualitative table.
- `run_experiments.py` repeats this for English, Spanish, and Turkish.

## Notes
- POS is cached offline; no tagging during training.
- Factors come from your corpus’s own transition stats, not hard-coded rules.
- Future work: context-aware POS at decode time, and regularization to avoid over-rewarding common transitions.
