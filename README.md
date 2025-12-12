# Part-of-Speech and Lemma Information in Subword Tokenization

Byte-pair encoding and the unigram language model build their vocabularies from character statistics alone. This repository contains the code, the data preparation and the paper for a controlled evaluation of tokenizers that also use the part-of-speech tags and lemmas a standard tagger provides, in English, Spanish, Danish, Tamil and Turkish. Weighting BPE merges by the probability of the part-of-speech transition at each word boundary leaves boundary accuracy and compression unchanged, weighting merges inside content words has an effect that depends on the language, and Lemma2Word, which segments each word at the boundary its lemma implies, raises boundary recall by 23 to 77 points at a cost of four to eight percent more tokens. A language-model probe orders the tokenizers differently, with the unigram model lowest in bits per character and Lemma2Word highest in every language.

## Paper

[Part-of-Speech and Lemma Information in Subword Tokenization: A Controlled Evaluation in Five Languages](paper/main.pdf) (`paper/main.pdf`; the LaTeX source is in `paper/`).

## Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/fetch_corpus.py     # Wikipedia paragraphs
experiments/tag_all.sh                           # Stanza tags and lemmas
experiments/run_all.sh                           # 5 languages x 8 tokenizers x 3 seeds
experiments/run_sweep.sh                         # 2,000 and 32,000 types
experiments/lm_queue.sh                          # language-model probes
.venv/bin/python experiments/llm_segment.py      # segmentation study (needs ANTHROPIC_API_KEY)
.venv/bin/python experiments/aggregate.py        # paper/numbers.tex, tables, figures
```

The metrics of every run are committed under `results/`, so the numbers in the paper can be checked without retraining. The gold morpheme boundaries come from MorphScore (Arnett, Hudspeth and O'Connor, 2025) and the text from the Wikipedia snapshot of 1 November 2023, both through Hugging Face datasets.
