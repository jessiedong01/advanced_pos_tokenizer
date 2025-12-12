#!/usr/bin/env bash
# Tag every language as soon as its corpus and Stanza model exist; three threads each.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for lang in en es da ta tr; do
  (
    until [ -s "data/corpus/$lang.test.txt" ] && grep -q "downloaded $lang" logs/stanza_download.log; do sleep 20; done
    python experiments/tag_corpus.py --lang "$lang" > "logs/tag_$lang.log" 2>&1
  ) &
done
wait
echo "ALL TAGGED"
