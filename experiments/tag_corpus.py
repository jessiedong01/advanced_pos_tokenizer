#!/usr/bin/env python3
"""Tag the sampled Wikipedia paragraphs with Stanza and align the tags to
whitespace-delimited words.

The tokenizers in this repository split text on whitespace, so every
whitespace word needs one UPOS tag and one lemma. Stanza's own tokens differ
from whitespace words (punctuation is split off, some languages have
multi-word tokens), so each whitespace word takes the tag and lemma of the
first non-punctuation Stanza word that starts inside it, or PUNCT if there is
none.

Output: data/tagged/<lang>.<split>.jsonl, one paragraph per line with
  {"text": ..., "words": [[surface, upos, lemma, start, end], ...]}
"""
import argparse, json, re, sys, time
from pathlib import Path
import stanza
from stanza.models.common.doc import Document

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\S+")

def align(text, doc):
    starts = {}
    for sent in doc.sentences:
        for tok in sent.tokens:
            for w in tok.words:
                starts.setdefault(tok.start_char, []).append((w.upos, w.lemma or w.text))
    words = []
    for m in WORD_RE.finditer(text):
        tags = [t for pos in range(m.start(), m.end()) for t in starts.get(pos, [])]
        chosen = next(((u, l) for u, l in tags if u != "PUNCT"), tags[0] if tags else ("X", m.group()))
        words.append([m.group(), chosen[0], chosen[1], m.start(), m.end()])
    return words

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()
    nlp = stanza.Pipeline(args.lang, processors="tokenize,pos,lemma", use_gpu=False, verbose=False,
                          tokenize_no_ssplit=False, pos_batch_size=3000, lemma_batch_size=3000)
    out_dir = ROOT / "data" / "tagged"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        paras = (ROOT / "data" / "corpus" / f"{args.lang}.{split}.txt").read_text(encoding="utf-8").splitlines()
        t0 = time.time()
        with open(out_dir / f"{args.lang}.{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(0, len(paras), args.batch):
                batch = paras[i:i + args.batch]
                docs = nlp.bulk_process([Document([], text=p) for p in batch])
                for text, doc in zip(batch, docs):
                    f.write(json.dumps({"text": text, "words": align(text, doc)}, ensure_ascii=False) + "\n")
                print(f"[{args.lang} {split}] {min(i + args.batch, len(paras))}/{len(paras)} paragraphs, {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
