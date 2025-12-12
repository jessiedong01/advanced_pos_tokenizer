#!/usr/bin/env python3
"""Sample Wikipedia paragraphs for each language into train / test files.

Articles are streamed from the 20231101 Wikimedia dump. The first articles of
a dump are mostly short meta pages, so an offset is skipped first. Paragraphs
between 200 and 1,500 characters are kept, shuffled with a fixed seed, and
split into a training and a held-out file. Only the training file is ever
tagged or used to train a tokenizer.
"""
import argparse, random, re
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"

def paragraphs(text):
    for p in re.split(r"\n\s*\n|\n", text):
        p = re.sub(r"\s+", " ", p).strip()
        if 200 <= len(p) <= 1500 and not p.startswith(("==", "{{", "|", "*")):
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "es", "da", "ta", "tr"])
    ap.add_argument("--skip", type=int, default=2000)
    ap.add_argument("--articles", type=int, default=6000)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        paras = []
        for i, row in enumerate(ds.skip(args.skip)):
            if i >= args.articles:
                break
            paras.extend(paragraphs(row["text"]))
        rng = random.Random(args.seed)
        rng.shuffle(paras)
        seen, uniq = set(), []
        for p in paras:
            if p not in seen:
                seen.add(p); uniq.append(p)
        train, test = uniq[:args.train], uniq[args.train:args.train + args.test]
        (OUT / f"{lang}.train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (OUT / f"{lang}.test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
        print(f"{lang}: {len(uniq)} paragraphs from {args.articles} articles -> train {len(train)}, test {len(test)}", flush=True)

if __name__ == "__main__":
    main()
