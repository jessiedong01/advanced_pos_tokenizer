#!/usr/bin/env python3
import argparse, os, json, pandas as pd, numpy as np
from utils.metrics import jaccard, avg_token_length, token_length_entropy, kl_divergence

def load_model(dirpath):
    with open(os.path.join(dirpath, "vocab.json"), "r", encoding="utf8") as f:
        vocab = json.load(f)
    with open(os.path.join(dirpath, "merges.json"), "r", encoding="utf8") as f:
        merges = json.load(f)
    return set(vocab), merges

def histogram(vocab):
    # simple token frequency proxy: length-based histogram here since training is offline
    d = {}
    for t in vocab:
        d[t] = d.get(t, 0) + 1
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pos_json", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    base_dir = None
    pos_dir = None
    for d in os.listdir(args.models_dir):
        if d.startswith("baseline_"):
            base_dir = os.path.join(args.models_dir, d)
        if d.startswith("pos_"):
            pos_dir = os.path.join(args.models_dir, d)
    if base_dir is None or pos_dir is None:
        raise RuntimeError("Expected both baseline_* and pos_* subdirs")

    base_vocab, base_merges = load_model(base_dir)
    pos_vocab, pos_merges = load_model(pos_dir)

    overlap = jaccard(base_vocab, pos_vocab)
    base_avg = avg_token_length(base_vocab)
    pos_avg = avg_token_length(pos_vocab)
    base_ent = token_length_entropy(base_vocab)
    pos_ent = token_length_entropy(pos_vocab)
    kl = kl_divergence(histogram(base_vocab), histogram(pos_vocab))

    df = pd.DataFrame([{
        "baseline_vocab_size": len(base_vocab),
        "pos_vocab_size": len(pos_vocab),
        "jaccard_vocab_overlap": round(overlap, 4),
        "baseline_avg_token_len": round(base_avg, 3),
        "pos_avg_token_len": round(pos_avg, 3),
        "baseline_len_entropy": round(base_ent, 3),
        "pos_len_entropy": round(pos_ent, 3),
        "kl_vocabs": round(kl, 4)
    }])
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    df.to_csv(args.report, index=False)
    print(df)

if __name__ == "__main__":
    main()
