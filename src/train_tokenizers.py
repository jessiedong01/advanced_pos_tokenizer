#!/usr/bin/env python3
import argparse, os, json
from tokenizer_models import BPETokenizer

def read_lines(path):
    with open(path, "r", encoding="utf8") as f:
        return [l.strip() for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pos_json", default=None)
    ap.add_argument("--pos_transitions", default=None)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pos_component", choices=["transitions","penalties","both"], default="both")
    args = ap.parse_args()

    lines = read_lines(args.corpus)

    # baseline
    base = BPETokenizer(use_pos=False)
    stats_b = base.train(lines, vocab_size=args.vocab_size)
    base_dir = os.path.join(args.outdir, f"baseline_{args.vocab_size}")
    base.save(base_dir)

    # pos-aware
    pos = BPETokenizer(pos_json=args.pos_json, pos_factors_json=args.pos_transitions, use_pos=True, pos_component=args.pos_component)
    stats_p = pos.train(lines, vocab_size=args.vocab_size)
    pos_dir = os.path.join(args.outdir, f"pos_{args.vocab_size}_{args.pos_component}")
    pos.save(pos_dir)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "train_summary.json"), "w", encoding="utf8") as f:
        json.dump({"baseline": stats_b, "pos": stats_p}, f, indent=2)

    print("[baseline]", stats_b, "->", base_dir)
    print("[pos]", stats_p, "->", pos_dir)

if __name__ == "__main__":
    main()
