#!/usr/bin/env python3
"""Bits per character of a small language model trained on each tokenizer's output.

The same training paragraphs are encoded with the tokenizer of a finished
run, a four-layer transformer language model is trained on the token stream
for a fixed number of passes over the text, and its held-out negative log-likelihood is
divided by the number of characters in the held-out text. The character
count is the same for every tokenizer of a language, so the score compares
tokenizers directly. Output: lm.json in the run directory.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run import METHODS, WrappedPosBPE, WrappedHF, WrappedSPM
from posbpe import PosBPE
from l2w import Lemma2Word

def load_wrapped(run_dir, method):
    kind = METHODS[method]["kind"]
    if kind == "posbpe":
        return WrappedPosBPE(PosBPE.load(run_dir))
    if kind == "l2w":
        return WrappedPosBPE(PosBPE.load(run_dir), Lemma2Word.load(run_dir))
    if kind == "hf_bpe":
        from tokenizers import Tokenizer
        return WrappedHF(Tokenizer.from_file(str(run_dir / "hf_bpe.json")))
    import sentencepiece as spm
    return WrappedSPM(spm.SentencePieceProcessor(model_file=str(run_dir / "spm.model")))

class LM(nn.Module):
    def __init__(self, vocab, d=256, layers=4, heads=4, ctx=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.out(self.norm(h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<lang>/<method>_v<vocab>_s<seed>")
    ap.add_argument("--epochs", type=float, default=6.0, help="passes over the training text; the step count follows from the token count")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lm.json", help="output file name inside the run directory")
    args = ap.parse_args()
    run_dir = ROOT / args.run
    metrics = json.loads((run_dir / "metrics.json").read_text())
    lang, method = metrics["lang"], metrics["method"]
    wrapped = load_wrapped(run_dir, method)
    train = [json.loads(l)["text"] for l in open(ROOT / "data/tagged" / f"{lang}.train.jsonl", encoding="utf-8")]
    test = [json.loads(l)["text"] for l in open(ROOT / "data/tagged" / f"{lang}.test.jsonl", encoding="utf-8")]
    idx = list(range(len(train))); random.Random(metrics["seed"]).shuffle(idx)
    train = [train[i] for i in idx[:metrics["n_train_paragraphs"]]]
    vocab = {t: i + 2 for i, t in enumerate(wrapped.vocab())}          # 0 = paragraph break, 1 = unknown
    def ids(paras):
        out, unk = [], 0
        for p in paras:
            for t in wrapped.encode_paragraph(p):
                i = vocab.get(t, 1); unk += i == 1; out.append(i)
            out.append(0)
        return np.array(out, dtype=np.int64), unk
    tr, _ = ids(train); te, te_unk = ids(test)
    n_chars = sum(len(p) + 1 for p in test)                             # one newline per paragraph
    steps = math.ceil(args.epochs * len(tr) / (args.batch * args.ctx))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = LM(len(vocab) + 2, ctx=args.ctx).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t0 = time.time(); model.train()
    for step in range(steps):
        starts = np.random.randint(0, len(tr) - args.ctx - 1, size=args.batch)
        x = torch.from_numpy(np.stack([tr[s:s + args.ctx] for s in starts])).to(dev)
        y = torch.from_numpy(np.stack([tr[s + 1:s + args.ctx + 1] for s in starts])).to(dev)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval(); nll = 0.0
    with torch.no_grad():                                               # non-overlapping windows over the held-out stream
        for s in range(0, len(te) - 1, args.ctx):
            x = torch.from_numpy(te[s:s + args.ctx][None]).to(dev)
            y = torch.from_numpy(te[s + 1:s + args.ctx + 1][None]).to(dev)
            if y.shape[1] < x.shape[1]:
                x = x[:, :y.shape[1]]
            nll += nn.functional.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1), reduction="sum").item()
    bpc = nll / math.log(2) / n_chars
    out = {"bits_per_char": bpc, "nll_per_token": nll / (len(te) - 1), "test_tokens": int(len(te)), "test_chars": n_chars,
           "test_unknown_tokens": int(te_unk), "train_tokens": int(len(tr)), "steps": steps, "epochs": args.epochs, "batch": args.batch, "ctx": args.ctx,
           "params": sum(p.numel() for p in model.parameters()), "seconds": time.time() - t0, "lm_seed": args.seed}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    print(f"{args.run}: {bpc:.4f} bits/char, {out['nll_per_token']:.3f} nats/token, {len(te)} test tokens, {te_unk} unknown", flush=True)

if __name__ == "__main__":
    main()
