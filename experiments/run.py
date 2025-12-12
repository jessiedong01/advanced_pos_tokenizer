#!/usr/bin/env python3
"""Train one tokenizer on one language and evaluate it.

Every method sees the same training paragraphs (a seeded sample of the
tagged Wikipedia text) and the same vocabulary budget. Evaluation covers
held-out paragraphs (compression and where tokens cross word boundaries) and
the MorphScore word list (boundary precision and recall against the
lemma-derived gold boundaries).

Output: results/<lang>/<method>_v<vocab>_s<seed>/metrics.json, plus the
trained model and the held-out segmentation for the cross-seed comparisons.
"""
import argparse, json, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from posbpe import PosBPE, transition_table, MARK
from l2w import Lemma2Word, is_word

CONTENT = {"NOUN": 2.0, "PROPN": 2.0, "VERB": 2.0, "ADJ": 2.0}
METHODS = {
    "bpe":         dict(kind="posbpe", beta=0.0, bf=1.0),
    "pos":         dict(kind="posbpe", beta=1.0, bf=0.98),
    "pos_trans":   dict(kind="posbpe", beta=1.0, bf=1.0),
    "pos_pen":     dict(kind="posbpe", beta=0.0, bf=0.98),
    "pos_content": dict(kind="posbpe", beta=0.0, bf=1.0, w_intra=CONTENT),
    "l2w":         dict(kind="l2w"),
    "hf_bpe":      dict(kind="hf_bpe"),
    "unigram":     dict(kind="unigram"),
}
MORPH_FILE = {"en": "english", "es": "spanish", "da": "danish", "ta": "tamil", "tr": "turkish"}
SEP = "\x00"   # unmergeable break between Lemma2Word pieces

# ----------------------------------------------------------------------------- tokenizers
class Wrapped:
    """Uniform interface: tokens of a paragraph, and pieces of a single word."""
    def encode_paragraph(self, text): raise NotImplementedError
    def encode_word(self, word): raise NotImplementedError
    def vocab(self): raise NotImplementedError

class WrappedPosBPE(Wrapped):
    def __init__(self, model, l2w=None):
        self.model, self.l2w = model, l2w
    def _word_symbols(self, word):
        if self.l2w is None:
            return list(word)
        syms = []
        for k, piece in enumerate(Lemma2Word.pieces(word, self.l2w.cuts(word))):
            if k > 0:
                syms.append(None)
            syms.extend(piece)
        return syms
    def encode_paragraph(self, text):
        syms = []
        for k, w in enumerate(text.split()):
            if k > 0:
                syms.append(MARK)
            syms.extend(self._word_symbols(w))
        return self.model.encode_symbols(syms)
    def encode_word(self, word, cuts=None):
        if cuts is None:
            return self.model.encode_symbols(self._word_symbols(word))
        syms = []
        for k, piece in enumerate(Lemma2Word.pieces(word, cuts)):
            if k > 0:
                syms.append(None)
            syms.extend(piece)
        return self.model.encode_symbols(syms)
    def vocab(self):
        return sorted(self.model.vocab)

class WrappedHF(Wrapped):
    def __init__(self, tok): self.tok = tok
    def encode_paragraph(self, text): return self.tok.encode(text).tokens
    def encode_word(self, word): return self.tok.encode(word).tokens
    def vocab(self): return sorted(self.tok.get_vocab())

class WrappedSPM(Wrapped):
    def __init__(self, sp): self.sp = sp
    def encode_paragraph(self, text): return self.sp.encode(text, out_type=str)
    def encode_word(self, word): return self.sp.encode(word, out_type=str)
    def vocab(self): return sorted(self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size()))

def train_tokenizer(method, paragraphs, vocab_size, out, log):
    spec = METHODS[method]
    texts = [" ".join(w[0] for w in words) for words in paragraphs]
    if spec["kind"] == "posbpe":
        model = PosBPE(beta=spec["beta"], boundary_factor=spec["bf"], w_intra=spec.get("w_intra"),
                       transitions=transition_table(paragraphs))
        model.train(paragraphs, vocab_size, log=log); model.save(out)
        return WrappedPosBPE(model)
    if spec["kind"] == "l2w":
        l2w = Lemma2Word().fit(paragraphs); l2w.save(out)
        log(f"lexicon {len(l2w.lexicon)} words, {len(l2w._pre)} prefixes, {len(l2w._suf)} suffixes")
        pre = [[[SEP.join(Lemma2Word.pieces(w[0], l2w.cuts(w[0]))), *w[1:]] for w in words] for words in paragraphs]
        model = PosBPE().train(pre, vocab_size, log=log); model.save(out)
        return WrappedPosBPE(model, l2w)
    if spec["kind"] == "hf_bpe":
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=MARK, prepend_scheme="always")
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False, limit_alphabet=5000, initial_alphabet=[MARK])
        tok.train_from_iterator(texts, trainer); tok.save(str(out / "hf_bpe.json"))
        return WrappedHF(tok)
    if spec["kind"] == "unigram":
        import sentencepiece as spm
        spm.SentencePieceTrainer.train(sentence_iterator=iter(texts), model_prefix=str(out / "spm"), vocab_size=vocab_size,
                                       model_type="unigram", character_coverage=1.0, normalization_rule_name="identity",
                                       add_dummy_prefix=True, hard_vocab_limit=False, num_threads=2, minloglevel=2,
                                       max_sentence_length=20000)
        return WrappedSPM(spm.SentencePieceProcessor(model_file=str(out / "spm.model")))
    raise ValueError(method)

# ----------------------------------------------------------------------------- metrics
def strip(tok):
    return tok.replace(MARK, "")

def compression(wrapped, test_paragraphs, transitions):
    """Tokens per word, characters per token, and which word boundaries a token absorbs."""
    n_tok = n_word = n_char = 0
    lead = trail = cross = 0
    absorbed_p, all_p, absorbed_n, all_n = 0.0, 0.0, 0, 0
    segmentations = []
    for words in test_paragraphs:
        text = " ".join(w[0] for w in words)
        toks = wrapped.encode_paragraph(text)
        segmentations.append(toks)
        n_tok += len(toks); n_word += len(words); n_char += sum(len(strip(t)) for t in toks)
        for t in toks:
            core = strip(t)
            if not core:
                continue
            if t.startswith(MARK): lead += 1
            if t.endswith(MARK): trail += 1
            if MARK in t.strip(MARK): cross += 1
        # locate each marker (boundary between word k and k+1) in the token stream
        stream = "".join(toks)
        offset = 0 if not stream.startswith(MARK) else 1    # hf/spm prepend a marker to the first word
        pos_in_stream, marker_tok = 0, []
        tok_of_pos = []
        for ti, t in enumerate(toks):
            tok_of_pos.extend([ti] * len(t))
        cursor = offset
        for k in range(len(words) - 1):
            cursor += len(words[k][0])            # marker sits right after word k
            ti = tok_of_pos[cursor] if cursor < len(tok_of_pos) else None
            cursor += 1
            if ti is None:
                continue
            a, b = words[k][1], words[k + 1][1]
            p = transitions.get(a, {}).get(b, 0.0)
            all_p += p; all_n += 1
            if len(toks[ti]) > 1:                 # the marker shares a token with characters of a word
                absorbed_p += p; absorbed_n += 1
    return {
        "tokens_per_word": n_tok / n_word, "chars_per_token": n_char / n_tok,
        "share_leading_marker": lead / n_tok, "share_trailing_marker": trail / n_tok, "share_cross_word": cross / n_tok,
        "boundaries_absorbed": absorbed_n / max(1, all_n),
        "mean_transition_p_all": all_p / max(1, all_n),
        "mean_transition_p_absorbed": absorbed_p / max(1, absorbed_n),
    }, segmentations

def morph_items(lang):
    df = pd.read_csv(ROOT / "data" / "morphscore" / f"{MORPH_FILE[lang]}_data.csv", keep_default_na=False)
    df = df[(df["unique"] == "unique") & (df["lemma"].str.lower() == df["stem"].str.lower())]
    items = []
    for r in df.itertuples():
        w, pre, stem, post = str(r.wordform), str(r.preceding_part), str(r.stem), str(r.following_part)
        if not is_word(w) or (pre + stem + post).lower() != w.lower() or len(w) < 2:   # letters and combining marks only
            continue
        cuts = set()
        if pre: cuts.add(len(pre))
        if post: cuts.add(len(pre) + len(stem))
        items.append({"word": w, "cuts": cuts, "split": r.data_split, "freq": float(r.word_freq), "pos": r.pos, "lemma": str(r.lemma)})
    return items

def pred_cuts(pieces):
    cuts, c = set(), 0
    for p in pieces:
        c += len(strip(p))
        cuts.add(c)
    cuts.discard(0)
    return cuts

def prf(rows):
    """rows: (gold cuts, predicted cuts, weight, word length). Micro-averaged precision/recall/F1,
    unweighted and frequency-weighted, plus tokens per word."""
    out = {}
    for name, use_w in (("", False), ("weighted_", True)):
        tp = fp = fn = 0.0
        for g, p, w, n in rows:
            wt = w if use_w else 1.0
            tp += wt * len(g & p); fp += wt * len(p - g); fn += wt * len(g - p)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        out[name + "precision"] = P; out[name + "recall"] = R
        out[name + "f1"] = 2 * P * R / (P + R) if P + R else 0.0
    out["n_words"] = len(rows)
    out["tokens_per_word"] = float(np.mean([n for _, _, _, n in rows])) if rows else 0.0
    return out

def morph_eval(encode, items, seen_words):
    rows = []
    for it in items:
        pieces = encode(it)
        if "".join(strip(p) for p in pieces) != it["word"]:
            continue
        p = pred_cuts(pieces); p.discard(len(it["word"]))
        rows.append((it, it["cuts"], p, len(pieces)))
    def subset(cond):
        return prf([(g, p, it["freq"], n) for it, g, p, n in rows if cond(it)])
    return {"all": subset(lambda it: True),
            "test_split": subset(lambda it: it["split"] == "test"),
            "unseen": subset(lambda it: it["word"].lower() not in seen_words),
            "seen": subset(lambda it: it["word"].lower() in seen_words)}

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = ROOT / "results" / args.lang / f"{args.method}_v{args.vocab}_s{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    def log(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    train = [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{args.lang}.train.jsonl", encoding="utf-8")]
    test = [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{args.lang}.test.jsonl", encoding="utf-8")]
    idx = list(range(len(train))); random.Random(args.seed).shuffle(idx)
    train = [train[i] for i in idx[:args.n_train]]
    seen_words = {w[0].lower() for words in train for w in words}
    log(f"{args.lang} {args.method} seed {args.seed}: {len(train)} training paragraphs, {sum(len(w) for w in train)} words")

    t0 = time.time()
    wrapped = train_tokenizer(args.method, train, args.vocab, out, log)
    train_seconds = time.time() - t0
    vocab = wrapped.vocab()
    log(f"trained in {train_seconds:.0f}s, vocab {len(vocab)}")

    transitions = transition_table(train)
    comp, segmentations = compression(wrapped, test, transitions)
    log(f"held-out: {comp['tokens_per_word']:.3f} tokens/word, {comp['chars_per_token']:.2f} chars/token, cross-word {comp['share_cross_word']:.3%}")

    items = morph_items(args.lang)
    morph = {"lexicon": morph_eval(lambda it: wrapped.encode_word(it["word"]), items, seen_words)}
    if args.method == "l2w":
        import stanza
        nlp = stanza.Pipeline(args.lang, processors="tokenize,pos,lemma", tokenize_pretokenized=True, use_gpu=False, verbose=False)
        doc = nlp([[it["word"]] for it in items])
        lemmas = [s.words[0].lemma or it["word"] for s, it in zip(doc.sentences, items)]
        for it, lem in zip(items, lemmas):
            it["stanza_lemma"] = lem
        morph["online"] = morph_eval(lambda it: wrapped.encode_word(it["word"], wrapped.l2w.cuts_online(it["word"], it["stanza_lemma"])), items, seen_words)
    m = morph["lexicon"]["all"]
    log(f"morph (all {m['n_words']} words): P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}")

    metrics = {"lang": args.lang, "method": args.method, "seed": args.seed, "vocab_size": len(vocab), "vocab_budget": args.vocab,
               "n_train_paragraphs": len(train), "n_train_words": sum(len(w) for w in train), "train_seconds": train_seconds,
               "heldout": comp, "morph": morph, "morph_filter": "letters"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "test_tokens.json").write_text(json.dumps(segmentations, ensure_ascii=False), encoding="utf-8")
    log("DONE")

if __name__ == "__main__":
    main()
