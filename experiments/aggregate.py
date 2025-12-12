#!/usr/bin/env python3
"""Collect every run into paper/numbers.tex, paper/tables/*.tex and paper/figures/*.

Every number quoted in the paper is a macro written here, so the text can
never drift from the results. Means and standard deviations are over the
three training seeds, each of which samples different training paragraphs.
"""
import glob, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, PAPER = ROOT / "results", ROOT / "paper"
sys.path.insert(0, str(ROOT / "experiments"))
from posbpe import transition_table, MARK

LANGS = ["en", "es", "da", "ta", "tr"]
LANG_NAME = {"en": "English", "es": "Spanish", "da": "Danish", "ta": "Tamil", "tr": "Turkish"}
LANG_MACRO = {"en": "English", "es": "Spanish", "da": "Danish", "ta": "Tamil", "tr": "Turkish"}
METHODS = [  # key, table label, macro prefix
    ("bpe", "BPE, free marker", "Bpe"),
    ("pos", "POS-weighted BPE", "Pos"),
    ("pos_trans", "\\quad transitions only", "PosTrans"),
    ("pos_pen", "\\quad penalty only", "PosPen"),
    ("pos_content", "Content-word-weighted BPE", "PosContent"),
    ("l2w", "Lemma2Word + BPE", "Ltw"),
    ("hf_bpe", "BPE, word-initial marker", "HfBpe"),
    ("unigram", "Unigram LM", "Uni"),
]
MAIN = ["bpe", "pos", "pos_content", "l2w", "hf_bpe", "unigram"]
COLORS = {"bpe": "#6f6f6f", "pos": "#2a78d6", "pos_content": "#7fb3f0", "l2w": "#1baf7a", "hf_bpe": "#b0b0b0", "unigram": "#eb6834",
          "pos_trans": "#2a78d6", "pos_pen": "#2a78d6"}
INK, INK2, GRID = "#111111", "#555555", "#dddddd"
macros = []

def macro(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

def fmt_int(n):
    return f"{int(n):,}"

def stat(xs):
    xs = np.asarray(xs, float)
    return xs.mean(), (xs.std(ddof=1) if len(xs) > 1 else 0.0)

def cell(xs, scale=1.0, d=1, bold=False):
    m, s = stat(xs)
    txt = f"{scale * m:.{d}f}\\,{{\\scriptsize$\\pm${scale * s:.{d}f}}}"
    return f"\\textbf{{{txt}}}" if bold else txt

def load_runs(vocab=8000, tag=""):
    runs = defaultdict(list)   # (lang, method) -> [metrics]
    for p in RES.glob(f"*/*_v{vocab}_s*{tag}/metrics.json"):
        if (tag == "" and re.search(r"_s\d+$", p.parent.name) is None):
            continue
        m = json.loads(p.read_text())
        runs[(m["lang"], m["method"])].append(m)
    return runs

# ----------------------------------------------------------------------------- derived diagnostics
def strip(t):
    return t.replace(MARK, "")

def boundary_set(tokens):
    cuts, c = set(), 0
    for t in tokens:
        c += len(strip(t))
        cuts.add(c)
    return cuts

def cross_seed(runs):
    """Vocabulary Jaccard and held-out segmentation agreement between seeds."""
    out = {}
    for (lang, method), ms in runs.items():
        dirs = {m["seed"]: RES / lang / f"{method}_v{m['vocab_budget']}_s{m['seed']}" for m in ms}
        seeds = sorted(dirs)
        vocabs = {s: set(json.loads((dirs[s] / "vocab.json").read_text())) for s in seeds}
        segs = {s: json.loads((dirs[s] / "test_tokens.json").read_text()) for s in seeds}
        jac, agree = [], []
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                jac.append(len(vocabs[a] & vocabs[b]) / len(vocabs[a] | vocabs[b]))
                tp = fp = 0
                for ta, tb in zip(segs[a], segs[b]):
                    A, B = boundary_set(ta), boundary_set(tb)
                    tp += len(A & B); fp += len(A ^ B)
                agree.append(2 * tp / (2 * tp + fp))
        out[(lang, method)] = {"jaccard": jac, "agreement": agree}
    return out

def cross_word_diagnostics(runs):
    """For tokens that span a word boundary: how likely is the POS transition they span, compared with all boundaries."""
    out = {}
    tagged = {lang: [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{lang}.test.jsonl", encoding="utf-8")] for lang in LANGS}
    train = {lang: [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{lang}.train.jsonl", encoding="utf-8")] for lang in LANGS}
    for (lang, method), ms in runs.items():
        vals = []
        for m in ms:
            idx = list(range(len(train[lang]))); random.Random(m["seed"]).shuffle(idx)
            trans = transition_table([train[lang][i] for i in idx[:m["n_train_paragraphs"]]])
            segs = json.loads((RES / lang / f"{method}_v{m['vocab_budget']}_s{m['seed']}" / "test_tokens.json").read_text())
            p_all, p_span, n_all, n_span, spanned_pairs, spanned_tokens = 0.0, 0.0, 0, 0, Counter(), Counter()
            for words, toks in zip(tagged[lang], segs):
                stream = "".join(toks)
                offset = 1 if stream.startswith(MARK) else 0
                tok_of_pos = [ti for ti, t in enumerate(toks) for _ in t]
                cursor = offset
                for k in range(len(words) - 1):
                    cursor += len(words[k][0])
                    ti = tok_of_pos[cursor] if cursor < len(tok_of_pos) else None
                    cursor += 1
                    if ti is None:
                        continue
                    a, b = words[k][1], words[k + 1][1]
                    p = trans.get(a, {}).get(b, 0.0)
                    p_all += p; n_all += 1
                    t = toks[ti]
                    if MARK in t.strip(MARK):
                        p_span += p; n_span += 1; spanned_pairs[(a, b)] += 1; spanned_tokens[t] += 1
            vals.append({"p_all": p_all / max(1, n_all), "p_span": p_span / max(1, n_span), "share_span": n_span / max(1, n_all),
                         "pairs": spanned_pairs, "tokens": spanned_tokens})
        out[(lang, method)] = vals
    return out

# ----------------------------------------------------------------------------- tables
def tex_token(tok, lang):
    """A token for a LaTeX table: visible space marker, escaped specials, Tamil in a Tamil font."""
    s = tok.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("$", "\\$").replace("{", "\\{").replace("}", "\\}")
    if lang == "ta":   # the marker glyph is not in the Tamil font, so it is set in the main font between Tamil runs
        return "\\textvisiblespace{}".join(f"\\tamil{{{part}}}" if part else "" for part in s.split(MARK))
    return "\\texttt{{{}}}".format(s.replace(MARK, "\\textvisiblespace{}"))

def examples_table(runs):
    """Held-out evaluation words of 8 to 14 characters, chosen by a fixed rule, segmented by every main method (seed 0)."""
    from lm_eval import load_wrapped
    from run import morph_items
    lines = ["\\begin{table}[!ht]", "\\centering",
             "\\caption{Example segmentations of evaluation words absent from the training text (seed 0). Words are the first, middle and last of the alphabetically sorted unseen forms without capitals of 8 to 14 characters with at least one gold boundary; the word-initial-marker BPE is omitted for space. Vertical bars mark boundaries; the gold column shows the lemma-derived boundaries.}",
             "\\label{tab:examples}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{llllll}", "\\toprule",
             "Lang. & Gold & BPE, free marker & Content-weighted & Lemma2Word & Unigram \\\\", "\\midrule"]
    for lang in LANGS:
        if not runs.get((lang, "bpe")):
            continue
        train = [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{lang}.train.jsonl", encoding="utf-8")]
        idx = list(range(len(train))); random.Random(0).shuffle(idx)
        seen = {w[0].lower() for i in idx[:runs[(lang, "bpe")][0]["n_train_paragraphs"]] for w in train[i]}
        cands = sorted((it for it in morph_items(lang) if 8 <= len(it["word"]) <= 14 and it["cuts"] and it["word"].lower() not in seen
                        and not any(ch.isupper() for ch in it["word"])),
                       key=lambda it: it["word"])
        picks = [cands[0], cands[len(cands) // 2], cands[-1]] if len(cands) >= 3 else cands
        toks = {k: load_wrapped(RES / lang / f"{k}_v8000_s0", k) for k in MAIN}
        for it in picks:
            w = it["word"]
            def seg(pieces):
                return tex_token("|".join(strip(p) for p in pieces if strip(p)), lang)
            gold_idx = [0] + sorted(it["cuts"]) + [len(w)]
            gold = tex_token("|".join(w[a:b] for a, b in zip(gold_idx, gold_idx[1:])), lang)
            cells = [seg(toks[k].encode_word(w)) for k in MAIN if k not in ("pos", "hf_bpe")]
            lines.append(f"{LANG_NAME[lang][:2]} & {gold} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "examples.tex").write_text("\n".join(lines) + "\n")

def llm_table(runs):
    """The language model as a segmenter, scored on its 400-word sample per language, next to the
    seed-0 tokenizers scored on exactly the same words."""
    from lm_eval import load_wrapped
    from run import morph_items, prf, pred_cuts
    summaries = {p.stem.split("_")[1]: json.loads(p.read_text()) for p in (RES / "llm").glob("summary_*.json")} if (RES / "llm").exists() else {}
    if not summaries:
        return
    lines = ["\\begin{table}[!ht]", "\\centering",
             "\\caption{A language model (Claude Sonnet) prompted to segment words into morphemes, scored on a fixed random sample of the gold word forms, beside three tokenizers (seed 0) scored on the same words. \\emph{Invalid}: answers that changed the characters of the word and were excluded. Boundary precision, recall and F1 in \\%; tokens per word is the number of pieces.}",
             "\\label{tab:llm}", "\\small", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lrrcccccccc}", "\\toprule",
             " & & & \\multicolumn{4}{c}{language model} & \\multicolumn{3}{c}{recall of tokenizers on the same words} \\\\",
             "\\cmidrule(lr){4-7}\\cmidrule(lr){8-10}",
             "Language & words & invalid & P & R & F1 & pieces/word & Lemma2Word & Unigram & BPE, free \\\\", "\\midrule"]
    for lang in LANGS:
        s = summaries.get(lang)
        if not s:
            continue
        items = morph_items(lang)
        rng = random.Random(0); rng.shuffle(items)
        sample = items[:s["n_sample"]]
        cache = {}
        for line in (RES / "llm" / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line); cache[d["word"]] = d
        scored = [it for it in sample if cache.get(it["word"], {}).get("valid")]
        rec = {}
        for key in ("l2w", "unigram", "bpe"):
            tok = load_wrapped(RES / lang / f"{key}_v8000_s0", key)
            rows = []
            for it in scored:
                pieces = tok.encode_word(it["word"]); p = pred_cuts(pieces); p.discard(len(it["word"]))
                rows.append((it["cuts"], p, it["freq"], len(pieces)))
            rec[key] = 100 * prf(rows)["recall"]
            macro(f"nLlmSample{ {'l2w': 'Ltw', 'unigram': 'Uni', 'bpe': 'Bpe'}[key] }R{LANG_MACRO[lang]}", f"{rec[key]:.1f}")
        for met, short in (("precision", "P"), ("recall", "R"), ("f1", "F")):
            macro(f"nLlm{short}{LANG_MACRO[lang]}", f"{100 * s[met]:.1f}")
        macro(f"nLlmN{LANG_MACRO[lang]}", fmt_int(s["n_words"])); macro(f"nLlmInvalid{LANG_MACRO[lang]}", str(s["invalid"]))
        macro(f"nLlmTpw{LANG_MACRO[lang]}", f"{s['tokens_per_word']:.2f}")
        lines.append(f"{LANG_NAME[lang]} & {s['n_words']} & {s['invalid']} & {100 * s['precision']:.1f} & {100 * s['recall']:.1f} & {100 * s['f1']:.1f} & {s['tokens_per_word']:.2f} & {rec['l2w']:.1f} & {rec['unigram']:.1f} & {rec['bpe']:.1f} \\\\")
    tin = sum(s["input_tokens"] for s in summaries.values()); tout = sum(s["output_tokens"] for s in summaries.values())
    macro("nLlmInputTokens", fmt_int(tin)); macro("nLlmOutputTokens", fmt_int(tout))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "llm.tex").write_text("\n".join(lines) + "\n")

def table_by_language(runs, getter, name, caption, label, scale=100.0, d=1, higher_better=True, methods=None):
    methods = methods or [m for m in METHODS]
    rows = []
    for key, lab, pre in methods:
        cells = []
        for lang in LANGS:
            ms = runs.get((lang, key), [])
            if not ms:
                cells.append("--"); continue
            vals = [getter(m) for m in ms]
            cells.append((key, lang, vals))
        rows.append((key, lab, pre, cells))
    # best per language among the main non-oracle methods
    best = {}
    for lang in LANGS:
        cand = [(stat(c[2])[0], c[0]) for _, _, _, cells in rows for c in cells if c != "--" and c[1] == lang]
        if cand:
            best[lang] = (max if higher_better else min)(cand)[1]
    lines = ["\\begin{table}[!ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{l" + "c" * len(LANGS) + "}", "\\toprule", "System & " + " & ".join(LANG_NAME[l] for l in LANGS) + " \\\\", "\\midrule"]
    for key, lab, pre, cells in rows:
        parts = []
        for c in cells:
            if c == "--":
                parts.append("--"); continue
            parts.append(cell(c[2], scale, d, bold=(best.get(c[1]) == key)))
            macro(f"n{name}{pre}{LANG_MACRO[c[1]]}", f"{scale * stat(c[2])[0]:.{d}f}")
            macro(f"n{name}{pre}{LANG_MACRO[c[1]]}Sd", f"{scale * stat(c[2])[1]:.{d}f}")
        lines.append(f"{lab} & " + " & ".join(parts) + " \\\\")
        if key == "pos_pen":
            lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / f"{label.split(':')[1]}.tex").write_text("\n".join(lines) + "\n")

def main():
    runs = load_runs()
    counts = Counter(len(v) for v in runs.values())
    print("runs per (lang, method):", dict(counts))
    (PAPER / "tables").mkdir(parents=True, exist_ok=True); (PAPER / "figures").mkdir(parents=True, exist_ok=True)

    # ---- data statistics
    for lang in LANGS:
        ms = runs.get((lang, "bpe"), [])
        if ms:
            macro(f"nTrainWords{LANG_MACRO[lang]}", fmt_int(np.mean([m["n_train_words"] for m in ms])))
            macro(f"nMorphWords{LANG_MACRO[lang]}", fmt_int(ms[0]["morph"]["lexicon"]["all"]["n_words"]))
            macro(f"nMorphTest{LANG_MACRO[lang]}", fmt_int(ms[0]["morph"]["lexicon"]["test_split"]["n_words"]))
            macro(f"nMorphUnseen{LANG_MACRO[lang]}", fmt_int(ms[0]["morph"]["lexicon"]["unseen"]["n_words"]))
        test = [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{lang}.test.jsonl", encoding="utf-8")]
        macro(f"nTestWords{LANG_MACRO[lang]}", fmt_int(sum(len(w) for w in test)))
    macro("nSeeds", str(max(counts)))
    macro("nVocabBudget", "8,000")

    # ---- main tables
    table_by_language(runs, lambda m: m["morph"]["lexicon"]["all"]["f1"], "F", "Boundary F1 (\\%) against the lemma-derived morpheme boundaries of MorphScore, all unique word forms. Mean $\\pm$ standard deviation over three training seeds; best per language in bold.", "tab:f1")
    table_by_language(runs, lambda m: m["morph"]["lexicon"]["all"]["precision"], "P", "Boundary precision (\\%).", "tab:precision")
    table_by_language(runs, lambda m: m["morph"]["lexicon"]["all"]["recall"], "R", "Boundary recall (\\%).", "tab:recall")
    table_by_language(runs, lambda m: m["heldout"]["tokens_per_word"], "Tpw", "Tokens per whitespace word on held-out Wikipedia paragraphs (lower is more compact).", "tab:tpw", scale=1.0, d=3, higher_better=False)
    table_by_language(runs, lambda m: m["heldout"]["share_cross_word"], "Cross", "Share of held-out tokens that span a word boundary (\\%).", "tab:cross")
    lm_runs = {k: [m for m in v if (RES / m["lang"] / f"{m['method']}_v{m['vocab_budget']}_s{m['seed']}" / "lm.json").exists()] for k, v in runs.items()}
    for k, v in lm_runs.items():
        for m in v:
            m["lm"] = json.loads((RES / m["lang"] / f"{m['method']}_v{m['vocab_budget']}_s{m['seed']}" / "lm.json").read_text())
    lm_runs = {k: v for k, v in lm_runs.items() if v}
    if lm_runs:
        table_by_language(lm_runs, lambda m: m["lm"]["bits_per_char"], "Bpc", "Bits per character of a 7M-parameter language model trained for six passes over the same text with each tokenizer (lower is better). Mean $\\pm$ standard deviation over the three tokenizer seeds.", "tab:bpc", scale=1.0, d=3, higher_better=False)

    # ---- step-count control for the probe: the same tokenizer trained for 10% fewer and 10% more passes
    base = RES / "en" / "bpe_v8000_s0"
    for name, suffix in (("Low", "5.4"), ("High", "6.6")):
        p = base / f"lm_epochs{suffix}.json"
        if p.exists():
            macro(f"nBpcControl{name}", f"{json.loads(p.read_text())['bits_per_char']:.3f}")
    if (base / "lm.json").exists():
        macro("nBpcControlBase", f"{json.loads((base / 'lm.json').read_text())['bits_per_char']:.3f}")

    # ---- cross-seed stability
    cs = cross_seed(runs)
    table_by_language({k: [{"v": x} for x in v["jaccard"]] for k, v in cs.items()}, lambda m: m["v"], "Jac", "Vocabulary reuse: Jaccard overlap (\\%) between the vocabularies learned from different training samples (three seed pairs).", "tab:jaccard")
    table_by_language({k: [{"v": x} for x in v["agreement"]] for k, v in cs.items()}, lambda m: m["v"], "Agree", "Segmentation agreement: boundary F1 (\\%) between the held-out segmentations of tokenizers trained on different samples (three seed pairs).", "tab:agreement")

    # ---- cross-word diagnostics
    cw = cross_word_diagnostics(runs)
    for (lang, method), vals in cw.items():
        pre = {k: p for k, _, p in METHODS}[method]
        macro(f"nSpanP{pre}{LANG_MACRO[lang]}", f"{100 * np.mean([v['p_span'] for v in vals]):.1f}")
        macro(f"nAllP{pre}{LANG_MACRO[lang]}", f"{100 * np.mean([v['p_all'] for v in vals]):.1f}")
        macro(f"nSpanShare{pre}{LANG_MACRO[lang]}", f"{100 * np.mean([v['share_span'] for v in vals]):.1f}")
    lines = ["\\begin{table}[!ht]", "\\centering",
             "\\caption{Word boundaries spanned by a token on held-out text: share of boundaries (\\%), mean probability (\\%) of the POS transition at spanned boundaries, and the same mean over all boundaries. Seed 0; the most frequent spanned transition and token are shown for the POS-weighted model.}",
             "\\label{tab:span}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{llccclc}", "\\toprule",
             "Language & System & spanned (\\%) & $P$, spanned & $P$, all & top transition & top token \\\\", "\\midrule"]
    for lang in LANGS:
        for key, lab, pre in [m for m in METHODS if m[0] in ("bpe", "pos", "pos_content")]:
            vals = cw.get((lang, key))
            if not vals:
                continue
            v0 = vals[0]
            pair, tok = (v0["pairs"].most_common(1) or [(("--", "--"), 0)])[0][0], (v0["tokens"].most_common(1) or [("--", 0)])[0][0]
            tok = tex_token(tok, lang)
            lines.append(f"{LANG_NAME[lang] if key == 'bpe' else ''} & {lab} & {100 * np.mean([v['share_span'] for v in vals]):.1f} & {100 * np.mean([v['p_span'] for v in vals]):.1f} & {100 * np.mean([v['p_all'] for v in vals]):.1f} & {pair[0]}$\\to${pair[1]} & {tok} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "span.tex").write_text("\n".join(lines) + "\n")

    # ---- Lemma2Word variants: lexicon vs online, seen vs unseen, UD test split
    lines = ["\\begin{table}[!ht]", "\\centering",
             "\\caption{Lemma2Word boundary recall and precision (\\%) by word subset: all unique forms, forms absent from the tokenizer's training text, and forms from the UD test split. \\emph{Lexicon} uses only boundaries and affixes learned from the training text; \\emph{online} also lemmatizes the evaluation word with Stanza at tokenization time. Mean over three seeds.}",
             "\\label{tab:l2w}", "\\small", "\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{llcccccc}", "\\toprule",
             " & & \\multicolumn{2}{c}{all} & \\multicolumn{2}{c}{unseen} & \\multicolumn{2}{c}{UD test} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", "Language & Variant & R & P & R & P & R & P \\\\", "\\midrule"]
    for lang in LANGS:
        ms = runs.get((lang, "l2w"), [])
        if not ms:
            continue
        for variant, lab in (("lexicon", "lexicon"), ("online", "online")):
            parts = []
            for subset in ("all", "unseen", "test_split"):
                for met in ("recall", "precision"):
                    v = 100 * np.mean([m["morph"][variant][subset][met] for m in ms])
                    parts.append(f"{v:.1f}")
                    macro(f"nLtw{variant.capitalize()}{subset.replace('_split', '').capitalize()}{met[0].upper()}{LANG_MACRO[lang]}", f"{v:.1f}")
            lines.append(f"{LANG_NAME[lang] if variant == 'lexicon' else ''} & {lab} & " + " & ".join(parts) + " \\\\")
        bpe = runs.get((lang, "bpe"), [])
        if bpe:
            parts = []
            for subset in ("all", "unseen", "test_split"):
                for met in ("recall", "precision"):
                    parts.append(f"{100 * np.mean([m['morph']['lexicon'][subset][met] for m in bpe]):.1f}")
            lines.append(f" & BPE, free marker & " + " & ".join(parts) + " \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (PAPER / "tables" / "l2w.tex").write_text("\n".join(lines) + "\n")

    # ---- vocabulary sweep (seed 0)
    sweep = defaultdict(dict)
    for p in RES.glob("*/*_v*_s0/metrics.json"):
        m = json.loads(p.read_text())
        sweep[(m["lang"], m["method"])][m["vocab_budget"]] = m
    for (lang, method), d in sweep.items():
        pre = {k: p for k, _, p in METHODS}[method]
        for v, m in d.items():
            vname = {2000: "TwoK", 8000: "EightK", 32000: "ThirtyTwoK"}[v]      # macro names cannot contain digits
            macro(f"nSweepF{pre}{LANG_MACRO[lang]}V{vname}", f"{100 * m['morph']['lexicon']['all']['f1']:.1f}")
            macro(f"nSweepTpw{pre}{LANG_MACRO[lang]}V{vname}", f"{m['heldout']['tokens_per_word']:.2f}")

    # ---- LLM study
    llm_table(runs)

    examples_table(runs)
    figures(runs, lm_runs, sweep)
    (PAPER / "numbers.tex").write_text("\n".join(sorted(set(macros))) + "\n")
    print(f"wrote {len(set(macros))} macros")

# ----------------------------------------------------------------------------- figures
def figures(runs, lm_runs, sweep):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    labels = {k: lab for k, lab, _ in METHODS}

    def grouped_bars(getter, ylabel, fname, source=runs, methods=MAIN, scale=100.0, points=False):
        """Grouped bars from zero, or, for quantities whose differences are small relative to their
        value, markers with error bars on an axis that starts near the smallest value."""
        fig, ax = plt.subplots(figsize=(6.9, 2.3))
        width = 0.8 / len(methods)
        allv = []
        for j, key in enumerate(methods):
            xs, ms_, sds = [], [], []
            for i, lang in enumerate(LANGS):
                ms = source.get((lang, key), [])
                if not ms:
                    continue
                m, s = stat([getter(x) for x in ms]); xs.append(i + (j - (len(methods) - 1) / 2) * width); ms_.append(scale * m); sds.append(scale * s)
            allv += ms_
            if points:
                ax.errorbar(xs, ms_, yerr=sds, fmt="o", ms=4.5, color=COLORS[key], ecolor=COLORS[key], elinewidth=0.7, capsize=1.5, label=labels[key])
            else:
                ax.bar(xs, ms_, width=width * 0.95, color=COLORS[key], label=labels[key], yerr=sds, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
        if points and allv:
            ax.set_ylim(min(allv) - 0.08 * (max(allv) - min(allv)), max(allv) + 0.08 * (max(allv) - min(allv)))
            for i in range(1, len(LANGS)):
                ax.axvline(i - 0.5, color=GRID, lw=0.5)
        ax.set_xticks(range(len(LANGS)), [LANG_NAME[l] for l in LANGS])
        ax.set_ylabel(ylabel); ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
        fig.savefig(PAPER / "figures" / f"{fname}.pdf"); fig.savefig(PAPER / "figures" / f"{fname}.png", dpi=200); plt.close(fig)

    grouped_bars(lambda m: m["morph"]["lexicon"]["all"]["f1"], "Boundary F1 (%)", "fig_f1")
    grouped_bars(lambda m: m["heldout"]["tokens_per_word"], "Tokens per word", "fig_tpw", scale=1.0)
    if lm_runs:
        grouped_bars(lambda m: m["lm"]["bits_per_char"], "Bits per character", "fig_bpc", source=lm_runs, scale=1.0, points=True)

    # precision-recall per language, one panel each
    fig, axes = plt.subplots(1, len(LANGS), figsize=(6.9, 1.7), sharey=True)
    for ax, lang in zip(axes, LANGS):
        for key in MAIN:
            ms = runs.get((lang, key), [])
            if not ms:
                continue
            R = 100 * np.array([m["morph"]["lexicon"]["all"]["recall"] for m in ms]); P = 100 * np.array([m["morph"]["lexicon"]["all"]["precision"] for m in ms])
            ax.errorbar(R.mean(), P.mean(), xerr=stat(R)[1], yerr=stat(P)[1], fmt="o", ms=4, color=COLORS[key], ecolor=COLORS[key], elinewidth=0.6, capsize=1.5, label=labels[key])
        ax.set_title(LANG_NAME[lang]); ax.set_xlabel("Recall (%)"); ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("Precision (%)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12), columnspacing=1.0, handlelength=1.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(PAPER / "figures" / "fig_pr.pdf"); fig.savefig(PAPER / "figures" / "fig_pr.png", dpi=200); plt.close(fig)

    # vocabulary sweep: F1 against budget, seed 0
    if any(len(d) > 1 for d in sweep.values()):
        fig, axes = plt.subplots(1, len(LANGS), figsize=(6.9, 1.7), sharey=True)
        for ax, lang in zip(axes, LANGS):
            for key in MAIN:
                d = sweep.get((lang, key), {})
                vs = sorted(d)
                if len(vs) > 1:
                    ax.plot(vs, [100 * d[v]["morph"]["lexicon"]["all"]["f1"] for v in vs], marker="o", ms=3, lw=1.2, color=COLORS[key], label=labels[key])
            ax.set_xscale("log"); ax.set_xticks([2000, 8000, 32000], ["2k", "8k", "32k"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_title(LANG_NAME[lang]); ax.set_xlabel("Vocabulary budget"); ax.yaxis.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        axes[0].set_ylabel("Boundary F1 (%)")
        handles, labs = axes[0].get_legend_handles_labels()
        fig.legend(handles, labs, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.12), columnspacing=1.0, handlelength=1.2)
        fig.tight_layout(w_pad=0.6)
        fig.savefig(PAPER / "figures" / "fig_sweep.pdf"); fig.savefig(PAPER / "figures" / "fig_sweep.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    main()
