#!/usr/bin/env python3
"""Recompute the MorphScore block of every finished run of one language from its saved
tokenizer, for use after a change to the gold word filter. The Stanza lemmas needed by the
online Lemma2Word variant are computed once per language and cached.

    python experiments/rescore_morph.py <lang>
"""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run import morph_items, morph_eval
from lm_eval import load_wrapped

def main():
    lang = sys.argv[1]
    items = morph_items(lang)
    cache = ROOT / "data" / "morphscore" / f"{lang}_stanza_lemmas.json"
    lemmas = json.loads(cache.read_text()) if cache.exists() else {}
    missing = sorted({it["word"] for it in items} - set(lemmas))
    if missing:
        import stanza
        nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma", tokenize_pretokenized=True, use_gpu=False, verbose=False)
        doc = nlp([[w] for w in missing])
        for w, s in zip(missing, doc.sentences):
            lemmas[w] = s.words[0].lemma or w
        cache.write_text(json.dumps(lemmas, ensure_ascii=False))
    for it in items:
        it["stanza_lemma"] = lemmas[it["word"]]
    train = [json.loads(l)["words"] for l in open(ROOT / "data/tagged" / f"{lang}.train.jsonl", encoding="utf-8")]
    for run_dir in sorted((ROOT / "results" / lang).glob("*_v*_s*")):
        mp = run_dir / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        if m.get("morph_filter") == "letters":
            continue
        idx = list(range(len(train))); random.Random(m["seed"]).shuffle(idx)
        seen = {w[0].lower() for i in idx[:m["n_train_paragraphs"]] for w in train[i]}
        wrapped = load_wrapped(run_dir, m["method"])
        morph = {"lexicon": morph_eval(lambda it: wrapped.encode_word(it["word"]), items, seen)}
        if m["method"] == "l2w":
            morph["online"] = morph_eval(lambda it: wrapped.encode_word(it["word"], wrapped.l2w.cuts_online(it["word"], it["stanza_lemma"])), items, seen)
        m["morph"] = morph; m["morph_filter"] = "letters"
        mp.write_text(json.dumps(m, indent=1, ensure_ascii=False))
        print(f"{run_dir.name}: F1 {morph['lexicon']['all']['f1']:.3f} on {morph['lexicon']['all']['n_words']} words", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
