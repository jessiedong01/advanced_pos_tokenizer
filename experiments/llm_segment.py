#!/usr/bin/env python3
"""LLM-As-Tok: a language model asked to segment words into morphemes.

A fixed random sample of MorphScore words per language is sent to the model
in batches. The model must return every character of the word with plus
signs at the morpheme boundaries; an answer that changes the characters is
discarded and counted as invalid. Boundaries are scored against the
lemma-derived gold boundaries like every other method. Responses are cached
in results/llm/<lang>.jsonl so the study can be resumed and re-scored; the scores go to
results/llm/summary_<lang>.json.
"""
import argparse, json, os, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run import morph_items, prf

LANG_NAME = {"en": "English", "es": "Spanish", "da": "Danish", "ta": "Tamil", "tr": "Turkish"}
PROMPT = """Segment each {lang} word below into its morphemes: prefixes, the stem, and suffixes (inflectional and derivational).
Rules: keep every character of the word exactly as given and in order; insert a plus sign (+) at each morpheme boundary; a word with a single morpheme is returned unchanged; do not add explanations.
Answer with one line per word in the format  word: seg+men+ted  and nothing else.

{words}"""

def ask(client, model, lang, words):
    msg = client.messages.create(model=model, max_tokens=8000,
                                 messages=[{"role": "user", "content": PROMPT.format(lang=LANG_NAME[lang], words="\n".join(words))}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    # The model sometimes deliberates before its final list, so a word can appear on several lines;
    # the last line whose characters match the word wins, otherwise the last line for the word.
    out, wanted = {}, set(words)
    for line in text.splitlines():
        if ":" not in line:
            continue
        w, seg = line.split(":", 1)
        w, seg = w.strip(), seg.strip()
        if w in wanted and (seg.replace("+", "") == w or out.get(w, "").replace("+", "") != w):
            out[w] = seg
    return out, msg.usage.input_tokens, msg.usage.output_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["en", "es", "da", "ta", "tr"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic()
    out_dir = ROOT / "results" / "llm"; out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for lang in args.langs:
        items = morph_items(lang)
        rng = random.Random(args.seed); rng.shuffle(items)
        sample = items[:args.n]
        cache_path = out_dir / f"{lang}.jsonl"
        cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line); cache[d["word"]] = d
        todo = [it["word"] for it in sample if it["word"] not in cache]
        tin = tout = 0
        with open(cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.batch):
                batch = todo[i:i + args.batch]
                for attempt in range(3):
                    try:
                        answers, a, b = ask(client, args.model, lang, batch); tin += a; tout += b
                        break
                    except Exception as e:
                        print(f"[{lang}] retry after error: {e}", flush=True); time.sleep(10)
                else:
                    answers = {}
                for w in batch:
                    seg = answers.get(w, "")
                    valid = seg.replace("+", "") == w
                    d = {"word": w, "segmentation": seg, "valid": valid, "model": args.model}
                    cache[w] = d; f.write(json.dumps(d, ensure_ascii=False) + "\n")
                print(f"[{lang}] {min(i + args.batch, len(todo))}/{len(todo)} words", flush=True)
        rows, invalid = [], 0
        for it in sample:
            d = cache[it["word"]]
            if not d["valid"]:
                invalid += 1; continue
            cuts, c = set(), 0
            for piece in d["segmentation"].split("+"):
                c += len(piece); cuts.add(c)
            cuts.discard(len(it["word"]))
            rows.append((it["cuts"], cuts, it["freq"], d["segmentation"].count("+") + 1))
        s = prf(rows); s["invalid"] = invalid; s["n_sample"] = len(sample); s["input_tokens"] = tin; s["output_tokens"] = tout
        summary[lang] = s
        (out_dir / f"summary_{lang}.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        print(f"[{lang}] P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} on {s['n_words']} words ({invalid} invalid)", flush=True)

if __name__ == "__main__":
    main()
