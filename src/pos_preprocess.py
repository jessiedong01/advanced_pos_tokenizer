#!/usr/bin/env python3
import json, sys, os
from tqdm import tqdm
from rich import print

def main():
    if len(sys.argv) < 3:
        print("Usage: pos_preprocess.py <input_txt> <output_json> [lang=en]")
        sys.exit(1)
    input_file, output_file = sys.argv[1], sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"

    try:
        import stanza
    except ImportError:
        print("stanza is not installed. Run: pip install stanza")
        sys.exit(2)

    if not os.path.exists(input_file):
        print(f"[ERROR] missing input file: {input_file}")
        sys.exit(3)

    with open(input_file, "r", encoding="utf8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"[INFO] {len(sentences)} sentences")

    stanza.download(lang, verbose=False)
    nlp = stanza.Pipeline(lang, processors="tokenize,pos", use_gpu=True)

    out = []
    for i in tqdm(range(0, len(sentences), 16), desc="Tagging"):
        batch = sentences[i:i+16]
        doc = nlp("\n".join(batch))
        for sent in doc.sentences:
            out.append([{"word": w.text, "pos": w.upos} for w in sent.words])

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {output_file}")

if __name__ == "__main__":
    main()
