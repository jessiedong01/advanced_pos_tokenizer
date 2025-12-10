#!/usr/bin/env python3
import os, subprocess, json, argparse

LANG_FILES = {
    "en": ("data/train_en.txt", "data/en_pos.json", "data/en_transitions.json", "results/en_models", "results/en_report.csv"),
    "es": ("data/train_es.txt", "data/es_pos.json", "data/es_transitions.json", "results/es_models", "results/es_report.csv"),
    "tr": ("data/train_tr.txt", "data/tr_pos.json", "data/tr_transitions.json", "results/tr_models", "results/tr_report.csv"),
}

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=1000)
    args = ap.parse_args()

    for lang, (txt, pos_json, trans_json, model_dir, rep_csv) in LANG_FILES.items():
        # 1 tag
        run(["python", "src/pos_preprocess.py", txt, pos_json, lang])
        # 2 transitions
        run(["python", "src/pos_transition.py", pos_json, trans_json])
        # 3 train baseline + pos (both components). You can flip to "transitions" for ablation.
        run(["python", "src/train_tokenizers.py", "--corpus", txt, "--pos_json", pos_json, "--pos_transitions", trans_json,
             "--vocab_size", str(args.vocab_size), "--outdir", model_dir, "--pos_component", "both"])
        # 4 eval
        run(["python", "src/evaluate_tokenizers.py", "--corpus", txt, "--pos_json", pos_json, "--models_dir", model_dir, "--report", rep_csv])

    # Aggregate reports if all exist
    import pandas as pd
    rows = []
    for lang, (_, _, _, _, rep_csv) in LANG_FILES.items():
        if os.path.exists(rep_csv):
            df = pd.read_csv(rep_csv)
            df["lang"] = lang
            rows.append(df)
    if rows:
        big = pd.concat(rows, ignore_index=True)
        big.to_csv("results/summary.csv", index=False)
        print(big)

if __name__ == "__main__":
    main()
