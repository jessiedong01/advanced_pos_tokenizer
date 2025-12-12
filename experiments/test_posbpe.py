#!/usr/bin/env python3
"""Check that posbpe reproduces the merge scores of src/tokenizer_models.py.

Both implementations are trained on the same small tagged sample. At every
merge step the original scores all pairs from scratch; the top score it finds
must equal the score of the pair the indexed implementation merges, up to
tie-breaking between pairs with the same score.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from tokenizer_models import BPETokenizer, encode_corpus_lines, merge_pair
from posbpe import PosBPE, transition_table

paras = [json.loads(l)["words"] for l in open(ROOT / "data/tagged/en.train.jsonl", encoding="utf-8").readlines()[:60]]
lines = [" ".join(w[0] for w in words) for words in paras]
trans = transition_table(paras)

# the original reads POS from JSON files; build them from the same tags
pos_json = ROOT / "data/tagged/_test_pos.json"; fac_json = ROOT / "data/tagged/_test_factors.json"
pos_json.write_text(json.dumps([[{"word": w[0], "pos": w[1]} for w in words] for words in paras], ensure_ascii=False))
fac_json.write_text(json.dumps({"factors": {f"{a}->{b}": 1.6 - min(max(p, 0.01), 0.95) for a, d in trans.items() for b, p in d.items()}}))

for beta, bf, comp in ((0.0, 1.0, None), (1.0, 0.98, "both")):
    probe = BPETokenizer(pos_json=str(pos_json), use_pos=True)       # only for its type-level tag map
    typed = [[[w[0], probe.surf2pos.get(w[0], "X")] for w in words] for words in paras] if comp else paras
    trans = transition_table(typed)
    fac_json.write_text(json.dumps({"factors": {f"{a}->{b}": 1.6 - min(max(p, 0.01), 0.95) for a, d in trans.items() for b, p in d.items()}}))
    orig = BPETokenizer(pos_json=str(pos_json), pos_factors_json=str(fac_json), use_pos=comp is not None, pos_component=comp or "both")
    # 1. every pair's score before any merge must match the original's scoring from scratch
    probe_new = PosBPE(beta=beta, boundary_factor=bf, transitions=trans, unseen_adj=0.6).index(typed)
    seqs0 = encode_corpus_lines(lines)
    ref = orig._score_pairs_pos(seqs0, lines) if comp else orig._score_pairs_baseline(seqs0)
    diff = max(abs(float(ref.get(k, 0.0)) - v) for k, v in probe_new.score.items())
    diff = max(diff, max(abs(float(v) - probe_new.score.get(k, 0.0)) for k, v in ref.items()))
    print(f"beta={beta} boundary_factor={bf}: {len(ref)} pairs at step 0, largest score difference {diff:.2e}")
    # 2. merge order
    new = PosBPE(beta=beta, boundary_factor=bf, transitions=trans, unseen_adj=0.6).train(typed, vocab_size=len(set("".join(lines))) + 1 + 150)
    # The original re-derives word indices from the symbol sequence and stops counting words once a
    # marker has been merged into a token, so its transition weights refer to the wrong words after the
    # first such merge. Agreement is therefore checked up to that merge.
    seqs = encode_corpus_lines(lines); mismatches = 0; first_absorb = None
    for step, (a, b) in enumerate(new.merges):
        scores = orig._score_pairs_pos(seqs, lines) if comp else orig._score_pairs_baseline(seqs)
        top = max(scores.values()); mine = scores.get((a, b), 0.0)
        if abs(top - mine) > 1e-6 and first_absorb is None:
            mismatches += 1
            print(f"step {step}: merged {(a, b)!r} score {mine:.3f} but best was {top:.3f}")
        if "▁" in a + b and first_absorb is None and comp:
            first_absorb = step
        seqs = [merge_pair(s, (a, b), a + b) for s in seqs]
    print(f"beta={beta} boundary_factor={bf}: {len(new.merges)} merges, {mismatches} score mismatches"
          + (f" before the first marker-absorbing merge (step {first_absorb})" if first_absorb is not None else ""))
