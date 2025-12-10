#!/usr/bin/env python3
import json, sys, collections

def load_tagged(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def compute_transitions(tagged):
    counts = collections.defaultdict(lambda: collections.Counter())
    for sent in tagged:
        prev = None
        for tok in sent:
            pos = tok["pos"]
            if prev is not None:
                counts[prev][pos] += 1
            prev = pos
    probs = {}
    for a, ctr in counts.items():
        total = sum(ctr.values())
        if total == 0: 
            continue
        probs[a] = {b: ctr[b]/total for b in ctr}
    return probs

def make_reward_table(probs, floor=0.01, cap=0.95):
    rewards = {}
    for a, dic in probs.items():
        for b, p in dic.items():
            p = min(max(p, floor), cap)
            # factor in [0.6, 1.6] with higher prob -> lower factor (reward)
            factor = 1.6 - p
            rewards[f"{a}->{b}"] = round(float(factor), 4)
    return rewards

def main():
    if len(sys.argv) < 3:
        print("Usage: pos_transition.py <corpus_pos.json> <out_json>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    tagged = load_tagged(inp)
    probs = compute_transitions(tagged)
    rewards = make_reward_table(probs)
    with open(outp, "w", encoding="utf8") as f:
        json.dump({"probs": probs, "factors": rewards}, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote {outp}")

if __name__ == "__main__":
    main()
