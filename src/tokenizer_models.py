from collections import Counter, defaultdict
import json, os, time

SPACE = "▁"

def whitespace_tokenize(text):
    return text.strip().split()

def to_char_stream(sent):
    out = []
    for i, w in enumerate(whitespace_tokenize(sent)):
        if i > 0:
            out.append(SPACE)
        out.extend(list(w))
    return out

def encode_corpus_lines(lines):
    return [to_char_stream(line) for line in lines if line.strip()]

def merge_pair(seq, pair, token):
    merged = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
            merged.append(token)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged

def count_pairs(seqs):
    pairs = Counter()
    for s in seqs:
        for i in range(len(s) - 1):
            pairs[(s[i], s[i+1])] += 1
    return pairs

def load_pos_maps(pos_json_path):
    if not pos_json_path or not os.path.exists(pos_json_path):
        return None, None
    with open(pos_json_path, "r", encoding="utf8") as f:
        tagged = json.load(f)
    surface_ctr = defaultdict(Counter)
    sentences_pos = []
    for sent in tagged:
        pos_list = []
        for tok in sent:
            surface_ctr[tok["word"]][tok["pos"]] += 1
            pos_list.append(tok["pos"])
        sentences_pos.append(pos_list)
    surf2pos = {w: ctr.most_common(1)[0][0] for w, ctr in surface_ctr.items()}
    return surf2pos, sentences_pos

def pair_pos_type(left, right):
    if left == SPACE or right == SPACE:
        return "boundary"
    return "intra"

class BPETokenizer:
    def __init__(self, pos_json=None, pos_factors_json=None, use_pos=False, pos_component="both"):
        self.vocab = set()
        self.merges = []
        self.use_pos = use_pos
        self.pos_component = pos_component  # "transitions", "penalties", "both"
        self.surf2pos, self.sentences_pos = load_pos_maps(pos_json) if use_pos else (None, None)
        self.pos_factors = {}
        if use_pos and pos_factors_json and os.path.exists(pos_factors_json):
            with open(pos_factors_json, "r", encoding="utf8") as f:
                d = json.load(f)
            self.pos_factors = d.get("factors", {})

        self.iter_times = []

    def _score_pairs_baseline(self, seqs):
        return count_pairs(seqs)

    def _score_pairs_pos(self, seqs, orig_lines):
        counts = Counter()
        # precompute word pos per sentence
        sent_word_pos = []
        for line in orig_lines:
            words = whitespace_tokenize(line)
            pos_seq = [self.surf2pos.get(w, "X") if self.surf2pos else "X" for w in words]
            sent_word_pos.append((words, pos_seq))

        for si, s in enumerate(seqs):
            words, pos_seq = sent_word_pos[si]
            # map char tokens to word index (None at SPACE)
            wi = 0
            char_to_word = []
            for t in s:
                if t == SPACE:
                    wi += 1
                    char_to_word.append(None)
                else:
                    char_to_word.append(wi)

            for i in range(len(s) - 1):
                a, b = s[i], s[i+1]
                pair = (a, b)
                base = 1.0
                if pair_pos_type(a, b) == "boundary":
                    if self.pos_component in ("transitions", "both"):
                        lw = char_to_word[i]
                        rw = None
                        j = i+1
                        while j < len(s) and char_to_word[j] is None:
                            j += 1
                        if j < len(s):
                            rw = char_to_word[j]
                        if lw is not None and rw is not None and lw < len(pos_seq) and rw < len(pos_seq):
                            a_pos = pos_seq[lw]
                            b_pos = pos_seq[rw]
                            key = f"{a_pos}->{b_pos}"
                            factor = float(self.pos_factors.get(key, 1.0))
                            adj = 1.6 - min(max(factor, 0.6), 1.6)
                            base += adj
                    if self.pos_component in ("penalties", "both"):
                        # discourage merging across punctuation boundaries implicitly
                        # handled via POS tag "PUNCT" if present in maps; here we keep a small generic penalty
                        base *= 0.98
                counts[pair] += base
        return counts

    def train(self, lines, vocab_size=1000, profile_every=50):
        seqs = encode_corpus_lines(lines)
        self.vocab = set([SPACE])
        for s in seqs:
            for ch in s:
                self.vocab.add(ch)

        step = 0
        while len(self.vocab) < vocab_size:
            t0 = time.perf_counter()
            pair_scores = self._score_pairs_pos(seqs, lines) if self.use_pos else self._score_pairs_baseline(seqs)
            if not pair_scores:
                break
            best_pair, _ = pair_scores.most_common(1)[0]
            new_token = "".join(best_pair)
            self.vocab.add(new_token)
            self.merges.append((best_pair, new_token))
            seqs = [merge_pair(s, best_pair, new_token) for s in seqs]
            t1 = time.perf_counter()
            self.iter_times.append(t1 - t0)
            step += 1
        return {
            "vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "avg_iter_ms": (sum(self.iter_times)/max(1, len(self.iter_times))) * 1000.0
        }

    def save(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "merges.json"), "w", encoding="utf8") as f:
            json.dump([["".join(p[0]), p[1]] for p in self.merges], f, indent=2, ensure_ascii=False)
        with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf8") as f:
            json.dump(sorted(list(self.vocab)), f, indent=2, ensure_ascii=False)
