#!/usr/bin/env python3
"""POS-weighted byte-pair encoding.

A re-implementation of src/tokenizer_models.py with the same merge semantics
but an occurrence index, so that a vocabulary of several thousand merges
trains in minutes rather than hours.

Each paragraph becomes a symbol sequence: the characters of every whitespace
word, with the marker ▁ as a symbol of its own between words. A merge joins
two adjacent symbols everywhere they occur. The score of a candidate pair is
the sum over its occurrences of a weight:

  pair inside one word                         w_intra[pos of the word]
  pair joining a word to the following ▁       (1 + beta * P(pos_left -> pos_right)) * boundary_factor
  pair joining ▁ to the following word         boundary_factor
  any other pair (spans an absorbed marker)    1

The original implementation is beta = 1, boundary_factor = 0.98 and w_intra
all 1 ("pos_component=both"), and beta = 0, boundary_factor = 1 is plain BPE
over the same symbol sequence. The transition probability is clamped to
[0.01, 0.95] as in src/pos_transition.py; the original assigned unseen
transitions a weight of 0.6, which is replaced here by the floor 0.01.
"""
from __future__ import annotations
import heapq, json
from collections import defaultdict
from pathlib import Path

MARK = "▁"

def transition_table(paragraphs):
    """P(pos_right | pos_left) over adjacent whitespace words."""
    counts = defaultdict(lambda: defaultdict(int))
    for words in paragraphs:
        for (_, a, *_), (_, b, *_) in zip(words, words[1:]):
            counts[a][b] += 1
    return {a: {b: n / sum(d.values()) for b, n in d.items()} for a, d in counts.items()}

def word_symbols(word):
    return list(word)

class PosBPE:
    def __init__(self, beta=0.0, boundary_factor=1.0, w_intra=None, transitions=None, unseen_adj=0.01):
        self.beta = beta
        self.unseen_adj = unseen_adj
        self.boundary_factor = boundary_factor
        self.w_intra = w_intra or {}
        self.transitions = transitions or {}
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()

    # ------------------------------------------------------------------ training
    def _adj(self, a, b):
        p = self.transitions.get(a, {}).get(b)
        if p is None:
            return self.unseen_adj
        return min(max(p, 0.01), 0.95)

    def _weight(self, i, j):
        si, sj = self.sym[i], self.sym[j]
        if si == MARK:
            return self.boundary_factor
        if sj == MARK:
            lw, rw = self.wid[i], self.marker_next[j]
            w = 1.0
            if self.beta and lw >= 0 and rw >= 0:
                w += self.beta * self._adj(self.pos[lw], self.pos[rw])
            return w * self.boundary_factor
        if self.wid[i] == self.wid[j] and self.wid[i] >= 0:
            return self.w_intra.get(self.pos[self.wid[i]], 1.0)
        return 1.0

    def _add(self, i):
        j = self.nxt[i]
        if j < 0 or self.sym[i] is None or self.sym[j] is None:
            return
        pair = (self.sym[i], self.sym[j])
        self.occ[pair].add(i)
        self.score[pair] = self.score.get(pair, 0.0) + self._weight(i, j)
        self.version[pair] = self.version.get(pair, 0) + 1
        heapq.heappush(self.heap, (-self.score[pair], pair, self.version[pair]))

    def _remove(self, i):
        j = self.nxt[i]
        if j < 0 or self.sym[i] is None or self.sym[j] is None:
            return
        pair = (self.sym[i], self.sym[j])
        self.occ[pair].discard(i)
        self.score[pair] -= self._weight(i, j)
        self.version[pair] = self.version.get(pair, 0) + 1
        if self.occ[pair]:
            heapq.heappush(self.heap, (-self.score[pair], pair, self.version[pair]))

    def index(self, paragraphs):
        """Build the symbol sequence and the occurrence index with every pair's score.

        paragraphs: list of word lists [surface, upos, ...] as written by tag_corpus.py."""
        sym, wid, pos, marker_next = [], [], [], []
        for words in paragraphs:
            for k, (surface, upos, *_) in enumerate(words):
                if k > 0:
                    sym.append(MARK); wid.append(-1); marker_next.append(len(pos))
                for ch in surface:
                    if ch == "\x00":
                        sym.append(None); wid.append(-1); marker_next.append(-1)
                    else:
                        sym.append(ch); wid.append(len(pos)); marker_next.append(-1)
                pos.append(upos)
            sym.append(None); wid.append(-1); marker_next.append(-1)   # paragraph sentinel never merges
        n = len(sym)
        self.sym, self.wid, self.pos, self.marker_next = sym, wid, pos, marker_next
        self.nxt = list(range(1, n)) + [-1]
        self.prv = [-1] + list(range(n - 1))
        self.occ, self.score, self.version, self.heap = defaultdict(set), {}, {}, []
        self.vocab = {s for s in sym if s is not None}
        for i in range(n - 1):
            self._add(i)
        return self

    def train(self, paragraphs, vocab_size, log=None):
        self.index(paragraphs)
        while len(self.vocab) < vocab_size and self.heap:
            neg, pair, ver = heapq.heappop(self.heap)
            if self.version.get(pair) != ver or not self.occ.get(pair):
                continue
            a, b = pair
            new = a + b
            if new in self.vocab:            # the same string can arise from two different splits
                self.occ.pop(pair, None); self.score.pop(pair, None)
                continue
            for i in sorted(self.occ[pair]):
                j = self.nxt[i]
                if j < 0 or self.sym[i] != a or self.sym[j] != b:
                    continue
                p, k = self.prv[i], self.nxt[j]
                if p >= 0: self._remove(p)
                self._remove(i); self._remove(j)
                self.sym[i] = new; self.sym[j] = None
                self.nxt[i] = k
                if k >= 0: self.prv[k] = i
                if p >= 0: self._add(p)
                self._add(i)
            self.occ.pop(pair, None); self.score.pop(pair, None)
            self.merges.append(pair); self.vocab.add(new)
            if log and len(self.merges) % 1000 == 0:
                log(f"{len(self.merges)} merges, vocab {len(self.vocab)}")
        del self.sym, self.wid, self.nxt, self.prv, self.occ, self.score, self.version, self.heap
        return self

    # ------------------------------------------------------------------ encoding
    def _ranks(self):
        if not hasattr(self, "_rank"):
            self._rank = {pair: r for r, pair in enumerate(self.merges)}
        return self._rank

    def encode_symbols(self, symbols):
        """Apply the merges to a symbol sequence in rank order (lowest rank first, left to right
        among equal ranks). None entries are unmergeable breaks and are dropped from the output."""
        rank = self._ranks()
        sym = list(symbols)
        n = len(sym)
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(n - 1))
        heap = []
        for i in range(n - 1):
            r = rank.get((sym[i], sym[i + 1])) if sym[i] is not None and sym[i + 1] is not None else None
            if r is not None:
                heap.append((r, i))
        heapq.heapify(heap)
        alive = [True] * n
        while heap:
            r, i = heapq.heappop(heap)
            j = nxt[i]
            if not alive[i] or j < 0 or rank.get((sym[i], sym[j])) != r:
                continue
            sym[i] = sym[i] + sym[j]
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
                rk = rank.get((sym[i], sym[k])) if sym[k] is not None else None
                if rk is not None:
                    heapq.heappush(heap, (rk, i))
            p = prv[i]
            if p >= 0 and sym[p] is not None:
                rp = rank.get((sym[p], sym[i]))
                if rp is not None:
                    heapq.heappush(heap, (rp, p))
        return [sym[i] for i in range(n) if alive[i] and sym[i] is not None]

    def _encode_symbols_slow(self, symbols):
        """Reference implementation: rescan for the lowest-rank pair after every batch of merges."""
        rank = self._ranks()
        seq = list(symbols)
        while len(seq) > 1:
            best, best_r = None, None
            for a, b in zip(seq, seq[1:]):
                r = rank.get((a, b)) if a is not None and b is not None else None
                if r is not None and (best_r is None or r < best_r):
                    best, best_r = (a, b), r
            if best is None:
                break
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                    out.append(seq[i] + seq[i + 1]); i += 2
                else:
                    out.append(seq[i]); i += 1
            seq = out
        return [t for t in seq if t is not None]

    def encode_paragraph(self, text):
        symbols = []
        for k, w in enumerate(text.split()):
            if k > 0:
                symbols.append(MARK)
            symbols.extend(w)
        return self.encode_symbols(symbols)

    def encode_word(self, word):
        return self.encode_symbols(list(word))

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        (path / "merges.json").write_text(json.dumps(self.merges, ensure_ascii=False), encoding="utf-8")
        (path / "vocab.json").write_text(json.dumps(sorted(self.vocab), ensure_ascii=False), encoding="utf-8")
        (path / "config.json").write_text(json.dumps({"beta": self.beta, "boundary_factor": self.boundary_factor,
                                                      "w_intra": self.w_intra, "unseen_adj": self.unseen_adj}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        t = cls(cfg["beta"], cfg["boundary_factor"], cfg["w_intra"], unseen_adj=cfg.get("unseen_adj", 0.01))
        t.merges = [tuple(p) for p in json.loads((path / "merges.json").read_text(encoding="utf-8"))]
        t.vocab = set(json.loads((path / "vocab.json").read_text(encoding="utf-8")))
        return t
