#!/usr/bin/env python3
"""Lemma2Word: morpheme boundaries from lemma-to-word alignment.

Stanza gives every training word a lemma. The stem of a word is the lemma
when the lemma occurs inside the word, and otherwise the longest substring
the two share (at least three characters). The parts before and after the
stem are the affixes, so a word gets up to two boundaries. Two things are
learned from the training text: a lexicon that maps each word form to its
boundaries, and inventories of prefixes and suffixes with their counts.

A word is segmented from the lexicon when it is in it. Otherwise the longest
suffix in the inventory that leaves a stem of at least three characters is
split off, and likewise the longest prefix, provided the affix was seen at
least `min_count` times. With a lemmatizer available at tokenization time
(`segment_online`), the lemma of the word itself gives the stem directly.
"""
from __future__ import annotations
import json, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def is_word(s):
    """Letters and combining marks only, so that scripts with vowel signs (Tamil) are not excluded."""
    return bool(s) and all(ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc") for ch in s)

def common_stem(word, lemma, min_len=3):
    w, l = word.lower(), lemma.lower()
    if not w or not l:
        return None
    k = w.find(l)
    if l and k >= 0:
        return k, k + len(l)
    best = (0, 0)
    for i in range(len(w)):
        for j in range(len(l)):
            n = 0
            while i + n < len(w) and j + n < len(l) and w[i + n] == l[j + n]:
                n += 1
            if n > best[1] - best[0]:
                best = (i, i + n)
    return best if best[1] - best[0] >= min_len else None

def boundaries_from_stem(word, stem_span):
    s, e = stem_span
    cuts = set()
    if s > 0:
        cuts.add(s)
    if e < len(word):
        cuts.add(e)
    return cuts

class Lemma2Word:
    def __init__(self, min_count=20, min_stem=3):
        self.min_count, self.min_stem = min_count, min_stem
        self.lexicon: dict[str, list[int]] = {}
        self.prefixes: Counter = Counter()
        self.suffixes: Counter = Counter()

    def fit(self, paragraphs):
        votes = defaultdict(Counter)
        for words in paragraphs:
            for surface, _, lemma, *_ in words:
                if not is_word(surface) or not lemma:
                    continue
                span = common_stem(surface, lemma, self.min_stem)
                if span is None:
                    continue
                cuts = boundaries_from_stem(surface, span)
                votes[surface.lower()][tuple(sorted(cuts))] += 1
                s, e = span
                if s > 0:
                    self.prefixes[surface[:s].lower()] += 1
                if e < len(surface):
                    self.suffixes[surface[e:].lower()] += 1
        self.lexicon = {w: list(c.most_common(1)[0][0]) for w, c in votes.items()}
        self._suf = sorted((s for s, n in self.suffixes.items() if n >= self.min_count), key=len, reverse=True)
        self._pre = sorted((p for p, n in self.prefixes.items() if n >= self.min_count), key=len, reverse=True)
        return self

    def cuts(self, word):
        w = word.lower()
        if w in self.lexicon:
            return set(self.lexicon[w])
        cuts, lo, hi = set(), 0, len(w)
        for s in self._suf:
            if hi - len(s) - lo >= self.min_stem and w.endswith(s):
                hi -= len(s); cuts.add(hi); break
        for p in self._pre:
            if hi - lo - len(p) >= self.min_stem and w[lo:hi].startswith(p):
                lo += len(p); cuts.add(lo); break
        return cuts

    def cuts_online(self, word, lemma):
        span = common_stem(word, lemma, self.min_stem)
        return boundaries_from_stem(word, span) if span else self.cuts(word)

    @staticmethod
    def pieces(word, cuts):
        idx = [0] + sorted(c for c in cuts if 0 < c < len(word)) + [len(word)]
        return [word[a:b] for a, b in zip(idx, idx[1:])]

    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        (path / "l2w.json").write_text(json.dumps({"min_count": self.min_count, "min_stem": self.min_stem,
            "lexicon": self.lexicon, "prefixes": self.prefixes, "suffixes": self.suffixes}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads((Path(path) / "l2w.json").read_text(encoding="utf-8"))
        m = cls(d["min_count"], d["min_stem"])
        m.lexicon, m.prefixes, m.suffixes = d["lexicon"], Counter(d["prefixes"]), Counter(d["suffixes"])
        m._suf = sorted((s for s, n in m.suffixes.items() if n >= m.min_count), key=len, reverse=True)
        m._pre = sorted((p for p, n in m.prefixes.items() if n >= m.min_count), key=len, reverse=True)
        return m
