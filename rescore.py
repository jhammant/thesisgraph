#!/usr/bin/env python3
"""
rescore.py — re-score screened pairs after excluding extraction artefacts.

Inspecting the highest-scoring pairs showed the top of the ranking was not
plagiarism at all. The "matched text" read:

    i n f o r m a t i o n s y s t e m s

Some PDF producers space characters widely enough that pdfplumber emits every
LETTER as a separate word. Two theses with that pathology share tens of
thousands of spurious runs, and a single long word becomes a 20+ "word" run.
Across a 400-thesis sample, 3.8% of theses have >25% single-character tokens
(median 6.8%, which is normal: "a", "I", maths variables).

A run is counted here only if it looks like prose:
    * at most 30% single-character tokens, and
    * at least 5 tokens of 4+ characters.

Both figures are reported so the effect of the filter is visible rather than
silently applied.
"""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_corpus as S, harvest_corpus as H, run_analysis as R

MAX_SINGLE_FRAC = 0.30
MIN_LONG_TOKENS = 5

_S: dict[str, list] = {}


def strings(doc_id):
    v = _S.get(doc_id)
    if v is None:
        v = S.doc_tokens(doc_id)
        if len(_S) > 300:
            _S.clear()
        _S[doc_id] = v
    return v


def clean(run, toks):
    seg = toks[run[2]:run[3]]
    if not seg:
        return False
    ones = sum(1 for t in seg if len(t) == 1)
    if ones / len(seg) > MAX_SINGLE_FRAC:
        return False
    return sum(1 for t in seg if len(t) >= 4) >= MIN_LONG_TOKENS


def main():
    con = H.db()
    for col, typ in (("clean_runs", "INTEGER"), ("clean_long_runs", "INTEGER"),
                     ("clean_max_run", "INTEGER"), ("clean_words", "INTEGER"),
                     ("clean_share", "REAL")):
        try:
            con.execute(f"ALTER TABLE pair ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    con.commit()
    rows = con.execute(
        "SELECT a,b FROM pair WHERE clean_runs IS NULL AND "
        "(matched_words>=800 OR max_run>=40 OR long_runs>=5) "
        "ORDER BY a,b").fetchall()
    print(f"  {len(rows):,} pairs to re-score", file=sys.stderr)
    t0 = time.monotonic()
    batch, cur_a, A, Al, idx = [], None, None, None, None
    for k, (a, b) in enumerate(rows, 1):
        try:
            if a != cur_a:
                cur_a, A = a, np.load(S.tok_path(a))
                Al = A.tolist()
                idx = S.build_a_index(A)
            B = np.load(S.tok_path(b))
            runs, _tr = S.fast_runs(A, Al, idx, B, B.tolist())
            tb = strings(b)
            good = [r for r in runs if clean(r, tb)]
            lens = [r[1] - r[0] for r in good]
            cov = S._merge_len([(r[2], r[3]) for r in good])
            batch.append((len(good), sum(1 for L in lens if L >= 20),
                          max(lens, default=0), cov, cov / max(B.size, 1), a, b))
        except Exception:
            batch.append((0, 0, 0, 0, 0.0, a, b))
        if len(batch) >= 500:
            con.executemany("UPDATE pair SET clean_runs=?, clean_long_runs=?, "
                            "clean_max_run=?, clean_words=?, clean_share=? "
                            "WHERE a=? AND b=?", batch)
            con.commit(); batch = []
            el = time.monotonic() - t0
            print(f"    {k:,}/{len(rows):,}  {el/60:.1f} min  "
                  f"eta {(len(rows)-k)*el/k/60:.0f} min", file=sys.stderr)
    if batch:
        con.executemany("UPDATE pair SET clean_runs=?, clean_long_runs=?, "
                        "clean_max_run=?, clean_words=?, clean_share=? "
                        "WHERE a=? AND b=?", batch)
        con.commit()
    print("  done", file=sys.stderr)
    con.close()


if __name__ == "__main__":
    main()
