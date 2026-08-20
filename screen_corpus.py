#!/usr/bin/env python3
"""
screen_corpus.py — screen the WHOLE corpus for textual reuse.

All-pairs is impossible: 22,190 theses is 246 million pairs, ~86 days
single-core. This uses the standard near-duplicate architecture instead —
fingerprint once, index, and only run exact alignment on candidates.

WINNOWING (Schleimer, Wilkerson & Aiken 2003). Hash every 8-gram, slide a
window of 13 over the hash sequence and keep the minimum in each window. That
retains ~1/13 of fingerprints while GUARANTEEING that any shared passage of
n + w - 1 = 8 + 13 - 1 = 20 or more words shares at least one fingerprint.
Twenty words is exactly the threshold this project treats as a long run, so the
index is lossless for everything we care about and lossy only below it.

What this finds and what it does not
------------------------------------
FINDS   verbatim and near-verbatim textual reuse between deposited theses.
NOT     fabricated or falsified data, ghostwriting, or a purchased thesis.
        Those leave no textual trace and no amount of this will surface them.

Output is a RANKED LIST OF PAIRS TO LOOK AT. It is not a finding about any
person. Every filter here reduces false positives; none of them establishes
intent, and shared text has many innocent causes.

Stages
    fingerprint   winnowed fingerprints per document (parallel, cached)
    index         global postings, document-frequency pruning, candidate pairs
    exact         real verbatim runs for candidates only
    report        ranked residuals

Paths anchor to this file's location, not the shell's working directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
FP_DIR = HERE / "corpus" / "fingerprints"

NGRAM = R.NGRAM_N          # 8
WINDOW = 13                # guarantees detection of runs >= NGRAM+WINDOW-1 = 20
BASE = np.uint64(0x100000001B3)
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

# A fingerprint appearing in more than this share of documents is field-wide
# phrasing, not evidence. Pruning it is what stops the pair count exploding.
DF_MAX_FRAC = 0.004
MIN_SHARED_FP = 3          # candidate threshold


def log(m):
    print(m, file=sys.stderr, flush=True)


def _tok_hash(tok: str, _cache={}) -> int:
    h = _cache.get(tok)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"),
                                           digest_size=8).digest(), "big")
        _cache[tok] = h
    return h


def doc_tokens(doc_id: str) -> list[str]:
    """Retained (non-excluded) tokens, identical rules to the focal analysis."""
    with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
        rows = json.load(fh)["sentences"]
    out = []
    for r in rows:
        text, page, printed, bucket, heading, excl = H.unpack_sentence(r)
        if excl:
            continue
        out.extend(R.match_tokens(text))
    return out


def winnow(tokens: list[str]) -> np.ndarray:
    if len(tokens) < NGRAM + WINDOW:
        return np.empty(0, dtype=np.uint64)
    h = np.fromiter((_tok_hash(t) for t in tokens), dtype=np.uint64,
                    count=len(tokens))
    m = len(h) - NGRAM + 1
    grams = np.zeros(m, dtype=np.uint64)
    for k in range(NGRAM):                      # rolling polynomial hash
        grams = (grams * BASE + h[k:k + m]) & MASK
    if m < WINDOW:
        return np.unique(grams)
    sw = np.lib.stride_tricks.sliding_window_view(grams, WINDOW)
    # rightmost minimum per window keeps the selection stable
    idx = (WINDOW - 1) - np.argmin(sw[:, ::-1], axis=1)
    sel = np.unique(idx + np.arange(sw.shape[0]))
    return np.unique(grams[sel])


def fp_path(doc_id: str) -> Path:
    d = hashlib.sha256(doc_id.encode()).hexdigest()
    return FP_DIR / d[:2] / f"{d}.npy"


def _fp_one(doc_id: str):
    p = fp_path(doc_id)
    if p.exists():
        return doc_id, -1
    try:
        fps = winnow(doc_tokens(doc_id))
    except Exception as e:
        return doc_id, -2
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, fps)
    return doc_id, len(fps)


def fingerprint(limit, workers):
    con = H.db()
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    con.close()
    if limit:
        ids = ids[:limit]
    todo = [i for i in ids if not fp_path(i).exists()]
    log(f"  {len(ids):,} documents, {len(todo):,} need fingerprinting")
    if not todo:
        return
    t0 = time.monotonic()
    done = errs = total = 0
    with mp.Pool(workers) as pool:
        for doc_id, n in pool.imap_unordered(_fp_one, todo, chunksize=8):
            done += 1
            if n == -2:
                errs += 1
            elif n > 0:
                total += n
            if done % 500 == 0:
                el = time.monotonic() - t0
                log(f"    {done:,}/{len(todo):,}  {el/60:.1f} min  "
                    f"eta {(len(todo)-done)*el/done/60:.0f} min  errors {errs}")
    log(f"  done: {total:,} fingerprints stored, {errs} errors")


def build_index(limit):
    """Global postings, DF pruning, and candidate pair counting."""
    con = H.db()
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    con.close()
    if limit:
        ids = ids[:limit]
    ids = [i for i in ids if fp_path(i).exists()]
    n = len(ids)
    log(f"  loading fingerprints for {n:,} documents ...")
    hashes, owners = [], []
    for k, d in enumerate(ids):
        a = np.load(fp_path(d))
        if a.size:
            hashes.append(a)
            owners.append(np.full(a.size, k, dtype=np.int32))
        if (k + 1) % 2000 == 0:
            log(f"    {k+1:,}/{n:,}")
    Hh = np.concatenate(hashes)
    Oo = np.concatenate(owners)
    del hashes, owners
    log(f"  {Hh.size:,} fingerprints total ({Hh.nbytes/1e9:.2f} GB)")

    log("  sorting ...")
    order = np.argsort(Hh, kind="stable")
    Hh, Oo = Hh[order], Oo[order]
    del order

    log("  grouping and pruning ...")
    starts = np.flatnonzero(np.r_[True, Hh[1:] != Hh[:-1]])
    counts = np.diff(np.r_[starts, Hh.size])
    df_max = max(3, int(n * DF_MAX_FRAC))
    keep = (counts >= 2) & (counts <= df_max)
    log(f"    {starts.size:,} distinct fingerprints; "
        f"{int((counts >= 2).sum()):,} shared by >=2 docs; "
        f"{int(keep.sum()):,} kept after DF<= {df_max} pruning")

    log("  counting co-occurrences ...")
    pair_codes = []
    ks, kc = starts[keep], counts[keep]
    for s, c in zip(ks.tolist(), kc.tolist()):
        docs = np.unique(Oo[s:s + c])
        if docs.size < 2:
            continue
        a, b = np.triu_indices(docs.size, 1)
        pair_codes.append(docs[a].astype(np.int64) * n + docs[b].astype(np.int64))
    if not pair_codes:
        log("  no candidate pairs")
        return
    codes = np.concatenate(pair_codes)
    del pair_codes
    log(f"    {codes.size:,} raw (pair, fingerprint) hits")
    uniq, cnt = np.unique(codes, return_counts=True)
    del codes
    sel = cnt >= MIN_SHARED_FP
    uniq, cnt = uniq[sel], cnt[sel]
    log(f"    {uniq.size:,} candidate pairs with >= {MIN_SHARED_FP} shared fingerprints")

    out = HERE / "corpus" / "candidates.npz"
    np.savez_compressed(out, pairs=uniq, counts=cnt,
                        ids=np.array(ids, dtype=object), n=n)
    log(f"  written {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fingerprint")
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 4))
    i = sub.add_parser("index")
    i.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("tokenize")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--workers", type=int, default=max(2,(os.cpu_count() or 4)-4))
    e2 = sub.add_parser("exact2")
    e2.add_argument("--workers", type=int, default=max(2,(os.cpu_count() or 4)-4))
    e2.add_argument("--min-fp", type=int, default=3)
    e = sub.add_parser("exact")
    e.add_argument("--workers", type=int, default=max(2,(os.cpu_count() or 4)-4))
    e.add_argument("--min-fp", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "fingerprint":
        fingerprint(a.limit, a.workers)
    elif a.cmd == "tokenize":
        tokenize(a.limit, a.workers)
    elif a.cmd == "exact":
        exact(a.workers, a.min_fp)
    elif a.cmd == "exact2":
        exact2(a.workers, a.min_fp)
    else:
        build_index(a.limit)
    return 0



# --------------------------------------------------------------------------- #
# Exact verification of candidates
# --------------------------------------------------------------------------- #
TOK_DIR = HERE / "corpus" / "tokens"


def tok_path(doc_id: str) -> Path:
    d = hashlib.sha256(doc_id.encode()).hexdigest()
    return TOK_DIR / d[:2] / f"{d}.npy"


def _tok_one(doc_id: str):
    p = tok_path(doc_id)
    if p.exists():
        return doc_id, -1
    try:
        toks = doc_tokens(doc_id)
        # Token ids are 64-bit blake2b digests: deterministic, identical across
        # processes without sharing a vocabulary, and collision-free in practice.
        # uint64: blake2b digests span the full unsigned range and overflow int64.
        arr = np.fromiter((_tok_hash(t) for t in toks), dtype=np.uint64,
                          count=len(toks))
    except Exception as e:
        return doc_id, f"{type(e).__name__}: {e}"[:120]
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return doc_id, arr.size


def tokenize(limit, workers):
    con = H.db()
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    con.close()
    if limit:
        ids = ids[:limit]
    todo = [i for i in ids if not tok_path(i).exists()]
    log(f"  {len(todo):,} of {len(ids):,} need token arrays")
    if not todo:
        return
    t0, done, errs = time.monotonic(), 0, 0
    with mp.Pool(workers) as pool:
        first_err = None
        for _, n in pool.imap_unordered(_tok_one, todo, chunksize=8):
            done += 1
            if isinstance(n, str):
                errs += 1
                first_err = first_err or n
            if done % 2000 == 0:
                log(f"    {done:,}/{len(todo):,}  {(time.monotonic()-t0)/60:.1f} min")
    log(f"  token arrays written, {errs} errors"
        + (f" (first: {first_err})" if errs else ""))


_TOKCACHE: dict[str, np.ndarray] = {}


def _load_tok(doc_id):
    a = _TOKCACHE.get(doc_id)
    if a is None:
        a = np.load(tok_path(doc_id))
        _TOKCACHE[doc_id] = a
    return a


def _exact_group(job):
    a_id, b_ids = job
    try:
        A = _load_tok(a_id)
    except Exception:
        return []
    if A.size < NGRAM:
        return []
    shim_a = SimpleNamespace(tokens=A.tolist())
    out = []
    for b_id in b_ids:
        try:
            B = _load_tok(b_id)
        except Exception:
            continue
        if B.size < NGRAM:
            continue
        runs, _ = R.verbatim_runs(shim_a, SimpleNamespace(tokens=B.tolist()))
        if not runs:
            continue
        lens = [r.length for r in runs]
        cov = set()
        for r in runs:
            cov.update(range(r.b0, r.b1))
        out.append((a_id, b_id, int(A.size), int(B.size), len(runs),
                    sum(1 for L in lens if L >= R.LONG_RUN_WORDS), max(lens),
                    len(cov), len(cov) / max(B.size, 1)))
    _TOKCACHE.clear()
    return out


def exact(workers, min_fp):
    z = np.load(HERE / "corpus" / "candidates.npz", allow_pickle=True)
    pairs, counts, ids, n = z["pairs"], z["counts"], list(z["ids"]), int(z["n"])
    sel = counts >= min_fp
    pairs = pairs[sel]
    log(f"  {pairs.size:,} candidate pairs with >= {min_fp} shared fingerprints")
    ai = (pairs // n).astype(np.int64)
    bi = (pairs % n).astype(np.int64)
    groups: dict[str, list] = {}
    for x, y in zip(ai.tolist(), bi.tolist()):
        groups.setdefault(ids[x], []).append(ids[y])
    jobs = sorted(groups.items())
    log(f"  {len(jobs):,} anchor documents")
    con = H.db()
    t0, done, written = time.monotonic(), 0, 0
    with mp.Pool(workers) as pool:
        for rows in pool.imap_unordered(_exact_group, jobs, chunksize=4):
            done += 1
            if rows:
                con.executemany(
                    "INSERT INTO pair(a,b,a_tokens,b_tokens,runs,long_runs,"
                    "max_run,matched_words,share) VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(a,b) DO NOTHING", rows)
                con.commit()
                written += len(rows)
            if done % 200 == 0:
                el = time.monotonic() - t0
                log(f"    {done:,}/{len(jobs):,} anchors  {el/60:.1f} min  "
                    f"eta {(len(jobs)-done)*el/done/60:.0f} min  {written:,} pairs")
    log(f"  {written:,} pairs with at least one shared run written")
    con.close()


# --------------------------------------------------------------------------- #
# Fast exact matching
# --------------------------------------------------------------------------- #
# R.verbatim_runs rebuilds A's n-gram index on every call and keys it on tuples
# of 8 arbitrary-precision ints. Over 222,881 candidate pairs that is millions
# of tuple allocations per pair and dominates everything else. This version is
# semantically identical but:
#   * hoists A's index so it is built ONCE per anchor, not once per pair
#     (15x fewer builds at the observed mean of 15 partners per anchor);
#   * keys the index on a vectorised int64 rolling hash instead of tuples,
#     verifying the 8 tokens on seed acceptance so a hash collision cannot
#     manufacture a run.
# `assert_same_as_reference` checks the output matches R.verbatim_runs exactly.

def gram_hashes(tok: np.ndarray) -> np.ndarray:
    m = tok.size - NGRAM + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    t = tok.astype(np.uint64, copy=False)
    g = np.zeros(m, dtype=np.uint64)
    for k in range(NGRAM):
        g = (g * BASE + t[k:k + m]) & MASK
    return g


def build_a_index(A: np.ndarray):
    gh = gram_hashes(A)
    idx: dict[int, list] = {}
    cap = R.NGRAM_POSTING_CAP
    for i, h in enumerate(gh.tolist()):
        lst = idx.get(h)
        if lst is None:
            idx[h] = [i]
        elif len(lst) < cap:
            lst.append(i)
    return idx


# A handful of pairs are pathological: highly repetitive text (glossaries, word
# lists, numeric tables) generates hundreds of thousands of seed runs, and the
# containment-drop is O(k^2) in that count. One observed pair took 1,391
# SECONDS while every other pair in its batch took under 0.1s. Past this cap we
# stop seeding, skip the containment pass and flag the pair as truncated: for
# such a pair the exact run count is not decision-relevant — that it shares a
# great deal of text is — and the flag keeps the truncation visible.
MAX_RUNS = 20000


def _merge_len(iv):
    """Length of the union of half-open intervals, without materialising them."""
    if not iv:
        return 0
    iv = sorted(iv)
    total = 0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + ce - cs


def fast_runs(A: np.ndarray, Al: list, idx: dict, B: np.ndarray, Bl: list):
    gh = gram_hashes(B)
    seen: dict[int, int] = {}
    found = set()
    la, lb = len(Al), len(Bl)
    n = NGRAM
    for j, h in enumerate(gh.tolist()):
        posts = idx.get(h)
        if not posts:
            continue
        for i in posts:
            d = i - j
            if seen.get(d, -1) > j:
                continue
            if Al[i:i + n] != Bl[j:j + n]:       # guard against hash collision
                continue
            a0, b0 = i, j
            while a0 > 0 and b0 > 0 and Al[a0 - 1] == Bl[b0 - 1]:
                a0 -= 1
                b0 -= 1
            a1, b1 = i + n, j + n
            while a1 < la and b1 < lb and Al[a1] == Bl[b1]:
                a1 += 1
                b1 += 1
            found.add((a0, a1, b0, b1))
            seen[d] = b1
        if len(found) > MAX_RUNS:
            truncated = True
            break
    else:
        truncated = False
    if truncated:
        kept = sorted(found, key=lambda r: (r[2], r[0], r[3], r[1]))
        return kept, True
    cand = sorted(found, key=lambda r: (-(r[1] - r[0]), r[0], r[2]))
    kept = []
    for a0, a1, b0, b1 in cand:
        if not any(k[0] <= a0 and a1 <= k[1] and k[2] <= b0 and b1 <= k[3]
                   for k in kept):
            kept.append((a0, a1, b0, b1))
    kept.sort(key=lambda r: (r[2], r[0], r[3], r[1]))
    return kept, False


def assert_same_as_reference(sample=40):
    """fast_runs must agree with R.verbatim_runs exactly, or corpus results are
    not comparable with the focal-pair analysis."""
    z = np.load(HERE / "corpus" / "candidates.npz", allow_pickle=True)
    pairs, counts, ids, n = z["pairs"], z["counts"], list(z["ids"]), int(z["n"])
    order = np.argsort(-counts)
    picks = list(order[:6]) + list(order[len(order)//3::max(1, len(order)//sample)])
    bad = 0
    for k in picks[:sample]:
        a, b = ids[int(pairs[k]) // n], ids[int(pairs[k]) % n]
        A, B = np.load(tok_path(a)), np.load(tok_path(b))
        Al, Bl = A.tolist(), B.tolist()
        mine, _tr = fast_runs(A, Al, build_a_index(A), B, Bl)
        ref, _ = R.verbatim_runs(SimpleNamespace(tokens=Al),
                                 SimpleNamespace(tokens=Bl))
        ref = [(r.a0, r.a1, r.b0, r.b1) for r in ref]
        if mine != ref:
            bad += 1
            log(f"    MISMATCH {a} ~ {b}: {len(mine)} vs {len(ref)}")
    log(f"  equivalence check: {len(picks[:sample])-bad}/{len(picks[:sample])} identical")
    return bad == 0


def _exact_group2(job):
    import gc
    gc.disable()                    # millions of transient objects; GC scanning dominates
    a_id, b_ids = job
    try:
        A = np.load(tok_path(a_id))
    except Exception:
        return []
    if A.size < NGRAM + 1:
        return []
    Al = A.tolist()
    idx = build_a_index(A)          # built ONCE for this anchor
    out = []
    for b_id in b_ids:
        try:
            B = np.load(tok_path(b_id))
        except Exception:
            continue
        if B.size < NGRAM + 1:
            continue
        Bl = B.tolist()
        runs, trunc = fast_runs(A, Al, idx, B, Bl)
        if not runs:
            continue
        lens = [r[1] - r[0] for r in runs]
        cov = _merge_len([(r[2], r[3]) for r in runs])
        out.append((a_id, b_id, int(A.size), int(B.size),
                    -len(runs) if trunc else len(runs),
                    sum(1 for L in lens if L >= R.LONG_RUN_WORDS), max(lens),
                    cov, cov / max(B.size, 1)))
    return out


def exact2(workers, min_fp):
    if not assert_same_as_reference():
        log("  ABORT: fast matcher disagrees with the reference implementation")
        return
    z = np.load(HERE / "corpus" / "candidates.npz", allow_pickle=True)
    pairs, counts, ids, n = z["pairs"], z["counts"], list(z["ids"]), int(z["n"])
    pairs = pairs[counts >= min_fp]
    con = H.db()
    have = {(a, b) for a, b in con.execute("SELECT a,b FROM pair")}
    groups: dict[str, list] = {}
    for code in pairs.tolist():
        a, b = ids[code // n], ids[code % n]
        if (a, b) in have:
            continue
        groups.setdefault(a, []).append(b)
    jobs = sorted(groups.items())
    todo = sum(len(v) for v in groups.values())
    log(f"  {todo:,} pairs still to verify across {len(jobs):,} anchors")
    t0, done, written = time.monotonic(), 0, 0
    with mp.Pool(workers) as pool:
        for rows in pool.imap_unordered(_exact_group2, jobs, chunksize=8):
            done += 1
            if rows:
                con.executemany(
                    "INSERT INTO pair(a,b,a_tokens,b_tokens,runs,long_runs,"
                    "max_run,matched_words,share) VALUES(?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(a,b) DO NOTHING", rows)
                con.commit()
                written += len(rows)
            if done % 500 == 0:
                el = time.monotonic() - t0
                log(f"    {done:,}/{len(jobs):,} anchors  {el/60:.1f} min  "
                    f"eta {(len(jobs)-done)*el/done/60:.0f} min  {written:,} pairs")
    log(f"  {written:,} pairs with at least one shared run written")
    con.close()

if __name__ == "__main__":
    sys.exit(main())
