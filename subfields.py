#!/usr/bin/env python3
"""
subfields.py — derive SUB-disciplines from the data rather than hand-writing them.

The 16 top-level categories are too coarse: "Biological sciences" spans genetics,
ecology and microbiology. Here each discipline is clustered on the embedding of
its theses' titles and abstracts, and each cluster is labelled with the terms
that distinguish it FROM ITS OWN PARENT discipline — so "Education" yields
sub-fields rather than repeating the word "education" fifteen times.

Deterministic throughout: farthest-first initialisation, not random seeding, so
the same corpus always yields the same sub-fields.
"""
from __future__ import annotations
import argparse, json, re, sqlite3, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
EMB = HERE / "corpus" / "title_emb.npy"
IDS = HERE / "corpus" / "title_emb_ids.json"
STOP = frozenset("""the of and to in a for on with an by from as at is are be its
this that study studies research thesis investigation analysis approach towards
toward using use used case new development role effect effects impact between
into within their there which what how why some more than other via based
understanding exploring examining assessment evaluation review its it""".split())


def embed_all():
    con = H.db()
    rows = con.execute("SELECT id,title,abstract FROM doc WHERE status='ok' "
                       "ORDER BY id").fetchall()
    con.close()
    ids = [r[0] for r in rows]
    texts = [((r[1] or "") + ". " + (r[2] or ""))[:900] for r in rows]
    model = R._get_embed_model()
    if model is None:
        print("no offline embedding model available", file=sys.stderr); return None
    print(f"  embedding {len(texts):,} titles+abstracts ...", file=sys.stderr)
    e = model.encode(texts, batch_size=256, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False).astype("float32")
    np.save(EMB, e); IDS.write_text(json.dumps(ids))
    return e, ids


def kmeans(X, k, iters=40):
    """Farthest-first init + Lloyd. Deterministic: no RNG anywhere."""
    n = X.shape[0]
    c0 = X.mean(0); c0 /= (np.linalg.norm(c0) or 1)
    cent = [X[int(np.argmin(X @ c0))]]
    for _ in range(k - 1):
        d = np.min(np.stack([1 - X @ c for c in cent]), axis=0)
        cent.append(X[int(np.argmax(d))])
    C = np.stack(cent)
    lab = None
    for _ in range(iters):
        lab = np.argmax(X @ C.T, axis=1)
        newC = np.zeros_like(C)
        for j in range(k):
            m = lab == j
            newC[j] = X[m].mean(0) if m.any() else C[j]
            nrm = np.linalg.norm(newC[j])
            if nrm: newC[j] /= nrm
        if np.allclose(newC, C, atol=1e-6): C = newC; break
        C = newC
    return np.argmax(X @ C.T, axis=1)


def terms(titles):
    c = Counter()
    for t in titles:
        for w in set(re.findall(r"[a-z][a-z\-]{3,}", (t or "").lower())):
            if w not in STOP:
                c[w] += 1
    return c


def build(min_size, per_cluster):
    if EMB.exists() and IDS.exists():
        E = np.load(EMB); ids = json.loads(IDS.read_text())
    else:
        got = embed_all()
        if not got: return
        E, ids = got
    pos = {d: i for i, d in enumerate(ids)}
    con = H.db()
    try:
        con.execute("ALTER TABLE doc ADD COLUMN subfield TEXT")
    except sqlite3.OperationalError:
        pass
    con.commit()
    rows = con.execute("SELECT id,discipline,title FROM doc WHERE status='ok'").fetchall()
    bydisc = {}
    for i, d, t in rows:
        if d and i in pos:
            bydisc.setdefault(d, []).append((i, t))
    updates = []
    print("=" * 96)
    print("SUB-DISCIPLINES (derived by clustering titles+abstracts within each discipline)")
    print("=" * 96)
    for disc in sorted(bydisc, key=lambda x: -len(bydisc[x])):
        items = bydisc[disc]
        if len(items) < min_size:
            continue
        idx = np.array([pos[i] for i, _ in items])
        X = E[idx]
        k = max(3, min(14, round(len(items) / per_cluster)))
        lab = kmeans(X, k)
        parent = terms([t for _, t in items])
        print(f"\n  {disc}  ({len(items):,} theses -> {k} sub-fields)")
        order = sorted(range(k), key=lambda j: -(lab == j).sum())
        for j in order:
            m = lab == j
            n = int(m.sum())
            if n < 12: continue
            sub_titles = [items[q][1] for q in np.where(m)[0]]
            tc = terms(sub_titles)
            # distinctive = frequent here, rare in the parent discipline
            sc = {w: (v / n) / ((parent[w] / len(items)) + 0.004)
                  for w, v in tc.items() if v >= max(3, n * 0.07)}
            top = [w for w, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:3]]
            name = " / ".join(top) if top else f"cluster {j}"
            print(f"     {n:>5}  {name}")
            for q in np.where(m)[0]:
                updates.append((f"{disc}: {name}", items[q][0]))
    con.executemany("UPDATE doc SET subfield=? WHERE id=?", updates)
    con.commit()
    print(f"\n  {len(updates):,} theses labelled with a sub-field")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-size", type=int, default=200)
    ap.add_argument("--per-cluster", type=int, default=260)
    a = ap.parse_args()
    build(a.min_size, a.per_cluster)
