#!/usr/bin/env python3
"""
hierarchy.py — a expandable subject tree, not a fixed set of sub-fields.

discipline -> sub-field -> sub-sub-field, by recursive bisecting k-means on
title+abstract embeddings. Every node is labelled with the terms that
distinguish it from ITS OWN PARENT, so labels get more specific as you descend
instead of repeating the parent's vocabulary.

Edges between any two nodes at any level are shared cited literature, so the
graph stays meaningful whatever cut of the tree is displayed.

Deterministic throughout: farthest-first init, no RNG.
"""
from __future__ import annotations
import json, re, sqlite3, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_corpus as H
from subfields import kmeans, terms, STOP

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus" / "hierarchy.json"
MIN_SPLIT = 90          # do not split a node smaller than this
MIN_CHILD = 25
MIN_EDGE = 10


def label_for(titles, parent_titles, fallback):
    tc, pc = terms(titles), terms(parent_titles)
    n, pn = len(titles), max(len(parent_titles), 1)
    sc = {w: (v / n) / ((pc[w] / pn) + 0.004)
          for w, v in tc.items() if v >= max(3, n * 0.06)}
    top = [w for w, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
    return " / ".join(top) if top else fallback


def main():
    E = np.load(HERE / "corpus" / "title_emb.npy")
    ids = json.loads((HERE / "corpus" / "title_emb_ids.json").read_text())
    pos = {d: i for i, d in enumerate(ids)}
    con = H.db()
    rows = con.execute("SELECT id,discipline,title,landing,pdf_url,words "
                       "FROM doc WHERE status='ok'").fetchall()
    title = {r[0]: r[2] for r in rows}
    link = {r[0]: (r[3] or r[4] or "") for r in rows}
    words = {r[0]: (r[5] or 0) for r in rows}
    bydisc = defaultdict(list)
    for i, d, t, *_ in rows:
        if d and i in pos:
            bydisc[d].append(i)

    cdb = sqlite3.connect(HERE / "corpus" / "citations.db")
    works = defaultdict(set)
    for src, s, y, t in cdb.execute(
            "SELECT src,surname,year,title FROM cite WHERE title<>'' AND length(title)>12"):
        works[src].add(hash((s, y, t)) & 0xFFFFFFFF)
    cdb.close()

    nodes, kids = [], defaultdict(list)

    def add(label, members, parent, depth, disc):
        nid = len(nodes)
        ws = set()
        for m in members:
            ws |= works.get(m, set())
        ex = [{"t": (title.get(m) or "?")[:70], "u": link.get(m, "")}
              for m in sorted(members, key=lambda x: (-words.get(x, 0), x))[:8]
              if link.get(m)]
        nodes.append({"id": nid, "label": label, "grp": disc, "depth": depth,
                      "n": len(members), "parent": parent, "ex": ex,
                      "_w": ws, "_m": members})
        if parent is not None:
            kids[parent].append(nid)
        return nid

    def split(nid):
        nd = nodes[nid]
        members = nd["_m"]
        if len(members) < MIN_SPLIT or nd["depth"] >= 2:
            return
        idx = np.array([pos[m] for m in members])
        X = E[idx]
        k = max(2, min(8, round(len(members) / 110)))
        if k < 2:
            return
        lab = kmeans(X, k)
        groups = [[members[q] for q in np.where(lab == j)[0]] for j in range(k)]
        groups = [g for g in groups if len(g) >= MIN_CHILD]
        if len(groups) < 2:
            return
        pt = [title.get(m) for m in members]
        for g in sorted(groups, key=lambda g: -len(g)):
            cl = label_for([title.get(m) for m in g], pt, f"group {len(g)}")
            cid = add(cl, g, nid, nd["depth"] + 1, nd["grp"])
            split(cid)

    for d in sorted(bydisc, key=lambda x: -len(bydisc[x])):
        rid = add(d, bydisc[d], None, 0, d)
        split(rid)

    print(f"  {len(nodes)} tree nodes "
          f"(depth 0: {sum(1 for n in nodes if n['depth']==0)}, "
          f"1: {sum(1 for n in nodes if n['depth']==1)}, "
          f"2: {sum(1 for n in nodes if n['depth']==2)})", file=sys.stderr)

    edges = []
    for i, j in combinations(range(len(nodes)), 2):
        a, b = nodes[i], nodes[j]
        if a["parent"] == j or b["parent"] == i:
            continue
        inter = len(a["_w"] & b["_w"])
        if inter < MIN_EDGE:
            continue
        un = len(a["_w"] | b["_w"]) or 1
        edges.append({"s": i, "t": j, "w": inter, "j": round(inter / un, 4)})
    print(f"  {len(edges)} edges across all levels", file=sys.stderr)

    for n in nodes:
        n.pop("_w"); n.pop("_m")
    OUT.write_text(json.dumps({"nodes": nodes, "edges": edges,
                               "kids": {str(k): v for k, v in kids.items()}},
                              separators=(",", ":")), encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
    con.close()


if __name__ == "__main__":
    main()
