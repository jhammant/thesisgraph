#!/usr/bin/env python3
"""build_graph.py — assemble the interactive graph datasets from the studies.

Four linked graphs, all derived from artefacts already produced:
  cocite     canonical works, linked when cited together by the same thesis
  bridges    disciplines, linked by how much cited literature they share
  methods    disciplines <-> research methods, weighted by prevalence
  reuse      theses linked by surviving verbatim text overlap
"""
from __future__ import annotations
import csv, json, sqlite3, sys, math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus" / "graph.json"
TOP_WORKS = 260
MIN_COCITE = 12

m = sqlite3.connect(HERE / "corpus" / "corpus.db")
cdb = sqlite3.connect(HERE / "corpus" / "citations.db")
disc = {r[0]: r[1] for r in m.execute("SELECT id,discipline FROM doc")}
LINK = {r[0]: (r[1], r[2], r[3]) for r in m.execute(
    "SELECT id,landing,pdf_url,title FROM doc")}
WORDS = {r[0]: (r[1] or 0) for r in m.execute("SELECT id,words FROM doc")}

def exemplars(ids, k=8):
    """Representative theses for an aggregate node: the most substantial ones,
    chosen deterministically so the same data always lists the same examples."""
    out = []
    for d in sorted(ids, key=lambda x: (-WORDS.get(x, 0), x))[:k]:
        t = LINK.get(d)
        if t and (t[0] or t[1]):
            out.append({"t": (t[2] or "?")[:70], "u": t[0] or t[1]})
    return out
year = {r[0]: r[1] for r in m.execute("SELECT id,year FROM doc")}

# ---------------------------------------------------------------- co-citation
print("co-citation ...", file=sys.stderr)
work_theses = defaultdict(set)
for src, s, y, t in cdb.execute(
        "SELECT src,surname,year,title FROM cite WHERE title<>'' AND length(title)>12"):
    work_theses[(s, y, t)].add(src)
top = sorted(work_theses.items(), key=lambda kv: -len(kv[1]))[:TOP_WORKS]
idx = {k: i for i, (k, _) in enumerate(top)}
by_thesis = defaultdict(list)
for k, srcs in top:
    for s in srcs:
        by_thesis[s].append(idx[k])
co = Counter()
for s, ws in by_thesis.items():
    for a, b in combinations(sorted(set(ws)), 2):
        co[(a, b)] += 1
nodes = []
for k, srcs in top:
    s, y, t = k
    dc = Counter(disc.get(x) for x in srcs if disc.get(x))
    nodes.append({"id": idx[k], "label": f"{s.title()} ({y})",
                  "title": t[:70], "n": len(srcs),
                  "grp": dc.most_common(1)[0][0] if dc else "—",
                  "q": f"{s} {y} {t[:60]}",     # for a scholar search
                  "ex": exemplars(srcs)})
edges = [{"s": a, "t": b, "w": w} for (a, b), w in co.items() if w >= MIN_COCITE]
cocite = {"nodes": nodes, "edges": edges}
print(f"  {len(nodes)} works, {len(edges)} co-citation edges", file=sys.stderr)

# ------------------------------------------------------------------- bridges
print("discipline bridges ...", file=sys.stderr)
d_works = defaultdict(set)
for k, srcs in work_theses.items():
    if len(srcs) < 3:
        continue
    for s in srcs:
        d = disc.get(s)
        if d:
            d_works[d].add(k)
dn = sorted(d_works, key=lambda d: -len(d_works[d]))
d_theses = defaultdict(list)
for _t, _d in disc.items():
    if _d:
        d_theses[_d].append(_t)
bn = [{"id": i, "label": d, "n": len(d_works[d]), "grp": d,
       "ex": exemplars(d_theses.get(d, []))} for i, d in enumerate(dn)]
be = []
for i, j in combinations(range(len(dn)), 2):
    A, B = d_works[dn[i]], d_works[dn[j]]
    inter = len(A & B)
    if inter < 20:
        continue
    be.append({"s": i, "t": j, "w": inter,
               "j": round(inter / len(A | B), 4)})
bridges = {"nodes": bn, "edges": be}
print(f"  {len(bn)} disciplines, {len(be)} shared-literature edges", file=sys.stderr)


# ------------------------------------------------------------------ subfields
print("subfield bridges ...", file=sys.stderr)
sub = {r[0]: r[1] for r in m.execute("SELECT id,subfield FROM doc WHERE subfield IS NOT NULL")}
s_works = defaultdict(set); s_n = Counter()
for k, srcs in work_theses.items():
    if len(srcs) < 3:
        continue
    for x in srcs:
        sf = sub.get(x)
        if sf:
            s_works[sf].add(k)
for x, sf in sub.items():
    s_n[sf] += 1
keep = [x for x in s_works if s_n[x] >= 40 and len(s_works[x]) >= 60]
keep.sort(key=lambda x: -s_n[x])
sub_theses = defaultdict(list)
for _t, _sf in sub.items():
    sub_theses[_sf].append(_t)
sn = [{"id": i, "label": (x.split(": ", 1)[1][:34] if ": " in x else x[:34]),
       "title": x.split(":")[0], "n": s_n[x], "grp": x.split(":")[0],
       "ex": exemplars(sub_theses.get(x, []))} for i, x in enumerate(keep)]
se = []
for i, j in combinations(range(len(keep)), 2):
    A, B = s_works[keep[i]], s_works[keep[j]]
    inter = len(A & B)
    if inter < 12:
        continue
    se.append({"s": i, "t": j, "w": inter, "j": round(inter / len(A | B), 4)})
subfields = {"nodes": sn, "edges": se}
print(f"  {len(sn)} sub-fields, {len(se)} edges", file=sys.stderr)

# ------------------------------------------------------------------- methods
print("methods ...", file=sys.stderr)
METHODS = ["thematic analysis", "grounded theory", "discourse analysis",
           "mixed methods", "ethnography", "case study", "systematic review",
           "meta-analysis", "action research", "phenomenology",
           "content analysis", "narrative inquiry", "survey", "RCT"]
rows = list(csv.DictReader((HERE / "corpus" / "studies" /
                            "method_features.csv").open(encoding="utf-8")))
tot = Counter(); hit = Counter()
for r in rows:
    d = disc.get(r["id"])
    if not d:
        continue
    tot[d] += 1
    for meth in METHODS:
        v = r.get(f"design::{meth}")
        if v and int(v) > 0:
            hit[(d, meth)] += 1
mn, mi = [], {}
for d in sorted(tot, key=lambda x: -tot[x]):
    mi[("d", d)] = len(mn)
    mn.append({"id": len(mn), "label": d, "n": tot[d], "grp": "discipline",
               "kind": "discipline"})
for meth in METHODS:
    mi[("m", meth)] = len(mn)
    mn.append({"id": len(mn), "label": meth,
               "n": sum(hit[(d, meth)] for d in tot), "grp": "method",
               "kind": "method"})
me = []
for (d, meth), k in hit.items():
    pc = k / max(tot[d], 1)
    if pc >= 0.02:
        me.append({"s": mi[("d", d)], "t": mi[("m", meth)],
                   "w": round(100 * pc, 1)})
methods = {"nodes": mn, "edges": me}
print(f"  {len(mn)} nodes, {len(me)} edges", file=sys.stderr)

# --------------------------------------------------------------------- reuse
print("text reuse ...", file=sys.stderr)
pairs = m.execute("""SELECT a,b,clean_max_run,clean_long_runs,clean_words
    FROM pair WHERE clean_max_run>=150 AND a_tokens>=20000 AND b_tokens>=20000
    ORDER BY clean_max_run DESC LIMIT 600""").fetchall()
seen = {}
rn, re_ = [], []
titles = {r[0]: r[1] for r in m.execute("SELECT id,title FROM doc")}
for a, b, mx, lr, w in pairs:
    for d in (a, b):
        if d not in seen:
            seen[d] = len(rn)
            _l = LINK.get(d) or ("", "", "")
            rn.append({"id": seen[d], "label": (titles.get(d) or "?")[:52],
                       "title": f"{disc.get(d,'?')} {year.get(d,'?')}",
                       "n": 1, "grp": disc.get(d) or "—",
                       "u": _l[0] or _l[1], "pdf": _l[1]})
    re_.append({"s": seen[a], "t": seen[b], "w": mx})
for e in re_:
    rn[e["s"]]["n"] += 1
    rn[e["t"]]["n"] += 1
reuse = {"nodes": rn, "edges": re_}
print(f"  {len(rn)} theses, {len(re_)} reuse edges", file=sys.stderr)

OUT.write_text(json.dumps({"cocite": cocite, "bridges": bridges,
                           "subfields": subfields,
                           "methods": methods, "reuse": reuse},
                          separators=(",", ":"), sort_keys=True), encoding="utf-8")
print(f"written {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
