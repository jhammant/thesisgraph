#!/usr/bin/env python3
"""
bridges.py — find where fields actually cross over, and make it readable.

A Jaccard number between two disciplines is not actionable. What is actionable:
  1. WHICH works bridge two fields that otherwise barely overlap
  2. WHICH theses on each side cite them
  3. WHERE in those theses the discussion happens — page, chapter, and the
     sentence itself, with a deep link straight to that PDF page.

Stage 1 scores candidate bridging works. Stage 2 extracts in-text citation
context only for those, which keeps the expensive full-text pass small.
"""
from __future__ import annotations
import argparse, gzip, json, multiprocessing as mp, os, re, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus" / "bridges.json"
CTX = HERE / "corpus" / "context.db"
MIN_SIDE = 2          # theses per side before a work counts as bridging
MAX_WORKS = 9000
# A bridge must be SPECIFIC. Braun & Clarke is cited by 4% of the entire corpus
# across all 16 disciplines — that is universal methods literature, not a
# crossover, and "Education and Medicine both use thematic analysis" is not a
# discovery. Genuine bridges are works a handful of fields cite and nobody else.
MAX_TOTAL_CITING = 150      # cited by more theses than this => too universal
MAX_FIELDS = 6             # spread across more fields than this => not a bridge
SNIPPET_WORDS = 38    # proportionate quotation, always with page + attribution


def field_of(disc, sub):
    return sub or disc


# A stored title came from norm_title(), which lowercases and strips stopwords:
# "A Realist Theory of Science" became "realist theory science". That reads as
# broken data. The original is still in cite.raw, so recover it.
_PUB = re.compile(r"\s(?:[A-Z][A-Za-z.& ]{2,28}:|In\b|Journal\b|Proceedings\b|"
                  r"pp?\.|vol\.|Vol\.|doi|https?://|Available)")


def clean_title(raw: str, year: int) -> str:
    m = re.search(r"\(?\b%d[a-z]?\b\)?" % year, raw or "")
    if not m:
        return ""
    t = raw[m.end():].lstrip(" .,):;")
    cut = _PUB.search(t)
    if cut and cut.start() > 12:
        t = t[:cut.start()]
    t = re.split(r"(?<=[a-z\"\'\)])\.\s+(?=[A-Z])", t)[0]
    # Some styles wrap the title in quotes and put the journal after the closing
    # quote; others leave a bare volume number trailing.
    t = re.split(r"['\u2019\"\u201d],\s+[A-Z]", t)[0]
    t = re.sub(r"\s+", " ", t).strip(" .,;:")
    t = t.strip("'\u2018\u2019\"\u201c\u201d")
    t = re.sub(r"[\s,]+\d{1,4}\s*$", "", t)          # trailing volume/page
    t = re.sub(r",\s*[A-Z][A-Za-z ]{2,24}$", "", t)   # trailing journal name
    t = t.strip(" .,;:")
    if not (6 <= len(t) <= 160):
        return ""
    # reject fragments that are mostly punctuation, digits or initials
    letters = sum(ch.isalpha() for ch in t)
    if letters < len(t) * 0.6 or len(t.split()) < 2:
        return ""
    return t


def score_works():
    m = H.db()
    disc = {r[0]: (r[1], r[2]) for r in m.execute(
        "SELECT id,discipline,subfield FROM doc WHERE status='ok'")}
    meta = {r[0]: r for r in m.execute(
        "SELECT id,title,year,landing,pdf_url,discipline,subfield FROM doc")}
    m.close()
    c = sqlite3.connect(HERE / "corpus" / "citations.db")
    work_src = defaultdict(set)
    titles = defaultdict(Counter)
    for src, s, y, t, raw in c.execute(
            "SELECT src,surname,year,title,raw FROM cite "
            "WHERE title<>'' AND length(title)>12"):
        # (surname, year) identifies a work well enough within a reference list;
        # grouping on the mangled title split one book across several nodes.
        work_src[(s, y)].add(src)
        ct = clean_title(raw, y)
        if ct:
            titles[(s, y)][ct] += 1
    c.close()
    DISPLAY = {k: v.most_common(1)[0][0] for k, v in titles.items() if v}

    # how often does each pair of DISCIPLINES co-occur across all works? rare
    # pairs are the interesting ones.
    pair_n = Counter()
    disc_n = Counter()
    for w, srcs in work_src.items():
        ds = {disc[s][0] for s in srcs if s in disc and disc[s][0]}
        for d in ds:
            disc_n[d] += 1
        for a in ds:
            for b in ds:
                if a < b:
                    pair_n[(a, b)] += 1
    total_works = len(work_src)

    scored = []
    for w, srcs in work_src.items():
        if w not in DISPLAY:
            continue                       # no recoverable title: not displayable
        if len(srcs) > MAX_TOTAL_CITING:
            continue                      # universal literature, not a bridge
        # Field = SUB-field where we have one: "zebrafish models" crossing into
        # "cancer therapy" is the interesting grain, not "Biology" and "Medicine".
        by = defaultdict(list)
        by_disc = defaultdict(list)
        for s in srcs:
            if s not in disc or not disc[s][0]:
                continue
            by[field_of(disc[s][0], disc[s][1])].append(s)
            by_disc[disc[s][0]].append(s)
        fields = [d for d, v in by.items() if len(v) >= MIN_SIDE]
        discs = [d for d, v in by_disc.items() if len(v) >= MIN_SIDE]
        if len(fields) < 2 or len(discs) < 2:
            continue                      # must cross a DISCIPLINE boundary
        if len(by) > MAX_FIELDS:
            continue
        # surprise: the rarest discipline pair this work joins
        best = None
        for i, a in enumerate(sorted(discs)):
            for b in sorted(discs)[i + 1:]:
                exp = (disc_n[a] * disc_n[b]) / max(total_works, 1)
                obs = pair_n[(a, b)] if a < b else pair_n[(b, a)]
                lift = exp / max(obs, 1)          # high when that pair is rare
                # specificity: reward a work only a few fields cite
                spec = 1.0 / len(by)
                sc = lift * spec * min(len(by_disc[a]), len(by_disc[b]))
                if best is None or sc > best[0]:
                    best = (sc, a, b)
        if best:
            scored.append({"key": [w[0], w[1], DISPLAY[w]],
                           "score": round(best[0], 3),
                           "a": best[1], "b": best[2],
                           "fields": {d: sorted(v) for d, v in by.items()
                                      if len(v) >= MIN_SIDE},
                           "discs": sorted(discs),
                           "n": len(srcs)})
    scored.sort(key=lambda x: (-x["score"], -x["n"], x["key"]))
    return scored[:MAX_WORKS], meta


# ---------------------------------------------------------------- context
_WANT: dict[str, set] = {}


def _ctx_one(doc_id):
    want = _WANT.get(doc_id)
    if not want:
        return doc_id, []
    surnames = sorted({s for s, y in want})
    rx = re.compile(r"\b(" + "|".join(re.escape(s) for s in surnames) + r")\b",
                    re.IGNORECASE)
    yr = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
    try:
        with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
    except Exception:
        return doc_id, []
    out, seen = [], set()
    for r in rows:
        text, pg, printed, bucket, heading, excl = H.unpack_sentence(r)
        if "references" in (excl or ""):
            continue
        if not rx.search(text):
            continue
        for m in rx.finditer(text):
            lo = max(0, m.start() - 60)
            for ym in yr.finditer(text[lo:m.end() + 70]):
                k = (m.group(1).lower(), int(ym.group(1)))
                if k in want and (k, pg) not in seen:
                    seen.add((k, pg))
                    words = text.split()
                    snip = " ".join(words[:SNIPPET_WORDS])
                    if len(words) > SNIPPET_WORDS:
                        snip += " …"
                    out.append((doc_id, k[0], k[1], snip, pg, printed or "",
                                bucket or "", (heading or "")[:80]))
    return doc_id, out


def _init(w):
    global _WANT
    _WANT = w


def build(workers):
    print("scoring bridging works ...", file=sys.stderr)
    works, meta = score_works()
    print(f"  {len(works)} candidate bridging works", file=sys.stderr)
    want = defaultdict(set)
    for w in works:
        s, y, t = w["key"]
        for d, srcs in w["fields"].items():
            for src in srcs:
                want[src].add((s, int(y)))
    print(f"  extracting citation context from {len(want):,} theses ...", file=sys.stderr)
    con = sqlite3.connect(CTX)
    con.executescript("""
      DROP TABLE IF EXISTS ctx;
      CREATE TABLE ctx(src TEXT, surname TEXT, year INT, snippet TEXT,
                       page INT, printed TEXT, bucket TEXT, heading TEXT);
      CREATE INDEX ctx_k ON ctx(surname,year);
      CREATE INDEX ctx_s ON ctx(src);""")
    t0, n = time.monotonic(), 0
    with mp.Pool(workers, initializer=_init, initargs=(dict(want),)) as pool:
        for doc_id, rows in pool.imap_unordered(_ctx_one, sorted(want), chunksize=16):
            if rows:
                con.executemany("INSERT INTO ctx VALUES(?,?,?,?,?,?,?,?)", rows)
            n += 1
            if n % 2000 == 0:
                con.commit()
                print(f"    {n:,}/{len(want):,}  {(time.monotonic()-t0)/60:.1f} min",
                      file=sys.stderr)
    con.commit()
    got = con.execute("SELECT COUNT(*) FROM ctx").fetchone()[0]
    print(f"  {got:,} in-text citation passages located", file=sys.stderr)

    # assemble the payload: only works we actually found passages for
    have = defaultdict(int)
    for s, y, k in con.execute("SELECT surname,year,COUNT(*) FROM ctx GROUP BY 1,2"):
        have[(s, y)] = k
    payload = []
    for w in works:
        s, y, t = w["key"]
        if have.get((s, int(y)), 0) < 2:
            continue
        sides = {}
        for d, srcs in w["fields"].items():
            ent = []
            for src in srcs:
                rows = con.execute(
                    "SELECT snippet,page,printed,bucket,heading FROM ctx "
                    "WHERE src=? AND surname=? AND year=? LIMIT 2",
                    (src, s, int(y))).fetchall()
                if not rows:
                    continue
                md = meta.get(src)
                if not md:
                    continue
                pdf = md[4] or md[3] or ""
                ent.append({"id": src, "t": (md[1] or "?")[:110], "y": md[2],
                            "sf": md[6] or md[5] or "",
                            "u": md[3] or pdf, "pdf": pdf,
                            "p": [{"s": r[0], "pg": r[1], "pr": r[2],
                                   "b": r[3], "h": r[4]} for r in rows]})
            if ent:
                sides[d] = ent          # every citing thesis, not a sample
        if len(sides) >= 2:
            payload.append({"w": f"{s.title()} ({y})", "title": t,
                            "score": w["score"], "a": w["a"], "b": w["b"],
                            "n": w["n"], "sides": sides})
    payload.sort(key=lambda x: -x["score"])
    OUT.write_text(json.dumps({"works": payload[:2500]}, separators=(",", ":")),
                   encoding="utf-8")
    print(f"  written {OUT} ({OUT.stat().st_size/1e6:.1f} MB, "
          f"{len(payload[:2500])} bridges)", file=sys.stderr)
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 6))
    build(ap.parse_args().workers)
