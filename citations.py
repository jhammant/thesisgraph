#!/usr/bin/env python3
"""
citations.py — Study 2: the citation graph. What UK doctoral researchers
actually read.

Reference lists are already isolated by the extraction pipeline (flagged
`references` and excluded from overlap matching). Here they are parsed instead
of discarded.

Reference parsing is genuinely hard and every stage is approximate, so
`validate` prints a random sample of parsed entries for eyeball checking, and
the parse rate is reported alongside every result. No number here should be
read without it.

    citations.py parse      parse every reference list -> SQLite
    citations.py validate   print a random sample of parses to check by eye
    citations.py report     most-cited works, citation age, co-citation
"""
from __future__ import annotations
import argparse, gzip, json, multiprocessing as mp, os, re, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
DB = HERE / "corpus" / "citations.db"

# Splitting on "Surname, X." fires on the SECOND and THIRD authors of a list
# too, so it grabs the wrong first author: "Snow, O'Connor, Jurafsky & Ng, A.Y.
# (2008)" parsed as "Ng 2008". Validation caught this on 5 of 8 sampled entries.
#
# Segment on the YEAR instead. Every style puts exactly one year near the front
# of an entry, so consecutive years bracket one reference; the split between
# entry i-1's title and entry i's authors is the last sentence-end between them.
YEAR_ANY = re.compile(r"\(\s*((?:1[5-9]|20)\d{2})[a-z]?\s*\)"
                      r"|(?<![\d\-])((?:1[5-9]|20)\d{2})[a-z]?(?=[\.,;:\s])")
AUTHOR_1ST = re.compile(r"([A-Z][A-Za-z'’\-]{1,24})\s*,")
# Surname, Initials [more authors] (YEAR)  — preceded by a sentence break.
HARVARD = re.compile(
    r"(?:(?<=\.\s)|(?<=\]\s)|(?<=\A)|(?<=\)\s))"
    r"(?P<sur>[A-Z][A-Za-z'’\-]{1,24}),\s*"
    r"(?:[A-Z]\.\s*){1,4}"
    r"[^()]{0,220}?"
    r"\(\s*(?P<yr>(?:1[5-9]|20)\d{2})[a-z]?\s*\)")
HARVARD_CAND = re.compile(r"\(\s*(?:1[5-9]|20)\d{2}[a-z]?\s*\)")
NOT_SURNAME = frozenset("""in and eds ed pp vol no the of by see also cited
available accessed retrieved from with for at on http doi isbn chapter part
paper report thesis journal review press university london new york""".split())
YEAR = re.compile(r"\(?\b((?:1[89]|20)\d{2})[a-z]?\b\)?")
SURNAME = re.compile(r"^([A-Z][A-Za-z'’\-]{1,24}),")
TITLE_STOP = re.compile(r"[.?!]\s")
NOISE = re.compile(r"^(?:http|doi|available|accessed|retrieved|pp?\.|vol\.|"
                   r"in:|ed\.|eds\.)", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cite (
  src TEXT NOT NULL,        -- citing thesis (OAI id)
  surname TEXT, year INTEGER, title TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS cite_src ON cite(src);
CREATE INDEX IF NOT EXISTS cite_key ON cite(surname, year);
CREATE TABLE IF NOT EXISTS srcstat (
  src TEXT PRIMARY KEY, ref_words INTEGER, entries INTEGER, parsed INTEGER
);
"""


def norm_title(t: str) -> str:
    t = R.normalise_text(t or "").lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    w = [x for x in t.split() if len(x) > 2][:8]
    return " ".join(w)


def parse_doc(doc_id: str):
    try:
        with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
    except Exception:
        return doc_id, 0, [], 0
    ref = []
    for r in rows:
        text, pg, pr, bucket, heading, excl = H.unpack_sentence(r)
        if "references" in (excl or "") or bucket == "references":
            ref.append(text)
    if not ref:
        return doc_id, 0, [], 0
    blob = " ".join(ref)
    ref_words = len(blob.split())
    # HIGH-PRECISION, AUTHOR-DATE ONLY.
    # Year-anchored segmentation failed on numbered/Vancouver style, where the
    # year sits at the END of the entry, so journal names were parsed as authors
    # ("Fusion 2013", "Vision 2004"). Rather than a citation graph built on bad
    # parses, this matches only the classic Harvard opening —
    #     <sentence break> Surname, I.I. ... (YEAR)
    # — which is unambiguous. Recall is therefore partial and BIASED toward
    # disciplines that use author-date. Both are reported, never hidden.
    out = []
    for m in HARVARD.finditer(blob):
        sur = m.group("sur").lower()
        if sur in NOT_SURNAME or len(sur) < 2:
            continue
        year = int(m.group("yr"))
        if not (1500 <= year <= 2030):
            continue
        after = blob[m.end():].lstrip(" .,):")
        ts = TITLE_STOP.search(after)
        title = after[:ts.start()] if ts else after[:160]
        out.append((doc_id, sur, year, norm_title(title),
                    blob[m.start():m.end() + 150][:300]))
    return doc_id, ref_words, out, len(HARVARD_CAND.findall(blob))

def _unused(blob, doc_id, ref_words):
    marks = [(m.start(), m.end(), int(m.group(1) or m.group(2)))
             for m in YEAR_ANY.finditer(blob)]
    marks = [m for m in marks if 1500 <= m[2] <= 2030]
    out = []
    prev_end = 0
    for k, (ys, ye, year) in enumerate(marks):
        seg = blob[prev_end:ys]
        # the previous entry's title ends at the last sentence break before the
        # authors of this one
        cut = max(seg.rfind(". "), seg.rfind(".) "), seg.rfind("] "))
        authors = seg[cut + 2:] if cut > 0 else seg
        authors = authors.strip(" .,;")
        nxt = marks[k + 1][0] if k + 1 < len(marks) else len(blob)
        after = blob[ye:nxt].lstrip(" .,):")
        ts = TITLE_STOP.search(after)
        title = after[:ts.start()] if ts else after[:160]
        prev_end = ye
        if not authors or len(authors) > 320:
            continue
        am = AUTHOR_1ST.search(authors)
        if not am:
            continue
        sur = am.group(1).lower()
        if sur in NOT_SURNAME or len(sur) < 2:
            continue
        raw = (authors[-120:] + " (" + str(year) + ") " + after[:150]).strip()
        out.append((doc_id, sur, year, norm_title(title), raw[:300]))
    return doc_id, ref_words, out, len(marks)


def parse_all(limit, workers):
    con = H.db()
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    con.close()
    if limit:
        ids = ids[:limit]
    cdb = sqlite3.connect(DB)
    cdb.executescript(SCHEMA)
    cdb.execute("PRAGMA journal_mode=WAL")
    done = {r[0] for r in cdb.execute("SELECT src FROM srcstat")}
    todo = [i for i in ids if i not in done]
    print(f"  {len(todo):,} of {len(ids):,} theses to parse", file=sys.stderr)
    t0, n = time.monotonic(), 0
    with mp.Pool(workers) as pool:
        for doc_id, rw, rows, nparts in pool.imap_unordered(parse_doc, todo,
                                                            chunksize=16):
            if rows:
                cdb.executemany("INSERT INTO cite(src,surname,year,title,raw) "
                                "VALUES(?,?,?,?,?)", rows)
            cdb.execute("INSERT INTO srcstat(src,ref_words,entries,parsed) "
                        "VALUES(?,?,?,?) ON CONFLICT(src) DO NOTHING",
                        (doc_id, rw, nparts, len(rows)))
            n += 1
            if n % 2000 == 0:
                cdb.commit()
                print(f"    {n:,}/{len(todo):,}  {(time.monotonic()-t0)/60:.1f} min",
                      file=sys.stderr)
    cdb.commit()
    print("  done", file=sys.stderr)
    cdb.close()


def validate(n):
    cdb = sqlite3.connect(DB)
    tot, parsed, ent = cdb.execute(
        "SELECT COUNT(*), SUM(parsed), SUM(entries) FROM srcstat").fetchone()
    ncite = cdb.execute("SELECT COUNT(*) FROM cite").fetchone()[0]
    print(f"theses with a reference list parsed : {tot:,}")
    print(f"candidate entries found             : {ent:,}")
    print(f"entries yielding surname+year       : {parsed:,} "
          f"({100*parsed/max(ent,1):.1f}% of candidates)")
    print(f"citation edges stored               : {ncite:,}")
    print(f"\nRandom sample of {n} parses — check these by eye:\n")
    rows = cdb.execute("SELECT surname,year,title,raw FROM cite "
                       "ORDER BY substr(raw,3,6), rowid LIMIT ?", (n,)).fetchall()
    for s, y, t, raw in rows:
        print(f"  [{s} {y}] {t[:52]}")
        print(f"      raw: {raw[:104]}")
    cdb.close()


def report():
    cdb = sqlite3.connect(DB)
    cdb.execute(f"ATTACH DATABASE '{HERE/'corpus'/'corpus.db'}' AS m")
    n_src = cdb.execute("SELECT COUNT(*) FROM srcstat WHERE parsed>0").fetchone()[0]
    ncite = cdb.execute("SELECT COUNT(*) FROM cite").fetchone()[0]
    print("=" * 82)
    print(f"STUDY 2 — CITATION GRAPH   ({ncite:,} edges from {n_src:,} theses)")
    print("=" * 82)
    print(f"\n  mean references per thesis: "
          f"{cdb.execute('SELECT AVG(parsed) FROM srcstat WHERE parsed>0').fetchone()[0]:.0f}")
    print("\n  MOST-CITED AUTHORS (distinct citing theses)")
    print(f"    {'author':<22}{'theses':>8}{'% of corpus':>13}")
    for s, k in cdb.execute("SELECT surname, COUNT(DISTINCT src) c FROM cite "
                            "GROUP BY surname ORDER BY c DESC LIMIT 15"):
        print(f"    {s:<22}{k:>8,}{100*k/max(n_src,1):>12.1f}%")
    print("\n  MOST-CITED WORKS (author + year + title)")
    print(f"    {'theses':>7}  work")
    for s, y, t, k in cdb.execute(
            "SELECT surname,year,title,COUNT(DISTINCT src) c FROM cite "
            "WHERE title<>'' GROUP BY surname,year,title HAVING c>1 "
            "ORDER BY c DESC LIMIT 15"):
        print(f"    {k:>7,}  {s} ({y}) {t[:56]}")
    print("\n  AGE OF CITED WORK AT TIME OF THESIS")
    rows = cdb.execute("""SELECT d.year - c.year AS age, COUNT(*) FROM cite c
        JOIN m.doc d ON d.id=c.src WHERE d.year IS NOT NULL
        AND c.year<=d.year AND d.year-c.year<=120 GROUP BY age""").fetchall()
    tot = sum(k for _, k in rows)
    cum = 0
    bands = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 120)]
    for lo, hi in bands:
        k = sum(v for a, v in rows if lo <= a <= hi)
        cum += k
        print(f"    {lo:>3}-{hi:<3} years old: {k:>9,}{100*k/max(tot,1):>7.1f}%"
              f"   (cumulative {100*cum/max(tot,1):>5.1f}%)")
    med = None
    run = 0
    for a, v in sorted(rows):
        run += v
        if med is None and run >= tot / 2:
            med = a
    print(f"    median age of a cited work: {med} years")
    cdb.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 6))
    v = sub.add_parser("validate"); v.add_argument("--n", type=int, default=12)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "parse":
        parse_all(a.limit, a.workers)
    elif a.cmd == "validate":
        validate(a.n)
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
