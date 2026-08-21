#!/usr/bin/env python3
"""
thesisdoc.py — per-thesis drill-down: metadata, abstract, and a derived table
of contents with a deep link to each section's first page.

The extraction pipeline already records, for every sentence, its heading,
chapter bucket and PDF page. That is enough to reconstruct a navigable contents
list for a thesis nobody indexed — so you can jump straight to its methodology
chapter rather than scrolling a 300-page PDF.
"""
from __future__ import annotations
import gzip, json, re, sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
OUT = HERE / "corpus" / "docs.json"
MAX_TOC = 28
NOISE = re.compile(r"^\s*(\d+|[ivxlcdm]+|page\s*\d+|table|figure|appendix\s*$)\s*$", re.I)


def toc_for(rows, npages=0):
    """Collapse the sentence stream into a contents list: first page per heading.

    Two things corrupt a naive version. The thesis's OWN contents pages list
    every heading in the first few pages, so headings appear long before the
    sections they name; and heading detection yields fragments. So: drop pages
    carrying many distinct headings (those are contents pages), require the
    page sequence to increase, and reject fragments.
    """
    per_page = {}
    for r in rows:
        text, pg, printed, bucket, heading, excl = H.unpack_sentence(r)
        h = (heading or "").strip()
        if h:
            per_page.setdefault(pg, set()).add(h)
    contents_pages = {pg for pg, hs in per_page.items() if len(hs) >= 4}

    seen = OrderedDict()
    for r in rows:
        text, pg, printed, bucket, heading, excl = H.unpack_sentence(r)
        h = (heading or "").strip()
        if not h or len(h) < 4 or len(h) > 90 or NOISE.match(h):
            continue
        if pg in contents_pages:
            continue
        if h[0].islower() or re.match(r"^(and|or|the|of|in|to|for)\b", h, re.I):
            continue                                   # heading-detector fragment
        # A heading numbered "5.5.7" cannot legitimately sit on page 10 of a
        # 339-page thesis: that is the thesis's own contents list being read as
        # headings. Reject a numbered heading whose chapter number is
        # incompatible with how far into the document it appears.
        nm = re.match(r"^(\d{1,2})[.\s]", h)
        if nm and npages:
            chap = int(nm.group(1))
            if chap >= 2 and pg < npages * 0.06 * chap:
                continue
        if h not in seen:
            seen[h] = {"h": h, "pg": pg, "pr": printed or "", "b": bucket or ""}
    # a contents list runs forwards
    out, last = [], -1
    for e in seen.values():
        if e["pg"] >= last:
            out.append(e)
            last = e["pg"]
    if len(out) <= MAX_TOC:
        return out
    # keep the structurally important ones: chapter-level and bucket changes
    keep, last_b = [], None
    for e in out:
        if re.match(r"^\s*(chapter|part|section)\b", e["h"], re.I) or e["b"] != last_b:
            keep.append(e)
            last_b = e["b"]
    if len(keep) < 8:
        keep = out[:MAX_TOC]
    return keep[:MAX_TOC]


def main():
    want = set()
    b = json.loads((HERE / "corpus" / "bridges.json").read_text(encoding="utf-8"))
    # bridges.json carries titles, not ids; re-derive ids from the context db
    import sqlite3
    ctx = sqlite3.connect(HERE / "corpus" / "context.db")
    want = {r[0] for r in ctx.execute("SELECT DISTINCT src FROM ctx")}
    ctx.close()
    con = H.db()
    meta = {r[0]: r for r in con.execute(
        "SELECT id,title,creator,year,publisher,discipline,subfield,abstract,"
        "pages,words,landing,pdf_url FROM doc WHERE status='ok'")}
    con.close()
    docs = {}
    for k, doc_id in enumerate(sorted(want), 1):
        m = meta.get(doc_id)
        if not m:
            continue
        try:
            with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
                rows = json.load(fh)["sentences"]
        except Exception:
            continue
        docs[doc_id] = {
            "t": m[1] or "?", "a": (m[2] or "").split(";")[0].strip(), "y": m[3],
            "i": m[4] or "", "d": m[5] or "", "sf": m[6] or "",
            "ab": (m[7] or "")[:700], "pg": m[8], "w": m[9],
            "u": m[10] or "", "pdf": m[11] or m[10] or "",
            "toc": toc_for(rows, m[8] or 0),
        }
        if k % 1000 == 0:
            print(f"  {k:,}/{len(want):,}", file=sys.stderr)
    OUT.write_text(json.dumps(docs, separators=(",", ":")), encoding="utf-8")
    ntoc = sum(len(d["toc"]) for d in docs.values())
    print(f"  {len(docs):,} theses, {ntoc:,} contents entries, "
          f"{OUT.stat().st_size/1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
