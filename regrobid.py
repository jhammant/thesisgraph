#!/usr/bin/env python3
"""regrobid.py — rebuild the citation graph with GROBID over the stored text.

No re-crawling. The reference sections are already in corpus/text/, so this
reads them, segments each into entries with refseg, and parses the entries with
a local GROBID service. Re-fetching 24,656 PDFs would cost ~68 hours at the
repository's published Crawl-delay; this costs a few hours of local CPU and
touches the network not at all.

Writes to a SEPARATE database (corpus/citations_grobid.db) so the existing graph
stays intact and the two can be compared before anything switches over.

    docker run -d --name grobid -p 8070:8070 grobid/grobid:0.9.1-crf
    regrobid.py --workers 4
    regrobid.py --compare          # old vs new, per discipline

Resumable: a thesis already recorded is skipped, so it can be stopped and
restarted freely.
"""
from __future__ import annotations
import argparse, gzip, html, json, os, queue, re, sqlite3, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

import harvest_corpus as H
from citations import NOT_SURNAME
from refseg import segment, looks_bibliographic

# GROBID will label a prose fragment as an author if a segmenter hands it one
# ("moreover, 2000, 'While the semiclassical approach...'"). These are the
# function words that show up as surnames when that happens; a single-character
# surname is likewise an artefact, not a person.
_BAD_SUR = set(NOT_SURNAME) | {
    "moreover", "however", "therefore", "furthermore", "thus", "hence",
    "although", "because", "whereas", "meanwhile", "nevertheless", "since",
    "while", "when", "where", "which", "that", "this", "these", "those",
    "there", "here", "then", "than", "such", "same", "other", "another",
    "figure", "table", "section", "appendix", "note", "notes", "source",
    "abstract", "introduction", "conclusion", "references", "bibliography"}

HERE = Path(__file__).resolve().parent
CORPUS_DB = HERE / "corpus" / "corpus.db"
OLD_DB = HERE / "corpus" / "citations.db"
OUT_DB = HERE / "corpus" / "citations_grobid.db"
GROBID = os.environ.get("GROBID_URL", "http://localhost:8070")
CHUNK = 250                      # entries per GROBID request

SCHEMA = """
CREATE TABLE IF NOT EXISTS cite (
  src TEXT, surname TEXT, year INTEGER, title TEXT, venue TEXT, doi TEXT, raw TEXT);
CREATE INDEX IF NOT EXISTS cite_src ON cite(src);
CREATE INDEX IF NOT EXISTS cite_work ON cite(surname, year);
CREATE TABLE IF NOT EXISTS srcstat (
  src TEXT PRIMARY KEY, ref_words INTEGER, segments INTEGER,
  parsed INTEGER, biblio INTEGER, note TEXT);
"""

_BIBL = re.compile(r"<biblStruct[\s>].*?</biblStruct>", re.S)
_SUR = re.compile(r"<surname>([^<]+)</surname>")
_YEAR = re.compile(r'when="(\d{4})')
_TITLE_A = re.compile(r'<title level="[am]"[^>]*>([^<]{4,})</title>')
_TITLE_J = re.compile(r'<title level="j"[^>]*>([^<]{2,})</title>')
_DOI = re.compile(r'<idno type="DOI"[^>]*>([^<]+)</idno>', re.I)


def refs_blob(doc_id: str) -> str:
    """The reference section of a thesis, from the stored sentence file."""
    try:
        with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
    except Exception:
        return ""
    out = []
    for r in rows:
        text, _pg, _pr, bucket, _hd, excl = H.unpack_sentence(r)
        if "references" in (excl or "") or bucket == "references":
            out.append(text)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def grobid_parse(segs: list[str], timeout: int = 300) -> list[tuple]:
    """POST entries to GROBID's citation endpoint; return structured rows.

    A row needs a surname and a year, plus a title or a venue. ACS and Vancouver
    entries carry no article title at all, so demanding one drops whole
    disciplines; the journal name identifies the work well enough to place it in
    a co-citation graph.
    """
    rows = []
    for i in range(0, len(segs), CHUNK):
        batch = segs[i:i + CHUNK]
        data = urllib.parse.urlencode([("citations", s) for s in batch]).encode()
        req = urllib.request.Request(
            f"{GROBID}/api/processCitationList", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/xml"})
        for attempt in range(3):
            try:
                tei = urllib.request.urlopen(req, timeout=timeout).read()
                break
            except Exception:
                if attempt == 2:
                    tei = b""
                time.sleep(2 * (attempt + 1))
        tei = tei.decode("utf-8", "replace")
        for j, b in enumerate(_BIBL.findall(tei)):
            sur = _SUR.search(b)
            yr = _YEAR.search(b)
            if not sur or not yr:
                continue
            ta, tj = _TITLE_A.search(b), _TITLE_J.search(b)
            if not ta and not tj:
                continue
            surname = html.unescape(sur.group(1)).lower().strip(" .,'\u2019")
            if len(surname) < 2 or surname in _BAD_SUR:
                continue
            year = int(yr.group(1))
            if not (1500 <= year <= 2027):
                continue
            doi = _DOI.search(b)
            raw = batch[j][:300] if j < len(batch) else ""
            rows.append((surname, year,
                         html.unescape(ta.group(1)).strip() if ta else "",
                         html.unescape(tj.group(1)).strip() if tj else "",
                         (doi.group(1).strip() if doi else ""), raw))
    return rows


def worker(q: queue.Queue, out: queue.Queue, stop: threading.Event):
    while not stop.is_set():
        try:
            doc_id = q.get(timeout=1)
        except queue.Empty:
            return
        try:
            blob = refs_blob(doc_id)
            if not blob:
                out.put((doc_id, 0, 0, [], -1, "no reference section"))
                continue
            segs = segment(blob)
            ok, why = looks_bibliographic(blob, segs)
            if not ok:
                out.put((doc_id, len(blob.split()), len(segs), [], 0, why))
                continue
            rows = grobid_parse(segs)
            out.put((doc_id, len(blob.split()), len(segs), rows, 1, "ok"))
        except Exception as e:                       # never lose a worker
            out.put((doc_id, 0, 0, [], 0, f"error: {type(e).__name__}: {e}"[:180]))
        finally:
            q.task_done()


def run(limit: int | None, workers: int):
    if urllib.request.urlopen(f"{GROBID}/api/isalive", timeout=10).read()[:4] != b"true":
        sys.exit(f"GROBID not answering at {GROBID}")
    db = sqlite3.connect(OUT_DB)
    db.executescript(SCHEMA)
    done = {r[0] for r in db.execute("SELECT src FROM srcstat")}

    src = sqlite3.connect(CORPUS_DB)
    ids = [r[0] for r in src.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id") if r[0] not in done]
    if limit:
        ids = ids[:limit]
    if not ids:
        print("nothing to do — all theses already processed")
        return
    print(f"{len(ids):,} theses to parse ({len(done):,} already done), "
          f"{workers} workers")

    q, out, stop = queue.Queue(), queue.Queue(), threading.Event()
    for i in ids:
        q.put(i)
    threads = [threading.Thread(target=worker, args=(q, out, stop), daemon=True)
               for _ in range(workers)]
    for t in threads:
        t.start()

    t0, n, tot_rows, skipped = time.time(), 0, 0, 0
    try:
        while n < len(ids):
            doc_id, words, nsegs, rows, biblio, note = out.get()
            db.execute("INSERT OR REPLACE INTO srcstat VALUES (?,?,?,?,?,?)",
                       (doc_id, words, nsegs, len(rows), biblio, note))
            if rows:
                db.executemany(
                    "INSERT INTO cite (src,surname,year,title,venue,doi,raw) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [(doc_id, *r) for r in rows])
            tot_rows += len(rows)
            skipped += (biblio == 0)
            n += 1
            if n % 200 == 0:
                db.commit()
                el = time.time() - t0
                rate = n / el
                print(f"  {n:,}/{len(ids):,}  {tot_rows:,} refs  "
                      f"{rate:.1f} th/s  eta {(len(ids)-n)/rate/60:.0f} min",
                      flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved, rerun to resume")
        stop.set()
    finally:
        db.commit()
    el = time.time() - t0
    print(f"\ndone: {n:,} theses, {tot_rows:,} references in {el/60:.1f} min")
    print(f"  {skipped:,} sections were not reference lists (excluded, not failed)")
    print(f"  -> {OUT_DB}")


def compare():
    """Old parser against new, per discipline."""
    if not OUT_DB.exists():
        sys.exit("no citations_grobid.db yet — run regrobid.py first")
    c = sqlite3.connect(CORPUS_DB)
    c.execute(f"ATTACH DATABASE '{OLD_DB}' AS o")
    c.execute(f"ATTACH DATABASE '{OUT_DB}' AS n")
    rows = c.execute("""
      SELECT d.discipline, COUNT(DISTINCT n.src),
             IFNULL(SUM(o.parsed),0), IFNULL(SUM(n.parsed),0),
             SUM(CASE WHEN n.biblio=0 THEN 1 ELSE 0 END)
      FROM doc d JOIN n.srcstat n ON n.src=d.id
                 LEFT JOIN o.srcstat o ON o.src=d.id
      WHERE d.discipline<>'' GROUP BY d.discipline ORDER BY d.discipline""").fetchall()
    print(f"{'discipline':<32}{'theses':>8}{'old':>10}{'new':>10}{'lift':>8}{'excl':>7}")
    to = tn = 0
    for disc, th, old, new, excl in rows:
        to += old; tn += new
        print(f"{disc:<32}{th:>8,}{old:>10,}{new:>10,}"
              f"{(f'{new/old:.2f}x' if old else 'inf'):>8}{excl:>7,}")
    print(f"\n{'TOTAL':<32}{'':>8}{to:>10,}{tn:>10,}{(tn/to if to else 0):>7.2f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only this many theses")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare", action="store_true", help="old vs new by discipline")
    a = ap.parse_args(argv)
    if a.compare:
        return compare()
    run(a.limit, a.workers)


if __name__ == "__main__":
    sys.exit(main())
