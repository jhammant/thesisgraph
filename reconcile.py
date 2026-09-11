#!/usr/bin/env python3
"""reconcile.py — resolve parsed citations to canonical works via OpenAlex.

Why this matters more than more theses: works currently dedupe on
(surname, year), so every 2006 Braun is the same node, "Smith 2011" collides
across unrelated fields, and a mis-OCR'd title silently becomes a second work.
A canonical id (OpenAlex id / DOI) fixes all three, and turns the map's field
gaps from a caveat into a finding.

Two stages, deliberately split by what each is good at:

  1. MATCH (this script, no model). Query OpenAlex by title, score the
     candidates on title similarity, year distance and author surname. Most
     citations resolve or fail unambiguously and never need a model — sending
     millions of rows through an LLM would be slower and less accurate than
     string comparison against an authority file.

  2. ADJUDICATE (adjudicate.py, local model). Only the genuinely ambiguous
     middle — a plausible title with the wrong year, an initials-only author,
     a truncated OCR title — goes to a local model, which is what "many similar
     items, mechanical judgement" is for.

Rate limits: the OpenAlex common pool is used, not the polite pool, because the
polite pool wants an email address in the query string and this tool does not
send the user's address to a third party without being asked. Add
--mailto you@example.com to opt in and go considerably faster.

    reconcile.py --min-theses 2          # graph-relevant works first
    reconcile.py --status
"""
from __future__ import annotations
import argparse, json, os, queue, re, sqlite3, sys, threading, time, unicodedata
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CITE_DB = HERE / "corpus" / "citations_grobid.db"
OUT_DB = HERE / "corpus" / "works.db"
API = "https://api.openalex.org/works"

SCHEMA = """
CREATE TABLE IF NOT EXISTS work (
  key TEXT PRIMARY KEY,        -- surname|year|normalised-title
  surname TEXT, year INTEGER, title TEXT, n_theses INTEGER,
  openalex TEXT, doi TEXT, matched_title TEXT, matched_year INTEGER,
  score REAL, verdict TEXT,    -- accept | reject | ambiguous | adjudicated-*
  candidates TEXT);            -- JSON, kept only for ambiguous rows
CREATE INDEX IF NOT EXISTS work_verdict ON work(verdict);
"""

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_STOP = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "from"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def toks(s: str) -> set[str]:
    return {w for w in norm(s).split() if w not in _STOP and len(w) > 1}


def sim(a: str, b: str) -> float:
    """Token-set Jaccard. Robust to word order and to OCR dropping a word."""
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fetch(title: str, mailto: str | None, tries: int = 4) -> list[dict]:
    q = {"search": title[:250], "per_page": "5",
         "select": "id,doi,title,publication_year,authorships"}
    if mailto:
        q["mailto"] = mailto
    url = f"{API}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "thesisgraph/0.1 (academic text-mining; +github.com/jhammant/thesisgraph)"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read()).get("results", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):                 # backed off, not broken
                time.sleep(2 ** attempt * 2)
                continue
            return []
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def score(cand: dict, surname: str, year: int, title: str) -> tuple[float, bool]:
    """(0-1 score, surname matched). Year and author are corroboration, not
    the primary signal — a citation's year is often the reprint, not the work."""
    s = sim(title, cand.get("title") or "")
    cy = cand.get("publication_year")
    if cy and year:
        d = abs(cy - year)
        s += 0.12 if d == 0 else 0.05 if d == 1 else -0.10 if d > 3 else 0.0
    sur = norm(surname)
    hit = any(sur and sur in norm(a.get("author", {}).get("display_name", ""))
              for a in cand.get("authorships", [])[:12])
    if hit:
        s += 0.18
    return max(0.0, min(1.0, s)), hit


def collect_works(min_theses: int) -> list[tuple]:
    # read-only: the reparse may still be writing to this database
    c = sqlite3.connect(f"file:{CITE_DB}?mode=ro", uri=True, timeout=30)
    rows = c.execute("""
        SELECT surname, year, title, COUNT(DISTINCT src) n
        FROM cite WHERE title <> '' AND length(title) > 12
        GROUP BY surname, year, LOWER(title)
        HAVING n >= ? ORDER BY n DESC""", (min_theses,)).fetchall()
    return rows


def run(min_theses: int, limit: int | None, mailto: str | None,
        rate: float, workers: int):
    if not CITE_DB.exists():
        sys.exit(f"no {CITE_DB} — run regrobid.py first")
    db = sqlite3.connect(OUT_DB)
    db.executescript(SCHEMA)
    done = {r[0] for r in db.execute("SELECT key FROM work")}

    works = collect_works(min_theses)
    todo = []
    for sur, yr, title, n in works:
        key = f"{sur}|{yr}|{norm(title)[:120]}"
        if key not in done:
            todo.append((key, sur, yr, title, n))
    if limit:
        todo = todo[:limit]
    print(f"{len(works):,} distinct works cited by >={min_theses} theses; "
          f"{len(todo):,} to reconcile ({len(done):,} already done)")
    if not todo:
        return

    # Each lookup is ~1.3s of network latency, so throughput is bound by
    # round-trips, not by OpenAlex's rate limit. Threads keep several in flight
    # while staying inside the documented ~10/s ceiling.
    t0 = time.time()
    counts = {"accept": 0, "reject": 0, "ambiguous": 0}
    work_q, res_q = queue.Queue(), queue.Queue()
    for item in todo:
        work_q.put(item)
    gate = threading.Semaphore(1)
    last = [0.0]

    def throttle():
        with gate:                       # keep requests spaced globally
            wait = rate - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()

    def w():
        while True:
            try:
                key, sur, yr, title, n = work_q.get_nowait()
            except queue.Empty:
                return
            try:
                throttle()
                cands = fetch(title, mailto)
                scored = sorted(((*score(cd, sur, yr, title), cd) for cd in cands),
                                key=lambda x: -x[0])
                if not scored or scored[0][0] < 0.45:
                    verdict, best, cj = "reject", None, None
                elif scored[0][0] >= 0.72 and scored[0][1]:
                    verdict, best, cj = "accept", scored[0][2], None
                else:
                    verdict, best = "ambiguous", scored[0][2]
                    cj = json.dumps([{"openalex": c["id"], "title": c.get("title"),
                                      "year": c.get("publication_year"),
                                      "authors": [a.get("author", {}).get("display_name")
                                                  for a in c.get("authorships", [])[:4]],
                                      "score": round(sc, 3)} for sc, _h, c in scored[:5]])
                res_q.put((key, sur, yr, title, n, best, cj, verdict,
                           scored[0][0] if scored else 0.0))
            except Exception:
                res_q.put((key, sur, yr, title, n, None, None, "reject", 0.0))
            finally:
                work_q.task_done()

    threads = [threading.Thread(target=w, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()

    for i in range(1, len(todo) + 1):
        key, sur, yr, title, n, best, cj, verdict, sc = res_q.get()
        counts[verdict] += 1
        db.execute("INSERT OR REPLACE INTO work VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (key, sur, yr, title, n,
                    (best or {}).get("id", ""), (best or {}).get("doi", "") or "",
                    (best or {}).get("title", "") or "",
                    (best or {}).get("publication_year") or 0,
                    round(sc, 3), verdict, cj))
        if i % 200 == 0:
            db.commit()
            el = time.time() - t0
            print(f"  {i:,}/{len(todo):,}  accept {counts['accept']:,} "
                  f"reject {counts['reject']:,} ambiguous {counts['ambiguous']:,}  "
                  f"{i/el:.1f}/s  eta {(len(todo)-i)/(i/el)/60:.0f} min", flush=True)
    acc, rej, amb = counts["accept"], counts["reject"], counts["ambiguous"]
    db.commit()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min: "
          f"{acc:,} accepted, {rej:,} rejected, {amb:,} ambiguous")
    print(f"  ambiguous rows are what adjudicate.py sends to a local model")


def status():
    if not OUT_DB.exists():
        sys.exit("no works.db yet")
    db = sqlite3.connect(OUT_DB)
    print(f"{'verdict':<22}{'works':>10}{'theses':>12}")
    for v, n, t in db.execute(
            "SELECT verdict, COUNT(*), SUM(n_theses) FROM work GROUP BY verdict ORDER BY 2 DESC"):
        print(f"{v:<22}{n:>10,}{(t or 0):>12,}")
    tot, doi = db.execute(
        "SELECT COUNT(*), SUM(doi<>'') FROM work").fetchone()
    print(f"\n  {doi or 0:,}/{tot:,} works carry a DOI "
          f"({100*(doi or 0)/max(tot,1):.1f}%)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-theses", type=int, default=2,
                    help="only works cited by at least this many theses (default 2)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mailto", help="opt in to the OpenAlex polite pool (faster)")
    ap.add_argument("--rate", type=float, default=0.11,
                    help="seconds between requests (default 0.11 ~ 9/s)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        return status()
    run(a.min_theses, a.limit, a.mailto, a.rate, a.workers)


if __name__ == "__main__":
    sys.exit(main())
