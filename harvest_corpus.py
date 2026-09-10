#!/usr/bin/env python3
"""
harvest_corpus.py — build a thesis corpus and screen it for textual overlap at
scale, so the focal-pair result from run_analysis.py can be read against a real
distribution of thousands of pairs rather than five control points.

Design constraints this respects:

  * The repository publishes `Crawl-delay: 10`. Every request to a repository
    host is spaced by that delay, by default. Nothing here bursts.
  * PDFs are NEVER retained. Each is downloaded, extracted, fingerprinted and
    deleted, so a 29k-thesis corpus costs ~8 GB of extracted text rather than
    ~72 GB of PDFs. The disk this was built on had 77 GiB free.
  * Every stage is resumable and idempotent. Interrupting is safe.
  * Extraction, segmentation, exclusion and matching are imported from
    run_analysis.py, so a corpus pair is measured by the identical code that
    measured the focal pair. That is the whole point.

Stages

    harvest   OAI-PMH metadata for a repository  -> SQLite
    fetch     download + extract + store text    -> corpus/text/*.json.gz
    screen    all-pairs verbatim overlap          -> SQLite pair table
    report    the distribution, and where the focal pair sits

Usage

    python harvest_corpus.py harvest
    python harvest_corpus.py fetch  --limit 400
    python harvest_corpus.py screen
    python harvest_corpus.py report

Paths anchor to this file's location, not the shell's working directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from array import array
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R           # identical extraction / segmentation / matching

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
TEXT_DIR = CORPUS / "text"
TMP_DIR = CORPUS / "tmp"
DB_PATH = CORPUS / "corpus.db"

# Politeness. robots.txt on the target repository asks for Crawl-delay: 10.
DEFAULT_DELAY = 10.0
CONTACT = "papercompare-research (one-off academic text-mining study)"
USER_AGENT = f"PaperCompareBot/0.1 (+{CONTACT}) python-urllib"

REPOSITORIES = {
    "whiterose": {
        "name": "White Rose eTheses Online",
        "oai": "https://etheses.whiterose.ac.uk/cgi/oai2",
        "host": "etheses.whiterose.ac.uk",
    },
}

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}

# A thesis is "doctoral" if its metadata says so. White Rose mixes PhD, EngD,
# EdD, DClinPsy and masters-by-research; masters are excluded.
DOCTORAL_RE = re.compile(
    r"\b(ph\.?d|d\.?phil|doctor|doctoral|eng\.?d|ed\.?d|dclin|d\.?b\.?a|md)\b",
    re.IGNORECASE)
MASTERS_RE = re.compile(r"\b(m\.?phil|m\.?sc|m\.?a\b|masters?|mres)\b", re.IGNORECASE)

EDUCATION_RE = re.compile(
    r"\b(education|teacher|teaching|pedagog|curriculum|classroom|school|pupil|"
    r"student teacher|learner|literacy|numeracy|higher education|"
    r"initial teacher|EFL|ESL|TESOL|language learning)\b", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc (
  id TEXT PRIMARY KEY,          -- OAI identifier
  repo TEXT NOT NULL,
  title TEXT, creator TEXT, year INTEGER, publisher TEXT,
  dctype TEXT, rights TEXT, landing TEXT, pdf_url TEXT,
  is_doctoral INTEGER DEFAULT 0,
  is_education INTEGER DEFAULT 0,
  -- populated by fetch
  status TEXT DEFAULT 'pending', -- pending|ok|failed|skipped
  http_status INTEGER, bytes INTEGER, sha256 TEXT,
  pages INTEGER, no_text_pages INTEGER,
  sentences INTEGER, tokens INTEGER, words INTEGER,
  error TEXT
);
CREATE INDEX IF NOT EXISTS doc_status ON doc(status);
CREATE INDEX IF NOT EXISTS doc_filters ON doc(is_doctoral, is_education);

CREATE TABLE IF NOT EXISTS harvest_state (
  repo TEXT PRIMARY KEY, token TEXT, complete INTEGER DEFAULT 0, seen INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pair (
  a TEXT NOT NULL, b TEXT NOT NULL,
  a_tokens INTEGER, b_tokens INTEGER,
  runs INTEGER, long_runs INTEGER, max_run INTEGER,
  matched_words INTEGER, share REAL,
  PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS pair_share ON pair(share);
CREATE INDEX IF NOT EXISTS pair_max ON pair(max_run);

CREATE TABLE IF NOT EXISTS screen_state (
  a TEXT PRIMARY KEY
);
"""


# Columns added by the `verify` stage. Kept as a migration so an in-flight
# database written by an earlier run is upgraded in place rather than rebuilt.
DOC_COLUMNS = [
    ("abstract", "TEXT"), ("discipline", "TEXT"),
    ("discipline_conf", "REAL"), ("discipline_alt", "TEXT"),
    ("discipline_method", "TEXT"),
]

VERIFY_COLUMNS = [
    ("unique_runs", "INTEGER"), ("unique_long_runs", "INTEGER"),
    ("unique_max_run", "INTEGER"), ("unique_words", "INTEGER"),
    ("unique_share", "REAL"), ("common_runs", "INTEGER"),
    ("quoted_runs", "INTEGER"), ("same_author", "INTEGER"),
    ("same_inst", "INTEGER"), ("title_sim", "INTEGER"), ("verdict", "TEXT"),
]


def db() -> sqlite3.Connection:
    CORPUS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    have = {r[1] for r in con.execute("PRAGMA table_info(pair)")}
    for col, typ in VERIFY_COLUMNS:
        if col not in have:
            con.execute(f"ALTER TABLE pair ADD COLUMN {col} {typ}")
    have = {r[1] for r in con.execute("PRAGMA table_info(doc)")}
    for col, typ in DOC_COLUMNS:
        if col not in have:
            con.execute(f"ALTER TABLE doc ADD COLUMN {col} {typ}")
    con.commit()
    return con


# --------------------------------------------------------------------------- #
# Polite HTTP
# --------------------------------------------------------------------------- #

class Fetcher:
    """One request per host per `delay` seconds, with retry and Retry-After."""

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self._last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            gap = self.delay - (time.monotonic() - last)
            if gap > 0:
                time.sleep(gap)
        self._last[host] = time.monotonic()

    def get(self, url: str, dest: Path | None = None, tries: int = 3
            ) -> tuple[int, bytes | None]:
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(tries):
            self._wait(host)
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                "From": CONTACT,
            })
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    if dest is not None:
                        n = 0
                        with dest.open("wb") as fh:
                            while True:
                                chunk = resp.read(1 << 16)
                                if not chunk:
                                    break
                                fh.write(chunk)
                                n += len(chunk)
                        return resp.status, None
                    return resp.status, resp.read()
            except urllib.error.HTTPError as e:
                ra = e.headers.get("Retry-After") if e.headers else None
                if e.code in (429, 503) and attempt < tries - 1:
                    time.sleep(float(ra) if (ra or "").isdigit() else self.delay * 3)
                    continue
                return e.code, None
            except Exception:
                if attempt < tries - 1:
                    time.sleep(self.delay * 2)
                    continue
                return -1, None
        return -1, None


# --------------------------------------------------------------------------- #
# Stage 1 — OAI-PMH metadata harvest
# --------------------------------------------------------------------------- #

def _text(el, path: str) -> str:
    n = el.find(path, OAI_NS)
    return (n.text or "").strip() if n is not None and n.text else ""


def _all(el, path: str) -> list[str]:
    return [(n.text or "").strip() for n in el.findall(path, OAI_NS) if n.text]


def harvest(repo_key: str, delay: float, max_pages: int | None) -> int:
    repo = REPOSITORIES[repo_key]
    con, f = db(), Fetcher(delay)
    row = con.execute("SELECT token, complete FROM harvest_state WHERE repo=?",
                      (repo_key,)).fetchone()
    token, complete = (row[0], row[1]) if row else (None, 0)
    if complete:
        log(f"  {repo['name']}: metadata harvest already complete")
        return 0
    added = pages_done = 0
    total = None
    while True:
        if token:
            url = f"{repo['oai']}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
        else:
            url = f"{repo['oai']}?verb=ListRecords&metadataPrefix=oai_dc"
        status, body = f.get(url)
        if status != 200 or not body:
            log(f"  HTTP {status} — stopping; rerun to resume")
            break
        try:
            root = ET.fromstring(body)
        except ET.ParseError as e:
            log(f"  XML parse error ({e}) — stopping")
            break
        err = root.find("oai:error", OAI_NS)
        if err is not None:
            log(f"  OAI error: {err.get('code')} {err.text}")
            con.execute("INSERT INTO harvest_state(repo,token,complete) VALUES(?,?,1) "
                        "ON CONFLICT(repo) DO UPDATE SET complete=1", (repo_key, None))
            con.commit()
            break
        rows = []
        for rec in root.findall(".//oai:record", OAI_NS):
            hdr = rec.find("oai:header", OAI_NS)
            if hdr is None:
                continue
            oid = _text(hdr, "oai:identifier")
            if hdr.get("status") == "deleted" or not oid:
                continue
            md = rec.find(".//oai_dc:dc", OAI_NS)
            if md is None:
                continue
            title = _text(md, "dc:title")
            creator = "; ".join(_all(md, "dc:creator"))
            date = _text(md, "dc:date")
            publisher = _text(md, "dc:publisher")
            types = _all(md, "dc:type")
            rights = _text(md, "dc:rights")
            desc = " ".join(_all(md, "dc:description"))[:4000]
            ids = _all(md, "dc:identifier")
            landing = next((i for i in ids if "/id/eprint/" in i and
                            not i.lower().endswith(".pdf")), "")
            pdfs = [i for i in ids if i.lower().endswith(".pdf")]
            # Prefer a whole-thesis file over per-chapter splits.
            pdfs.sort(key=lambda u: (
                0 if re.search(r"whole|full|thesis|final|complete", u, re.I) else 1,
                len(u)))
            pdf = pdfs[0] if pdfs else ""
            ym = re.match(r"(\d{4})", date or "")
            blob = " ".join([title, desc, " ".join(types)])
            doctoral = bool(DOCTORAL_RE.search(blob)) or (
                not MASTERS_RE.search(blob) and "thesis" in " ".join(types).lower())
            rows.append((
                oid, repo_key, title, creator, int(ym.group(1)) if ym else None,
                publisher, "; ".join(types), rights, landing, pdf,
                1 if doctoral else 0, 1 if EDUCATION_RE.search(blob) else 0,
                desc,
            ))
        con.executemany(
            "INSERT INTO doc(id,repo,title,creator,year,publisher,dctype,rights,"
            "landing,pdf_url,is_doctoral,is_education,abstract) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  abstract=COALESCE(excluded.abstract, doc.abstract),"
            "  title=COALESCE(doc.title, excluded.title)", rows)
        added += len(rows)
        pages_done += 1
        rt = root.find(".//oai:resumptionToken", OAI_NS)
        if total is None and rt is not None and rt.get("completeListSize"):
            total = int(rt.get("completeListSize"))
        token = (rt.text or "").strip() if rt is not None and rt.text else None
        con.execute("INSERT INTO harvest_state(repo,token,complete,seen) VALUES(?,?,?,?) "
                    "ON CONFLICT(repo) DO UPDATE SET token=excluded.token, seen=excluded.seen",
                    (repo_key, token, 0, added))
        con.commit()
        have = con.execute("SELECT COUNT(*) FROM doc WHERE repo=?", (repo_key,)).fetchone()[0]
        log(f"  page {pages_done}: +{len(rows)} records, {have:,} stored"
            + (f" of {total:,}" if total else ""))
        if not token:
            con.execute("UPDATE harvest_state SET complete=1 WHERE repo=?", (repo_key,))
            con.commit()
            log("  harvest complete")
            break
        if max_pages and pages_done >= max_pages:
            log(f"  stopped after {max_pages} pages (resumable)")
            break
    con.close()
    return added


# --------------------------------------------------------------------------- #
# Stage 2 — fetch, extract, store, delete the PDF
# --------------------------------------------------------------------------- #

# Stored sentence tuple layout (schema 2). Downstream studies index by these
# rather than by position literals.
SENT_TEXT, SENT_PAGE, SENT_PRINTED, SENT_BUCKET, SENT_HEADING, SENT_EXCL = range(6)


def unpack_sentence(row: list) -> tuple[str, int, str, str, str, str]:
    """Tolerant unpack: schema 1 had 4 fields, schema 2 has 6."""
    if len(row) >= 6:
        return (row[0], row[1], row[2], row[3], row[4], row[5])
    return (row[0], row[1], "", row[2], "", row[3])


def store_path(doc_id: str) -> Path:
    h = hashlib.sha256(doc_id.encode()).hexdigest()
    return TEXT_DIR / h[:2] / f"{h}.json.gz"


def extract_and_store(doc_id: str, pdf_path: Path, meta: dict) -> dict:
    """Extract with the identical pipeline used for the focal pair."""
    spec = R.DocSpec(key="CORPUS", short="CORPUS", label=meta.get("title", "")[:120],
                     surname="", surname_regex=r"(?!x)x", year=meta.get("year") or 0,
                     filename=pdf_path.name, landing_url="", pdf_url="",
                     expected_pages=None, role="control")
    pages = R.extract_pages(pdf_path)
    R.detect_printed_folios(pages)
    doc = R.build_document(spec, pages)
    payload = {
        "id": doc_id,
        "title": meta.get("title"), "creator": meta.get("creator"),
        "year": meta.get("year"), "publisher": meta.get("publisher"),
        "pages": len(pages),
        "no_text_pages": len([p for p in pages if p.source == "none"]),
        # Field order is FIXED — see SENT_* below. Re-fetching the corpus costs
        # ~80 hours at the published crawl delay, so anything a downstream study
        # might want is stored now rather than recovered later.
        "schema": 2,
        "sentences": [
            [s.text, s.pdf_page_start, s.printed_page or "", s.bucket,
             s.heading or "", ";".join(s.exclusions)]
            for s in doc.sentences
        ],
        "modal_x0": doc.modal_x0,
        "modal_size": doc.modal_size,
        "folio_pages": doc.folio_pages,
    }
    out = store_path(doc_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return {
        "pages": len(pages),
        "no_text_pages": payload["no_text_pages"],
        "sentences": len(doc.sentences),
        "tokens": len(doc.tokens),
        "words": sum(s.n_tokens for s in doc.sentences),
    }


def fetch(limit: int, delay: float, only_doctoral: bool, only_education: bool,
          keep_pdfs: bool) -> None:
    con, f = db(), Fetcher(delay)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    where = ["status='pending'", "pdf_url<>''"]
    if only_doctoral:
        where.append("is_doctoral=1")
    if only_education:
        where.append("is_education=1")
    # Deterministic pseudo-random sample: ordering by id alone would bias the
    # slice towards the oldest deposits, which is exactly the wrong shape for a
    # baseline. Sorting by a hash of the id is reproducible and unbiased.
    cand = con.execute(
        f"SELECT id,title,creator,year,publisher,pdf_url FROM doc "
        f"WHERE {' AND '.join(where)}").fetchall()
    cand.sort(key=lambda r: hashlib.blake2b(r[0].encode(), digest_size=8).digest())
    rows = cand[:limit]
    log(f"  {len(rows)} documents to fetch "
        f"(delay {delay:g}s/request -> ~{len(rows)*delay/60:.0f} min minimum)")
    ok = failed = 0
    for i, (doc_id, title, creator, year, publisher, url) in enumerate(rows, 1):
        if store_path(doc_id).exists():
            con.execute("UPDATE doc SET status='ok' WHERE id=?", (doc_id,))
            con.commit()
            continue
        tmp = TMP_DIR / (hashlib.sha256(doc_id.encode()).hexdigest()[:16] + ".pdf")
        status, _ = f.get(url, dest=tmp)
        if status != 200 or not tmp.exists() or tmp.stat().st_size < 1024:
            con.execute("UPDATE doc SET status='failed', http_status=?, "
                        "error='download' WHERE id=?", (status, doc_id))
            con.commit()
            failed += 1
            tmp.unlink(missing_ok=True)
            continue
        try:
            with tmp.open("rb") as fh:
                if fh.read(5) != b"%PDF-":
                    raise ValueError("not a PDF")
            size = tmp.stat().st_size
            sha = R.sha256_file(tmp)
            stats = extract_and_store(doc_id, tmp, {
                "title": title, "creator": creator, "year": year,
                "publisher": publisher})
            con.execute(
                "UPDATE doc SET status='ok', http_status=?, bytes=?, sha256=?, "
                "pages=?, no_text_pages=?, sentences=?, tokens=?, words=?, error=NULL "
                "WHERE id=?",
                (status, size, sha, stats["pages"], stats["no_text_pages"],
                 stats["sentences"], stats["tokens"], stats["words"], doc_id))
            ok += 1
        except Exception as e:
            con.execute("UPDATE doc SET status='failed', http_status=?, error=? "
                        "WHERE id=?", (status, f"{type(e).__name__}: {e}"[:200], doc_id))
            failed += 1
        finally:
            con.commit()
            if not keep_pdfs:
                tmp.unlink(missing_ok=True)   # never retain the PDF
        if i % 10 == 0 or i == len(rows):
            log(f"  {i}/{len(rows)}  ok={ok} failed={failed}")
    con.close()
    log(f"  fetch done: {ok} stored, {failed} failed")


# --------------------------------------------------------------------------- #
# Stage 3 — all-pairs verbatim screening
# --------------------------------------------------------------------------- #

_VOCAB: dict[str, int] = {}
_DOCS: dict[str, array] = {}
_META: dict[str, dict] = {}
_BARRIER: int = 0          # globally unique, decrements across ALL documents


_TOKSENT: dict[str, array] = {}
_SENTS: dict[str, list] = {}


def load_sentences(doc_id: str) -> list:
    """Sentence texts, loaded on demand (only needed for the quotation test)."""
    if doc_id not in _SENTS:
        with gzip.open(store_path(doc_id), "rt", encoding="utf-8") as fh:
            _SENTS[doc_id] = json.load(fh)["sentences"]
        if len(_SENTS) > 64:                       # bounded cache
            for k in list(_SENTS)[:16]:
                if k != doc_id:
                    _SENTS.pop(k, None)
    return _SENTS[doc_id]


def load_doc_tokens(doc_id: str) -> tuple[array, int]:
    """Rebuild the retained-sentence token stream as an int array.

    Uses the identical exclusion rules and tokenisation as the focal analysis;
    barriers between non-adjacent retained sentences are unique negative ids so
    a run can never bridge excluded material.
    """
    p = store_path(doc_id)
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    toks = array("i")
    tsent = array("i")
    prev = None
    words = 0
    global _BARRIER
    for idx, row in enumerate(payload["sentences"]):
        text, page, printed, bucket, heading, excl = unpack_sentence(row)
        words += len(R.match_tokens(text))
        if excl:
            continue
        if prev is not None and idx != prev + 1:
            # Barriers must be unique ACROSS documents, not just within one, or
            # two documents could share a barrier value and a run could bridge
            # excluded material on both sides.
            _BARRIER -= 1
            toks.append(_BARRIER)
            tsent.append(-1)
        for t in R.match_tokens(text):
            v = _VOCAB.get(t)
            if v is None:
                v = len(_VOCAB) + 1
                _VOCAB[t] = v
            toks.append(v)
            tsent.append(idx)
        prev = idx
    _TOKSENT[doc_id] = tsent
    _META[doc_id] = {"pages": payload.get("pages"), "title": payload.get("title"),
                     "creator": payload.get("creator"), "year": payload.get("year"),
                     "words": words}
    return toks, words


def screen(limit: int | None) -> None:
    con = db()
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    if limit:
        ids = ids[:limit]
    log(f"  loading {len(ids)} documents ...")
    for k, doc_id in enumerate(ids, 1):
        try:
            _DOCS[doc_id] = load_doc_tokens(doc_id)[0]
        except Exception as e:
            log(f"    skip {doc_id}: {e}")
        if k % 100 == 0:
            log(f"    {k}/{len(ids)} loaded, vocab {len(_VOCAB):,}")
    ids = [i for i in ids if i in _DOCS]
    n = len(ids)
    total_pairs = n * (n - 1) // 2
    log(f"  {n} documents, {total_pairs:,} pairs, vocab {len(_VOCAB):,}")
    done = {r[0] for r in con.execute("SELECT a FROM screen_state").fetchall()}
    t0 = time.monotonic()
    for i, a_id in enumerate(ids):
        if a_id in done:
            continue
        A = _DOCS[a_id]
        shim_a = SimpleNamespace(tokens=A)
        rows = []
        for b_id in ids[i + 1:]:
            B = _DOCS[b_id]
            runs, _ = R.verbatim_runs(shim_a, SimpleNamespace(tokens=B))
            if not runs:
                continue
            lens = [r.length for r in runs]
            cov = set()
            for r in runs:
                cov.update(range(r.b0, r.b1))
            bw = _META.get(b_id, {}).get("words") or len(B) or 1
            rows.append((a_id, b_id, len(A), len(B), len(runs),
                         sum(1 for L in lens if L >= R.LONG_RUN_WORDS), max(lens),
                         len(cov), len(cov) / bw))
        if rows:
            con.executemany(
                "INSERT INTO pair(a,b,a_tokens,b_tokens,runs,long_runs,max_run,"
                "matched_words,share) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(a,b) DO NOTHING", rows)
        con.execute("INSERT INTO screen_state(a) VALUES(?) "
                    "ON CONFLICT(a) DO NOTHING", (a_id,))
        con.commit()
        el = time.monotonic() - t0
        if (i + 1) % 10 == 0 or i == n - 1:
            log(f"  {i+1}/{n} anchors, {el/60:.1f} min elapsed, "
                f"{con.execute('SELECT COUNT(*) FROM pair').fetchone()[0]:,} "
                f"non-zero pairs")
    con.close()


# --------------------------------------------------------------------------- #
# Stage 3b — false-positive suppression
# --------------------------------------------------------------------------- #
#
# Four filters, weakest to strongest:
#
#   same author       An author's own MPhil feeding their PhD, or a corrected
#                     redeposit. Reuse of your own words is a different question
#                     from reuse of someone else's.
#   duplicate deposit The same thesis under two OAI identifiers.
#   quotation         Both sides carry quotation marks or an attribution — the
#                     two documents quote the same third source.
#   THIRD-DOCUMENT    The strong one, and only possible because we have a
#                     corpus. If a run shared by A and B also appears IN FULL in
#                     some unrelated thesis C, then a shared source can and
#                     evidently does generate it, so it is not evidence of A->B
#                     transfer. This is the empirical form of the argument the
#                     direction check makes on semantic grounds.

def _norm_author(s: str) -> str:
    if not s:
        return ""
    s = R.normalise_text(s.split(";")[0]).lower()
    s = re.sub(r"[^a-z,\s'\-]", " ", s)
    if "," in s:
        sur, rest = s.split(",", 1)
    else:
        parts = s.split()
        sur, rest = (parts[-1] if parts else ""), " ".join(parts[:-1])
    sur = re.sub(r"\s+", "", sur.strip())
    return f"{sur}|{rest.strip()[:1]}" if sur else ""


_QUOTE_RE = re.compile(r"[\"'‘’“”]")


def _run_is_quoted(doc_id: str, t0: int, t1: int) -> bool:
    """True if the run sits inside quoted or attributed text in this document."""
    ts = _TOKSENT.get(doc_id)
    if ts is None:
        return False
    idxs = {ts[k] for k in range(t0, min(t1, len(ts))) if ts[k] >= 0}
    if not idxs:
        return False
    sents = load_sentences(doc_id)
    lo, hi = min(idxs), max(idxs)
    for j in range(max(0, lo - R.ATTRIB_WINDOW), min(len(sents), hi + R.ATTRIB_WINDOW + 1)):
        text = unpack_sentence(sents[j])[SENT_TEXT]
        if j in idxs and len(_QUOTE_RE.findall(text)) >= 2:
            return True
        if R.ATTRIBUTION_RE.search(text):
            return True
    return False


def verify(limit: int | None) -> None:
    from rapidfuzz import fuzz

    con = db()
    N = R.NGRAM_N
    pairs = con.execute(
        "SELECT a,b,runs,share FROM pair WHERE runs>0 ORDER BY a,b").fetchall()
    if not pairs:
        log("  no non-zero pairs to verify — run `screen` first")
        return
    involved = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    if limit:
        ids = ids[:limit]
    log(f"  {len(pairs):,} non-zero pairs across {len(involved)} documents; "
        f"corpus of {len(ids)} for the third-document test")

    log("  loading token streams ...")
    for k, d in enumerate(ids, 1):
        if d not in _DOCS:
            try:
                _DOCS[d] = load_doc_tokens(d)[0]
            except Exception:
                pass
        if k % 100 == 0:
            log(f"    {k}/{len(ids)}")

    # 1. recompute runs and collect every n-gram that appears in one
    log("  recomputing runs and collecting their n-grams ...")
    pair_runs: dict[tuple[str, str], list] = {}
    wanted: set[tuple] = set()
    for a, b, _, _ in pairs:
        A, B = _DOCS.get(a), _DOCS.get(b)
        if A is None or B is None:
            continue
        runs, _ = R.verbatim_runs(SimpleNamespace(tokens=A),
                                  SimpleNamespace(tokens=B))
        pair_runs[(a, b)] = runs
        for r in runs:
            for k in range(r.a0, r.a1 - N + 1):
                wanted.add(tuple(A[k:k + N]))
    log(f"    {len(wanted):,} distinct n-grams appear in some matched run")

    # 2. which corpus documents contain each of those n-grams?
    log("  scanning the corpus for those n-grams ...")
    posting: dict[tuple, set] = {g: set() for g in wanted}
    for k, d in enumerate(ids, 1):
        T = _DOCS.get(d)
        if T is None:
            continue
        for i in range(len(T) - N + 1):
            g = tuple(T[i:i + N])
            s = posting.get(g)
            if s is not None:
                s.add(d)
        if k % 50 == 0:
            log(f"    {k}/{len(ids)} scanned")

    # 3. classify every run, then every pair
    log("  classifying ...")
    meta = {r[0]: r for r in con.execute(
        "SELECT id,title,creator,publisher FROM doc")}
    rows = []
    for (a, b), runs in sorted(pair_runs.items()):
        A = _DOCS[a]
        uniq, common, quoted = [], 0, 0
        for r in runs:
            grams = [tuple(A[k:k + N]) for k in range(r.a0, r.a1 - N + 1)]
            third = None
            for g in grams:
                s = posting.get(g, set()) - {a, b}
                third = s if third is None else (third & s)
                if not third:
                    break
            if third:
                common += 1
                continue
            if _run_is_quoted(b, r.b0, r.b1) and _run_is_quoted(a, r.a0, r.a1):
                quoted += 1
                continue
            uniq.append(r)
        lens = [r.length for r in uniq]
        cov: set = set()
        for r in uniq:
            cov.update(range(r.b0, r.b1))
        bw = (_META.get(b, {}) or {}).get("words") or len(_DOCS[b]) or 1
        ma, mb = meta.get(a), meta.get(b)
        same_author = int(bool(ma and mb and _norm_author(ma[2])
                               and _norm_author(ma[2]) == _norm_author(mb[2])))
        same_inst = int(bool(ma and mb and ma[3] and ma[3] == mb[3]))
        tsim = int(fuzz.token_set_ratio(ma[1] or "", mb[1] or "")) if ma and mb else 0
        share = len(cov) / bw
        if same_author or tsim >= 95:
            verdict = "same_author_or_duplicate"
        elif not uniq:
            verdict = "fully_explained"
        elif not lens or max(lens) < R.LONG_RUN_WORDS:
            verdict = "short_only"
        else:
            verdict = "residual"
        rows.append((len(uniq), sum(1 for L in lens if L >= R.LONG_RUN_WORDS),
                     max(lens, default=0), len(cov), share, common, quoted,
                     same_author, same_inst, tsim, verdict, a, b))
    con.executemany(
        "UPDATE pair SET unique_runs=?, unique_long_runs=?, unique_max_run=?, "
        "unique_words=?, unique_share=?, common_runs=?, quoted_runs=?, "
        "same_author=?, same_inst=?, title_sim=?, verdict=? WHERE a=? AND b=?", rows)
    con.commit()
    tot = len(rows)
    res = sum(1 for r in rows if r[10] == "residual")
    log(f"  verified {tot:,} pairs — {res:,} survive as 'residual' "
        f"({100*res/max(tot,1):.2f}%)")
    con.close()


def focal_check(limit: int | None) -> None:
    """Test the FOCAL pair's runs against the corpus.

    For each of the focal pair's verbatim runs, ask whether any unrelated
    education thesis in the corpus contains the same run in full. That is a
    direct empirical test of the 'they both drew on a common source'
    explanation, which no amount of pairwise analysis can settle on its own.
    """
    con = db()
    N = R.NGRAM_N
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    if limit:
        ids = ids[:limit]
    if not ids:
        log("  corpus is empty — run `fetch` first")
        return
    log(f"  loading {len(ids)} corpus documents ...")
    for k, d in enumerate(ids, 1):
        if d not in _DOCS:
            try:
                _DOCS[d] = load_doc_tokens(d)[0]
            except Exception:
                pass
        if k % 100 == 0:
            log(f"    {k}/{len(ids)}")

    log("  rebuilding the focal pair with the identical pipeline ...")
    specs = R.load_documents()
    early = next(x for x in specs if x.role == "focal-earlier")
    late = next(x for x in specs if x.role == "focal-later")
    info = R.ensure_pdfs((early, late), True, R.PDF_DIR)
    fdocs = {}
    for spec in (early, late):
        pages = R.load_or_extract(spec, Path(info[spec.key]["path"]),
                                  info[spec.key]["sha256"])
        fdocs[spec.key] = R.build_document(spec, pages)

    def to_ints(doc):
        out = array("i")
        global _BARRIER
        for t, s in zip(doc.tokens, doc.tok_sent):
            if s < 0:
                _BARRIER -= 1
                out.append(_BARRIER)
            else:
                v = _VOCAB.get(t)
                if v is None:
                    v = len(_VOCAB) + 1
                    _VOCAB[t] = v
                out.append(v)
        return out

    A = to_ints(fdocs[early.key])
    B = to_ints(fdocs[late.key])
    runs, _ = R.verbatim_runs(SimpleNamespace(tokens=A), SimpleNamespace(tokens=B))
    log(f"  focal pair: {len(runs)} verbatim runs")

    wanted = set()
    for r in runs:
        for k in range(r.a0, r.a1 - N + 1):
            wanted.add(tuple(A[k:k + N]))
    posting = {g: set() for g in wanted}
    log(f"  scanning corpus for {len(wanted):,} focal n-grams ...")
    for k, d in enumerate(ids, 1):
        T = _DOCS.get(d)
        if T is None:
            continue
        for i in range(len(T) - N + 1):
            g = tuple(T[i:i + N])
            s = posting.get(g)
            if s is not None:
                s.add(d)
        if k % 50 == 0:
            log(f"    {k}/{len(ids)} scanned")

    bands = [(8, 11), (12, 15), (16, 19), (20, 29), (30, 999)]
    tab = {b: [0, 0] for b in bands}          # [total, explained by a third thesis]
    explained_examples = []
    for r in runs:
        grams = [tuple(A[k:k + N]) for k in range(r.a0, r.a1 - N + 1)]
        third = None
        for g in grams:
            s = posting.get(g, set())
            third = s if third is None else (third & s)
            if not third:
                break
        band = next(b for b in bands if b[0] <= r.length <= b[1])
        tab[band][0] += 1
        if third:
            tab[band][1] += 1
            if len(explained_examples) < 8:
                explained_examples.append((r.length, sorted(third)[:2]))

    print()
    print("=" * 74)
    print("FOCAL PAIR vs CORPUS — the third-document test")
    print("=" * 74)
    print(f"  corpus: {len(ids)} unrelated education doctorates")
    print()
    print(f"  {'run length':>12}{'runs':>8}{'also in a 3rd thesis':>24}{'unexplained':>13}")
    print("  " + "-" * 58)
    tt = te = 0
    for b in bands:
        t, e = tab[b]
        tt += t
        te += e
        lbl = f"{b[0]}-{b[1]}" if b[1] < 999 else f"{b[0]}+"
        print(f"  {lbl:>12}{t:>8}{e:>18} ({100*e/max(t,1):>4.1f}%){t-e:>13}")
    print("  " + "-" * 58)
    print(f"  {'TOTAL':>12}{tt:>8}{te:>18} ({100*te/max(tt,1):>4.1f}%){tt-te:>13}")
    if explained_examples:
        print()
        print("  examples of runs a third thesis also contains:")
        for L, who in explained_examples[:5]:
            print(f"    {L:>3}w  also in: {', '.join(w.split(':')[-1] for w in who)}")
    print()
    print("  Runs in the 'unexplained' column are not reproduced by any thesis in")
    print("  this corpus. That does not prove no source exists — the corpus is a")
    print("  sample, not the literature — but it is the strongest available test")
    print("  of the shared-source explanation.")
    con.close()


# --------------------------------------------------------------------------- #
# Stage 4 — report the distribution
# --------------------------------------------------------------------------- #

def report() -> None:
    con = db()
    n = con.execute("SELECT COUNT(*) FROM doc WHERE status='ok'").fetchone()[0]
    screened = con.execute("SELECT COUNT(*) FROM screen_state").fetchone()[0]
    ids = [r[0] for r in con.execute(
        "SELECT id FROM doc WHERE status='ok' ORDER BY id").fetchall()]
    pos = {d: i for i, d in enumerate(ids)}
    pairs_done = sum(len(ids) - 1 - pos[a] for (a,) in
                     con.execute("SELECT a FROM screen_state").fetchall()
                     if a in pos)
    nz = con.execute("SELECT COUNT(*) FROM pair").fetchone()[0]

    print()
    print("=" * 74)
    print("CORPUS SCREEN — distribution of pairwise verbatim overlap")
    print("=" * 74)
    tot = con.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
    doc_ = con.execute("SELECT COUNT(*) FROM doc WHERE is_doctoral=1").fetchone()[0]
    edu = con.execute("SELECT COUNT(*) FROM doc WHERE is_education=1").fetchone()[0]
    fail = con.execute("SELECT COUNT(*) FROM doc WHERE status='failed'").fetchone()[0]
    print(f"  metadata records harvested : {tot:,}")
    print(f"    flagged doctoral         : {doc_:,}")
    print(f"    flagged education-related: {edu:,}")
    print(f"  full text extracted        : {n:,}   (failed {fail:,})")
    print(f"  anchors screened           : {screened:,} of {len(ids):,}")
    print(f"  pairs compared             : {pairs_done:,}")
    print(f"  pairs with >=1 shared run  : {nz:,}"
          + (f"  ({100*nz/pairs_done:.3f}%)" if pairs_done else ""))
    if not pairs_done:
        print("\n  nothing screened yet — run `screen` first")
        return

    zero = pairs_done - nz
    print()
    print("  Longest verbatim run per pair")
    print(f"    {'band':>12}{'pairs':>12}{'% of pairs':>13}")
    print("    " + "-" * 35)
    bands = [(0, 0), (8, 11), (12, 15), (16, 19), (20, 29), (30, 49), (50, 10**6)]
    for lo, hi in bands:
        if lo == 0:
            c = zero
            lbl = "no match"
        else:
            c = con.execute("SELECT COUNT(*) FROM pair WHERE max_run BETWEEN ? AND ?",
                            (lo, hi)).fetchone()[0]
            lbl = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
        print(f"    {lbl:>12}{c:>12,}{100*c/pairs_done:>12.4f}%")

    print()
    print("  How exceptional is the focal pair, against this corpus?")
    # Read the focal figures from the run's own summary rather than pinning
    # them to any particular pair of documents.
    focal = {"share": 0.0321, "max_run": 65, "long_runs": 54, "runs": 509}
    try:
        _s = json.loads((HERE / "out" / "summary.json").read_text(encoding="utf-8"))
        _f = _s["pairs"][_s["focal_pair"]]
        focal = {"share": _f["matched_share_of_b_all"],
                 "max_run": _f["verbatim_max_run"],
                 "long_runs": _f["verbatim_long_runs"],
                 "runs": _f["verbatim_runs"]}
    except Exception:
        pass
    for label, col, val in (("longest run >= 65 words", "max_run", 65),
                            ("runs >= 20 words >= 54", "long_runs", 54),
                            ("shared runs >= 509", "runs", 509)):
        c = con.execute(f"SELECT COUNT(*) FROM pair WHERE {col} >= ?",
                        (val,)).fetchone()[0]
        print(f"    corpus pairs matching '{label}': {c:,} "
              f"({100*c/pairs_done:.5f}% of pairs)")
    c = con.execute("SELECT COUNT(*) FROM pair WHERE share >= ?",
                    (focal["share"],)).fetchone()[0]
    print(f"    corpus pairs with share >= 3.21%: {c:,} "
          f"({100*c/pairs_done:.5f}% of pairs)")

    verified = con.execute("SELECT COUNT(*) FROM pair WHERE verdict IS NOT NULL"
                           ).fetchone()[0]
    if verified:
        print()
        print("  False-positive suppression (verified pairs)")
        print(f"    {'verdict':>26}{'pairs':>10}{'% of non-zero':>15}")
        print("    " + "-" * 51)
        for v, lbl in (("same_author_or_duplicate", "same author / duplicate"),
                       ("fully_explained", "every run explained"),
                       ("short_only", "nothing >=20w survives"),
                       ("residual", "RESIDUAL — needs a human")):
            c = con.execute("SELECT COUNT(*) FROM pair WHERE verdict=?",
                            (v,)).fetchone()[0]
            print(f"    {lbl:>26}{c:>10}{100*c/max(verified,1):>14.2f}%")
        cr = con.execute("SELECT SUM(runs),SUM(common_runs),SUM(quoted_runs),"
                         "SUM(unique_runs) FROM pair WHERE verdict IS NOT NULL"
                         ).fetchone()
        if cr and cr[0]:
            print()
            print(f"    runs before filtering            : {cr[0]:,}")
            print(f"    dropped: also in a 3rd thesis    : {cr[1] or 0:,} "
                  f"({100*(cr[1] or 0)/cr[0]:.1f}%)")
            print(f"    dropped: quoted/attributed both  : {cr[2] or 0:,} "
                  f"({100*(cr[2] or 0)/cr[0]:.1f}%)")
            print(f"    surviving runs                   : {cr[3] or 0:,} "
                  f"({100*(cr[3] or 0)/cr[0]:.1f}%)")
        rs = con.execute("SELECT COUNT(*) FROM pair WHERE verdict='residual' "
                         "AND unique_max_run>=?", (65,)).fetchone()[0]
        print()
        print(f"    residual pairs with a surviving run >= 65 words "
              f"(the focal pair's longest): {rs:,}")

    print()
    print("  Top 15 corpus pairs by longest verbatim run")
    print(f"    {'max':>5}{'>=20w':>7}{'runs':>7}{'share':>8}  documents")
    print("    " + "-" * 92)
    for a, b, mx, lr, rn, sh in con.execute(
            "SELECT a,b,max_run,long_runs,runs,share FROM pair "
            "ORDER BY max_run DESC, runs DESC LIMIT 15"):
        ta = con.execute("SELECT title,creator,year FROM doc WHERE id=?", (a,)).fetchone()
        tb = con.execute("SELECT title,creator,year FROM doc WHERE id=?", (b,)).fetchone()
        def nm(t):
            if not t:
                return "?"
            who = (t[1] or "?").split(";")[0].strip()
            return f"{who} {t[2] or '?'}"
        print(f"    {mx:>5}{lr:>7}{rn:>7}{100*sh:>7.2f}%  {nm(ta)}  ~vs~  {nm(tb)}")
    print()
    print("  Note: high-overlap corpus pairs are usually INNOCENT — the same")
    print("  author's masters and doctorate, co-authored or same-lab work, or a")
    print("  thesis deposited twice. Every pair above needs a human before it")
    print("  means anything at all.")
    con.close()


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="OAI-PMH metadata -> SQLite")
    h.add_argument("--repo", default="whiterose", choices=sorted(REPOSITORIES))
    h.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    h.add_argument("--max-pages", type=int, default=None)
    h.add_argument("--refresh", action="store_true",
                   help="re-run the metadata pass to backfill new fields "
                        "(e.g. abstracts) on rows harvested by an older version")

    f = sub.add_parser("fetch", help="download + extract + store (PDFs deleted)")
    f.add_argument("--limit", type=int, default=100)
    f.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    f.add_argument("--doctoral", action="store_true", help="doctoral only")
    f.add_argument("--education", action="store_true", help="education-related only")
    f.add_argument("--keep-pdfs", action="store_true",
                   help="retain PDFs (off by default; the corpus does not fit)")

    s = sub.add_parser("screen", help="all-pairs verbatim overlap")
    s.add_argument("--limit", type=int, default=None)

    v = sub.add_parser("verify", help="suppress false positives on screened pairs")
    v.add_argument("--limit", type=int, default=None)

    fc = sub.add_parser("focal-check",
                        help="test the focal pair's runs against the corpus")
    fc.add_argument("--limit", type=int, default=None)

    sub.add_parser("report", help="distribution and focal-pair percentile")
    sub.add_parser("status", help="one-line progress")

    a = ap.parse_args(argv)
    if a.cmd == "harvest":
        log(f"[harvest] {REPOSITORIES[a.repo]['name']} (delay {a.delay:g}s)")
        if a.refresh:
            con = db()
            con.execute("DELETE FROM harvest_state WHERE repo=?", (a.repo,))
            con.commit()
            con.close()
            log("  --refresh: restarting the metadata pass to backfill fields")
        harvest(a.repo, a.delay, a.max_pages)
    elif a.cmd == "fetch":
        log(f"[fetch] delay {a.delay:g}s, PDFs "
            f"{'RETAINED' if a.keep_pdfs else 'deleted after extraction'}")
        fetch(a.limit, a.delay, a.doctoral, a.education, a.keep_pdfs)
    elif a.cmd == "screen":
        log("[screen] all-pairs verbatim overlap")
        screen(a.limit)
    elif a.cmd == "verify":
        log("[verify] false-positive suppression")
        verify(a.limit)
    elif a.cmd == "focal-check":
        log("[focal-check] focal pair against the corpus")
        focal_check(a.limit)
    elif a.cmd == "report":
        report()
    elif a.cmd == "status":
        con = db()
        for k, q in (("metadata", "SELECT COUNT(*) FROM doc"),
                     ("doctoral", "SELECT COUNT(*) FROM doc WHERE is_doctoral=1"),
                     ("education", "SELECT COUNT(*) FROM doc WHERE is_education=1"),
                     ("extracted", "SELECT COUNT(*) FROM doc WHERE status='ok'"),
                     ("failed", "SELECT COUNT(*) FROM doc WHERE status='failed'"),
                     ("screened", "SELECT COUNT(*) FROM screen_state"),
                     ("nonzero pairs", "SELECT COUNT(*) FROM pair")):
            print(f"  {k:<16}{con.execute(q).fetchone()[0]:>10,}")
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
