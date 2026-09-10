#!/usr/bin/env python3
"""
run_analysis.py — reproducible textual-overlap analysis between two documents,
measured against a control baseline.

The documents to compare are NOT built in. You supply them in `documents.json`
beside this file (see `documents.example.json` for the shape, and .gitignore,
which keeps yours out of version control). Two documents carry the roles
`focal-earlier` and `focal-later`; every other document is a control.

All pairwise combinations are run, so the baseline is a spread rather than a
single point. A raw similarity score between two documents in the same field
means very little on its own; the controls are what make it interpretable.

To see the whole pipeline work end to end without supplying anything, run
`--selftest`: it generates synthetic fixture documents with a deliberately
planted shared passage and asserts that the pipeline recovers it.

Every path anchors to this file's location, never the shell's cwd.
Every artefact is deterministic: no wall-clock, no RNG, no reliance on set or
dict iteration order.  Two consecutive runs over the same PDFs produce
byte-identical matches.csv, report.md and viewer.html.

Usage
    python run_analysis.py                 # run over documents.json
    python run_analysis.py --selftest      # synthetic fixtures + assertions
    python run_analysis.py --skip-download # use PDFs already in ./pdfs
    python run_analysis.py --no-embeddings # skip pass 3

Pinned requirements: see REQUIREMENTS below and requirements.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Paths — anchored to this file, never to the shell's working directory.
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
PDF_DIR = HERE / "pdfs"
CACHE_DIR = HERE / "cache"
OUT_DIR = HERE / "out"
SELFTEST_FIXTURE_DIR = HERE / "selftest_fixtures"
SELFTEST_OUT_DIR = HERE / "out_selftest"

REQUIREMENTS = [
    "pdfplumber==0.11.10",
    "pypdf==6.16.1",
    "rapidfuzz==3.14.5",
    "reportlab==5.0.0",
    "requests==2.34.2",
    # Optional, for the third (embedding) pass only. The pipeline reports
    # "not run" and continues cleanly if these are absent.
    "sentence-transformers==5.7.0",
    "torch==2.13.0",
    "numpy==2.5.2",
    # Used only by check_viewer.py, to drive headless Chromium over CDP.
    "websocket-client==1.9.0",
]

# --------------------------------------------------------------------------- #
# Tunables (all thresholds in one place so the methods section can quote them)
# --------------------------------------------------------------------------- #

NGRAM_N = 8                 # verbatim seed length, in words
NGRAM_POSTING_CAP = 40      # max postings indexed per n-gram
MIN_SENT_TOKENS = 8         # sentences shorter than this skip passes 2 and 3
FUZZ_CUTOFF = 85            # rapidfuzz token_set_ratio threshold
EMBED_CUTOFF = 0.90         # cosine threshold for pass 3
EMBED_ROUND = 4             # cosine values rounded to this many dp for determinism
LINE_Y_TOL = 3.0            # pt: vertical tolerance when grouping words into lines
QUOTE_INDENT_PT = 12.0      # pt beyond modal body x0 that marks a block quote
QUOTE_LINE_FRACTION = 0.60  # fraction of a sentence's lines that must be indented
ATTRIB_WINDOW = 2           # sentences either side that may carry the attribution
CITATION_WINDOW = 3         # sentences either side searched for the earlier author
FOLIO_BAND = 0.08           # top/bottom fraction of page searched for printed folio
HEADING_SIZE_DELTA = 0.6    # pt above modal body size to count as a heading
LONG_RUN_WORDS = 20         # "long run" threshold used by the self-test and report
CDIST_CHUNK = 256           # rows of A per rapidfuzz.process.cdist call
NEAR_PAIR_CAP = 400_000     # safety cap on stored pass-2 pairs per document pair
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Document registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DocSpec:
    key: str
    short: str            # short label used in tables
    label: str            # full human label
    surname: str          # author surname, for the citation check
    surname_regex: str    # tolerant pattern matching that surname in running text
    year: int
    filename: str
    landing_url: str      # EPrints landing page ("" if the PDF URL is direct)
    pdf_url: str          # direct PDF URL, or "" to scrape it from the landing page
    expected_pages: int | None
    role: str             # "focal-earlier" | "focal-later" | "control"


DOCUMENTS_JSON = HERE / "documents.json"


def load_documents() -> tuple[DocSpec, ...]:
    """Read the documents to compare from documents.json.

    Deliberately not hard-coded. This tool measures textual overlap, which is a
    measurement and not an accusation, and a built-in worked example would name
    real people as the standing illustration of one. Supply your own documents,
    or run --selftest, which exercises the entire pipeline on synthetic fixtures
    with a known planted passage.
    """
    if not DOCUMENTS_JSON.exists():
        raise SystemExit(
            f"no {DOCUMENTS_JSON.name} found.\n\n"
            f"Copy documents.example.json to documents.json and fill in the "
            f"documents you want to compare:\n"
            f"  cp documents.example.json documents.json\n\n"
            f"Exactly two entries must have role 'focal-earlier' and "
            f"'focal-later'; any others are controls.\n"
            f"To see the pipeline run without supplying anything:\n"
            f"  python run_analysis.py --selftest"
        )
    raw = json.loads(DOCUMENTS_JSON.read_text(encoding="utf-8"))
    docs = []
    for d in raw.get("documents", []):
        sur = d.get("surname") or ""
        docs.append(DocSpec(
            key=d["key"], short=d.get("short", d["key"]),
            label=d.get("label", d["key"]), surname=sur,
            surname_regex=d.get("surname_regex")
            or (r"\b" + re.escape(sur) + r"\b" if sur else r"(?!x)x"),
            year=int(d.get("year") or 0), filename=d["filename"],
            landing_url=d.get("landing_url", ""), pdf_url=d.get("pdf_url", ""),
            expected_pages=d.get("expected_pages"),
            role=d.get("role", "control")))
    roles = [x.role for x in docs]
    for need in ("focal-earlier", "focal-later"):
        if roles.count(need) != 1:
            raise SystemExit(f"{DOCUMENTS_JSON.name}: exactly one document must "
                             f"have role '{need}' (found {roles.count(need)})")
    return tuple(docs)


# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "Ĳ": "IJ", "ĳ": "ij", "Œ": "OE", "œ": "oe",
    "Æ": "AE", "æ": "ae",
}
# Single-quote family. The 2009 deposit uses U+201E / U+201F as single quotes
# (a cp1252 mangling); both are mapped here so the two producers agree.
_SQUOTES = "‘’‚‛′´`"
_DQUOTES = "“”„‟″«»"
_DASHES_WORD = "‐‑"                      # true hyphens
_DASHES_SEP = "‒–—―−"     # figure/en/em/horizontal/minus
_SPACES = "           " \
          "    　"
_ZERO_WIDTH = "​‌‍⁠﻿­"

_NORM_TABLE = {}
for _src, _dst in _LIGATURES.items():
    _NORM_TABLE[ord(_src)] = _dst
for _c in _SQUOTES:
    _NORM_TABLE[ord(_c)] = "'"
for _c in _DQUOTES:
    _NORM_TABLE[ord(_c)] = '"'
for _c in _DASHES_WORD:
    _NORM_TABLE[ord(_c)] = "-"
for _c in _DASHES_SEP:
    _NORM_TABLE[ord(_c)] = "–"   # one canonical separating dash
for _c in _SPACES:
    _NORM_TABLE[ord(_c)] = " "
for _c in _ZERO_WIDTH:
    _NORM_TABLE[ord(_c)] = None
_NORM_TABLE[ord("…")] = "..."
_NORM_TABLE[ord("\t")] = " "


def normalise_text(s: str) -> str:
    """Normalise ligatures, quote/dash variants and exotic spaces.

    Runs before tokenising so that the same words match across two different
    PDF producers.  Display text keeps its punctuation; only the shapes are
    unified.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_NORM_TABLE)
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


_DASH_SEP_RE = re.compile("[–]")
_INTRAWORD_HYPHEN_RE = re.compile(r"(?<=[a-z0-9])-(?=[a-z0-9])")
_NONWORD_RE = re.compile(r"[^a-z0-9\s]")


def match_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, return the word list used for matching.

    Intra-word hyphens are deleted rather than split on, so that a compound
    broken across a line ("well-" / "known") and the same compound set solid
    both reduce to one token.  Separating dashes become whitespace.
    """
    t = text.lower()
    t = _DASH_SEP_RE.sub(" ", t)
    t = _INTRAWORD_HYPHEN_RE.sub("", t)
    t = _NONWORD_RE.sub(" ", t)
    return t.split()


# --------------------------------------------------------------------------- #
# PDF acquisition
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scrape_first_pdf_link(landing_url: str) -> str:
    import requests
    resp = requests.get(landing_url, headers={"User-Agent": USER_AGENT}, timeout=120)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]+\.pdf)"', resp.text, flags=re.IGNORECASE)
    if not hrefs:
        raise RuntimeError(f"no PDF link found on landing page {landing_url}")
    # Deterministic: first PDF link in document order.
    href = hrefs[0]
    if href.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", landing_url)
        href = (m.group(1) if m else "") + href
    return href


def ensure_pdfs(docs: Sequence[DocSpec], skip_download: bool,
                pdf_dir: Path | None = None) -> dict[str, dict]:
    """Download any missing PDFs, then hash and page-count every one."""
    pdf_dir = pdf_dir or PDF_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, dict] = {}
    for spec in docs:
        path = pdf_dir / spec.filename
        source_url = spec.pdf_url
        if not path.exists():
            if skip_download:
                raise SystemExit(f"missing {path} and --skip-download was given")
            import requests
            if not source_url:
                source_url = scrape_first_pdf_link(spec.landing_url)
                log(f"  scraped PDF link for {spec.key}: {source_url}")
            log(f"  downloading {spec.key} -> {path.name}")
            resp = requests.get(source_url, headers={"User-Agent": USER_AGENT},
                                timeout=600, stream=True)
            resp.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.replace(path)
        with path.open("rb") as fh:
            head = fh.read(5)
        if head != b"%PDF-":
            raise SystemExit(f"{path} is not a PDF (starts {head!r})")
        from pypdf import PdfReader
        pages = len(PdfReader(str(path)).pages)
        info[spec.key] = {
            "key": spec.key,
            "label": spec.label,
            "path": str(path),
            "filename": spec.filename,
            "sha256": sha256_file(path),
            "pages": pages,
            "bytes": path.stat().st_size,
            "source_url": source_url or spec.pdf_url or spec.landing_url,
            "landing_url": spec.landing_url,
        }
    return info


def verify_focal_page_counts(info: dict[str, dict], docs: Sequence[DocSpec]) -> None:
    """Hard stop if a focal deposit is not the expected length."""
    problems = []
    for spec in docs:
        if spec.expected_pages is None:
            continue
        got = info[spec.key]["pages"]
        if got != spec.expected_pages:
            problems.append(
                f"  {spec.key} ({spec.label}): expected {spec.expected_pages} pages, "
                f"repository served {got}"
            )
    if problems:
        raise SystemExit(
            "STOP — focal page counts do not match.\n"
            + "\n".join(problems)
            + "\n\nThe repository has served a different deposit. Every downstream "
              "number would describe the wrong document. Nothing further was run."
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

@dataclass
class Line:
    text: str
    x0: float
    top: float
    bottom: float
    size: float
    bold: bool
    page: int

    def as_json(self) -> list:
        return [self.text, self.x0, self.top, self.bottom, self.size,
                1 if self.bold else 0, self.page]

    @staticmethod
    def from_json(row: list) -> "Line":
        return Line(row[0], row[1], row[2], row[3], row[4], bool(row[5]), row[6])


@dataclass
class Page:
    number: int
    width: float
    height: float
    lines: list[Line]
    source: str          # "pdfplumber" | "pypdf" | "none"
    printed_page: str | None = None


def _group_words_into_lines(words: list[dict], page_no: int) -> list[Line]:
    """Group word-level geometry into lines with a ~3pt vertical tolerance."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (round(w["top"], 1), round(w["x0"], 1),
                                      w.get("text", "")))
    lines: list[Line] = []
    bucket: list[dict] = []
    bucket_top = ws[0]["top"]
    for w in ws:
        if abs(w["top"] - bucket_top) <= LINE_Y_TOL:
            bucket.append(w)
        else:
            lines.append(_bucket_to_line(bucket, page_no))
            bucket = [w]
            bucket_top = w["top"]
    if bucket:
        lines.append(_bucket_to_line(bucket, page_no))
    lines = [ln for ln in lines if ln.text]
    lines.sort(key=lambda ln: (ln.top, ln.x0))
    return lines


def _bucket_to_line(bucket: list[dict], page_no: int) -> Line:
    bucket = sorted(bucket, key=lambda w: w["x0"])
    text = normalise_text(" ".join(w.get("text", "") for w in bucket))
    x0 = min(w["x0"] for w in bucket)
    top = min(w["top"] for w in bucket)
    bottom = max(w["bottom"] for w in bucket)
    sizes = [float(w.get("size") or 0.0) for w in bucket]
    size = max(sizes) if sizes else 0.0
    fonts = " ".join(str(w.get("fontname") or "") for w in bucket).lower()
    bold = ("bold" in fonts) or (",b" in fonts) or ("-bd" in fonts) or ("black" in fonts)
    # Round at extraction time so the cache round-trips bit-for-bit.
    return Line(text, round(x0, 2), round(top, 2), round(bottom, 2),
                round(size, 2), bold, page_no)


def extract_pages(pdf_path: Path) -> list[Page]:
    """Extract per-page lines with word-level geometry.

    pdfplumber is primary.  pypdf is a fallback both per-page (mixed PDFs have
    individual pages pdfplumber returns nothing for) and whole-document.
    """
    import pdfplumber

    pages: list[Page] = []
    pypdf_reader = None

    def pypdf_page_lines(idx: int, page_no: int) -> list[Line]:
        nonlocal pypdf_reader
        if pypdf_reader is None:
            from pypdf import PdfReader
            pypdf_reader = PdfReader(str(pdf_path))
        try:
            raw = pypdf_reader.pages[idx].extract_text() or ""
        except Exception:
            return []
        out = []
        y = 0.0
        for raw_line in raw.splitlines():
            t = normalise_text(raw_line)
            if t:
                out.append(Line(t, 0.0, round(y, 2), round(y + 10.0, 2),
                                0.0, False, page_no))
                y += 12.0
        return out

    try:
        pdf = pdfplumber.open(str(pdf_path))
    except Exception as exc:                                   # whole-document fallback
        log(f"  pdfplumber failed to open {pdf_path.name} ({exc}); using pypdf")
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        for i in range(len(reader.pages)):
            lines = pypdf_page_lines(i, i + 1)
            pages.append(Page(i + 1, 595.0, 842.0, lines,
                              "pypdf" if lines else "none"))
        return pages

    with pdf:
        for i, page in enumerate(pdf.pages):
            page_no = i + 1
            width = round(float(page.width), 2)
            height = round(float(page.height), 2)
            source = "pdfplumber"
            try:
                words = page.extract_words(
                    extra_attrs=["size", "fontname"],
                    use_text_flow=False,
                    keep_blank_chars=False,
                )
            except Exception:
                words = []
            lines = _group_words_into_lines(words, page_no)
            if not lines:                                       # per-page fallback
                lines = pypdf_page_lines(i, page_no)
                source = "pypdf" if lines else "none"
            pages.append(Page(page_no, width, height, lines, source))
    return pages


_FOLIO_RE = re.compile(
    r"^[\[\(\-–\s]*"
    r"(?:page\s*)?"
    r"(\d{1,4}|[ivxlcdm]{1,9}|[IVXLCDM]{1,9})"
    r"[\]\)\-–\s\.]*$"
)


def detect_printed_folios(pages: list[Page]) -> tuple[str, int]:
    """Find the printed folio: a short numeric or roman line alone in the top or
    bottom ~8% of the page.  Chooses whichever band the document uses more, then
    removes the folio line from the body text.

    Returns (band_used, folios_found).
    """
    cand: dict[str, dict[int, tuple[str, int]]] = {"top": {}, "bottom": {}}
    for pg in pages:
        if not pg.lines or pg.height <= 0:
            continue
        top_limit = pg.height * FOLIO_BAND
        bottom_limit = pg.height * (1.0 - FOLIO_BAND)
        for idx, ln in enumerate(pg.lines):
            if len(ln.text) > 12:
                continue
            m = _FOLIO_RE.match(ln.text)
            if not m:
                continue
            band = None
            if ln.bottom <= top_limit:
                band = "top"
            elif ln.top >= bottom_limit:
                band = "bottom"
            if band and pg.number not in cand[band]:
                cand[band][pg.number] = (m.group(1), idx)
    band = "bottom" if len(cand["bottom"]) >= len(cand["top"]) else "top"
    chosen = cand[band]
    for pg in pages:
        hit = chosen.get(pg.number)
        if hit:
            pg.printed_page = hit[0]
            pg.lines = [ln for k, ln in enumerate(pg.lines) if k != hit[1]]
    return band, len(chosen)


# --------------------------------------------------------------------------- #
# Extraction cache (keyed on the PDF's own hash, so it can never go stale)
# --------------------------------------------------------------------------- #

def load_or_extract(spec: DocSpec, pdf_path: Path, sha: str,
                    use_cache: bool = True,
                    cache_dir: Path | None = None) -> list[Page]:
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec.key}.{sha[:16]}.json"
    if use_cache and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        pages = [
            Page(p["n"], p["w"], p["h"], [Line.from_json(r) for r in p["lines"]],
                 p["src"], p["folio"])
            for p in raw["pages"]
        ]
        return pages
    log(f"  extracting {spec.key} ({pdf_path.name}) ...")
    pages = extract_pages(pdf_path)
    band, n_folios = detect_printed_folios(pages)
    log(f"    folio band '{band}', printed folios on {n_folios}/{len(pages)} pages")
    if use_cache:
        payload = {
            "sha256": sha,
            "pages": [
                {"n": p.number, "w": p.width, "h": p.height, "src": p.source,
                 "folio": p.printed_page, "lines": [ln.as_json() for ln in p.lines]}
                for p in pages
            ],
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")),
            encoding="utf-8",
        )
    return pages


# --------------------------------------------------------------------------- #
# Sentence splitting (abbreviation aware)
# --------------------------------------------------------------------------- #

ABBREV = {
    "al", "etc", "cf", "viz", "vs", "ca", "approx", "ibid", "op", "cit",
    "ed", "eds", "edn", "vol", "vols", "no", "nos", "pp", "p", "fig", "figs",
    "tab", "tabs", "ch", "chap", "chaps", "sec", "secs", "para", "paras",
    "dr", "prof", "mr", "mrs", "ms", "messrs", "st", "jr", "sr", "rev", "hon",
    "capt", "gen", "univ", "dept", "inc", "ltd", "co", "corp", "assoc", "est",
    "min", "max", "trans", "repr", "suppl", "ser", "pt", "pts", "ref", "refs",
    "e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "ph.d", "m.a", "b.a", "b.ed",
    "m.ed", "ed.d", "d.phil", "b.sc", "m.sc", "n.d", "c.f", "et",
}

_SENT_END_RE = re.compile(r'([.!?]+)(["\'\)\]’”]*)(\s+)')
_TRAIL_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_TRAIL_NUM_RE = re.compile(r"(?:^|\s)(\d{1,3})$")


def _suppress_boundary(before: str) -> bool:
    """True if the '.' closing `before` is an abbreviation/initial/list number."""
    stripped = before.rstrip()
    m = _TRAIL_WORD_RE.search(stripped)
    if m:
        word = m.group(1).lower().strip(".")
        if word in ABBREV:
            return True
        # Single capital letter => an initial, as in "Fairweather, P."
        if len(m.group(1)) == 1 and m.group(1).isupper():
            return True
        # Dotted initial run such as "P.A" in "Smith, P.A."
        if re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", m.group(1)) and len(word) <= 3:
            return True
    # A bare short numeral: list numbering, "p. 12", a table number.
    if _TRAIL_NUM_RE.search(stripped):
        return True
    return False


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split into sentences, returning (start, end) character spans."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end(2)
        nxt = text[m.end():m.end() + 3]
        if not nxt:
            continue
        c = nxt[0]
        if c in "\"'([‘“":
            c = nxt[1] if len(nxt) > 1 else ""
        if not (c.isupper() or c.isdigit()):
            continue
        if _suppress_boundary(text[start:m.start(1)]):
            continue
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


# --------------------------------------------------------------------------- #
# Headings, chapters, structural regexes
# --------------------------------------------------------------------------- #

# Deliberately case-SENSITIVE on the keyword and anchored to a heading-shaped
# line. Body text is full of cross-references like "section 4.6." or
# "Chapter 4 explicitly details how each criterion was addressed…", and matching
# those as chapter headings silently overwrites the real chapter state.
_CHAPTER_RE = re.compile(
    r"^\s*(?:Chapter|CHAPTER|Chapters|Part|PART|Section|SECTION)\s+"
    r"(?:\d{1,2}|[IVXLCDM]{1,6}|[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|"
    r"[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Ee]leven|[Tt]welve|ONE|TWO|THREE|FOUR|"
    r"FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE)\b"
)
_DOT_LEADER_RE = re.compile(r"\.{4,}")     # table-of-contents lines
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+\S")
_FRONT_HEADING_RE = re.compile(
    r"^\s*(abstract|acknowledgement|acknowledgment|declaration|dedication|"
    r"contents|table of contents|list of (?:tables|figures|abbreviations|"
    r"appendices|acronyms|contents)|abbreviations|glossary|preface|"
    r"references|bibliography|appendix|appendices)\b",
    re.IGNORECASE,
)

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d{1,2}[\.\)]?\s*)?(references?|bibliography|reference list|"
    r"works cited|list of references)\s*:?\s*$",
    re.IGNORECASE,
)
REF_SECTION_END_RE = re.compile(
    r"^\s*(?:appendix|appendices|chapter\b|glossary|index)\b", re.IGNORECASE
)
REF_ENTRY_RE = re.compile(
    r"^[\"'\(]?[A-Z][A-Za-z'’\-]{1,}\s*,\s*[A-Z]\.|"
    r"^[A-Z][A-Za-z'’\-]{1,}\s*,\s*[A-Z][a-z]+\s*(?:\(|,)|"
    r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)"
)

BOILERPLATE_HEADING_RE = re.compile(
    r"(declaration|acknowledgement|acknowledgment|dedication|copyright|"
    r"statement of originality|intellectual property|table of contents|"
    r"^\s*contents\s*$|list of (?:tables|figures|abbreviations|appendices|acronyms)|"
    r"^\s*abbreviations\s*$|glossary|ethics statement|ethical approval|"
    r"statement of ethics|ethical clearance|ethics form)",
    re.IGNORECASE,
)
BOILERPLATE_SENT_RE = re.compile(
    r"(i\s+(?:hereby\s+)?(?:declare|certify|confirm)\s+that|"
    r"has not been submitted (?:in whole or in part )?(?:for|to)|"
    r"in (?:partial )?fulfil?ment of the requirements? for the (?:degree|award)|"
    r"submitted in accordance with the requirements|"
    r"no portion of the work referred to|"
    r"ethical approval (?:was|has been) (?:granted|obtained|sought)|"
    r"approved by the (?:university )?(?:research )?ethics committee|"
    r"copyright rests with the author|"
    r"this (?:thesis|copy) has been supplied on condition)",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(
    r"^\s*(table|figure|fig\.?|chart|graph|diagram|box|plate|exhibit|image|map)\s*"
    r"\.?\s*\d+([\.\-:]\d+)?\b|^\s*(source|notes?|key)\s*:",
    re.IGNORECASE,
)

ATTRIBUTION_RE = re.compile(
    r"\(\s*[A-Z][A-Za-z'’\-]+(?:[^)\n]{0,60})?\b(?:19|20)\d{2}[a-z]?\b[^)\n]{0,30}\)"
    r"|\b[A-Z][A-Za-z'’\-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-z'’\-]+)?"
    r"\s*\(\s*(?:19|20)\d{2}[a-z]?\b"
    r"|\(\s*(?:ibid|op\.?\s*cit|loc\.?\s*cit)\b"
    r"|\bcited in\b"
)

CHAPTER_BUCKETS: tuple[tuple[str, str], ...] = (
    ("references", r"\b(references?|bibliograph|works cited)\b"),
    ("appendix", r"\bappendi(x|ces)\b"),
    ("front_matter", r"\b(abstract|acknowledge?ment|declaration|dedication|"
                     r"table of contents|list of (tables|figures|abbreviations|"
                     r"appendices|acronyms)|glossary|abbreviations|preface|contents)\b"),
    ("literature_review", r"\b(literature review|review of (the )?(related )?literature|"
                          r"literature|theoretical framework|conceptual framework|"
                          r"theoretical perspectives?|contextual review)\b"),
    ("methodology", r"\b(methodolog\w*|research design|research methods?|methods?|"
                    r"research approach|research strategy|design of the study)\b"),
    ("findings", r"\b(findings?|results?|data analysis|analysis of (the )?data|"
                 r"presentation of (the )?(data|findings)|data presentation)\b"),
    ("discussion", r"\b(discussion|interpretation of (the )?findings|"
                   r"discussion of (the )?findings)\b"),
    ("conclusion", r"\b(conclusions?|recommendations?|implications?|"
                   r"concluding (remarks|thoughts|comments)|reflections?)\b"),
    ("introduction", r"\b(introduction|introductory|background to the study|"
                     r"setting the scene|rationale)\b"),
)
_BUCKET_RES = tuple((name, re.compile(pat, re.IGNORECASE))
                    for name, pat in CHAPTER_BUCKETS)

BUCKET_ORDER = ("front_matter", "introduction", "literature_review", "methodology",
                "findings", "discussion", "conclusion", "references", "appendix",
                "unclassified")


_STRUCTURAL_BUCKETS = ("references", "appendix", "front_matter")


def classify_bucket(*texts: str) -> str:
    """Map heading text(s) to a canonical bucket.

    Structural buckets (references / appendix / front matter) win outright.
    Otherwise the bucket mentioned EARLIEST in the heading wins, because a
    chapter title leads with what the chapter is: "Chapter 6: Discussion and
    implications of research findings" is a discussion chapter, not a findings
    chapter, and a fixed priority order gets that backwards.
    """
    for t in texts:
        if not t:
            continue
        for name, rx in _BUCKET_RES:
            if name in _STRUCTURAL_BUCKETS and rx.search(t):
                return name
        best: tuple[int, int, str] | None = None
        for prio, (name, rx) in enumerate(_BUCKET_RES):
            if name in _STRUCTURAL_BUCKETS:
                continue
            m = rx.search(t)
            if m and (best is None or (m.start(), prio) < (best[0], best[1])):
                best = (m.start(), prio, name)
        if best is not None:
            return best[2]
    return "unclassified"


# --------------------------------------------------------------------------- #
# Direction-check regexes (applied to the earlier document's side of a match)
# --------------------------------------------------------------------------- #

DIRECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("own_study",
     r"\b(?:this|the present|the current)\s+(?:doctoral\s+|research\s+|empirical\s+)?"
     r"(?:thesis|study|research|investigation|enquiry|inquiry|project|dissertation)\b"),
    ("own_chapter_structure",
     r"\bthis chapter\b"
     r"|\b(?:the )?(?:following|next|preceding|previous|final|first|second|third|"
     r"fourth|fifth|sixth)\s+chapter\b"
     r"|\bchapter\s+(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"
     r"[^.]{0,60}\b(?:present|presents|presented|outline|outlines|describe|describes|"
     r"discuss|discusses|examine|examines|report|reports|conclude|concludes|"
     r"introduce|introduces|provide|provides|consider|considers|set out|sets out)\b"
     r"|\bin chapter\s+(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b"),
    ("own_findings",
     r"\b(?:findings?|results?|data|evidence)\s+(?:of|from|in)\s+"
     r"(?:this|the present|the current)\s+(?:study|research|thesis|investigation)\b"
     r"|\b(?:this|the present|the current)\s+(?:study|research|thesis|investigation)\s+"
     r"(?:found|found that|revealed|showed|shows|demonstrated|indicated|"
     r"identified|concluded|establishes|established)\b"
     r"|\bmy (?:findings|results|data|analysis)\b"),
    ("own_definition",
     r"\bfor the purposes?\s+of\s+(?:this|the present|the current)\s+"
     r"(?:study|thesis|research|investigation)\b"
     r"|\b(?:is|are|was|were)\s+defined\s+(?:in|for|within)\s+"
     r"(?:this|the present|the current)\s+(?:study|thesis|research)\b"
     r"|\boperational(?:ly)?\s+defin"
     r"|\bin this (?:thesis|study|research)[^.]{0,40}\b(?:refers? to|means?|"
     r"is taken to (?:mean|be)|is understood as)\b"),
    ("own_participants",
     r"\b(?:n\s*=\s*\d+)\b"
     r"|\b\d{1,4}\s+(?:student teachers?|participants?|respondents?|pupils?|"
     r"teachers?|trainees?|interviewees?|informants?|schools?|children)\b"
     r"[^.]{0,80}\b(?:were|was|took part|participated|completed|returned|"
     r"volunteered|recruited|selected|invited|sampled)\b"
     r"|\b(?:the sample|the cohort|the participants?|the respondents?)\s+"
     r"(?:comprised|consisted of|included|was made up of|numbered)\b"),
)
_DIRECTION_RES = tuple((name, re.compile(pat, re.IGNORECASE))
                       for name, pat in DIRECTION_PATTERNS)


def direction_categories(text: str) -> list[str]:
    return sorted(name for name, rx in _DIRECTION_RES if rx.search(text))


# --------------------------------------------------------------------------- #
# Shared-error detectors
# --------------------------------------------------------------------------- #

_DOUBLED_WORD_RE = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.IGNORECASE)
_SPACED_PUNCT_RE = re.compile(r"\s+([,;:!?])")
_SPLIT_HYPHEN_RE = re.compile(r"\b([A-Za-z]{2,})-\s+([a-z]{2,})\b")
_DOUBLED_PUNCT_RE = re.compile(r"([,;:!?])\1")
_MISSING_SPACE_RE = re.compile(r"\b([a-z]{2,}[,;:][a-z]{2,})\b")
_INTERNAL_CAPS_RE = re.compile(r"\b([a-z]{2,}[A-Z][a-z]*)\b")

# Frequent academic misspellings. A shared misspelling survives PDF extraction
# intact, unlike whitespace, so it is real evidence rather than an artefact.
COMMON_TYPOS = frozenset("""
acheive acheived acheivement accomodate accomodated acknowlege adress agressive
allready allthough alot apparant appearence arguement assement basicly becuase
beleive beleived benifit calender catagory cognative collegue commited comparision
concious consistant continous critisism decison definately dependant developement
diffrent dilemna disapear ecspecially embarass enviroment envolved existance
experiance explaination familar finaly foriegn fourty freind fullfil futher
goverment grammer guarentee harrass hierachy hieght ignorence immediatly
independant indispensible interupt irrelevent knowlege lenght liason libary
maintainance managment mispell neccessary necesary noticable occassion occured
occurence oppurtunity paralel particulary percieve perseverence persistant
personel perswade pharoah pheonix posession possable practicaly preceed
preferance priviledge probaly proffesional pronounciation publically pursuade
questionaire questionaires realy recieve recieved recomend recomended
refered refering relevent religous repetion responsability rythm seperate
seperated seperately seperation seige sieze similiar sincerly speach
strenght succesful sucessful supercede suprise suprised temperment tendancy
therefor thier threshhold tommorow truely twelth tyrany underate unfortunatly
untill useable vaccuum vegatable wich wierd writting
""".split())


def error_signatures(text: str) -> set[str]:
    """Idiosyncratic anomalies that could be carried over between documents.

    Only signals that SURVIVE word-geometry PDF extraction are useful here.
    Runs of whitespace are destroyed by extraction (words are re-joined with a
    single space), so a whitespace idiosyncrasy in the original deposit cannot
    be recovered and is not tested for. Doubled words, split hyphens, shared
    misspellings and odd internal capitals all survive intact.
    """
    sigs: set[str] = set()
    for m in _DOUBLED_WORD_RE.finditer(text):
        sigs.add("doubled_word:" + m.group(1).lower())
    for m in _SPLIT_HYPHEN_RE.finditer(text):
        sigs.add(f"split_hyphen:{m.group(1).lower()}-{m.group(2).lower()}")
    for m in _SPACED_PUNCT_RE.finditer(text):
        sigs.add("spaced_punct:" + m.group(1))
    for m in _DOUBLED_PUNCT_RE.finditer(text):
        sigs.add("doubled_punct:" + m.group(1))
    for m in _MISSING_SPACE_RE.finditer(text):
        sigs.add("missing_space:" + m.group(1).lower())
    for m in _INTERNAL_CAPS_RE.finditer(text):
        sigs.add("internal_caps:" + m.group(1).lower())
    for w in re.findall(r"[A-Za-z]{4,}", text):
        lw = w.lower()
        if lw in COMMON_TYPOS:
            sigs.add("misspelling:" + lw)
    return sigs


# --------------------------------------------------------------------------- #
# Document model
# --------------------------------------------------------------------------- #

@dataclass
class Sentence:
    idx: int
    text: str                 # display text (normalised, punctuation kept)
    pdf_page_start: int
    pdf_page_end: int
    printed_page: str | None
    heading: str
    chapter: str
    bucket: str
    x0: float                 # min x0 of the lines this sentence sits on
    indented: bool
    n_tokens: int
    exclusions: tuple[str, ...] = ()

    @property
    def included(self) -> bool:
        return not self.exclusions


@dataclass
class Document:
    spec: DocSpec
    pages: list[Page]
    sentences: list[Sentence]
    modal_size: float
    modal_x0: float
    no_text_pages: list[int]
    pypdf_pages: list[int]
    folio_band: str
    folio_pages: int
    # Matching arrays, over INCLUDED sentences only:
    tokens: list[str] = field(default_factory=list)
    tok_sent: list[int] = field(default_factory=list)
    tok_disp: list[int] = field(default_factory=list)
    disp_words: list[str] = field(default_factory=list)
    sent_tok_range: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.spec.key


def _doc_stats(pages: list[Page]) -> tuple[float, float, float, float]:
    """Modal body font size, modal body x0, modal leading, modal line width."""
    size_w: Counter = Counter()
    x0_w: Counter = Counter()
    widths: list[float] = []
    leadings: list[float] = []
    for pg in pages:
        prev = None
        for ln in pg.lines:
            n = max(len(ln.text), 1)
            if ln.size > 0:
                size_w[round(ln.size * 2) / 2] += n
            if len(ln.text) > 40:
                x0_w[round(ln.x0)] += 1
                widths.append(ln.bottom - ln.top)
            if prev is not None:
                d = ln.top - prev.top
                if 0 < d < 60:
                    leadings.append(d)
            prev = ln
    modal_size = size_w.most_common(1)[0][0] if size_w else 0.0
    if x0_w:
        best = max(x0_w.values())
        modal_x0 = float(min(k for k, v in x0_w.items() if v == best))
    else:
        modal_x0 = 0.0
    leadings.sort()
    modal_leading = leadings[len(leadings) // 2] if leadings else 14.0
    widths.sort()
    modal_height = widths[len(widths) // 2] if widths else 10.0
    return float(modal_size), modal_x0, float(modal_leading), float(modal_height)


def _is_heading(ln: Line, modal_size: float, modal_x0: float) -> bool:
    t = ln.text.strip()
    if not t or len(t) > 130:
        return False
    words = t.split()
    if len(words) > 22:
        return False
    if t.endswith((",", ";", "and", "or", "the", "of")):
        return False
    if _DOT_LEADER_RE.search(t):        # a contents-page line, not a heading
        return False
    # A real chapter heading is short. "Chapter 4 explicitly details how each
    # criterion was addressed as well as the cautionary…" is running prose.
    if _CHAPTER_RE.match(t) and len(words) <= 12:
        return True
    big = modal_size > 0 and ln.size >= modal_size + HEADING_SIZE_DELTA
    letters = [c for c in t if c.isalpha()]
    allcaps = bool(letters) and all(c.isupper() for c in letters) and len(words) <= 14
    if big and len(words) <= 20:
        return True
    if ln.bold and len(words) <= 16 and not t.endswith("."):
        return True
    if allcaps and len(t) >= 4 and not t.endswith("."):
        return True
    if _NUMBERED_HEADING_RE.match(t) and len(words) <= 14 and not t.endswith("."):
        return True
    if _FRONT_HEADING_RE.match(t) and len(words) <= 8:
        return True
    return False


def _reference_pages(pages: list[Page]) -> set[int]:
    """Pages that look like a reference list by entry density."""
    out = set()
    for pg in pages:
        body = [ln for ln in pg.lines if len(ln.text) > 15]
        if len(body) < 6:
            continue
        hits = sum(1 for ln in body if REF_ENTRY_RE.search(ln.text))
        if hits / len(body) >= 0.35:
            out.add(pg.number)
    return out


def build_document(spec: DocSpec, pages: list[Page]) -> Document:
    """Rebuild paragraphs, repair hyphenation, segment, tag and classify."""
    modal_size, modal_x0, modal_leading, _ = _doc_stats(pages)
    ref_dense_pages = _reference_pages(pages)

    # -- Walk lines in document order, building paragraphs and heading state.
    paragraphs: list[dict] = []
    cur_lines: list[Line] = []
    heading = ""
    chapter = ""
    pending_chapter_join = 0
    in_references = False

    def flush():
        nonlocal cur_lines
        if cur_lines:
            paragraphs.append({
                "lines": cur_lines,
                "heading": heading,
                "chapter": chapter,
                "in_references": in_references,
            })
            cur_lines = []

    all_lines: list[Line] = []
    for pg in pages:
        all_lines.extend(pg.lines)

    prev: Line | None = None
    para_x0: float | None = None
    for ln in all_lines:
        if _is_heading(ln, modal_size, modal_x0):
            flush()
            prev = None
            para_x0 = None
            if REFERENCE_HEADING_RE.match(ln.text):
                in_references = True
            elif REF_SECTION_END_RE.match(ln.text):
                in_references = False
            if _CHAPTER_RE.match(ln.text):
                chapter = ln.text
                # "CHAPTER FOUR" often sits above its descriptive title.
                pending_chapter_join = 2 if classify_bucket(ln.text) == "unclassified" else 0
            elif pending_chapter_join > 0:
                chapter = (chapter + " – " + ln.text).strip()
                pending_chapter_join = 0
            heading = ln.text
            continue
        pending_chapter_join = max(0, pending_chapter_join - 1)

        if prev is not None:
            brk = False
            if ln.page != prev.page:
                # Continue across a page break only if the previous page ended
                # mid-sentence and this page resumes lowercase.
                if re.search(r"[.!?][\"'\)\]]?$", prev.text) or not ln.text[:1].islower():
                    brk = True
            else:
                gap = ln.top - prev.bottom
                if gap > 0.75 * modal_leading:
                    brk = True
                elif para_x0 is not None and ln.x0 - para_x0 > 6.0 and gap > 0:
                    brk = True
                elif re.search(r"[.!?][\"'\)\]]?$", prev.text) and \
                        (prev.bottom - prev.top) > 0 and len(prev.text) < 45:
                    brk = True
            if brk:
                flush()
                para_x0 = None
        if not cur_lines:
            para_x0 = ln.x0
        cur_lines.append(ln)
        prev = ln
    flush()

    # -- Assemble paragraph text with hyphen repair, tracking line provenance.
    sentences: list[Sentence] = []
    for para in paragraphs:
        lines: list[Line] = para["lines"]
        buf: list[str] = []
        spans: list[tuple[int, int, int]] = []   # (char_start, char_end, line_index)
        pos = 0
        for i, ln in enumerate(lines):
            t = ln.text
            joined_without_sep = False
            if buf:
                prev_text = buf[-1]
                if prev_text.endswith("-") and t[:1].islower():
                    # Hyphenated line-end split, continuation lowercase: rejoin.
                    buf[-1] = prev_text[:-1]
                    pos -= 1
                    spans[-1] = (spans[-1][0], spans[-1][1] - 1, spans[-1][2])
                    joined_without_sep = True
                elif prev_text.endswith("-") and t[:1].isupper():
                    # Genuine compound with a proper noun: keep the hyphen, no space.
                    joined_without_sep = True
                if not joined_without_sep:
                    buf.append(" ")
                    pos += 1
            start = pos
            buf.append(t)
            pos += len(t)
            spans.append((start, pos, i))
        text = "".join(buf)
        text = re.sub(r"[ ]{2,}", " ", text).strip()
        if not text:
            continue
        # Recompute spans against the collapsed text is unnecessary: collapsing
        # only removes runs of spaces we inserted; map by proportional lookup.
        raw = "".join(buf)
        for s, e in split_sentences(raw):
            sent_raw = raw[s:e].strip()
            if not sent_raw:
                continue
            covered = [li for (ls, le, li) in spans if ls < e and le > s]
            if not covered:
                covered = [0]
            sl = [lines[i] for i in covered]
            pages_covered = sorted({l.page for l in sl})
            x0 = min(l.x0 for l in sl)
            n_ind = sum(1 for l in sl if l.x0 >= modal_x0 + QUOTE_INDENT_PT)
            indented = (n_ind / len(sl)) >= QUOTE_LINE_FRACTION
            toks = match_tokens(sent_raw)
            sentences.append(Sentence(
                idx=len(sentences),
                text=re.sub(r"[ ]{2,}", " ", sent_raw),
                pdf_page_start=pages_covered[0],
                pdf_page_end=pages_covered[-1],
                printed_page=None,
                heading=para["heading"],
                chapter=para["chapter"],
                bucket=classify_bucket(para["chapter"], para["heading"]),
                x0=x0,
                indented=indented,
                n_tokens=len(toks),
                exclusions=("references",) if para["in_references"] else (),
            ))

    # -- Printed page tag.
    folio_by_page = {pg.number: pg.printed_page for pg in pages}
    for s in sentences:
        s.printed_page = folio_by_page.get(s.pdf_page_start)

    # -- Exclusion classification (each category counted separately).
    n = len(sentences)
    attrib_flags = [bool(ATTRIBUTION_RE.search(s.text)) for s in sentences]
    for i, s in enumerate(sentences):
        ex = set(s.exclusions)
        if s.pdf_page_start in ref_dense_pages:
            ex.add("references")
        if CAPTION_RE.match(s.text):
            ex.add("caption")
        if BOILERPLATE_HEADING_RE.search(s.heading or "") or \
                BOILERPLATE_HEADING_RE.search(s.chapter or "") or \
                BOILERPLATE_SENT_RE.search(s.text):
            ex.add("boilerplate")
        if s.indented:
            lo = max(0, i - ATTRIB_WINDOW)
            hi = min(n, i + ATTRIB_WINDOW + 1)
            if any(attrib_flags[j] for j in range(lo, hi)):
                ex.add("blockquote_attributed")
            # An indented passage with no nearby attribution deliberately stays in.
        s.exclusions = tuple(sorted(ex))

    doc = Document(
        spec=spec,
        pages=pages,
        sentences=sentences,
        modal_size=modal_size,
        modal_x0=modal_x0,
        no_text_pages=sorted(p.number for p in pages if p.source == "none"),
        pypdf_pages=sorted(p.number for p in pages if p.source == "pypdf"),
        folio_band="",
        folio_pages=sum(1 for p in pages if p.printed_page),
    )
    _build_token_streams(doc)
    return doc


def _build_token_streams(doc: Document) -> None:
    """Flatten included sentences into a token stream with unique barriers."""
    tokens: list[str] = []
    tok_sent: list[int] = []
    tok_disp: list[int] = []
    disp_words: list[str] = []
    ranges: dict[int, tuple[int, int]] = {}
    prev_idx: int | None = None
    barrier = 0
    for s in doc.sentences:
        if not s.included:
            continue
        if prev_idx is not None and s.idx != prev_idx + 1:
            # Unique, per-document barrier token: can never match anything.
            tokens.append(f"\x00{doc.key}#{barrier}")
            tok_sent.append(-1)
            tok_disp.append(len(disp_words))
            disp_words.append("…")
            barrier += 1
        start = len(tokens)
        for w in s.text.split():
            di = len(disp_words)
            disp_words.append(w)
            for t in match_tokens(w):
                tokens.append(t)
                tok_sent.append(s.idx)
                tok_disp.append(di)
        if len(tokens) > start:
            ranges[s.idx] = (start, len(tokens))
        prev_idx = s.idx
    doc.tokens = tokens
    doc.tok_sent = tok_sent
    doc.tok_disp = tok_disp
    doc.disp_words = disp_words
    doc.sent_tok_range = ranges


# --------------------------------------------------------------------------- #
# Pass 1 — verbatim shared runs
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    a0: int
    a1: int          # exclusive, token indices into doc A
    b0: int
    b1: int          # exclusive, token indices into doc B

    @property
    def length(self) -> int:
        return self.a1 - self.a0


def verbatim_runs(a: Document, b: Document) -> tuple[list[Run], dict]:
    """Every shared run of NGRAM_N+ consecutive words, extended maximally.

    Seeds are extended left and right as far as they will go; runs whose A-span
    and B-span both sit inside another kept run are dropped, so one underlying
    passage yields one alignment rather than dozens of shifted ones.
    """
    A, B = a.tokens, b.tokens
    n = NGRAM_N
    index: dict[tuple, list[int]] = {}
    capped_ngrams = 0
    capped_postings = 0
    for i in range(len(A) - n + 1):
        g = tuple(A[i:i + n])
        lst = index.get(g)
        if lst is None:
            index[g] = [i]
        elif len(lst) < NGRAM_POSTING_CAP:
            lst.append(i)
        else:
            if len(lst) == NGRAM_POSTING_CAP:
                capped_ngrams += 1
                index[g] = lst + [-1]      # sentinel marks "this n-gram is capped"
            capped_postings += 1

    seen: dict[int, int] = {}              # diagonal -> b-end of last run found
    found: set[tuple[int, int, int, int]] = set()
    seeds = 0
    for j in range(len(B) - n + 1):
        g = tuple(B[j:j + n])
        posts = index.get(g)
        if not posts:
            continue
        for i in posts:
            if i < 0:
                continue
            d = i - j
            if seen.get(d, -1) > j:
                continue                    # already inside a run on this diagonal
            seeds += 1
            a0, b0 = i, j
            while a0 > 0 and b0 > 0 and A[a0 - 1] == B[b0 - 1]:
                a0 -= 1
                b0 -= 1
            a1, b1 = i + n, j + n
            while a1 < len(A) and b1 < len(B) and A[a1] == B[b1]:
                a1 += 1
                b1 += 1
            found.add((a0, a1, b0, b1))
            seen[d] = b1

    cand = sorted(found, key=lambda r: (-(r[1] - r[0]), r[0], r[2]))
    kept: list[Run] = []
    for a0, a1, b0, b1 in cand:
        contained = False
        for k in kept:
            if k.a0 <= a0 and a1 <= k.a1 and k.b0 <= b0 and b1 <= k.b1:
                contained = True
                break
        if not contained:
            kept.append(Run(a0, a1, b0, b1))
    kept.sort(key=lambda r: (r.b0, r.a0, r.b1, r.a1))
    stats = {
        "ngram_cap": NGRAM_POSTING_CAP,
        "ngrams_hitting_cap": capped_ngrams,
        "postings_dropped_by_cap": capped_postings,
        "seeds_extended": seeds,
        "runs_before_containment_drop": len(found),
        "runs_after_containment_drop": len(kept),
    }
    return kept, stats


# --------------------------------------------------------------------------- #
# Pass 2 — near-verbatim sentence similarity
# --------------------------------------------------------------------------- #

def near_verbatim_pairs(a: Document, b: Document) -> tuple[list[tuple[int, int, int]], dict]:
    """rapidfuzz token_set_ratio >= FUZZ_CUTOFF over every eligible sentence pair.

    ~250M pairs for the focal documents, so this uses process.cdist with
    score_cutoff, dtype uint8 and workers=-1, chunked over rows of A.
    """
    from rapidfuzz import fuzz, process
    import numpy as np

    a_idx = [s.idx for s in a.sentences if s.included and s.n_tokens >= MIN_SENT_TOKENS]
    b_idx = [s.idx for s in b.sentences if s.included and s.n_tokens >= MIN_SENT_TOKENS]
    a_txt = [" ".join(match_tokens(a.sentences[i].text)) for i in a_idx]
    b_txt = [" ".join(match_tokens(b.sentences[i].text)) for i in b_idx]

    out: list[tuple[int, int, int]] = []
    capped = False
    if a_txt and b_txt:
        for start in range(0, len(a_txt), CDIST_CHUNK):
            chunk = a_txt[start:start + CDIST_CHUNK]
            m = process.cdist(
                chunk, b_txt,
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZ_CUTOFF,
                dtype=np.uint8,
                workers=-1,
            )
            rows, cols = np.nonzero(m)
            for r, c in zip(rows.tolist(), cols.tolist()):
                out.append((a_idx[start + r], b_idx[c], int(m[r, c])))
            if len(out) > NEAR_PAIR_CAP:
                capped = True
                break
    out.sort(key=lambda t: (t[1], t[0], -t[2]))
    stats = {
        "a_sentences_eligible": len(a_idx),
        "b_sentences_eligible": len(b_idx),
        "comparisons": len(a_idx) * len(b_idx),
        "pairs": len(out),
        "cap_hit": capped,
        "cutoff": FUZZ_CUTOFF,
    }
    return out, stats


# --------------------------------------------------------------------------- #
# Pass 3 — sentence embeddings (optional, degrades to "not run")
# --------------------------------------------------------------------------- #

_EMBED_STATE: dict[str, object] = {"tried": False, "model": None, "reason": ""}
_ACTIVE_CACHE_DIR: list[Path | None] = [None]   # set by run_pipeline


def _get_embed_model():
    if _EMBED_STATE["tried"]:
        return _EMBED_STATE["model"]
    _EMBED_STATE["tried"] = True
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL, local_files_only=True, device="cpu")
        _EMBED_STATE["model"] = model
    except Exception as exc:
        _EMBED_STATE["model"] = None
        _EMBED_STATE["reason"] = f"{type(exc).__name__}: {exc}"[:200]
    return _EMBED_STATE["model"]


def _embed_document(doc: Document, sha: str):
    import numpy as np
    model = _get_embed_model()
    if model is None:
        return None, []
    idx = [s.idx for s in doc.sentences if s.included and s.n_tokens >= MIN_SENT_TOKENS]
    cdir = _ACTIVE_CACHE_DIR[0] or CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    tag = EMBED_MODEL.split("/")[-1]
    cache = cdir / f"emb.{doc.key}.{sha[:16]}.{tag}.npy"
    idx_cache = cdir / f"emb.{doc.key}.{sha[:16]}.{tag}.idx.json"
    if cache.exists() and idx_cache.exists():
        cached_idx = json.loads(idx_cache.read_text(encoding="utf-8"))
        if cached_idx == idx:
            return np.load(cache), idx
    texts = [doc.sentences[i].text for i in idx]
    log(f"    embedding {len(texts)} sentences from {doc.key} ...")
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    emb = emb.astype("float32")
    np.save(cache, emb)
    idx_cache.write_text(json.dumps(idx), encoding="utf-8")
    return emb, idx


def embedding_pairs(a: Document, b: Document, sha_a: str, sha_b: str):
    import numpy as np
    ea, ia = _embed_document(a, sha_a)
    if ea is None:
        return None, {"status": "not run", "reason": _EMBED_STATE["reason"] or
                      "no offline sentence-embedding model available"}
    eb, ib = _embed_document(b, sha_b)
    if eb is None:
        return None, {"status": "not run", "reason": _EMBED_STATE["reason"]}
    out: list[tuple[int, int, float]] = []
    capped = False
    for start in range(0, len(ia), 512):
        block = ea[start:start + 512]
        sim = block @ eb.T
        sim = np.round(sim.astype("float64"), EMBED_ROUND)
        rows, cols = np.nonzero(sim >= EMBED_CUTOFF)
        for r, c in zip(rows.tolist(), cols.tolist()):
            out.append((ia[start + r], ib[c], float(sim[r, c])))
        if len(out) > NEAR_PAIR_CAP:
            capped = True
            break
    out.sort(key=lambda t: (t[1], t[0], -t[2]))
    return out, {
        "status": "run",
        "model": EMBED_MODEL,
        "cutoff": EMBED_CUTOFF,
        "rounding_dp": EMBED_ROUND,
        "a_sentences_eligible": len(ia),
        "b_sentences_eligible": len(ib),
        "comparisons": len(ia) * len(ib),
        "pairs": len(out),
        "cap_hit": capped,
    }


# --------------------------------------------------------------------------- #
# Match records
# --------------------------------------------------------------------------- #

@dataclass
class MatchRecord:
    pair_id: str
    pair_kind: str
    pass_name: str
    match_id: str
    run_length_words: int
    similarity: float
    doc_a: str
    a_pdf_page_start: int
    a_pdf_page_end: int
    a_printed_page: str
    a_bucket: str
    a_heading: str
    a_sent_start: int
    a_sent_end: int
    a_disp_start: int
    a_disp_end: int
    a_text: str
    doc_b: str
    b_pdf_page_start: int
    b_pdf_page_end: int
    b_printed_page: str
    b_bucket: str
    b_heading: str
    b_sent_start: int
    b_sent_end: int
    b_disp_start: int
    b_disp_end: int
    b_text: str
    citation_class: str
    direction_flag: int
    direction_categories: str
    error_carryover: int
    error_carryover_detail: str


CSV_FIELDS = [
    "pair_id", "pair_kind", "pass", "match_id", "run_length_words", "similarity",
    "doc_a", "a_pdf_page_start", "a_pdf_page_end", "a_printed_page",
    "a_chapter_bucket", "a_heading", "a_sent_start", "a_sent_end",
    "doc_b", "b_pdf_page_start", "b_pdf_page_end", "b_printed_page",
    "b_chapter_bucket", "b_heading", "b_sent_start", "b_sent_end",
    "citation_class", "direction_flag", "direction_categories",
    "error_carryover", "error_carryover_detail",
]


def _span_text(doc: Document, t0: int, t1: int) -> tuple[str, int, int]:
    d0 = doc.tok_disp[t0]
    d1 = doc.tok_disp[t1 - 1]
    return " ".join(doc.disp_words[d0:d1 + 1]), d0, d1


def _sent_range(doc: Document, t0: int, t1: int) -> tuple[int, int]:
    s = [doc.tok_sent[k] for k in range(t0, t1) if doc.tok_sent[k] >= 0]
    if not s:
        return -1, -1
    return min(s), max(s)


def _citation_class(b: Document, s0: int, s1: int, surname_rx: re.Pattern) -> str:
    if s0 < 0:
        return "unattributed"
    inside = any(surname_rx.search(b.sentences[i].text)
                 for i in range(s0, min(s1 + 1, len(b.sentences))))
    if inside:
        return "attributed"
    lo = max(0, s0 - CITATION_WINDOW)
    hi = min(len(b.sentences), s1 + CITATION_WINDOW + 1)
    near = any(surname_rx.search(b.sentences[i].text) for i in range(lo, hi))
    return "adjacent-but-unattributed" if near else "unattributed"


def build_match_records(pair_id: str, pair_kind: str, a: Document, b: Document,
                        runs: list[Run],
                        near: list[tuple[int, int, int]] | None,
                        emb: list[tuple[int, int, float]] | None) -> list[MatchRecord]:
    surname_rx = re.compile(a.spec.surname_regex)
    recs: list[MatchRecord] = []

    def make(pass_name: str, seq: int, length: int, sim: float,
             a_s0: int, a_s1: int, a_text: str, a_d0: int, a_d1: int,
             b_s0: int, b_s1: int, b_text: str, b_d0: int, b_d1: int) -> MatchRecord:
        sa0 = a.sentences[a_s0] if a_s0 >= 0 else None
        sb0 = b.sentences[b_s0] if b_s0 >= 0 else None
        sa1 = a.sentences[a_s1] if a_s1 >= 0 else sa0
        sb1 = b.sentences[b_s1] if b_s1 >= 0 else sb0
        dirs = direction_categories(a_text)
        shared = sorted(error_signatures(a_text) & error_signatures(b_text))
        return MatchRecord(
            pair_id=pair_id, pair_kind=pair_kind, pass_name=pass_name,
            match_id=f"{pair_id}:{pass_name[:1]}{seq:06d}",
            run_length_words=length, similarity=round(float(sim), 4),
            doc_a=a.key,
            a_pdf_page_start=sa0.pdf_page_start if sa0 else -1,
            a_pdf_page_end=sa1.pdf_page_end if sa1 else -1,
            a_printed_page=(sa0.printed_page or "") if sa0 else "",
            a_bucket=sa0.bucket if sa0 else "unclassified",
            a_heading=(sa0.heading or "") if sa0 else "",
            a_sent_start=a_s0, a_sent_end=a_s1,
            a_disp_start=a_d0, a_disp_end=a_d1, a_text=a_text,
            doc_b=b.key,
            b_pdf_page_start=sb0.pdf_page_start if sb0 else -1,
            b_pdf_page_end=sb1.pdf_page_end if sb1 else -1,
            b_printed_page=(sb0.printed_page or "") if sb0 else "",
            b_bucket=sb0.bucket if sb0 else "unclassified",
            b_heading=(sb0.heading or "") if sb0 else "",
            b_sent_start=b_s0, b_sent_end=b_s1,
            b_disp_start=b_d0, b_disp_end=b_d1, b_text=b_text,
            citation_class=_citation_class(b, b_s0, b_s1, surname_rx),
            direction_flag=1 if dirs else 0,
            direction_categories=";".join(dirs),
            error_carryover=1 if shared else 0,
            error_carryover_detail=";".join(shared),
        )

    for k, r in enumerate(runs):
        a_text, ad0, ad1 = _span_text(a, r.a0, r.a1)
        b_text, bd0, bd1 = _span_text(b, r.b0, r.b1)
        as0, as1 = _sent_range(a, r.a0, r.a1)
        bs0, bs1 = _sent_range(b, r.b0, r.b1)
        recs.append(make("verbatim", k, r.length, 100.0,
                         as0, as1, a_text, ad0, ad1,
                         bs0, bs1, b_text, bd0, bd1))

    if near is not None:
        for k, (ai, bi, score) in enumerate(near):
            sa, sb = a.sentences[ai], b.sentences[bi]
            ra = a.sent_tok_range.get(ai)
            rb = b.sent_tok_range.get(bi)
            ad0, ad1 = (a.tok_disp[ra[0]], a.tok_disp[ra[1] - 1]) if ra else (-1, -1)
            bd0, bd1 = (b.tok_disp[rb[0]], b.tok_disp[rb[1] - 1]) if rb else (-1, -1)
            recs.append(make("near_verbatim", k, min(sa.n_tokens, sb.n_tokens),
                             score, ai, ai, sa.text, ad0, ad1,
                             bi, bi, sb.text, bd0, bd1))

    if emb is not None:
        for k, (ai, bi, cos) in enumerate(emb):
            sa, sb = a.sentences[ai], b.sentences[bi]
            ra = a.sent_tok_range.get(ai)
            rb = b.sent_tok_range.get(bi)
            ad0, ad1 = (a.tok_disp[ra[0]], a.tok_disp[ra[1] - 1]) if ra else (-1, -1)
            bd0, bd1 = (b.tok_disp[rb[0]], b.tok_disp[rb[1] - 1]) if rb else (-1, -1)
            recs.append(make("embedding", k, min(sa.n_tokens, sb.n_tokens),
                             cos, ai, ai, sa.text, ad0, ad1,
                             bi, bi, sb.text, bd0, bd1))

    recs.sort(key=lambda m: (m.pair_id, m.pass_name, -m.run_length_words,
                             m.b_sent_start, m.a_sent_start, m.match_id))
    return recs


# --------------------------------------------------------------------------- #
# Per-pair analysis
# --------------------------------------------------------------------------- #

def analyse_pair(pair_id: str, pair_kind: str, a: Document, b: Document,
                 runs: list[Run], run_stats: dict,
                 near: list | None, near_stats: dict,
                 emb: list | None, emb_stats: dict,
                 recs: list[MatchRecord]) -> dict:
    b_included_tokens = len([t for t in b.tok_sent if t >= 0])
    b_all_tokens = sum(s.n_tokens for s in b.sentences)
    covered: set[int] = set()
    for r in runs:
        covered.update(range(r.b0, r.b1))
    matched_words_b = len({b.tok_disp[i] for i in covered})

    lengths = sorted(r.length for r in runs)
    buckets = [(8, 11), (12, 15), (16, 19), (20, 29), (30, 49), (50, 10 ** 9)]
    dist = {}
    for lo, hi in buckets:
        label = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        dist[label] = sum(1 for L in lengths if lo <= L <= hi)

    by_bucket_matched: Counter = Counter()
    by_bucket_words: Counter = Counter()
    for r in runs:
        bs0, bs1 = _sent_range(b, r.b0, r.b1)
        bucket = b.sentences[bs0].bucket if bs0 >= 0 else "unclassified"
        by_bucket_matched[bucket] += 1
        by_bucket_words[bucket] += r.length
    bucket_totals: Counter = Counter()
    for s in b.sentences:
        if s.included:
            bucket_totals[s.bucket] += s.n_tokens

    verb = [m for m in recs if m.pass_name == "verbatim"]
    cite_counts = Counter(m.citation_class for m in verb)
    dir_flagged = [m for m in verb if m.direction_flag]
    err_flagged = [m for m in verb if m.error_carryover]

    surname_rx = re.compile(a.spec.surname_regex)
    mentions_all = sum(len(surname_rx.findall(s.text)) for s in b.sentences)
    mentions_body = sum(len(surname_rx.findall(s.text))
                        for s in b.sentences if s.included)
    mentions_refs = sum(len(surname_rx.findall(s.text))
                        for s in b.sentences if "references" in s.exclusions)
    mentions_sentences = sorted(
        {s.pdf_page_start for s in b.sentences if surname_rx.search(s.text)}
    )
    # Which work of the earlier author is being cited? A citation of the
    # earlier THESIS (its own year) is a different fact from a citation of some
    # other book by the same author, so the years are counted separately.
    year_counts: Counter = Counter()
    for s in b.sentences:
        for m in surname_rx.finditer(s.text):
            tail = s.text[m.end():m.end() + 40]
            ym = re.search(r"\b(19|20)\d{2}[a-z]?\b", tail)
            year_counts[ym.group(0) if ym else "no year adjacent"] += 1
    # Is the earlier document itself listed in the later document's bibliography?
    earlier_in_biblio = sorted(
        s.pdf_page_start for s in b.sentences
        if "references" in s.exclusions and surname_rx.search(s.text)
        and re.search(rf"\b{a.spec.year}[a-z]?\b", s.text)
    )

    comparisons = near_stats.get("comparisons", 0) or 1
    return {
        "pair_id": pair_id,
        "pair_kind": pair_kind,
        "doc_a": a.key, "doc_a_label": a.spec.label,
        "doc_b": b.key, "doc_b_label": b.spec.label,
        "verbatim_runs": len(runs),
        "verbatim_long_runs": sum(1 for L in lengths if L >= LONG_RUN_WORDS),
        "verbatim_max_run": lengths[-1] if lengths else 0,
        "verbatim_median_run": lengths[len(lengths) // 2] if lengths else 0,
        "verbatim_total_matched_words_b": matched_words_b,
        "b_included_tokens": b_included_tokens,
        "b_all_tokens": b_all_tokens,
        "matched_share_of_b_included": matched_words_b / max(b_included_tokens, 1),
        "matched_share_of_b_all": matched_words_b / max(b_all_tokens, 1),
        "run_length_distribution": dist,
        "runs_by_bucket": {k: by_bucket_matched.get(k, 0) for k in BUCKET_ORDER},
        "matched_words_by_bucket": {k: by_bucket_words.get(k, 0) for k in BUCKET_ORDER},
        "bucket_token_totals": {k: bucket_totals.get(k, 0) for k in BUCKET_ORDER},
        "near_verbatim_pairs": len(near) if near is not None else 0,
        "embedding_pairs": len(emb) if emb is not None else None,
        "comparisons": near_stats.get("comparisons", 0),
        "verbatim_per_million_comparisons": len(runs) * 1e6 / comparisons,
        "long_runs_per_million_comparisons": (
            sum(1 for L in lengths if L >= LONG_RUN_WORDS) * 1e6 / comparisons),
        "near_per_million_comparisons": (
            (len(near) if near is not None else 0) * 1e6 / comparisons),
        "embedding_per_million_comparisons": (
            (len(emb) * 1e6 / comparisons) if emb is not None else None),
        "citation_classes": {k: cite_counts.get(k, 0) for k in
                             ("attributed", "adjacent-but-unattributed", "unattributed")},
        "direction_flagged": len(dir_flagged),
        "direction_categories": dict(sorted(Counter(
            c for m in dir_flagged for c in m.direction_categories.split(";") if c
        ).items())),
        "error_carryover_matches": len(err_flagged),
        "author_a_mentions_in_b_total": mentions_all,
        "author_a_mentions_in_b_body": mentions_body,
        "author_a_mentions_in_b_reference_list": mentions_refs,
        "author_a_mention_pages_in_b": mentions_sentences,
        "author_a_mentions_by_cited_year": dict(sorted(year_counts.items())),
        "earlier_document_in_b_bibliography_pages": earlier_in_biblio,
        "earlier_document_year": a.spec.year,
        "run_stats": run_stats,
        "near_stats": near_stats,
        "embedding_stats": emb_stats,
    }


# --------------------------------------------------------------------------- #
# Output: matches.csv
# --------------------------------------------------------------------------- #

def write_matches_csv(path: Path, recs: list[MatchRecord]) -> None:
    rows = sorted(
        recs,
        key=lambda m: (m.pair_id, m.pass_name, -m.run_length_words,
                       m.b_sent_start, m.a_sent_start, m.match_id),
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_FIELDS)
        for m in rows:
            w.writerow([
                m.pair_id, m.pair_kind, m.pass_name, m.match_id,
                m.run_length_words, f"{m.similarity:.4f}",
                m.doc_a, m.a_pdf_page_start, m.a_pdf_page_end, m.a_printed_page,
                m.a_bucket, m.a_heading, m.a_sent_start, m.a_sent_end,
                m.doc_b, m.b_pdf_page_start, m.b_pdf_page_end, m.b_printed_page,
                m.b_bucket, m.b_heading, m.b_sent_start, m.b_sent_end,
                m.citation_class, m.direction_flag, m.direction_categories,
                m.error_carryover, m.error_carryover_detail,
            ])


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Output: report.md
# --------------------------------------------------------------------------- #

SYNTHETIC_BANNER = """> # ⚠️  SYNTHETIC SELF-TEST OUTPUT — NOT REAL FINDINGS  ⚠️
>
> **Every number in this file was produced from machine-generated fixture PDFs
> containing invented prose and a deliberately planted shared passage.**
> The documents described here do not exist. No real thesis, no real author and
> no real person is described anywhere in this file. Nothing here may be quoted,
> cited or relied upon as a finding about any real work.
>
> Real results live in `out/report.md`.

---

"""


def _pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def _fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def write_report(path: Path, ctx: dict) -> None:
    docs: dict[str, Document] = ctx["docs"]
    info: dict[str, dict] = ctx["pdf_info"]
    analyses: dict[str, dict] = ctx["analyses"]
    focal_id: str = ctx["focal_pair_id"]
    focal = analyses[focal_id]
    a = docs[ctx["focal_a"]]
    b = docs[ctx["focal_b"]]
    synthetic = ctx["synthetic"]

    L: list[str] = []
    add = L.append

    if synthetic:
        add(SYNTHETIC_BANNER.rstrip("\n"))
        add("")

    add("# Textual-overlap analysis: focal pair against a control baseline")
    add("")
    add(f"- **Earlier document (A):** {a.spec.label} — "
        f"{info[a.key]['pages']} PDF pages")
    add(f"- **Later document (B):** {b.spec.label} — "
        f"{info[b.key]['pages']} PDF pages")
    add(f"- **Controls:** " + "; ".join(
        docs[k].spec.label for k in ctx["doc_order"]
        if docs[k].spec.role == "control"))
    add("")
    add("All six pairwise combinations of the four documents were run through an "
        "identical pipeline, so the control baseline is a spread rather than a "
        "single point.")
    add("")

    # ---------------------------------------------------------------- summary
    add("## 1. Headline numbers (focal pair)")
    add("")
    add(f"| Measure | Value |")
    add(f"|---|---|")
    add(f"| Verbatim shared runs (≥{NGRAM_N} consecutive words, maximally extended) "
        f"| **{_fmt_int(focal['verbatim_runs'])}** |")
    add(f"| — of which ≥{LONG_RUN_WORDS} words | **{_fmt_int(focal['verbatim_long_runs'])}** |")
    add(f"| Longest verbatim run | **{focal['verbatim_max_run']} words** |")
    add(f"| Median verbatim run | {focal['verbatim_median_run']} words |")
    add(f"| Matched words in the later document (union, verbatim) "
        f"| **{_fmt_int(focal['verbatim_total_matched_words_b'])}** |")
    add(f"| Matched words as a share of the later document's *matched-set* words "
        f"| **{_pct(focal['matched_share_of_b_included'])}** |")
    add(f"| Matched words as a share of *all* the later document's words "
        f"| {_pct(focal['matched_share_of_b_all'])} |")
    add(f"| Near-verbatim sentence pairs (token_set_ratio ≥ {FUZZ_CUTOFF}) "
        f"| **{_fmt_int(focal['near_verbatim_pairs'])}** |")
    ep = focal["embedding_pairs"]
    add(f"| Embedding sentence pairs (cosine ≥ {EMBED_CUTOFF}) | "
        + (f"**{_fmt_int(ep)}**" if ep is not None else "*not run*") + " |")
    add(f"| Sentence-pair comparisons behind those rates | "
        f"{_fmt_int(focal['comparisons'])} |")
    add("")

    # ------------------------------------------------------- control baseline
    add("## 2. Control comparison — the point of the exercise")
    add("")
    add("A raw similarity score between two theses in the same subfield means "
        "nothing on its own. The table below gives every pairwise combination, "
        "normalised to matches per million sentence-pair comparisons so that "
        "documents of different lengths are comparable.")
    add("")
    add("| Pair | Kind | Verbatim runs | Runs/M | Long runs (≥20w) | Long runs/M | "
        "Longest run | Near-verbatim/M | Embedding/M | Matched words in later doc |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for pid in ctx["pair_order"]:
        an = analyses[pid]
        embpm = an["embedding_per_million_comparisons"]
        add("| {p} | {k} | {vr} | {vpm:.2f} | {lr} | {lpm:.3f} | {mx} | {npm:.2f} | "
            "{epm} | {mw} |".format(
                p=pid,
                k="**FOCAL**" if an["pair_kind"] == "focal" else "control",
                vr=_fmt_int(an["verbatim_runs"]),
                vpm=an["verbatim_per_million_comparisons"],
                lr=_fmt_int(an["verbatim_long_runs"]),
                lpm=an["long_runs_per_million_comparisons"],
                mx=an["verbatim_max_run"],
                npm=an["near_per_million_comparisons"],
                epm=(f"{embpm:.2f}" if embpm is not None else "n/r"),
                mw=_fmt_int(an["verbatim_total_matched_words_b"]),
            ))
    add("")
    ctrl = [analyses[p] for p in ctx["pair_order"] if analyses[p]["pair_kind"] != "focal"]
    if ctrl:
        cr = sorted(c["verbatim_per_million_comparisons"] for c in ctrl)
        cl = sorted(c["long_runs_per_million_comparisons"] for c in ctrl)
        cx = sorted(c["verbatim_max_run"] for c in ctrl)
        fr = focal["verbatim_per_million_comparisons"]
        fl = focal["long_runs_per_million_comparisons"]
        add(f"**Control spread (5 pairs).** Verbatim runs per million comparisons: "
            f"{cr[0]:.2f} – {cr[-1]:.2f} (median {cr[len(cr)//2]:.2f}). "
            f"Long runs (≥{LONG_RUN_WORDS} words) per million: "
            f"{cl[0]:.3f} – {cl[-1]:.3f}. "
            f"Longest verbatim run in any control pair: {cx[-1]} words.")
        add("")
        ratio = (fr / cr[-1]) if cr[-1] > 0 else float("inf")
        lratio = (fl / cl[-1]) if cl[-1] > 0 else float("inf")
        add(f"**Focal pair against that spread.** Verbatim runs per million: "
            f"{fr:.2f} — "
            + (f"{ratio:.1f}× the highest control pair."
               if math.isfinite(ratio) else "no control pair produced any run.")
            + f" Long runs per million: {fl:.3f} — "
            + (f"{lratio:.1f}× the highest control pair."
               if math.isfinite(lratio) else
               "**no control pair produced a single run of "
               f"{LONG_RUN_WORDS}+ words.**"))
        add("")

    # ------------------------------------------------------------ where it sits
    add("## 3. Where the overlap sits in the later document")
    add("")
    add("Headings are mapped to canonical buckets by regular expression. The "
        "`unclassified` row is reported so that a failure of that mapping is "
        "visible rather than hidden.")
    add("")
    add("| Chapter bucket | Verbatim runs | Matched words | Bucket words "
        "(matched set) | Density (matched words / bucket words) |")
    add("|---|---:|---:|---:|---:|")
    tot_words = 0
    for bucket in BUCKET_ORDER:
        runs_n = focal["runs_by_bucket"].get(bucket, 0)
        words = focal["matched_words_by_bucket"].get(bucket, 0)
        total = focal["bucket_token_totals"].get(bucket, 0)
        tot_words += total
        dens = (words / total) if total else 0.0
        add(f"| {bucket} | {_fmt_int(runs_n)} | {_fmt_int(words)} | "
            f"{_fmt_int(total)} | {_pct(dens)} |")
    unc = focal["bucket_token_totals"].get("unclassified", 0)
    add("")
    unc_share = unc / max(tot_words, 1)
    a_tot = sum(s.n_tokens for s in a.sentences if s.included) or 1
    a_unc = sum(s.n_tokens for s in a.sentences
                if s.included and s.bucket == "unclassified")
    a_unc_share = a_unc / a_tot
    add(f"**Unclassified share of the later document's matched-set words: "
        f"{_pct(unc_share)}.** For the earlier document — whose bucket labels "
        f"appear in the A-side columns of the tables below, but not in the "
        f"density table above — the unclassified share is "
        f"{_pct(a_unc_share)}.")
    if unc_share > 0.25:
        add("")
        add("> ⚠️ More than a quarter of the later document's words could not be "
            "assigned to a canonical chapter bucket. **The chapter-density table "
            "above is unreliable** and should not be used to argue about where "
            "overlap concentrates.")
    elif unc_share > 0.10:
        add("")
        add("> Note: the unclassified share is non-trivial. Treat the "
            "chapter-density table as indicative rather than exact.")
    add("")

    # --------------------------------------------------------- run length dist
    add("## 4. Length distribution of verbatim runs")
    add("")
    add("A single 40-word run means more than forty 8-word runs; the "
        "distribution matters more than the total.")
    add("")
    add("| Run length (words) | Focal pair | " +
        " | ".join(p for p in ctx["pair_order"] if p != focal_id) + " |")
    add("|---|---:|" + "---:|" * (len(ctx["pair_order"]) - 1))
    dist_keys = list(focal["run_length_distribution"].keys())
    for k in dist_keys:
        cells = [str(analyses[p]["run_length_distribution"].get(k, 0))
                 for p in ctx["pair_order"] if p != focal_id]
        add(f"| {k} | **{focal['run_length_distribution'][k]}** | "
            + " | ".join(cells) + " |")
    add("")

    longest = ctx["focal_verbatim_sorted"][:12]
    if longest:
        add("The longest verbatim runs in the focal pair (page numbers only; the "
            "text is carried by `viewer.html`, both documents being in copyright):")
        add("")
        add("| Words | A: PDF p. (printed) | A: bucket | B: PDF p. (printed) | "
            "B: bucket | Citation class | Direction flag |")
        add("|---:|---|---|---|---|---|---|")
        for m in longest:
            add("| {w} | {ap} ({apr}) | {ab} | {bp} ({bpr}) | {bb} | {cc} | {df} |"
                .format(w=m.run_length_words,
                        ap=m.a_pdf_page_start, apr=m.a_printed_page or "—",
                        ab=m.a_bucket,
                        bp=m.b_pdf_page_start, bpr=m.b_printed_page or "—",
                        bb=m.b_bucket, cc=m.citation_class,
                        df="yes" if m.direction_flag else "no"))
        add("")

    # ------------------------------------------------------------- direction
    add("## 5. Direction check")
    add("")
    add("A shared third source cannot generate a passage in which the earlier "
        "document describes its own study — its own chapter structure, its own "
        "findings, its own stated definitions, its own participant numbers. "
        "Those matches are the diagnostically important ones and are listed "
        "separately here.")
    add("")
    add(f"**Verbatim runs whose earlier-document side is self-descriptive: "
        f"{_fmt_int(focal['direction_flagged'])} of "
        f"{_fmt_int(focal['verbatim_runs'])}.**")
    add("")
    if focal["direction_categories"]:
        add("| Self-description category | Runs |")
        add("|---|---:|")
        for k, v in sorted(focal["direction_categories"].items()):
            add(f"| {k} | {v} |")
        add("")
    dflag = ctx["focal_direction_sorted"][:25]
    if dflag:
        add("Direction-flagged runs, longest first:")
        add("")
        add("| Words | A: PDF p. (printed) | B: PDF p. (printed) | B: bucket | "
            "Categories | Citation class |")
        add("|---:|---|---|---|---|---|")
        for m in dflag:
            add("| {w} | {ap} ({apr}) | {bp} ({bpr}) | {bb} | {c} | {cc} |".format(
                w=m.run_length_words, ap=m.a_pdf_page_start,
                apr=m.a_printed_page or "—", bp=m.b_pdf_page_start,
                bpr=m.b_printed_page or "—", bb=m.b_bucket,
                c=m.direction_categories or "—", cc=m.citation_class))
        add("")
    for pid in ctx["pair_order"]:
        if pid == focal_id:
            continue
        an = analyses[pid]
        if an["direction_flagged"]:
            add(f"Control pair {pid} also produced {an['direction_flagged']} "
                f"direction-flagged run(s); see `matches.csv`.")
    add("")

    # -------------------------------------------------------------- citation
    add("## 6. Citation check")
    add("")
    cc = focal["citation_classes"]
    total_v = max(sum(cc.values()), 1)
    add(f"For every verbatim run, the later document was searched within "
        f"±{CITATION_WINDOW} sentences for a citation naming the earlier "
        f"document's author.")
    add("")
    add("| Classification | Runs | Share |")
    add("|---|---:|---:|")
    for k in ("attributed", "adjacent-but-unattributed", "unattributed"):
        add(f"| {k} | {_fmt_int(cc[k])} | {100.0*cc[k]/total_v:.1f}% |")
    add("")
    mentions = focal["author_a_mentions_in_b_total"]
    add(f"### Does the later document name the earlier author at all?")
    add("")
    if mentions > 0:
        add(f"> **Yes — and this is the point that matters most.** The later "
            f"document names *{a.spec.surname}* **{_fmt_int(mentions)} time(s)** "
            f"in total, anywhere in the document, including its reference list "
            f"({_fmt_int(focal['author_a_mentions_in_b_body'])} of those fall "
            f"inside the matched set rather than in excluded material such as "
            f"the bibliography).")
        pages = focal["author_a_mention_pages_in_b"]
        add(">")
        add("> PDF pages of the later document on which that name appears: "
            + ", ".join(str(p) for p in pages[:60])
            + (" …" if len(pages) > 60 else "") + ".")
        biblio = focal["earlier_document_in_b_bibliography_pages"]
        add(">")
        if biblio:
            add(f"> **The earlier thesis itself is listed in the later "
                f"document's bibliography** — a reference-list entry naming "
                f"{a.spec.surname} and the year {focal['earlier_document_year']} "
                f"appears on PDF page(s) "
                + ", ".join(str(p) for p in biblio) + " of the later document.")
        else:
            add(f"> **The earlier thesis itself does not appear in the later "
                f"document's bibliography.** The name occurs, but no "
                f"reference-list entry pairs it with the year "
                f"{focal['earlier_document_year']}.")
        add(">")
        add("> Which work of that author is cited, by the year given alongside "
            "the name:")
        add(">")
        add("> | Year cited | Mentions |")
        add("> |---|---:|")
        for yr, n in sorted(focal["author_a_mentions_by_cited_year"].items()):
            add(f"> | {yr} | {n} |")
        add(">")
        add("> The contested question in a case like this is *not* whether "
            "textual overlap exists — the numbers above settle that — but "
            "whether the later work acknowledges the earlier one. It does name "
            "the earlier author. Whether that acknowledgement is adequate to "
            "the specific passages that overlap is a judgement about academic "
            "practice, and this analysis does not make it.")
    else:
        add(f"> **No.** The later document does not name *{a.spec.surname}* "
            f"anywhere at all, including in its reference list.")
    add("")

    # ------------------------------------------------------ error carry-over
    add("## 7. Carried-over errors and idiosyncratic punctuation")
    add("")
    add(f"Verbatim runs in which the same anomaly — a doubled word, spaced "
        f"punctuation, an odd internal space — is present on both sides: "
        f"**{_fmt_int(focal['error_carryover_matches'])}**.")
    add("")
    errs = ctx["focal_error_sorted"][:20]
    if errs:
        add("| Words | A: PDF p. | B: PDF p. | Shared anomaly signature |")
        add("|---:|---|---|---|")
        for m in errs:
            add(f"| {m.run_length_words} | {m.a_pdf_page_start} | "
                f"{m.b_pdf_page_start} | `{m.error_carryover_detail}` |")
        add("")
    add("**This check is weaker than it looks, and one class of evidence is "
        "simply unavailable.** Word-geometry extraction re-joins each line's "
        "words with a single space, so runs of whitespace and most spacing "
        "idiosyncrasies in the original deposits are destroyed before anything "
        "can be compared; a shared whitespace quirk therefore *cannot* be "
        "detected by this pipeline, and its absence from the table above is "
        "not evidence that none exists. What does survive extraction — and is "
        "what the detector actually tests — is doubled words, hyphens left "
        "split mid-word, doubled or space-preceded punctuation tokens, missing "
        "spaces after punctuation, odd internal capitals, and a list of "
        "frequent academic misspellings. A shared *doubled word* or a shared "
        "*misspelling* is the strongest signal in that set, because extraction "
        "does not invent either.")
    add("")

    # --------------------------------------------------------------- methods
    add("## 8. Methods")
    add("")
    add("### 8.1 Sources and integrity")
    add("")
    add("| Document | Role | PDF pages | SHA-256 | Bytes |")
    add("|---|---|---:|---|---:|")
    for k in ctx["doc_order"]:
        d = info[k]
        add(f"| {docs[k].spec.label} | {docs[k].spec.role} | {d['pages']} | "
            f"`{d['sha256']}` | {_fmt_int(d['bytes'])} |")
    add("")
    add("Source URLs:")
    add("")
    for k in ctx["doc_order"]:
        d = info[k]
        add(f"- `{docs[k].spec.key}` — {d['source_url']}")
    add("")
    exp = [(docs[k].spec.short, docs[k].spec.expected_pages)
           for k in ctx["doc_order"] if docs[k].spec.expected_pages]
    if exp:
        add("Expected page counts were declared for "
            + ", ".join(f"{n} ({p} pp)" for n, p in exp)
            + ". The pipeline aborts before any analysis if a page count "
              "differs, because a different deposit would make every "
              "downstream number describe the wrong document. All matched.")
    add("")
    add("### 8.2 Extraction")
    add("")
    add("Text is extracted per page with `pdfplumber`, at word level with x/y "
        "geometry and font size retained rather than via flat `extract_text()`, "
        "because line `x0` is needed for block-quote indentation and font size "
        "for heading detection. Words are grouped into lines with a "
        f"{LINE_Y_TOL:g}pt vertical tolerance. `pypdf` is a fallback both "
        "per-page (mixed PDFs contain individual pages `pdfplumber` returns "
        "nothing for) and for the whole document. Ligatures, smart quotes, dash "
        "variants and exotic spaces are normalised before tokenising, or the "
        "same words would not match across two different PDF producers.")
    add("")
    add("Paragraphs are rebuilt across line breaks using vertical gap, "
        "first-line indent and short-final-line cues. A hyphenated line-end "
        "split is rejoined **only** when the continuation begins lowercase; "
        "otherwise the hyphen is kept, so genuine compounds are not destroyed. "
        "For matching, intra-word hyphens are deleted rather than split on, so "
        "a compound broken across a line and the same compound set solid reduce "
        "to the same token.")
    add("")
    add(f"The printed folio is detected as a short numeric or roman line alone "
        f"in the top or bottom {FOLIO_BAND:.0%} of the page; whichever band a "
        f"document uses more often is adopted for that document, and the folio "
        f"line is removed from the body text.")
    add("")
    add("| Document | Pages | No text layer | pypdf fallback pages | "
        "Printed folio found | Sentences | Modal body x0 (pt) | Modal body size (pt) |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in ctx["doc_order"]:
        d = docs[k]
        add(f"| {d.spec.short} | {len(d.pages)} | {len(d.no_text_pages)} | "
            f"{len(d.pypdf_pages)} | {d.folio_pages} | {_fmt_int(len(d.sentences))} | "
            f"{d.modal_x0:g} | {d.modal_size:g} |")
    add("")
    total_no_text = sum(len(docs[k].no_text_pages) for k in ctx["doc_order"])
    if total_no_text == 0:
        add("**No page in any of the four documents lacked a text layer.**")
    else:
        add(f"**{total_no_text} page(s) across the four documents had no text "
            f"layer at all** and contribute nothing to the analysis:")
        for k in ctx["doc_order"]:
            if docs[k].no_text_pages:
                add(f"- {docs[k].spec.short}: pages "
                    + ", ".join(str(p) for p in docs[k].no_text_pages))
    add("")
    add("### 8.3 Segmentation and exclusions")
    add("")
    add("Sentence splitting is abbreviation-aware (`et al.`, `i.e.`, `e.g.`, "
        "`pp.`, `Dr.`, single-letter initials, bare list numerals), or every "
        "citation-dense sentence would be shredded. Each sentence carries its "
        "source document, PDF page, printed folio where detectable, and the "
        "nearest chapter/section heading.")
    add("")
    add("Four categories are excluded from the matched set and counted "
        "separately so the exclusions are auditable:")
    add("")
    add("| Document | Total sentences | references | blockquote_attributed | "
        "boilerplate | caption | Excluded (any) | Retained |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in ctx["doc_order"]:
        d = docs[k]
        c: Counter = Counter()
        for s in d.sentences:
            for e in s.exclusions:
                c[e] += 1
        excluded = sum(1 for s in d.sentences if not s.included)
        add(f"| {d.spec.short} | {_fmt_int(len(d.sentences))} | "
            f"{_fmt_int(c['references'])} | {_fmt_int(c['blockquote_attributed'])} | "
            f"{_fmt_int(c['boilerplate'])} | {_fmt_int(c['caption'])} | "
            f"{_fmt_int(excluded)} | {_fmt_int(len(d.sentences) - excluded)} |")
    add("")
    add(f"Block quotes are identified by testing line indentation against the "
        f"document's own modal body `x0` (≈ +{QUOTE_INDENT_PT:g}pt marks a "
        f"quote), then requiring an attribution within {ATTRIB_WINDOW} "
        f"sentences. **An indented passage with no nearby attribution stays in "
        f"the matched set** — that asymmetry is deliberate: unattributed "
        f"indented text is exactly what the analysis is looking for.")
    add("")
    add("### 8.4 Matching")
    add("")
    add(f"**Pass 1 — verbatim.** Text is lowercased and stripped of "
        f"punctuation. Every shared run of {NGRAM_N}+ consecutive words is "
        f"found by indexing {NGRAM_N}-grams of A and probing with B, then "
        f"extending each seed maximally in both directions. After extension, "
        f"any run whose A-span *and* B-span both sit inside another kept run is "
        f"dropped; without this, repeated phrasing yields dozens of shifted "
        f"alignments of one underlying passage. A single {NGRAM_N}-gram may be "
        f"indexed at most {NGRAM_POSTING_CAP} times so boilerplate phrases "
        f"cannot blow up combinatorially.")
    add("")
    add("| Pair | n-grams hitting the posting cap | Postings dropped | "
        "Seeds extended | Runs before containment drop | After |")
    add("|---|---:|---:|---:|---:|---:|")
    for pid in ctx["pair_order"]:
        rs = analyses[pid]["run_stats"]
        add(f"| {pid} | {_fmt_int(rs['ngrams_hitting_cap'])} | "
            f"{_fmt_int(rs['postings_dropped_by_cap'])} | "
            f"{_fmt_int(rs['seeds_extended'])} | "
            f"{_fmt_int(rs['runs_before_containment_drop'])} | "
            f"{_fmt_int(rs['runs_after_containment_drop'])} |")
    add("")
    add(f"**Pass 2 — near-verbatim.** Sentence-level similarity with "
        f"`rapidfuzz` `token_set_ratio` ≥ {FUZZ_CUTOFF}, to catch light "
        f"paraphrase and word-order edits. Run through "
        f"`rapidfuzz.process.cdist` with `score_cutoff`, `dtype=\"uint8\"` and "
        f"`workers=-1`, chunked over rows of A; the focal pair alone is "
        f"{_fmt_int(focal['comparisons'])} sentence pairs, which nested Python "
        f"loops would never finish. Sentences shorter than {MIN_SENT_TOKENS} "
        f"words are skipped, because `token_set_ratio` is noisy on short "
        f"strings.")
    add("")
    es = focal["embedding_stats"]
    if es.get("status") == "run":
        add(f"**Pass 3 — sentence embeddings.** An offline "
            f"`{es['model']}` model was available, so a third pass was run at "
            f"cosine ≥ {EMBED_CUTOFF}. Embeddings are computed once per "
            f"document and cached; cosines are rounded to {EMBED_ROUND} decimal "
            f"places before thresholding so that floating-point reduction order "
            f"cannot make the artefacts differ between runs.")
    else:
        add(f"**Pass 3 — sentence embeddings: _not run_.** No sentence-embedding "
            f"model was available offline "
            f"(`{es.get('reason', 'unavailable')}`). The pipeline degrades "
            f"cleanly rather than failing; passes 1 and 2 are unaffected.")
    add("")
    add("### 8.5 Determinism")
    add("")
    add("No wall-clock value, no random number, and no reliance on set or dict "
        "iteration order enters any artefact. Every collection is sorted before "
        "it is written. `matches.csv`, `report.md` and `viewer.html` were "
        "produced twice over the same PDFs and diffed byte-for-byte; see "
        "`determinism.txt`.")
    add("")

    # ----------------------------------------------------------- limitations
    add("## 9. What this establishes — and what it does not")
    add("")
    add("**It establishes** that particular word sequences appear in both "
        "documents, how long those sequences are, where in the later document "
        "they fall, how that compares with five control pairings of "
        "same-discipline education doctorates run through the identical "
        "pipeline, and whether a citation naming the earlier author appears "
        "near each one.")
    add("")
    add("**It does not establish intent.** Textual overlap is a measurement, "
        "not a motive. There are many routes to shared wording: a common source "
        "both documents drew on, a shared supervisor's phrasing, standard "
        "disciplinary formulations, quotation whose marks or attribution the "
        "extraction failed to recover, or material legitimately reproduced with "
        "permission.")
    add("")
    add("**It is not a finding of misconduct, and this document must not be "
        "read as one.** Only the awarding institutions — and, where relevant, "
        "the relevant national body — can investigate and make such a finding, "
        "under their own procedures, with access to material this analysis does "
        "not have: drafts, supervision records, and the authors' own accounts. "
        "Nothing here substitutes for that process.")
    add("")

    add("## 10. Limitations")
    add("")
    lim = [
        "**Extraction is imperfect.** Headings, block quotes and captions are "
        "detected by geometry and regular expression, not by understanding. "
        "Every exclusion rule has a false-negative rate, and an undetected "
        "block quote appears in the matched set as if it were the later "
        "author's own prose. Spot checks against the source pages are reported "
        "in §11.",

        "**Quotation marks are not used as an exclusion signal.** Short "
        "in-line quotations that carry quotation marks and an adjacent citation "
        "are not excluded — only *indented* block quotations with a nearby "
        "attribution are. Some matched runs will therefore be legitimate quoted "
        "material.",

        f"**The chapter-density table depends on heading detection.** The "
        f"unclassified share is reported above ({_pct(unc_share)}) precisely so "
        f"that a failure of the regex mapping is visible.",

        "**The citation check is a string search for a surname.** It cannot "
        "tell an adequate attribution from a token one, it will miss an "
        "attribution phrased without the surname (\"the same author\", an "
        "op-cit, a numbered reference style), and it counts a name in the "
        "bibliography the same way it counts one in the running text.",

        f"**`token_set_ratio` ≥ {FUZZ_CUTOFF} is a loose criterion.** It "
        f"returns 100 whenever one sentence's token set is a subset of the "
        f"other's, so pass-2 counts include a substantial number of pairs that "
        f"are similar only in vocabulary. Pass 2 is a sensitivity floor, not a "
        f"finding; the verbatim runs of pass 1 carry the weight.",

        "**Two controls, five control pairs.** The baseline is a spread across "
        "five pairings of four documents, not a large sample. Two of those "
        "pairings share a document with the focal pair, which is why all six "
        "combinations are reported rather than a single averaged figure.",

        "**Page numbers are PDF page indices unless a printed folio was "
        "recovered.** Both are given wherever both are known.",

        "**Runs are counted after maximal extension and containment removal.** "
        "A different de-duplication rule would give a different total; the "
        "length distribution and the longest-run figure are more robust than "
        "the raw count.",
    ]
    for item in lim:
        add(f"- {item}")
    add("")

    # ------------------------------------------------------- verification log
    add("## 11. Verification")
    add("")
    for line in ctx.get("verification_notes", ["*(not recorded)*"]):
        add(f"- {line}")
    add("")

    add("## 12. Artefacts")
    add("")
    add("| File | Contents |")
    add("|---|---|")
    add("| `matches.csv` | One row per matched pair, all six pairings, all "
        "passes: run length, similarity, both PDF pages, both chapter labels, "
        "citation classification, direction flag. |")
    add("| `report.md` | This file. |")
    add("| `viewer.html` | Self-contained side-by-side viewer for the focal "
        "pair. Opens with no server. |")
    add("| `app.html` | Interactive side-by-side application over the same "
        "data: match navigator, keyboard navigation, density minimaps, synced "
        "scrolling, deep links and a findings dashboard. Also opens with no "
        "server. |")
    add("| `summary.json` | Every statistic in this report, machine-readable. |")
    add("| `run_analysis.py` | The whole pipeline. |")
    add("| `check_viewer.py` | Renders `viewer.html` from `file://` in headless "
        "Chromium and asserts zero console errors, zero external references, "
        "working filters and working click-to-scroll cross-linking. |")
    add("| `check_app.py` | The same for `app.html`, plus the match list, "
        "keyboard navigation, minimaps, tab switching, chart rendering, synced "
        "scrolling and deep links. |")
    add("| `verification_notes.md` | The hand-made checks reproduced in §11. |")
    add("| `determinism.txt` | SHA-256 of each artefact from two consecutive "
        "runs. |")
    add("| `requirements.txt` | Pinned dependencies. |")
    add("")
    add("Both theses are in copyright. This report cites page numbers and "
        "reports counts; `viewer.html` and `app.html` carry the text for "
        "inspection locally, and are not published anywhere.")
    add("")

    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Output: viewer.html — self-contained, opens from file://
# --------------------------------------------------------------------------- #

VIEWER_CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a18;--mut:#6b6b66;--line:#e2e2dd;--card:#fff;
--v:#ffd94a;--v2:#ffb01f;--n:#bfe3ff;--e:#d9d0ff;--acc:#0b5ed7;--warn:#b42318}
*{box-sizing:border-box}
html,body{margin:0;padding:0;height:100%}
body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
display:flex;flex-direction:column}
header{border-bottom:1px solid var(--line);background:var(--card);flex:0 0 auto}
.hd{padding:10px 14px;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
.hd h1{font-size:15px;margin:0;font-weight:650}
.hd .sub{color:var(--mut);font-size:12px}
.banner{background:#7a0d0d;color:#fff;padding:12px 14px;font-weight:700;
font-size:13px;line-height:1.5}
.ctl{padding:8px 14px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;
border-top:1px solid var(--line);font-size:12px}
.ctl label{display:inline-flex;gap:5px;align-items:center;white-space:nowrap}
.ctl select,.ctl input[type=search]{font:inherit;padding:3px 6px;
border:1px solid var(--line);border-radius:5px;background:var(--card);color:var(--fg)}
.ctl .grp{display:flex;gap:10px;align-items:center;padding-right:14px;
border-right:1px solid var(--line)}
.ctl .grp:last-child{border-right:0}
#stat{color:var(--mut);margin-left:auto;font-variant-numeric:tabular-nums}
main{flex:1 1 auto;display:flex;min-height:0}
.pane{flex:1 1 50%;overflow-y:auto;padding:0 18px 60vh 18px;min-width:0}
.pane+.pane{border-left:1px solid var(--line)}
.pane h2{position:sticky;top:0;background:var(--bg);margin:0;padding:9px 0;
font-size:12.5px;font-weight:650;border-bottom:1px solid var(--line);z-index:5}
.pg{margin:0 0 4px 0}
.pgn{color:var(--mut);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
margin:16px 0 4px;padding-top:8px;border-top:1px dashed var(--line)}
.hh{font-weight:700;margin:12px 0 4px;font-size:13.5px}
.s{display:inline}
.s.ex{color:#9a9a94}
mark{background:var(--v);padding:.5px 0;border-radius:2px;cursor:pointer;
scroll-margin-top:80px}
mark.p-near{background:var(--n)}
mark.p-emb{background:var(--e)}
mark.long{background:var(--v2)}
mark.off{background:transparent!important;cursor:text}
mark.sel{outline:2px solid var(--acc);outline-offset:1px}
.legend{font-size:11px;color:var(--mut);display:flex;gap:12px;align-items:center}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;
vertical-align:-1px;margin-right:3px}
.empty{color:var(--mut);padding:20px 0;font-style:italic}
@media (prefers-color-scheme:dark){
:root{--bg:#16161a;--fg:#e8e8e4;--mut:#9a9a94;--line:#2e2e34;--card:#1e1e23;
--v:#8a6d00;--v2:#b08800;--n:#1d4a6b;--e:#3b2f6b;--acc:#6fa8ff}
mark{color:#fff}
}
"""

VIEWER_JS = r"""
(function(){
var D = window.__DATA__;
var byId = {};
D.matches.forEach(function(m){ byId[m.i] = m; });

var elMin  = document.getElementById('minlen');
var elCite = document.getElementById('cite');
var elDir  = document.getElementById('dironly');
var elErr  = document.getElementById('erronly');
var elSrch = document.getElementById('srch');
var passes = Array.prototype.slice.call(document.querySelectorAll('.passchk'));
var stat   = document.getElementById('stat');
var marks  = Array.prototype.slice.call(document.querySelectorAll('mark'));

function activeSet(){
  var minlen = parseInt(elMin.value, 10);
  var cite   = elCite.value;
  var dirOnly= elDir.checked;
  var errOnly= elErr.checked;
  var q      = (elSrch.value || '').trim().toLowerCase();
  var wanted = {};
  passes.forEach(function(c){ if(c.checked) wanted[c.value] = 1; });
  var on = {};
  var n = 0;
  for (var k = 0; k < D.matches.length; k++){
    var m = D.matches[k];
    if (!wanted[m.p]) continue;
    if (m.w < minlen) continue;
    if (cite !== 'all' && m.c !== cite) continue;
    if (dirOnly && !m.d) continue;
    if (errOnly && !m.e) continue;
    if (q && (m.t || '').toLowerCase().indexOf(q) < 0) continue;
    on[m.i] = 1; n++;
  }
  return {on:on, n:n};
}

function apply(){
  var r = activeSet();
  for (var i = 0; i < marks.length; i++){
    var el = marks[i];
    var ids = el.getAttribute('data-m').split(',');
    var vis = false, longest = 0, pass = 'verbatim';
    for (var j = 0; j < ids.length; j++){
      if (r.on[ids[j]]){
        vis = true;
        var m = byId[ids[j]];
        if (m.w > longest){ longest = m.w; pass = m.p; }
      }
    }
    el.classList.toggle('off', !vis);
    el.classList.toggle('long', vis && longest >= 20 && pass === 'verbatim');
    el.classList.toggle('p-near', vis && pass === 'near_verbatim');
    el.classList.toggle('p-emb', vis && pass === 'embedding');
  }
  stat.textContent = r.n.toLocaleString() + ' of ' +
    D.matches.length.toLocaleString() + ' matches shown';
}

function clearSel(){
  for (var i = 0; i < marks.length; i++) marks[i].classList.remove('sel');
}

function jump(e){
  var el = e.currentTarget;
  if (el.classList.contains('off')) return;
  var r = activeSet();
  var ids = el.getAttribute('data-m').split(',').filter(function(x){ return r.on[x]; });
  if (!ids.length) return;
  // Prefer the longest visible match anchored here.
  ids.sort(function(x,y){ return byId[y].w - byId[x].w; });
  var id = ids[0];
  var side = el.getAttribute('data-side');
  var other = side === 'A' ? 'B' : 'A';
  var target = document.querySelector('mark[data-side="' + other +
                                      '"][data-m~="' + id + '"]') ||
               document.querySelector('mark[data-side="' + other + '"][data-m*="' + id + '"]');
  clearSel();
  el.classList.add('sel');
  if (target){
    target.classList.add('sel');
    target.scrollIntoView({block:'center', behavior:'smooth'});
  }
  var m = byId[id];
  stat.textContent = m.w + ' words | ' + m.p + ' | sim ' + m.s +
    ' | A p.' + m.ap + ' | B p.' + m.bp + ' | ' + m.c +
    (m.d ? ' | direction-flagged' : '') + (m.e ? ' | shared anomaly' : '');
}

marks.forEach(function(el){ el.addEventListener('click', jump); });
[elMin, elCite].forEach(function(el){ el.addEventListener('change', apply); });
[elDir, elErr].forEach(function(el){ el.addEventListener('change', apply); });
passes.forEach(function(c){ c.addEventListener('change', apply); });
elSrch.addEventListener('input', apply);
apply();
})();
"""


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _pane_html(doc: Document, side: str,
               seg_by_sent: dict[int, list[tuple[int, int, list[str]]]]) -> str:
    """Render one document as page blocks of sentences, with <mark> spans."""
    out: list[str] = []
    cur_page = None
    cur_head = None
    for s in doc.sentences:
        if s.pdf_page_start != cur_page:
            cur_page = s.pdf_page_start
            folio = s.printed_page
            label = f"PDF p. {cur_page}" + (f" · printed p. {folio}" if folio else "")
            out.append(f'<div class="pgn" id="{side}-p{cur_page}">{_esc(label)}</div>')
            cur_head = None
        head = s.heading or ""
        if head and head != cur_head:
            cur_head = head
            out.append(f'<div class="hh">{_esc(head)}</div>')
        cls = "s ex" if not s.included else "s"
        title = ""
        if not s.included:
            title = f' title="excluded: {_esc(", ".join(s.exclusions))}"'
        segs = seg_by_sent.get(s.idx)
        if not segs:
            out.append(f'<span class="{cls}" id="{side}-s{s.idx}"{title}>'
                       f'{_esc(s.text)}</span> ')
            continue
        words = s.text.split()
        parts: list[str] = []
        pos = 0
        for w0, w1, mids in segs:
            if w0 > pos:
                parts.append(_esc(" ".join(words[pos:w0])))
                parts.append(" ")
            parts.append(
                f'<mark data-side="{side}" data-m="{",".join(mids)}">'
                f'{_esc(" ".join(words[w0:w1 + 1]))}</mark>'
            )
            parts.append(" ")
            pos = w1 + 1
        if pos < len(words):
            parts.append(_esc(" ".join(words[pos:])))
        out.append(f'<span class="{cls}" id="{side}-s{s.idx}"{title}>'
                   + "".join(parts).strip() + "</span> ")
    return "\n".join(out)


def _segments(doc: Document, recs: list[MatchRecord], side: str
              ) -> dict[int, list[tuple[int, int, list[str]]]]:
    """Per-sentence highlight segments, merged where matches overlap."""
    # sentence -> list of (word_start, word_end_inclusive, match_id)
    raw: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    sent_disp_start: dict[int, int] = {}
    # Map each sentence's first display-word index so offsets can be local.
    for s in doc.sentences:
        r = doc.sent_tok_range.get(s.idx)
        if r:
            sent_disp_start[s.idx] = doc.tok_disp[r[0]]
    for m in recs:
        s0 = m.a_sent_start if side == "A" else m.b_sent_start
        s1 = m.a_sent_end if side == "A" else m.b_sent_end
        d0 = m.a_disp_start if side == "A" else m.b_disp_start
        d1 = m.a_disp_end if side == "A" else m.b_disp_end
        if s0 < 0 or d0 < 0:
            continue
        for sidx in range(s0, s1 + 1):
            base = sent_disp_start.get(sidx)
            if base is None:
                continue
            nwords = len(doc.sentences[sidx].text.split())
            lo = max(0, d0 - base)
            hi = min(nwords - 1, d1 - base)
            if hi >= lo:
                raw[sidx].append((lo, hi, m.match_id))
    out: dict[int, list[tuple[int, int, list[str]]]] = {}
    for sidx, items in raw.items():
        points = sorted({p for lo, hi, _ in items for p in (lo, hi + 1)})
        segs: list[tuple[int, int, list[str]]] = []
        for i in range(len(points) - 1):
            lo, hi = points[i], points[i + 1] - 1
            mids = sorted(mid for a, b, mid in items if a <= lo and hi <= b)
            if mids:
                if segs and segs[-1][1] + 1 == lo and segs[-1][2] == mids:
                    segs[-1] = (segs[-1][0], hi, mids)
                else:
                    segs.append((lo, hi, mids))
        if segs:
            out[sidx] = segs
    return out


def write_viewer(path: Path, a: Document, b: Document,
                 recs: list[MatchRecord], ctx: dict) -> None:
    synthetic = ctx["synthetic"]
    focal = ctx["analyses"][ctx["focal_pair_id"]]
    recs = sorted(recs, key=lambda m: (m.pass_name, -m.run_length_words, m.match_id))
    seg_a = _segments(a, recs, "A")
    seg_b = _segments(b, recs, "B")

    data = {
        "matches": [
            {
                "i": m.match_id,
                "p": m.pass_name,
                "w": m.run_length_words,
                "s": m.similarity,
                "ap": m.a_pdf_page_start,
                "bp": m.b_pdf_page_start,
                "c": m.citation_class,
                "d": m.direction_flag,
                "e": m.error_carryover,
                "t": m.b_text[:400],
            }
            for m in sorted(recs, key=lambda x: x.match_id)
        ]
    }
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).replace("</", "<\\/")

    banner = ""
    if synthetic:
        banner = (
            '<div class="banner">⚠️ SYNTHETIC SELF-TEST OUTPUT — NOT REAL '
            'FINDINGS. Every passage and every match shown below was generated '
            'by the self-test from fixture PDFs of invented prose containing a '
            'deliberately planted shared passage. No real thesis and no real '
            'person is shown here. Real results live in out/viewer.html.</div>'
        )

    title = "Textual-overlap viewer" + (" — SYNTHETIC SELF-TEST" if synthetic else "")
    sub = (f"{len(recs):,} matches · verbatim runs ≥{NGRAM_N} words · "
           f"near-verbatim token_set_ratio ≥{FUZZ_CUTOFF}"
           + (f" · embedding cosine ≥{EMBED_CUTOFF}"
              if focal["embedding_stats"].get("status") == "run" else ""))

    passes_present = sorted({m.pass_name for m in recs}) or ["verbatim"]
    pass_boxes = "".join(
        f'<label><input type="checkbox" class="passchk" value="{p}"'
        f'{" checked" if p == "verbatim" else ""}> {p}</label>'
        for p in passes_present
    )

    html_out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>{VIEWER_CSS}</style>
</head>
<body>
{banner}
<header>
  <div class="hd">
    <h1>{_esc(title)}</h1>
    <span class="sub">{_esc(sub)}</span>
    <span class="legend">
      <span><span class="sw" style="background:#ffd94a"></span>verbatim</span>
      <span><span class="sw" style="background:#ffb01f"></span>verbatim ≥20w</span>
      <span><span class="sw" style="background:#bfe3ff"></span>near-verbatim</span>
      <span><span class="sw" style="background:#d9d0ff"></span>embedding</span>
    </span>
  </div>
  <div class="ctl">
    <span class="grp">
      <label>Min run length
        <select id="minlen">
          <option value="8">8+</option>
          <option value="12">12+</option>
          <option value="16">16+</option>
          <option value="20">20+</option>
          <option value="30">30+</option>
          <option value="40">40+</option>
        </select>
      </label>
    </span>
    <span class="grp">
      <label>Citation class
        <select id="cite">
          <option value="all">all</option>
          <option value="attributed">attributed</option>
          <option value="adjacent-but-unattributed">adjacent-but-unattributed</option>
          <option value="unattributed">unattributed</option>
        </select>
      </label>
    </span>
    <span class="grp">{pass_boxes}</span>
    <span class="grp">
      <label><input type="checkbox" id="dironly"> direction-flagged only</label>
      <label><input type="checkbox" id="erronly"> shared anomaly only</label>
    </span>
    <span class="grp">
      <label><input type="search" id="srch" placeholder="search matched text"
             size="22"></label>
    </span>
    <span id="stat"></span>
  </div>
</header>
<main>
  <section class="pane" id="paneA">
    <h2>EARLIER — {_esc(a.spec.label)}</h2>
    {_pane_html(a, "A", seg_a)}
  </section>
  <section class="pane" id="paneB">
    <h2>LATER — {_esc(b.spec.label)}</h2>
    {_pane_html(b, "B", seg_b)}
  </section>
</main>
<script type="application/json" id="payload">{payload}</script>
<script>
window.__DATA__ = JSON.parse(document.getElementById('payload').textContent);
</script>
<script>{VIEWER_JS}</script>
</body>
</html>
"""
    path.write_text(html_out, encoding="utf-8")


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

def pair_id(a: DocSpec, b: DocSpec) -> str:
    return f"{a.key}~{b.key}"


def run_pipeline(docs: Sequence[DocSpec], *, pdf_dir: Path, cache_dir: Path,
                 out_dir: Path, synthetic: bool, skip_download: bool,
                 use_embeddings: bool, enforce_pages: bool,
                 extra_notes: Sequence[str] = ()) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ACTIVE_CACHE_DIR[0] = cache_dir

    log("[1/6] acquiring PDFs")
    info = ensure_pdfs(docs, skip_download, pdf_dir)
    for spec in docs:
        d = info[spec.key]
        log(f"  {spec.key}: {d['pages']} pages, sha256 {d['sha256'][:16]}…")
    if enforce_pages:
        verify_focal_page_counts(info, docs)
        log("  focal page counts verified against the expected deposit lengths")

    log("[2/6] extracting and segmenting")
    built: dict[str, Document] = {}
    for spec in docs:
        pages = load_or_extract(spec, Path(info[spec.key]["path"]),
                                info[spec.key]["sha256"], cache_dir=cache_dir)
        doc = build_document(spec, pages)
        built[spec.key] = doc
        excl = sum(1 for s in doc.sentences if not s.included)
        log(f"  {spec.key}: {len(doc.sentences)} sentences "
            f"({excl} excluded, {len(doc.sentences)-excl} retained), "
            f"{len(doc.tokens)} tokens, {len(doc.no_text_pages)} pages with no text")

    order = [s.key for s in docs]
    by_year = sorted(docs, key=lambda s: (s.year, s.key))
    pairs: list[tuple[DocSpec, DocSpec]] = []
    for i in range(len(by_year)):
        for j in range(i + 1, len(by_year)):
            pairs.append((by_year[i], by_year[j]))

    focal_a = next(s.key for s in docs if s.role == "focal-earlier")
    focal_b = next(s.key for s in docs if s.role == "focal-later")
    focal_id = f"{focal_a}~{focal_b}"

    log(f"[3/6] matching {len(pairs)} pairwise combinations")
    analyses: dict[str, dict] = {}
    all_recs: list[MatchRecord] = []
    focal_recs: list[MatchRecord] = []
    for sa, sb in pairs:
        a, b = built[sa.key], built[sb.key]
        pid = pair_id(sa, sb)
        kind = "focal" if pid == focal_id else "control"
        log(f"  {pid} ({kind})")
        runs, rstats = verbatim_runs(a, b)
        log(f"    pass 1: {len(runs)} verbatim runs "
            f"(longest {max((r.length for r in runs), default=0)} words)")
        near, nstats = near_verbatim_pairs(a, b)
        log(f"    pass 2: {len(near)} near-verbatim pairs over "
            f"{nstats['comparisons']:,} comparisons")
        if use_embeddings:
            emb, estats = embedding_pairs(a, b, info[sa.key]["sha256"],
                                          info[sb.key]["sha256"])
        else:
            emb, estats = None, {"status": "not run",
                                 "reason": "disabled with --no-embeddings"}
        log(f"    pass 3: {estats['status']}"
            + (f" — {len(emb)} pairs" if emb is not None else ""))
        recs = build_match_records(pid, kind, a, b, runs, near, emb)
        analyses[pid] = analyse_pair(pid, kind, a, b, runs, rstats,
                                     near, nstats, emb, estats, recs)
        all_recs.extend(recs)
        if pid == focal_id:
            focal_recs = recs

    log("[4/6] writing matches.csv")
    write_matches_csv(out_dir / "matches.csv", all_recs)

    fv = sorted([m for m in focal_recs if m.pass_name == "verbatim"],
                key=lambda m: (-m.run_length_words, m.b_sent_start, m.match_id))
    ctx = {
        "docs": built,
        "doc_order": order,
        "pdf_info": info,
        "analyses": analyses,
        "pair_order": [focal_id] + sorted(p for p in analyses if p != focal_id),
        "focal_pair_id": focal_id,
        "focal_a": focal_a,
        "focal_b": focal_b,
        "synthetic": synthetic,
        "focal_verbatim_sorted": fv,
        "focal_direction_sorted": [m for m in fv if m.direction_flag],
        "focal_error_sorted": [m for m in fv if m.error_carryover],
        "focal_records": focal_recs,
        "all_records": all_recs,
    }

    notes = list(extra_notes)
    total_no_text = sum(len(built[k].no_text_pages) for k in order)
    notes.append(
        f"Pages with no text layer, across all four documents: {total_no_text}."
        + ("" if total_no_text else " Extraction covered every page.")
    )
    f = analyses[focal_id]
    tot = sum(f["bucket_token_totals"].values()) or 1
    notes.append(
        f"Unclassified share of the later document's matched-set words: "
        f"{100.0 * f['bucket_token_totals'].get('unclassified', 0) / tot:.1f}%."
    )
    nb = HERE / "verification_notes.md"
    if nb.exists():
        for line in nb.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("- "):
                notes.append(line[2:].strip())
    ctx["verification_notes"] = notes

    log("[5/6] writing report.md")
    write_report(out_dir / "report.md", ctx)

    log("[6/6] writing viewer.html and app.html")
    write_viewer(out_dir / "viewer.html", built[focal_a], built[focal_b],
                 focal_recs, ctx)
    write_app(out_dir / "app.html", built[focal_a], built[focal_b],
              focal_recs, ctx)

    summary = {
        "synthetic": synthetic,
        "documents": {k: {kk: vv for kk, vv in info[k].items() if kk != "path"}
                      for k in order},
        "extraction": {
            k: {
                "pages": len(built[k].pages),
                "pages_no_text_layer": built[k].no_text_pages,
                "pages_via_pypdf_fallback": built[k].pypdf_pages,
                "printed_folios_found": built[k].folio_pages,
                "sentences": len(built[k].sentences),
                "sentences_retained": sum(1 for s in built[k].sentences if s.included),
                "exclusions": dict(sorted(Counter(
                    e for s in built[k].sentences for e in s.exclusions).items())),
                "tokens_retained": len(built[k].tokens),
                "modal_body_x0": built[k].modal_x0,
                "modal_body_size": built[k].modal_size,
                "bucket_tokens": dict(sorted(Counter(
                    {b: sum(s.n_tokens for s in built[k].sentences
                            if s.included and s.bucket == b)
                     for b in BUCKET_ORDER}).items())),
            } for k in order
        },
        "pairs": {p: analyses[p] for p in sorted(analyses)},
        "focal_pair": focal_id,
        "thresholds": {
            "ngram_n": NGRAM_N, "ngram_posting_cap": NGRAM_POSTING_CAP,
            "min_sentence_tokens": MIN_SENT_TOKENS, "fuzz_cutoff": FUZZ_CUTOFF,
            "embed_cutoff": EMBED_CUTOFF, "quote_indent_pt": QUOTE_INDENT_PT,
            "attrib_window": ATTRIB_WINDOW, "citation_window": CITATION_WINDOW,
            "long_run_words": LONG_RUN_WORDS,
        },
        "requirements": REQUIREMENTS,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return ctx


# --------------------------------------------------------------------------- #
# Self-test: synthetic fixtures with a deliberately planted shared passage
# --------------------------------------------------------------------------- #

FIXTURE_DOCS: tuple[DocSpec, ...] = (
    DocSpec("FIXA", "FIXA", "Marchetti, R. (2004), University of Northmoor [FIXTURE]",
            "Marchetti", r"\bMarchetti\b", 2004, "fixture_a_2004.pdf", "", "", None,
            "focal-earlier"),
    DocSpec("FIXB", "FIXB", "Okonkwo, D. (2011), University of Eastfield [FIXTURE]",
            "Okonkwo", r"\bOkonkwo\b", 2011, "fixture_b_2011.pdf", "", "", None,
            "focal-later"),
    DocSpec("FIXC", "FIXC", "Lindqvist, S. (2016), University of Westgate [FIXTURE]",
            "Lindqvist", r"\bLindqvist\b", 2016, "fixture_c_2016.pdf", "", "", None,
            "control"),
    DocSpec("FIXD", "FIXD", "Vasquez, M. (2019), University of Southbrook [FIXTURE]",
            "Vasquez", r"\bVasquez\b", 2019, "fixture_d_2019.pdf", "", "", None,
            "control"),
)

# Each fixture gets its OWN vocabulary and its OWN sentence templates. Sharing
# either would make the control pairs look exactly as similar as the focal pair
# and the self-test would be worthless.
FIXTURE_STYLE: dict[str, dict] = {
    "FIXA": {
        "n": ["curriculum", "mentor", "placement", "portfolio", "workshop",
              "cohort", "seminar", "moderator", "handbook", "rubric",
              "timetable", "induction"],
        "v": ["shaped", "constrained", "rewarded", "displaced", "anchored",
              "unsettled", "consolidated", "mediated"],
        "a": ["procedural", "tacit", "provisional", "sedimented", "granular",
              "situated", "recursive", "brittle"],
        "t": [
            "The {a1} {n1} {v1} how each {n2} was understood by those involved.",
            "Where the {n1} became {a1}, the {n2} {v1} what could reasonably follow.",
            "Participants described the {n1} as {a1}, and the {n2} {v1} that account.",
            "One consequence was that a {a1} {n1} {v1} the standing of the {n2}.",
            "Little in the {n1} {v1} the {a1} character of the {n2} itself.",
            "Read together, the {n1} and the {n2} {v1} a {a1} pattern of practice.",
        ],
    },
    "FIXB": {
        "n": ["provision", "gatekeeper", "trajectory", "threshold", "caseload",
              "referral", "panel", "dossier", "clinic", "roster",
              "allocation", "briefing"],
        "v": ["fractured", "amplified", "deferred", "obscured", "hardened",
              "loosened", "redirected", "flattened"],
        "a": ["contingent", "layered", "opaque", "durable", "uneven",
              "improvised", "porous", "entrenched"],
        "t": [
            "Across sites, a {a1} {n1} {v1} the assumptions carried into the {n2}.",
            "It was the {n2}, not the {n1}, that {v1} outcomes in {a1} ways.",
            "Respondents returned to the {n1} whenever the {a1} {n2} {v1}.",
            "That the {n1} {v1} so {a1} a {n2} went largely unremarked.",
            "Nothing about the {a1} {n1} {v1} the ordinary working of the {n2}.",
            "Between the {n1} and the {n2} lay a {a1} space that {v1} everything.",
        ],
    },
    "FIXC": {
        "n": ["lexicon", "transcript", "corpus", "utterance", "prompt",
              "annotation", "register", "glossary", "recording", "elicitation",
              "turn", "gloss"],
        "v": ["clustered", "diverged", "collapsed", "stabilised", "drifted",
              "recurred", "attenuated", "compounded"],
        "a": ["marked", "idiomatic", "sparse", "formulaic", "elliptical",
              "salient", "hedged", "polysemous"],
        "t": [
            "Counts drawn from the {n1} {v1} against a {a1} baseline in the {n2}.",
            "Every {a1} {n1} {v1} once the {n2} was segmented by speaker.",
            "The {n2} showed {a1} behaviour wherever the {n1} {v1}.",
            "Frequency in the {n1} {v1}; the {a1} {n2} did not.",
            "Coders disagreed about the {n1}, though the {a1} {n2} {v1} reliably.",
            "Neither the {n1} nor the {a1} {n2} {v1} under closer reading.",
        ],
    },
    "FIXD": {
        "n": ["scaffold", "artefact", "iteration", "prototype", "sandbox",
              "checkpoint", "dashboard", "backlog", "sprint", "walkthrough",
              "storyboard", "telemetry"],
        "v": ["accelerated", "stalled", "surfaced", "buried", "reframed",
              "duplicated", "streamlined", "fragmented"],
        "a": ["modular", "brittle", "legible", "opaque", "iterative",
              "lightweight", "redundant", "coupled"],
        "t": [
            "Each {a1} {n1} {v1} work that the {n2} had previously absorbed.",
            "Teams reported that the {n1} {v1} once the {n2} grew {a1}.",
            "A {a1} {n2} {v1} far more than any change to the {n1}.",
            "Whether the {n1} {v1} depended entirely on how {a1} the {n2} was.",
            "The {n1} looked {a1}; the {n2} {v1} regardless.",
            "Only after the {n2} {v1} did the {a1} {n1} become worth keeping.",
        ],
    },
}

# --- The planted passages ---------------------------------------------------
# PLANT_MAIN carries self-descriptive ("direction") language and a deliberate
# doubled word, so the direction check and the error-carry-over check are both
# exercised by a passage we know the ground truth for.
PLANT_MAIN = (
    "This study examined 120 student teachers drawn from four partner schools "
    "over a single academic year, and the findings of this study indicate that "
    "reflective capacity develops unevenly across the the cohort. Chapter four "
    "presents the questionnaire returns alongside the interview material, while "
    "chapter five draws the two strands together into a single interpretive "
    "account of professional learning."
)
PLANT_ATTRIB = (
    "sustained mentoring appears to matter more than the total number of "
    "observed lessons a trainee completes during a placement year"
)
PLANT_ADJACENT = (
    "the evidence assembled here points consistently towards a gradual and "
    "cumulative shift in how novices frame their own classroom decisions"
)
PLANT_BLOCKQUOTE = (
    "the language of competence has quietly displaced the older vocabulary of "
    "judgement, and with it a whole tradition of practical reasoning about "
    "teaching has become difficult to articulate"
)
PLANT_INDENT_NOATTRIB = (
    "what counts as evidence of progress is settled long before any individual "
    "trainee arrives, and is rarely revisited once the programme documentation "
    "has been signed off"
)
PLANT_IN_REFS = (
    "the assessment framework was revised three times during the period covered "
    "by this account and each revision narrowed the range of admissible evidence"
)


def _mix(*nums: int) -> int:
    x = 0x9E3779B9
    for n in nums:
        x = (x ^ ((n + 0x9E3779B9 + ((x << 6) & 0xFFFFFFFF) + (x >> 2)) & 0xFFFFFFFF))
        x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _gen_sentence(key: str, i: int) -> str:
    st = FIXTURE_STYLE[key]
    salt = sum(ord(c) for c in key)
    tpl = st["t"][_mix(salt, i, 1) % len(st["t"])]
    return tpl.format(
        n1=st["n"][_mix(salt, i, 2) % len(st["n"])],
        n2=st["n"][_mix(salt, i, 3) % len(st["n"])],
        v1=st["v"][_mix(salt, i, 4) % len(st["v"])],
        a1=st["a"][_mix(salt, i, 5) % len(st["a"])],
    )


def _gen_blocks(key: str, n_paras: int) -> list[tuple[str, str]]:
    """Build the block stream for one fixture: headings, paragraphs, quotes."""
    chapters = [
        "CHAPTER ONE: INTRODUCTION",
        "CHAPTER TWO: LITERATURE REVIEW",
        "CHAPTER THREE: METHODOLOGY",
        "CHAPTER FOUR: FINDINGS",
        "CHAPTER FIVE: DISCUSSION AND CONCLUSION",
    ]
    blocks: list[tuple[str, str]] = []
    per = max(1, n_paras // len(chapters))
    counter = 0
    for ci, ch in enumerate(chapters):
        blocks.append(("heading", ch))
        for _ in range(per):
            sents = [_gen_sentence(key, counter + k) for k in range(5)]
            counter += 5
            blocks.append(("para", " ".join(sents)))
    return blocks


def _inject_plants(key: str, blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Insert the planted passages. FIXA is the source; FIXB reuses them."""
    b = list(blocks)

    def at(frac: float) -> int:
        return max(2, min(len(b) - 1, int(len(b) * frac)))

    if key == "FIXA":
        b.insert(at(0.18), ("para", PLANT_MAIN))
        b.insert(at(0.34), ("para", "Considered carefully, " + PLANT_ATTRIB + "."))
        b.insert(at(0.50), ("para", "Taken as a whole, " + PLANT_ADJACENT + "."))
        b.insert(at(0.62), ("para", "It is worth stressing that " + PLANT_BLOCKQUOTE + "."))
        b.insert(at(0.74), ("para", "On this reading, " + PLANT_INDENT_NOATTRIB + "."))
        b.insert(at(0.86), ("para", "For completeness, " + PLANT_IN_REFS + "."))
    elif key == "FIXB":
        # 1. Long plant, no citation anywhere near it -> unattributed.
        b.insert(at(0.16), ("para", PLANT_MAIN))
        # 2. Citation inside the matched sentence -> attributed.
        b.insert(at(0.30), ("para", "As Marchetti (2004) observed, " + PLANT_ATTRIB + "."))
        # 3. Citation two sentences later -> adjacent-but-unattributed.
        b.insert(at(0.44), ("para",
                            "Taken as a whole, " + PLANT_ADJACENT + ". "
                            "That reading is not universally shared. "
                            "A comparable case is set out by Marchetti (2004)."))
        # 4. Indented WITH an attribution right after -> must be EXCLUDED.
        #    Both blocks go in at the same index so nothing can drift between
        #    them and push the attribution outside the ATTRIB_WINDOW.
        _i = at(0.56)
        b.insert(_i, ("para", "(Marchetti, 2004, p. 88)"))
        b.insert(_i, ("quote", PLANT_BLOCKQUOTE + "."))
        # 5. Indented with NO attribution nearby -> must be RETAINED.
        b.insert(at(0.70), ("quote", PLANT_INDENT_NOATTRIB + "."))
        # 6. Inside the reference list -> must be EXCLUDED.
        b.append(("heading", "REFERENCES"))
        b.append(("para", "Marchetti, R. (2004) Learning to notice. "
                          "Northmoor: Northmoor University Press."))
        b.append(("para", "Ellery, T. (1998) Placements and progression. "
                          "Eastfield: Eastfield Academic."))
        b.append(("para", PLANT_IN_REFS + "."))
        b.append(("para", "Nandi, K. (2007) The mentoring relation. "
                          "Westgate: Westgate Books."))
        b.append(("para", "Orwin, L. (2010) Evidence and judgement. "
                          "Southbrook: Southbrook Press."))
    else:
        b.append(("heading", "REFERENCES"))
        for i, nm in enumerate(("Baytas, F.", "Coleridge, H.", "Duarte, P.",
                                "Ferreira, L.", "Gaskell, R.")):
            b.append(("para", f"{nm} ({1995 + i * 4}) A study of practice. "
                              f"Placeholder: Placeholder Press."))
    return b


def write_fixture_pdf(path: Path, key: str, title: str, n_paras: int) -> None:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    W, H = A4
    LEFT, RIGHT, TOP, BOT = 72.0, 72.0, 72.0, 64.0
    BODY_W = W - LEFT - RIGHT
    QUOTE_INDENT = 24.0
    LEAD = 14.0

    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(title)
    c.setAuthor("run_analysis.py self-test fixture")
    y = H - TOP
    page = 1

    def wrap(text: str, font: str, size: float, width: float) -> list[str]:
        words = text.split()
        lines, cur = [], []
        for w in words:
            trial = (" ".join(cur + [w]))
            if c.stringWidth(trial, font, size) <= width or not cur:
                cur.append(w)
            else:
                lines.append(" ".join(cur))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))
        return lines

    def folio_and_break():
        nonlocal y, page
        c.setFont("Helvetica", 9)
        c.drawCentredString(W / 2.0, BOT - 22.0, str(page))
        c.showPage()
        page += 1
        y = H - TOP

    for kind, text in _inject_plants(key, _gen_blocks(key, n_paras)):
        if kind == "heading":
            font, size, x, width = "Helvetica-Bold", 14.0, LEFT, BODY_W
            gap = 10.0
        elif kind == "quote":
            font, size, x, width = "Helvetica", 10.0, LEFT + QUOTE_INDENT, \
                BODY_W - 2 * QUOTE_INDENT
            gap = 8.0
        else:
            font, size, x, width = "Helvetica", 10.0, LEFT, BODY_W
            gap = 8.0
        lines = wrap(text, font, size, width)
        if y - (len(lines) * LEAD + gap) < BOT:
            folio_and_break()
        y -= gap
        c.setFont(font, size)
        for ln in lines:
            if y < BOT:
                folio_and_break()
                c.setFont(font, size)
            c.drawString(x, y, ln)
            y -= LEAD
    c.setFont("Helvetica", 9)
    c.drawCentredString(W / 2.0, BOT - 22.0, str(page))
    c.showPage()
    c.save()


# --- unit assertions --------------------------------------------------------

def _unit_tests() -> list[str]:
    notes: list[str] = []

    def ok(cond: bool, what: str):
        if not cond:
            raise AssertionError(f"UNIT TEST FAILED: {what}")
        notes.append(f"unit ok: {what}")

    # normalisation
    ok(normalise_text("eﬃcient ﬁnal ﬂow") == "efficient final flow",
       "ligatures ﬃ/ﬁ/ﬂ are normalised")
    ok(normalise_text("“quoted” and ‘single’")
       == '"quoted" and \'single\'', "smart quotes are normalised")
    ok(normalise_text("range 1990–2000 and dash — here")
       == "range 1990–2000 and dash – here", "dash variants collapse to one shape")
    ok(match_tokens("well-known state-of-the-art") == ["wellknown", "stateoftheart"],
       "intra-word hyphens are deleted, not split on")
    ok(match_tokens("A — B") == ["a", "b"],
       "separating dashes become token boundaries")

    # sentence splitting
    def sents(t):
        return [t[s:e] for s, e in split_sentences(t)]
    ok(len(sents("Smith et al. found that it worked. The next claim follows.")) == 2,
       "'et al.' does not split a sentence")
    ok(len(sents("This holds, i.e. always. Then something else happens.")) == 2,
       "'i.e.' does not split a sentence")
    ok(len(sents("See pp. 23-45. The argument resumes there.")) == 2,
       "'pp.' does not split a sentence")
    ok(len(sents("Reported by Fairweather-Blake, P. A. Smith agreed with that.")) == 1,
       "single-letter initials do not split a sentence")
    ok(len(sents("1. Introduction to the area. The topic is broad.")) == 2,
       "a bare list numeral does not split a sentence")
    ok(len(sents("It was published in 2009. The sequel came later.")) == 2,
       "a four-digit year still ends a sentence")
    ok(len(sents("Dr. Jones disagreed with that view entirely.")) == 1,
       "'Dr.' does not split a sentence")

    # direction / attribution / caption / boilerplate regexes
    ok(direction_categories("This study examined 120 student teachers who were "
                            "recruited from four schools.")
       == ["own_participants", "own_study"], "direction check fires on own study")
    ok(direction_categories("The weather in April is variable.") == [],
       "direction check stays silent on ordinary prose")
    ok(bool(ATTRIBUTION_RE.search("(Marchetti, 2004, p. 88)")),
       "attribution regex matches a parenthetical citation")
    ok(bool(ATTRIBUTION_RE.search("as Marchetti (2004) argued")),
       "attribution regex matches a narrative citation")
    ok(not ATTRIBUTION_RE.search("the cohort numbered 2004 trainees"),
       "attribution regex does not fire on a bare number")
    ok(bool(CAPTION_RE.match("Table 4.2 Distribution of responses")),
       "caption regex matches a table caption")
    ok(bool(BOILERPLATE_SENT_RE.search(
        "I hereby declare that this thesis is my own work.")),
       "boilerplate regex matches a declaration")

    # error signatures
    ok("doubled_word:the" in error_signatures("across the the cohort"),
       "doubled-word anomaly is detected")
    ok("misspelling:questionaire" in error_signatures("the questionaire returns"),
       "a shared misspelling is detected")
    ok("split_hyphen:self-esteem" in error_signatures("low self- esteem levels"),
       "a hyphen left split mid-word is detected")
    ok("internal_caps:teacherstraining" in
       error_signatures("the teachersTraining record"),
       "an odd internal capital is detected")
    ok(error_signatures("a normal sentence") == set(),
       "no anomaly is reported for clean text")

    # chapter buckets
    ok(classify_bucket("CHAPTER TWO: LITERATURE REVIEW") == "literature_review",
       "heading maps to the literature_review bucket")
    ok(classify_bucket("CHAPTER THREE – METHODOLOGY") == "methodology",
       "heading maps to the methodology bucket")
    ok(classify_bucket("Some unusual chapter name") == "unclassified",
       "an unmatched heading falls through to 'unclassified'")
    ok(classify_bucket("Chapter 6: Discussion and implications of research "
                       "findings") == "discussion",
       "the bucket mentioned earliest in a chapter title wins")
    ok(classify_bucket("Appendix 3: Findings tables") == "appendix",
       "a structural bucket (appendix) wins outright")
    ok(not _is_heading(Line("Chapter 4 explicitly details how each criterion was "
                            "addressed as well as the cautionary", 72.0, 100.0,
                            110.0, 12.0, False, 1), 12.0, 72.0),
       "a body-text cross-reference beginning 'Chapter 4' is not a heading")
    ok(not _is_heading(Line("section 4.6.", 72.0, 100.0, 110.0, 12.0, False, 1),
                       12.0, 72.0),
       "a lowercase cross-reference 'section 4.6.' is not a heading")
    ok(_is_heading(Line("Chapter 5: Research findings", 72.0, 100.0, 110.0,
                        12.0, True, 1), 12.0, 72.0),
       "a genuine short bold chapter heading is a heading")
    ok(not _is_heading(Line("Chapter 6: Discussion .................. 223", 72.0,
                            100.0, 110.0, 12.0, True, 1), 12.0, 72.0),
       "a contents-page line with dot leaders is not a heading")

    # pass 3 degrades cleanly when no model is present
    saved = dict(_EMBED_STATE)
    try:
        _EMBED_STATE["tried"] = True
        _EMBED_STATE["model"] = None
        _EMBED_STATE["reason"] = "simulated: no model"
        dummy = Document(FIXTURE_DOCS[0], [], [], 0.0, 0.0, [], [], "", 0)
        res, st = embedding_pairs(dummy, dummy, "x", "y")
        ok(res is None and st["status"] == "not run",
           "pass 3 degrades cleanly to 'not run' with no offline model")
    finally:
        _EMBED_STATE.clear()
        _EMBED_STATE.update(saved)
    return notes


def _probe(text: str, n: int = 7) -> str:
    return " ".join(match_tokens(text)[:n])


def selftest(use_embeddings: bool) -> int:
    log("=" * 72)
    log("SELF-TEST — synthetic fixtures, planted passage, asserted recovery")
    log("=" * 72)

    notes = _unit_tests()
    log(f"  {len(notes)} unit assertions passed")

    fixture_dir = SELFTEST_FIXTURE_DIR
    if fixture_dir.resolve() == PDF_DIR.resolve():
        raise SystemExit("refusing to write fixtures into the real pdfs/ directory")
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    cache_dir = SELFTEST_OUT_DIR / "cache"
    if SELFTEST_OUT_DIR.exists():
        shutil.rmtree(SELFTEST_OUT_DIR)
    SELFTEST_OUT_DIR.mkdir(parents=True)

    failures: list[str] = []
    try:
        for spec in FIXTURE_DOCS:
            write_fixture_pdf(fixture_dir / spec.filename, spec.key,
                              spec.label, n_paras=90)
        log(f"  wrote 4 fixture PDFs to {fixture_dir}")

        ctx = run_pipeline(FIXTURE_DOCS, pdf_dir=fixture_dir, cache_dir=cache_dir,
                           out_dir=SELFTEST_OUT_DIR, synthetic=True,
                           skip_download=True, use_embeddings=use_embeddings,
                           enforce_pages=False,
                           extra_notes=["SYNTHETIC self-test run — not real data."])

        an = ctx["analyses"]
        focal_id = ctx["focal_pair_id"]
        verb = [m for m in ctx["focal_records"] if m.pass_name == "verbatim"]
        all_verb = [m for m in ctx["all_records"] if m.pass_name == "verbatim"]

        def check(cond: bool, what: str):
            if cond:
                log(f"  PASS  {what}")
            else:
                log(f"  FAIL  {what}")
                failures.append(what)

        # -- extraction sanity
        no_text = sum(len(d.no_text_pages) for d in ctx["docs"].values())
        check(no_text == 0, "every fixture page yielded a text layer")
        check(all(d.folio_pages >= 3 for d in ctx["docs"].values()),
              "printed folios were detected in all four fixtures")

        # -- the plant is recovered at full length
        probe_main = _probe(PLANT_MAIN, 8)
        main_hits = [m for m in verb if probe_main in " ".join(match_tokens(m.b_text))]
        longest = max((m.run_length_words for m in main_hits), default=0)
        check(bool(main_hits), "the planted passage is recovered in the focal pair")
        check(longest >= 55,
              f"the planted passage is recovered at full length "
              f"(got {longest} words, expected >= 55)")

        # -- assert on LONG runs, not on totals
        focal_long = an[focal_id]["verbatim_long_runs"]
        ctrl_long = {p: an[p]["verbatim_long_runs"] for p in an if p != focal_id}
        ctrl_max = {p: an[p]["verbatim_max_run"] for p in an if p != focal_id}
        check(focal_long >= 4,
              f"the focal fixture pair yields >= 4 runs of {LONG_RUN_WORDS}+ words "
              f"(got {focal_long})")
        check(all(v == 0 for v in ctrl_long.values()),
              f"no control fixture pair yields any run of {LONG_RUN_WORDS}+ words "
              f"(got {ctrl_long})")
        check(max(ctrl_max.values(), default=0) < LONG_RUN_WORDS,
              f"longest control-pair run stays under {LONG_RUN_WORDS} words "
              f"(got {max(ctrl_max.values(), default=0)})")

        # -- exclusion behaviour, including the deliberate asymmetry
        def present(plant: str) -> bool:
            pr = _probe(plant, 7)
            return any(pr in " ".join(match_tokens(m.b_text)) for m in all_verb)

        check(not present(PLANT_BLOCKQUOTE),
              "an indented passage WITH an attribution nearby is excluded")
        check(present(PLANT_INDENT_NOATTRIB),
              "an indented passage WITHOUT an attribution nearby is RETAINED "
              "(the deliberate asymmetry)")
        check(not present(PLANT_IN_REFS),
              "a passage inside the reference list is excluded")

        # -- citation classification
        def cls_for(plant: str) -> set[str]:
            pr = _probe(plant, 7)
            return {m.citation_class for m in verb
                    if pr in " ".join(match_tokens(m.b_text))}

        check(cls_for(PLANT_ATTRIB) == {"attributed"},
              f"a match citing the earlier author in-sentence is 'attributed' "
              f"(got {cls_for(PLANT_ATTRIB)})")
        check(cls_for(PLANT_ADJACENT) == {"adjacent-but-unattributed"},
              f"a match with the citation two sentences later is "
              f"'adjacent-but-unattributed' (got {cls_for(PLANT_ADJACENT)})")
        check(cls_for(PLANT_MAIN) == {"unattributed"},
              f"a match with no citation nearby is 'unattributed' "
              f"(got {cls_for(PLANT_MAIN)})")

        # -- direction flag and error carry-over
        check(all(m.direction_flag == 1 for m in main_hits),
              "the self-descriptive plant is direction-flagged")
        check(any("own_study" in m.direction_categories for m in main_hits)
              and any("own_chapter_structure" in m.direction_categories
                      for m in main_hits),
              "direction categories include own_study and own_chapter_structure")
        check(any(m.error_carryover == 1 and "doubled_word:the" in
                  m.error_carryover_detail for m in main_hits),
              "the planted doubled word is reported as a carried-over anomaly")

        # -- author mention count
        check(an[focal_id]["author_a_mentions_in_b_total"] >= 3,
              "the earlier author's surname is counted everywhere it appears in "
              "the later document, reference list included")

        # -- artefacts exist and carry the banner
        rep = (SELFTEST_OUT_DIR / "report.md").read_text(encoding="utf-8")
        vw = (SELFTEST_OUT_DIR / "viewer.html").read_text(encoding="utf-8")
        check(rep.lstrip().startswith("> # ⚠️  SYNTHETIC SELF-TEST OUTPUT"),
              "the synthetic report.md opens with an unmissable banner")
        check('class="banner"' in vw and "SYNTHETIC SELF-TEST OUTPUT" in vw,
              "the synthetic viewer.html opens with an unmissable banner")
        check("http://" not in vw.replace("http://www.w3.org", "") and
              "https://" not in vw and "src=" not in vw,
              "the synthetic viewer.html references nothing external")
        check((SELFTEST_OUT_DIR / "matches.csv").exists(),
              "matches.csv was written")

        # -- determinism of the artefacts, over FIXED input PDFs
        # (reportlab embeds a creation timestamp, so regenerating the fixtures
        #  would change their hashes; the PDFs are NOT regenerated here.)
        first = {n: sha256_file(SELFTEST_OUT_DIR / n)
                 for n in ("matches.csv", "report.md", "viewer.html",
                           "app.html")}
        again = SELFTEST_OUT_DIR / "_rerun"
        run_pipeline(FIXTURE_DOCS, pdf_dir=fixture_dir, cache_dir=cache_dir,
                     out_dir=again, synthetic=True, skip_download=True,
                     use_embeddings=use_embeddings, enforce_pages=False,
                     extra_notes=["SYNTHETIC self-test run — not real data."])
        second = {n: sha256_file(again / n)
                  for n in ("matches.csv", "report.md", "viewer.html",
                            "app.html")}
        check(first == second,
              "two consecutive runs over the same fixture PDFs are byte-identical")
    finally:
        # Fixtures are deleted so they can never contaminate the real run.
        if fixture_dir.exists():
            shutil.rmtree(fixture_dir)
        log(f"  removed fixture PDFs from {fixture_dir}")

    log("-" * 72)
    if failures:
        log(f"SELF-TEST FAILED — {len(failures)} check(s) did not pass:")
        for f in failures:
            log(f"  - {f}")
        return 1
    log("SELF-TEST PASSED — all checks green. Synthetic artefacts are in "
        f"{SELFTEST_OUT_DIR}/ and are banner-marked as not real.")
    return 0


# --------------------------------------------------------------------------- #
# Determinism check over the real run
# --------------------------------------------------------------------------- #

def determinism_check(docs: Sequence[DocSpec], use_embeddings: bool) -> int:
    names = ("matches.csv", "report.md", "viewer.html", "app.html")
    first = {n: sha256_file(OUT_DIR / n) for n in names}
    scratch = OUT_DIR / "_recheck"
    run_pipeline(docs, pdf_dir=PDF_DIR, cache_dir=CACHE_DIR, out_dir=scratch,
                 synthetic=False, skip_download=True,
                 use_embeddings=use_embeddings, enforce_pages=True)
    second = {n: sha256_file(scratch / n) for n in names}
    lines = ["Determinism check — two consecutive runs over the same PDFs.", ""]
    same = True
    for n in names:
        eq = first[n] == second[n]
        same = same and eq
        lines.append(f"{'IDENTICAL' if eq else 'DIFFERENT':>10}  {n}")
        lines.append(f"            run 1 sha256 {first[n]}")
        lines.append(f"            run 2 sha256 {second[n]}")
    lines.append("")
    lines.append("RESULT: " + ("all artefacts byte-identical."
                               if same else "ARTEFACTS DIFFER."))
    (OUT_DIR / "determinism.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    shutil.rmtree(scratch)
    log("\n".join(lines))
    return 0 if same else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reproducible textual-overlap analysis between UK doctoral "
                    "theses, benchmarked against control pairs.")
    ap.add_argument("--selftest", action="store_true",
                    help="generate synthetic fixture PDFs, assert the pipeline "
                         "recovers a planted passage, then delete them")
    ap.add_argument("--skip-download", action="store_true",
                    help="use the PDFs already in ./pdfs")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip pass 3 even if an offline model is available")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the extraction cache")
    ap.add_argument("--determinism-check", action="store_true",
                    help="after the run, produce the artefacts a second time "
                         "and diff them byte-for-byte into out/determinism.txt")
    args = ap.parse_args(argv)

    (HERE / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n",
                                           encoding="utf-8")

    if args.selftest:
        return selftest(use_embeddings=not args.no_embeddings)

    cache = CACHE_DIR
    if args.no_cache:
        cache = CACHE_DIR / "_disabled"
        if cache.exists():
            shutil.rmtree(cache)

    run_pipeline(load_documents(), pdf_dir=PDF_DIR, cache_dir=cache, out_dir=OUT_DIR,
                 synthetic=False, skip_download=args.skip_download,
                 use_embeddings=not args.no_embeddings, enforce_pages=True)
    log(f"\nArtefacts written to {OUT_DIR}/")
    for n in ("matches.csv", "report.md", "viewer.html", "app.html",
              "summary.json"):
        p = OUT_DIR / n
        if p.exists():
            log(f"  {n:<16} {p.stat().st_size:>12,} bytes")

    if args.determinism_check:
        return determinism_check(load_documents(),
                                 use_embeddings=not args.no_embeddings)
    return 0


# --------------------------------------------------------------------------- #
# Output: app.html — richer interactive side-by-side application
# --------------------------------------------------------------------------- #

APP_CSS = """
:root{
 --bg:#f7f7f5; --panel:#ffffff; --fg:#17171a; --mut:#6c6c72; --line:#e4e4df;
 --acc:#0b5ed7; --acc-fg:#fff; --chip:#eef1f6;
 --v1:#ffe27a; --v2:#ffc53d; --v3:#ff9d2e; --near:#a8d8ff; --emb:#d6c9ff;
 --ok:#137a4b; --warn:#b42318; --shadow:0 1px 2px rgba(0,0,0,.06);
}
:root[data-theme="dark"], :root.dark{
 --bg:#131317; --panel:#1b1b21; --fg:#e9e9e6; --mut:#9b9ba3; --line:#2c2c34;
 --acc:#6fa8ff; --acc-fg:#0b1220; --chip:#23232b;
 --v1:#6f5a12; --v2:#8f6d0d; --v3:#a35a10; --near:#1e4a6d; --emb:#3a2f6d;
 --ok:#4ec78d; --warn:#ff8b80; --shadow:0 1px 2px rgba(0,0,0,.4);
}
@media (prefers-color-scheme:dark){
 :root:not([data-theme="light"]){
  --bg:#131317; --panel:#1b1b21; --fg:#e9e9e6; --mut:#9b9ba3; --line:#2c2c34;
  --acc:#6fa8ff; --acc-fg:#0b1220; --chip:#23232b;
  --v1:#6f5a12; --v2:#8f6d0d; --v3:#a35a10; --near:#1e4a6d; --emb:#3a2f6d;
  --ok:#4ec78d; --warn:#ff8b80; --shadow:0 1px 2px rgba(0,0,0,.4);
 }
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);display:flex;flex-direction:column;
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
button,select,input{font:inherit;color:inherit}
.banner{background:#7a0d0d;color:#fff;padding:12px 16px;font-weight:700;font-size:13px}

/* ---------- header ---------- */
header{background:var(--panel);border-bottom:1px solid var(--line);flex:0 0 auto}
.top{display:flex;align-items:center;gap:14px;padding:9px 14px;flex-wrap:wrap}
.brand{font-weight:680;font-size:14.5px;white-space:nowrap}
.brand small{display:block;font-weight:400;font-size:11px;color:var(--mut)}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{background:var(--chip);border:1px solid var(--line);border-radius:999px;
 padding:3px 10px;font-size:11.5px;white-space:nowrap}
.chip b{font-variant-numeric:tabular-nums}
.tabs{display:flex;gap:4px;margin-left:auto}
.tab{background:transparent;border:1px solid transparent;border-radius:7px;
 padding:5px 12px;cursor:pointer;font-size:12.5px;color:var(--mut)}
.tab[aria-selected="true"]{background:var(--acc);color:var(--acc-fg);font-weight:600}
.iconbtn{background:var(--chip);border:1px solid var(--line);border-radius:7px;
 padding:5px 9px;cursor:pointer;font-size:12.5px}
.iconbtn:hover{border-color:var(--acc)}

/* ---------- layout ---------- */
main{flex:1 1 auto;display:flex;min-height:0}
aside{width:310px;flex:0 0 310px;border-right:1px solid var(--line);
 background:var(--panel);display:flex;flex-direction:column;min-height:0}
aside.hidden{display:none}
.filters{padding:11px 12px;border-bottom:1px solid var(--line);
 display:flex;flex-direction:column;gap:9px}
.frow{display:flex;flex-direction:column;gap:4px}
.frow label{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.frow select,.frow input[type=search]{width:100%;padding:5px 7px;border:1px solid var(--line);
 border-radius:6px;background:var(--bg)}
.rangewrap{display:flex;align-items:center;gap:8px}
.rangewrap input[type=range]{flex:1;accent-color:var(--acc)}
.rangewrap b{font-variant-numeric:tabular-nums;min-width:34px;text-align:right;font-size:12px}
.checks{display:flex;flex-wrap:wrap;gap:4px 12px}
.checks label{display:inline-flex;gap:5px;align-items:center;font-size:12px;
 text-transform:none;letter-spacing:0;color:var(--fg)}
.listhead{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;
 align-items:center;gap:8px;font-size:11.5px;color:var(--mut)}
.listhead select{margin-left:auto;padding:3px 5px;border:1px solid var(--line);
 border-radius:6px;background:var(--bg);font-size:11.5px}
#list{overflow-y:auto;flex:1 1 auto;padding:4px 0}
.item{padding:7px 12px;border-bottom:1px solid var(--line);cursor:pointer;font-size:12px}
.item:hover{background:var(--chip)}
.item.on{background:var(--acc);color:var(--acc-fg)}
.item.on .mut,.item.on .badge{color:inherit;opacity:.85}
.item .l1{display:flex;gap:7px;align-items:baseline}
.item .w{font-weight:700;font-variant-numeric:tabular-nums}
.item .mut{color:var(--mut);font-size:11px}
.item .l2{color:var(--mut);font-size:11px;margin-top:2px;
 overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;
 border:1px solid currentColor;border-radius:4px;padding:0 4px;opacity:.8}
.b-dir{color:var(--warn)}
.b-att{color:var(--ok)}

/* ---------- panes ---------- */
.panes{flex:1 1 auto;display:flex;min-width:0}
.panewrap{flex:1 1 50%;display:flex;min-width:0;border-left:1px solid var(--line)}
.pane{flex:1 1 auto;overflow-y:auto;padding:0 16px 60vh 16px;min-width:0;
 scroll-behavior:smooth}
.pane h2{position:sticky;top:0;background:var(--bg);margin:0;padding:9px 0 7px;
 font-size:12px;font-weight:650;border-bottom:1px solid var(--line);z-index:5}
.pane h2 span{color:var(--mut);font-weight:400}
.pgn{color:var(--mut);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
 margin:15px 0 4px;padding-top:7px;border-top:1px dashed var(--line)}
.hh{font-weight:700;margin:11px 0 4px;font-size:13px}
.s{display:inline}
.s.ex{color:var(--mut);opacity:.65}
mark{background:var(--v1);padding:.5px 0;border-radius:2px;cursor:pointer;
 scroll-margin-top:90px;color:inherit}
mark.m{background:var(--v2)}
mark.l{background:var(--v3)}
mark.p-near{background:var(--near)}
mark.p-emb{background:var(--emb)}
mark.off{background:transparent!important;cursor:text}
mark.sel{outline:2.5px solid var(--acc);outline-offset:1.5px;border-radius:3px}

/* ---------- minimap ---------- */
.mini{width:15px;flex:0 0 15px;background:var(--panel);border-left:1px solid var(--line);
 position:relative;cursor:pointer}
.mini i{position:absolute;left:2px;right:2px;height:2.5px;border-radius:2px;
 background:var(--v2);display:block}
.mini i.big{background:var(--v3);height:4px}
.mini i.sel{background:var(--acc);height:5px;left:0;right:0}
.mini .vp{position:absolute;left:0;right:0;background:rgba(11,94,215,.14);
 border-top:1px solid var(--acc);border-bottom:1px solid var(--acc);pointer-events:none}

/* ---------- findings ---------- */
#findings{flex:1 1 auto;overflow-y:auto;padding:20px 26px 60px;display:none}
#findings.show{display:block}
.panes.hide{display:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;
 max-width:1500px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:14px 16px;box-shadow:var(--shadow)}
.card h3{margin:0 0 3px;font-size:13px}
.card p.sub{margin:0 0 12px;color:var(--mut);font-size:11.5px}
.card.wide{grid-column:1/-1}
table.dt{width:100%;border-collapse:collapse;font-size:12px}
table.dt th,table.dt td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
table.dt th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;
 letter-spacing:.04em}
table.dt td.n,table.dt th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.focal td{background:var(--chip);font-weight:650}
.scrollx{overflow-x:auto}
.note{font-size:11.5px;color:var(--mut);margin-top:10px;line-height:1.5}
svg .ax{stroke:var(--line)}
svg text{fill:var(--mut);font-size:10px}
svg text.val{fill:var(--fg);font-size:10px;font-variant-numeric:tabular-nums}
.empty{padding:26px;color:var(--mut);font-style:italic;text-align:center}
kbd{background:var(--chip);border:1px solid var(--line);border-bottom-width:2px;
 border-radius:4px;padding:0 4px;font-size:11px;font-family:inherit}
"""

APP_JS = r"""
(function(){
"use strict";
var D = window.__DATA__;
var M = D.matches, byId = {};
for (var i=0;i<M.length;i++) byId[M[i].i] = M[i];

var $ = function(id){ return document.getElementById(id); };
var paneA=$('paneA'), paneB=$('paneB'), listEl=$('list');
var miniA=$('miniA'), miniB=$('miniB');

/* ---- index marks by match id ---- */
var markIdx = {};
var allMarks = Array.prototype.slice.call(document.querySelectorAll('mark'));
allMarks.forEach(function(el){
  var side = el.getAttribute('data-side');
  el.getAttribute('data-m').split(',').forEach(function(id){
    var e = markIdx[id] || (markIdx[id] = {});
    if (!e[side]) e[side] = el;
  });
});

/* ---- state ---- */
var DEF = {min:8, cite:'all', bucket:'all', dir:false, err:false, q:'',
           passes:{verbatim:true, near_verbatim:false, embedding:false},
           sort:'len', sync:false, theme:'auto'};
var S = load();
function load(){
  try{
    var raw = localStorage.getItem('papercompare.app');
    if(!raw) return JSON.parse(JSON.stringify(DEF));
    var o = JSON.parse(raw), d = JSON.parse(JSON.stringify(DEF));
    for (var k in d) if (raw.indexOf('"'+k+'"')>=0 && o[k]!==undefined) d[k]=o[k];
    return d;
  }catch(e){ return JSON.parse(JSON.stringify(DEF)); }
}
function save(){
  try{ localStorage.setItem('papercompare.app', JSON.stringify(S)); }catch(e){}
}

var current = null, filtered = [];

/* ---- filtering ---- */
function passes(m){
  if (!S.passes[m.p]) return false;
  if (m.w < S.min) return false;
  if (S.cite !== 'all' && m.c !== S.cite) return false;
  if (S.bucket !== 'all' && m.bb !== S.bucket) return false;
  if (S.dir && !m.d) return false;
  if (S.err && !m.e) return false;
  if (S.q){
    var q = S.q.toLowerCase();
    if ((m.t||'').toLowerCase().indexOf(q) < 0 &&
        (m.ta||'').toLowerCase().indexOf(q) < 0) return false;
  }
  return true;
}
function recompute(){
  filtered = M.filter(passes);
  filtered.sort(S.sort === 'len'
    ? function(a,b){ return b.w-a.w || (a.bs-b.bs); }
    : S.sort === 'docB'
      ? function(a,b){ return a.bs-b.bs || b.w-a.w; }
      : function(a,b){ return a.as-b.as || b.w-a.w; });
}

/* ---- rendering ---- */
function renderMarks(){
  var on = {};
  for (var i=0;i<filtered.length;i++) on[filtered[i].i]=filtered[i];
  for (var k=0;k<allMarks.length;k++){
    var el = allMarks[k], ids = el.getAttribute('data-m').split(',');
    var best = null;
    for (var j=0;j<ids.length;j++){
      var m = on[ids[j]];
      if (m && (!best || m.w > best.w)) best = m;
    }
    el.className = best
      ? (best.p==='near_verbatim' ? 'p-near'
         : best.p==='embedding' ? 'p-emb'
         : best.w>=30 ? 'l' : best.w>=20 ? 'm' : '')
      : 'off';
  }
  if (current && on[current]) markSel(current);
}
function markSel(id){
  document.querySelectorAll('mark.sel').forEach(function(e){ e.classList.remove('sel'); });
  var e = markIdx[id];
  if (e){ if(e.A) e.A.classList.add('sel'); if(e.B) e.B.classList.add('sel'); }
}
function esc(s){ return String(s).replace(/[&<>]/g, function(c){
  return c==='&'?'&amp;':c==='<'?'&lt;':'&gt;'; }); }

function renderList(){
  var f = document.createDocumentFragment();
  var LIMIT = 800;
  var shown = filtered.slice(0, LIMIT);
  for (var i=0;i<shown.length;i++){
    var m = shown[i];
    var d = document.createElement('div');
    d.className = 'item' + (m.i===current ? ' on' : '');
    d.setAttribute('data-id', m.i);
    d.innerHTML =
      '<div class="l1"><span class="w">'+m.w+'w</span>'+
      '<span class="mut">A p.'+m.ap+' &rarr; B p.'+m.bp+'</span>'+
      (m.d ? ' <span class="badge b-dir">dir</span>' : '')+
      (m.c==='attributed' ? ' <span class="badge b-att">cited</span>' : '')+
      (m.p!=='verbatim' ? ' <span class="badge">'+(m.p==='embedding'?'emb':'near')+'</span>' : '')+
      '</div><div class="l2">'+esc((m.t||'').slice(0,110))+'</div>';
    f.appendChild(d);
  }
  listEl.innerHTML='';
  if (!shown.length){
    var e=document.createElement('div');
    e.className='empty'; e.textContent='No matches with these filters.';
    listEl.appendChild(e);
  } else {
    listEl.appendChild(f);
    if (filtered.length > LIMIT){
      var more=document.createElement('div');
      more.className='empty';
      more.textContent='… and '+(filtered.length-LIMIT).toLocaleString()+
        ' more (narrow the filters to see them listed)';
      listEl.appendChild(more);
    }
  }
  $('count').textContent = filtered.length.toLocaleString()+' of '+
    M.length.toLocaleString()+' matches';
}

function renderMini(){
  [[miniA,'posA'],[miniB,'posB']].forEach(function(cfg){
    var el=cfg[0], key=cfg[1];
    var h='';
    var step = Math.max(1, Math.ceil(filtered.length/900));
    for (var i=0;i<filtered.length;i+=step){
      var m=filtered[i];
      h += '<i data-id="'+m.i+'" class="'+(m.w>=20?'big':'')+
           (m.i===current?' sel':'')+'" style="top:'+(m[key]*100).toFixed(3)+'%"></i>';
    }
    el.innerHTML = h + '<div class="vp"></div>';
  });
  updateViewport();
}
function updateViewport(){
  [[paneA,miniA],[paneB,miniB]].forEach(function(c){
    var p=c[0], mm=c[1].querySelector('.vp');
    if(!mm) return;
    var tot=p.scrollHeight||1;
    mm.style.top=(100*p.scrollTop/tot)+'%';
    mm.style.height=Math.max(1.5,100*p.clientHeight/tot)+'%';
  });
}

/* ---- selection & navigation ---- */
function select(id, doScroll){
  current = id;
  markSel(id);
  listEl.querySelectorAll('.item.on').forEach(function(e){ e.classList.remove('on'); });
  var row = listEl.querySelector('.item[data-id="'+cssq(id)+'"]');
  if (row){ row.classList.add('on');
    var rt=row.offsetTop, lh=listEl.clientHeight;
    if (rt < listEl.scrollTop || rt > listEl.scrollTop+lh-40)
      listEl.scrollTop = rt - lh/2;
  }
  miniA.querySelectorAll('i.sel').forEach(function(e){e.classList.remove('sel');});
  miniB.querySelectorAll('i.sel').forEach(function(e){e.classList.remove('sel');});
  var mi=miniA.querySelector('i[data-id="'+cssq(id)+'"]'); if(mi) mi.classList.add('sel');
  var mj=miniB.querySelector('i[data-id="'+cssq(id)+'"]'); if(mj) mj.classList.add('sel');

  var m = byId[id], e = markIdx[id] || {};
  if (doScroll){
    scrollTo(paneA, e.A || $('A-s'+ (m?m.as:0)));
    scrollTo(paneB, e.B || $('B-s'+ (m?m.bs:0)));
  }
  if (m){
    $('status').innerHTML =
      '<b>'+m.w+' words</b> · '+m.p.replace('_',' ')+' · similarity '+m.s+
      ' · A PDF p.'+m.ap+(m.app?' (printed '+m.app+')':'')+' ['+m.ab+']'+
      ' &rarr; B PDF p.'+m.bp+(m.bpp?' (printed '+m.bpp+')':'')+' ['+m.bb+']'+
      ' · '+m.c + (m.d ? ' · direction-flagged: '+m.dc : '') +
      (m.e ? ' · shared anomaly' : '');
    try{ history.replaceState(null,'','#m='+encodeURIComponent(id)); }catch(err){}
  }
}
function cssq(s){ return String(s).replace(/"/g,'\\"'); }
function scrollTo(pane, el){
  if (!el) return;
  var top = el.offsetTop - pane.clientHeight/2;
  pane.scrollTop = Math.max(0, top);
}
function step(delta){
  if (!filtered.length) return;
  var idx = -1;
  for (var i=0;i<filtered.length;i++) if (filtered[i].i===current){ idx=i; break; }
  idx = idx < 0 ? (delta>0?0:filtered.length-1) : idx+delta;
  if (idx < 0) idx = filtered.length-1;
  if (idx >= filtered.length) idx = 0;
  select(filtered[idx].i, true);
}

/* ---- synced scrolling ---- */
var syncing = false;
function syncFrom(src, dst, key){
  if (!S.sync || syncing || !filtered.length) return;
  syncing = true;
  var frac = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
  var best=null, bd=2;
  for (var i=0;i<filtered.length;i++){
    var d = Math.abs(filtered[i][key] - frac);
    if (d < bd){ bd=d; best=filtered[i]; }
  }
  if (best){
    var e = markIdx[best.i] || {};
    var el = (dst===paneA) ? e.A : e.B;
    if (el) scrollTo(dst, el);
  }
  setTimeout(function(){ syncing=false; }, 80);
}

/* ---- charts (inline SVG, no libraries) ---- */
function bars(el, rows, opts){
  opts = opts || {};
  var W=440, rowH=22, padL=opts.padL||140, padR=54, H=rows.length*rowH+10;
  var max = 0;
  rows.forEach(function(r){ if (r.v > max) max = r.v; });
  max = max || 1;
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%" height="'+H+'" '+
        'preserveAspectRatio="xMinYMin meet" role="img">';
  rows.forEach(function(r,i){
    var y=i*rowH+4, w=(W-padL-padR)*(r.v/max);
    s+='<text x="'+(padL-8)+'" y="'+(y+11)+'" text-anchor="end">'+esc(r.k)+'</text>';
    s+='<rect x="'+padL+'" y="'+y+'" width="'+Math.max(w,r.v>0?1.5:0)+'" height="13" rx="3" fill="'+
       (r.hi?'var(--acc)':'var(--v2)')+'"><title>'+esc(r.k+': '+r.label)+'</title></rect>';
    s+='<text class="val" x="'+(padL+Math.max(w,2)+6)+'" y="'+(y+11)+'">'+esc(r.label)+'</text>';
  });
  s+='</svg>';
  el.innerHTML=s;
}
function renderFindings(){
  var f = D.focal;
  bars($('chartControl'), D.pairs.map(function(p){
    return {k:p.id.replace('~',' → '), v:p.runs_per_m,
            label:p.runs_per_m.toFixed(2), hi:p.kind==='focal'};
  }), {padL:190});
  bars($('chartLong'), D.pairs.map(function(p){
    return {k:p.id.replace('~',' → '), v:p.long_per_m,
            label:p.long_per_m.toFixed(3), hi:p.kind==='focal'};
  }), {padL:190});
  bars($('chartBucket'), D.buckets.filter(function(b){ return b.total>0; })
    .map(function(b){
      return {k:b.name, v:b.density*100,
              label:(b.density*100).toFixed(2)+'%  ('+b.words.toLocaleString()+'w)'};
    }), {padL:120});
  bars($('chartLen'), D.runlen.map(function(r){
    return {k:r.k+' words', v:r.v, label:String(r.v)};
  }), {padL:100});
  bars($('chartCite'), D.cite.map(function(c){
    return {k:c.k, v:c.v, label:String(c.v)};
  }), {padL:190});
  bars($('chartDir'), D.direction.map(function(c){
    return {k:c.k, v:c.v, label:String(c.v)};
  }), {padL:190});
}

/* ---- wiring ---- */
function apply(rebuildList){
  recompute();
  renderMarks();
  if (rebuildList !== false) renderList();
  renderMini();
  save();
}

$('min').addEventListener('input', function(){
  S.min = parseInt(this.value,10); $('minv').textContent = S.min+'w'; apply();
});
$('cite').addEventListener('change', function(){ S.cite=this.value; apply(); });
$('bucket').addEventListener('change', function(){ S.bucket=this.value; apply(); });
$('dir').addEventListener('change', function(){ S.dir=this.checked; apply(); });
$('err').addEventListener('change', function(){ S.err=this.checked; apply(); });
$('sortby').addEventListener('change', function(){ S.sort=this.value; apply(); });
$('sync').addEventListener('change', function(){ S.sync=this.checked; save(); });
$('q').addEventListener('input', function(){ S.q=this.value; apply(); });
document.querySelectorAll('.passchk').forEach(function(c){
  c.addEventListener('change', function(){ S.passes[this.value]=this.checked; apply(); });
});
$('prev').addEventListener('click', function(){ step(-1); });
$('next').addEventListener('click', function(){ step(1); });
$('reset').addEventListener('click', function(){
  S = JSON.parse(JSON.stringify(DEF)); syncControls(); apply();
});
$('togglebar').addEventListener('click', function(){
  document.querySelector('aside').classList.toggle('hidden');
});
$('theme').addEventListener('click', function(){
  var r=document.documentElement;
  var cur=r.getAttribute('data-theme')||'auto';
  var nxt = cur==='auto' ? 'light' : cur==='light' ? 'dark' : 'auto';
  if (nxt==='auto') r.removeAttribute('data-theme'); else r.setAttribute('data-theme',nxt);
  S.theme=nxt; this.textContent = nxt==='auto'?'◐ auto':nxt==='light'?'☀ light':'☾ dark';
  save();
});
listEl.addEventListener('click', function(e){
  var it = e.target.closest ? e.target.closest('.item') : null;
  if (it) select(it.getAttribute('data-id'), true);
});
[miniA,miniB].forEach(function(mm){
  mm.addEventListener('click', function(e){
    var t=e.target;
    if (t && t.tagName==='I'){ select(t.getAttribute('data-id'), true); return; }
    var r=mm.getBoundingClientRect(), frac=(e.clientY-r.top)/r.height;
    var pane = (mm===miniA)?paneA:paneB;
    pane.scrollTop = frac*(pane.scrollHeight-pane.clientHeight);
  });
});
document.addEventListener('click', function(e){
  var mk = e.target.closest ? e.target.closest('mark') : null;
  if (!mk || mk.classList.contains('off')) return;
  var ids = mk.getAttribute('data-m').split('').length ? mk.getAttribute('data-m').split(',') : [];
  var vis = ids.filter(function(id){ return filtered.some(function(m){ return m.i===id; }); });
  if (!vis.length) return;
  vis.sort(function(a,b){ return byId[b].w - byId[a].w; });
  select(vis[0], true);
});
paneA.addEventListener('scroll', function(){ updateViewport(); syncFrom(paneA,paneB,'posA'); });
paneB.addEventListener('scroll', function(){ updateViewport(); syncFrom(paneB,paneA,'posB'); });
document.addEventListener('keydown', function(e){
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key==='j' || e.key==='ArrowDown'){ e.preventDefault(); step(1); }
  else if (e.key==='k' || e.key==='ArrowUp'){ e.preventDefault(); step(-1); }
  else if (e.key==='/'){ e.preventDefault(); $('q').focus(); }
  else if (e.key==='f'){ showTab(document.querySelector('.tab[data-v="findings"]')); }
  else if (e.key==='c'){ showTab(document.querySelector('.tab[data-v="compare"]')); }
});

function showTab(btn){
  if (!btn) return;
  document.querySelectorAll('.tab').forEach(function(b){
    b.setAttribute('aria-selected', String(b===btn));
  });
  var findings = btn.getAttribute('data-v')==='findings';
  $('findings').classList.toggle('show', findings);
  document.querySelector('.panes').classList.toggle('hide', findings);
  document.querySelector('aside').classList.toggle('hidden', findings);
  if (findings) renderFindings();
}
document.querySelectorAll('.tab').forEach(function(b){
  b.addEventListener('click', function(){ showTab(b); });
});

function syncControls(){
  $('min').value=S.min; $('minv').textContent=S.min+'w';
  $('cite').value=S.cite; $('bucket').value=S.bucket;
  $('dir').checked=S.dir; $('err').checked=S.err;
  $('sortby').value=S.sort; $('sync').checked=S.sync; $('q').value=S.q;
  document.querySelectorAll('.passchk').forEach(function(c){
    c.checked = !!S.passes[c.value];
  });
  if (S.theme && S.theme!=='auto'){
    document.documentElement.setAttribute('data-theme',S.theme);
    $('theme').textContent = S.theme==='light'?'☀ light':'☾ dark';
  }
}

syncControls();
apply();
var hash = (location.hash||'').match(/#m=(.+)$/);
if (hash && byId[decodeURIComponent(hash[1])]) select(decodeURIComponent(hash[1]), true);
else if (filtered.length) select(filtered[0].i, false);
window.addEventListener('resize', updateViewport);
})();
"""


def write_app(path: Path, a: Document, b: Document,
              recs: list[MatchRecord], ctx: dict) -> None:
    """Richer interactive application over the same data as viewer.html."""
    synthetic = ctx["synthetic"]
    analyses = ctx["analyses"]
    focal_id = ctx["focal_pair_id"]
    focal = analyses[focal_id]

    recs = sorted(recs, key=lambda m: (m.pass_name, -m.run_length_words, m.match_id))
    seg_a = _segments(a, recs, "A")
    seg_b = _segments(b, recs, "B")

    n_a = max(len(a.sentences), 1)
    n_b = max(len(b.sentences), 1)

    matches = []
    for m in sorted(recs, key=lambda x: x.match_id):
        matches.append({
            "i": m.match_id, "p": m.pass_name, "w": m.run_length_words,
            "s": m.similarity,
            "ap": m.a_pdf_page_start, "app": m.a_printed_page, "ab": m.a_bucket,
            "bp": m.b_pdf_page_start, "bpp": m.b_printed_page, "bb": m.b_bucket,
            "c": m.citation_class, "d": m.direction_flag,
            "dc": m.direction_categories, "e": m.error_carryover,
            "as": m.a_sent_start, "bs": m.b_sent_start,
            "posA": round(max(m.a_sent_start, 0) / n_a, 5),
            "posB": round(max(m.b_sent_start, 0) / n_b, 5),
            "t": m.b_text[:300], "ta": m.a_text[:300],
        })

    buckets = []
    for name in BUCKET_ORDER:
        total = focal["bucket_token_totals"].get(name, 0)
        words = focal["matched_words_by_bucket"].get(name, 0)
        buckets.append({"name": name, "total": total, "words": words,
                        "runs": focal["runs_by_bucket"].get(name, 0),
                        "density": (words / total) if total else 0.0})

    data = {
        "focal": {
            "runs": focal["verbatim_runs"],
            "long": focal["verbatim_long_runs"],
            "max": focal["verbatim_max_run"],
            "words": focal["verbatim_total_matched_words_b"],
            "share": focal["matched_share_of_b_included"],
            "near": focal["near_verbatim_pairs"],
            "emb": focal["embedding_pairs"],
            "comparisons": focal["comparisons"],
            "mentions": focal["author_a_mentions_in_b_total"],
        },
        "pairs": [
            {"id": p, "kind": analyses[p]["pair_kind"],
             "runs": analyses[p]["verbatim_runs"],
             "runs_per_m": round(analyses[p]["verbatim_per_million_comparisons"], 4),
             "long": analyses[p]["verbatim_long_runs"],
             "long_per_m": round(analyses[p]["long_runs_per_million_comparisons"], 5),
             "max": analyses[p]["verbatim_max_run"],
             "words": analyses[p]["verbatim_total_matched_words_b"]}
            for p in ctx["pair_order"]
        ],
        "buckets": buckets,
        "runlen": [{"k": k, "v": v}
                   for k, v in focal["run_length_distribution"].items()],
        "cite": [{"k": k, "v": focal["citation_classes"][k]}
                 for k in ("attributed", "adjacent-but-unattributed", "unattributed")],
        "direction": [{"k": k, "v": v}
                      for k, v in sorted(focal["direction_categories"].items())],
        "matches": matches,
    }
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).replace("</", "<\\/")

    banner = ""
    if synthetic:
        banner = ('<div class="banner">⚠️ SYNTHETIC SELF-TEST OUTPUT — NOT REAL '
                  'FINDINGS. Every passage and every match below was generated '
                  'from fixture PDFs of invented prose containing a deliberately '
                  'planted shared passage. No real thesis and no real person is '
                  'shown here.</div>')

    passes_present = sorted({m.pass_name for m in recs}) or ["verbatim"]
    pass_boxes = "".join(
        f'<label><input type="checkbox" class="passchk" value="{p}"'
        f'{" checked" if p == "verbatim" else ""}> {p.replace("_", " ")}</label>'
        for p in passes_present
    )
    bucket_opts = "".join(
        f'<option value="{k}">{k}</option>' for k in BUCKET_ORDER
        if focal["bucket_token_totals"].get(k, 0) > 0
    )

    f = data["focal"]
    chips = [
        f'<span class="chip"><b>{f["runs"]:,}</b> verbatim runs</span>',
        f'<span class="chip"><b>{f["long"]}</b> runs ≥20w</span>',
        f'<span class="chip">longest <b>{f["max"]}</b>w</span>',
        f'<span class="chip"><b>{100*f["share"]:.2f}%</b> of later doc</span>',
        f'<span class="chip">earlier author named <b>{f["mentions"]}</b>×</span>',
    ]

    title = "PaperCompare" + (" — SYNTHETIC SELF-TEST" if synthetic else "")
    pairs_rows = "".join(
        '<tr class="{cls}"><td>{id}</td><td>{kind}</td><td class="n">{runs}</td>'
        '<td class="n">{rpm:.2f}</td><td class="n">{lng}</td>'
        '<td class="n">{lpm:.3f}</td><td class="n">{mx}</td>'
        '<td class="n">{w:,}</td></tr>'.format(
            cls="focal" if p["kind"] == "focal" else "",
            id=html_escape(p["id"]), kind=p["kind"], runs=f'{p["runs"]:,}',
            rpm=p["runs_per_m"], lng=p["long"], lpm=p["long_per_m"],
            mx=p["max"], w=p["words"])
        for p in data["pairs"]
    )

    out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<style>{APP_CSS}</style>
</head>
<body>
{banner}
<header>
  <div class="top">
    <div class="brand">Textual-overlap explorer
      <small>{_esc(a.spec.label)} &nbsp;→&nbsp; {_esc(b.spec.label)}</small>
    </div>
    <div class="chips">{''.join(chips)}</div>
    <div class="tabs">
      <button class="tab" data-v="compare" aria-selected="true">Compare</button>
      <button class="tab" data-v="findings" aria-selected="false">Findings</button>
      <button class="iconbtn" id="togglebar" title="Show/hide sidebar">☰</button>
      <button class="iconbtn" id="theme" title="Theme">◐ auto</button>
    </div>
  </div>
  <div class="top" style="padding-top:0;border-top:1px solid var(--line)">
    <button class="iconbtn" id="prev" title="Previous match (k / ↑)">◀ prev</button>
    <button class="iconbtn" id="next" title="Next match (j / ↓)">next ▶</button>
    <span id="status" style="font-size:12px;color:var(--mut)"></span>
  </div>
</header>
<main>
  <aside>
    <div class="filters">
      <div class="frow">
        <label>Minimum run length</label>
        <div class="rangewrap">
          <input type="range" id="min" min="8" max="60" step="1" value="8">
          <b id="minv">8w</b>
        </div>
      </div>
      <div class="frow">
        <label>Citation class</label>
        <select id="cite">
          <option value="all">all</option>
          <option value="attributed">attributed</option>
          <option value="adjacent-but-unattributed">adjacent-but-unattributed</option>
          <option value="unattributed">unattributed</option>
        </select>
      </div>
      <div class="frow">
        <label>Chapter of later document</label>
        <select id="bucket"><option value="all">all</option>{bucket_opts}</select>
      </div>
      <div class="frow">
        <label>Matching pass</label>
        <div class="checks">{pass_boxes}</div>
      </div>
      <div class="frow">
        <label>Only show</label>
        <div class="checks">
          <label><input type="checkbox" id="dir"> direction-flagged</label>
          <label><input type="checkbox" id="err"> shared anomaly</label>
        </div>
      </div>
      <div class="frow">
        <label>Search matched text</label>
        <input type="search" id="q" placeholder="type to filter  ( / )">
      </div>
      <div class="frow">
        <div class="checks">
          <label><input type="checkbox" id="sync"> synced scrolling</label>
          <button class="iconbtn" id="reset" style="margin-left:auto">reset</button>
        </div>
      </div>
    </div>
    <div class="listhead">
      <span id="count"></span>
      <select id="sortby">
        <option value="len">longest first</option>
        <option value="docB">later doc order</option>
        <option value="docA">earlier doc order</option>
      </select>
    </div>
    <div id="list"></div>
  </aside>

  <div class="panes">
    <div class="panewrap">
      <section class="pane" id="paneA">
        <h2>EARLIER <span>— {_esc(a.spec.label)}</span></h2>
        {_pane_html(a, "A", seg_a)}
      </section>
      <div class="mini" id="miniA" title="Match density — click to jump"></div>
    </div>
    <div class="panewrap">
      <section class="pane" id="paneB">
        <h2>LATER <span>— {_esc(b.spec.label)}</span></h2>
        {_pane_html(b, "B", seg_b)}
      </section>
      <div class="mini" id="miniB" title="Match density — click to jump"></div>
    </div>
  </div>

  <div id="findings">
    <div class="grid">
      <div class="card wide">
        <h3>Control comparison — all six pairwise combinations</h3>
        <p class="sub">A raw similarity score between two theses in the same
          subfield means nothing on its own. Normalised to matches per million
          sentence-pair comparisons.</p>
        <div class="scrollx">
        <table class="dt">
          <thead><tr><th>Pair</th><th>Kind</th><th class="n">Verbatim runs</th>
          <th class="n">Runs/M</th><th class="n">Runs ≥20w</th>
          <th class="n">Long/M</th><th class="n">Longest</th>
          <th class="n">Matched words</th></tr></thead>
          <tbody>{pairs_rows}</tbody>
        </table>
        </div>
      </div>
      <div class="card"><h3>Verbatim runs per million comparisons</h3>
        <p class="sub">Focal pair highlighted.</p><div id="chartControl"></div></div>
      <div class="card"><h3>Runs of 20+ words per million</h3>
        <p class="sub">Length matters more than count.</p><div id="chartLong"></div></div>
      <div class="card"><h3>Overlap density by chapter of the later document</h3>
        <p class="sub">Matched words as a share of that chapter's words.</p>
        <div id="chartBucket"></div></div>
      <div class="card"><h3>Length distribution of verbatim runs</h3>
        <p class="sub">One 40-word run means more than forty 8-word runs.</p>
        <div id="chartLen"></div></div>
      <div class="card"><h3>Citation classification</h3>
        <p class="sub">Is the earlier author named within ±3 sentences?</p>
        <div id="chartCite"></div></div>
      <div class="card"><h3>Direction check — categories</h3>
        <p class="sub">Runs where the earlier text describes its own study.</p>
        <div id="chartDir"></div></div>
      <div class="card wide">
        <p class="note"><b>What this shows and what it does not.</b> Textual
        overlap is a measurement, not a motive, and it is not a finding of
        misconduct — only an institution can make that. Shared wording can
        arise from a common source, standard disciplinary formulation, or
        quotation the extraction did not recover. Roughly one match in six
        carries quotation marks on both sides. See <code>report.md</code> §9
        and §10 for the full statement and limitations, and §11 for the
        verification log. Keyboard: <kbd>j</kbd>/<kbd>k</kbd> next and previous
        match, <kbd>/</kbd> search, <kbd>c</kbd> compare, <kbd>f</kbd> findings.</p>
      </div>
    </div>
  </div>
</main>
<script type="application/json" id="payload">{payload}</script>
<script>
window.__DATA__ = JSON.parse(document.getElementById('payload').textContent);
</script>
<script>{APP_JS}</script>
</body>
</html>
"""
    path.write_text(out, encoding="utf-8")


def html_escape(s: str) -> str:
    return _esc(s)


if __name__ == "__main__":
    sys.exit(main())
