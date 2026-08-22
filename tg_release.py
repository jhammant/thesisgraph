#!/usr/bin/env python3
"""
tg_release.py — package the corpus's DERIVED data as an open dataset, and refuse
to ship it if any field looks like thesis prose.

The corpus itself can never be published. The theses in it are in copyright and
the UK text-and-data-mining exception (s.29A CDPA) permits computational
analysis of lawfully accessed works, not their redistribution. What the
exception does not touch is everything the analysis produced ABOUT those works:
counts, shares, labels, features, bibliographic edges. That derived layer is the
only publishable part of this project, and until now it existed only as a
private SQLite file and two CSVs nobody outside this machine could use.

So this builds `out/release/` — data files, a Gebru-style datasheet, a manifest
with per-column descriptions, checksums, and a licence note — and then tries to
break it.

WHY THE LEAK CHECK IS THE POINT
-------------------------------
A copyright promise that is only a promise is worth nothing. Somebody adds a
column six months from now, a `raw` reference string or an abstract rides along
with it, and the dataset that was safe on the day it was written is a
redistribution of in-copyright text on the day it is published. So the release
is not "assembled and checked over"; it is assembled into a staging directory,
put through four independent tests, and only moved into place if all four pass —
with a fifth that verifies the stamped result afterwards:

  1. FILE ALLOW-LIST      the set of files in data/ must be exactly the declared
                          set — no strays, nothing missing.
  2. COLUMN ALLOW-LIST    every header must equal its declared column list,
                          in order. A new column is a build failure, not a
                          silent addition, which is the only way a schema change
                          cannot quietly introduce a text field.
  3. VALUE POLICY         every column declares a kind, and a kind is a
                          full-match pattern: `int` and `bool` admit digits,
                          `float` a number, `url` one landing-page template.
                          Those four cannot hold a word at all. `id`, `token`
                          and `enum` DO admit letters — an OAI identifier has a
                          word-shaped repository segment, a token is one word
                          per row, an enum is a short phrase — so they are
                          capped on length (and an enum on cardinality too) and
                          then handed to test 4 alongside the `text` columns.
                          Only two columns in the release are declared `text`,
                          and each carries a character and word budget.
  4. CORPUS VERBATIM RUN  the test the first three cannot do. Every value of
                          every word-bearing column is concatenated per thesis
                          in file order and aligned against that thesis's own
                          extracted text, using the same tokenisation the
                          overlap analysis uses. The longest contiguous
                          verbatim run must fit the column's budget. Per-value
                          caps cannot catch reassembly across rows; this does,
                          and it does not care what kind the column declared:
                          120 single words in a `token` column reassemble into
                          120 contiguous words of a thesis exactly as a `text`
                          column would.
  5. CHECKSUM             on any later `tg release --check`, every file must
                          still match its SHA256SUMS entry and every file must
                          be covered by one. A build has no stamp yet, so this
                          costs nothing there; it is what makes checking a
                          release somebody handed you mean anything.

WHAT TEST 4 DOES NOT COVER
--------------------------
It aligns a row's values against the thesis that row is keyed to. Two things sit
outside that. Words lifted from thesis A and filed under a row belonging to
thesis B: nothing short of aligning every value against all 24,656 theses would
catch that, which is the all-pairs problem this project does not solve at build
time. And the subject tree, whose nodes are keyed to no thesis at all —
check_hierarchy caps its labels at the same length an enum value gets and the
tree has a bounded number of nodes, but nothing aligns them. What stands in
place of test 4 in both cases is the value policy alone — a bounded per-value
length everywhere, a bounded number of distinct values in an `enum`, a bounded
number of nodes in the tree — so a misfiled column or a relabelled node can hold
short fragments, not a document. That is a weaker guarantee than the per-thesis
one and it is stated as such in the datasheet rather than glossed.

By default every released thesis is aligned, because "leak check clean" over a
sample is a claim about the sample and gets reported as one: `--sample N` runs a
partial check and every line that reports it, in the log, the manifest and the
datasheet, says so and says the fraction.

Check 4 is what settles the awkward question about `cite.raw` — the
reference-list string each parsed citation was extracted from, and the one field
where "is this derived data or is this the thesis?" has no obvious answer.
Measured rather than argued. Aligned against all 24,656 theses, the parsed
citation fields reproduce at most 14 contiguous words of one — the parser caps a
title at 8 words and the run breaks at almost every row boundary — while the raw
strings they were parsed from reproduce 32 to 71. Fourteen words of a cited
work's title is the kind of bibliographic fact Crossref and OpenCitations
already publish openly; a 45-word window of a thesis page is the thesis. So the
parsed fields ship, `raw` does not, and the check enforces the line rather than
trusting it.

`--self-test` proves the guard fires. It builds a small clean release, asserts
it passes, then feeds the check ten deliberately bad releases built from real
corpus data — reference-list strings in a title column, a deposited abstract in
a title column, a body passage cut into per-row chunks that each pass every
per-value cap, that same passage smuggled one word per row through a `token`
column, a sentence of it inside an `enum` value, and a few words of it per row
dotted into an `id` column, prose in a numeric column, an undeclared column, a
stray file, and a release edited after it was stamped — and asserts every one is
rejected, for the expected reason. A guard that has never fired has not been
tested.

Determinism: every query is ordered, every collection is sorted before writing,
no wall-clock and no RNG reach any artefact. `--determinism-check` builds twice
and diffs byte for byte.

    tg release                          # build out/release/
    tg release --no-titles              # omit thesis titles
    tg release --check out/release      # re-run the leak check on a built release
    tg release --check out/release --sample 500   # a fast, PARTIAL check
    tg release --self-test              # prove the leak check fails on bad data
    tg release --determinism-check      # build twice, diff byte for byte
    tg release --out DIR --force        # overwrite DIR even if it is not a release
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "release"
HELP = "build out/release/ — a text-free open dataset, datasheet and leak check"

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

REPO_NAME = {"whiterose": "White Rose eTheses Online"}
REPO_LANDING = {"whiterose": "https://etheses.whiterose.ac.uk/id/eprint/{n}/"}
LANDING_RE = re.compile(r"^https://etheses\.whiterose\.ac\.uk/id/eprint/\d+/$")
OAI_ID_RE = re.compile(r"^oai:[a-z0-9.\-]+:\d+$")

DATA_LICENCE = "CC BY 4.0"
CODE_LICENCE = "MIT"

# --------------------------------------------------------------------------- #
# Column policy
# --------------------------------------------------------------------------- #
# Every released column declares a kind, and every kind is a full-match pattern
# a value has to satisfy. Four of them cannot express a word: `int` and `bool`
# are digits, `float` is a number, `url` is one landing-page template with a
# number in it. The other four can — an `id` has a word-shaped repository
# segment, a `token` is a word, an `enum` is a short phrase, `text` is text — so
# a per-value cap is all their policy can do, and a per-value cap is exactly
# what reassembly across rows walks around. Every one of those four is aligned
# against the thesis by test 4.

KINDS = ("id", "int", "float", "bool", "token", "url", "enum", "text")

# Kinds whose values can carry words a producer chose, and are therefore
# concatenated per thesis and aligned against that thesis's own text.
RUN_KINDS = frozenset({"id", "token", "enum", "text"})

ENUM_VALUE_CHARS = 80          # any single enum value
ENUM_VALUE_WORDS = 12
TOKEN_CHARS = 60               # a `token` is one whitespace-free word
URL_CHARS = 400
ID_CHARS = 80                  # an OAI identifier; the observed maximum is 33

# The verbatim-run budget for a word-bearing column that declares none of its
# own — every `id`, `token` and `enum` column. Two things fix it at eight words.
# MEASURED: with the budget disabled, over all 24,656 released theses and all
# sixteen such columns, the worst run any of them reaches is 6 contiguous words
# (`documents.publisher` — an awarding institution whose name is also printed on
# the title page); the `id` and `token` columns reach 5, and against a deposited
# abstract the worst is 4. ANCHORED: eight words is the shortest run the overlap
# analysis will report as a run at all, so a column with no business carrying
# prose may not reproduce one. A future harvest with longer institution names
# would fail this and a human would have to look, which is the right failure for
# a guard to have. Per-column figures for any given build are in the manifest,
# under leak_check.verbatim_run_test.by_column.
DEFAULT_RUN = 8
DEFAULT_ABSTRACT_RUN = 8


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    desc: str
    max_card: int = 0          # enum: distinct values allowed
    max_chars: int = 0         # text: characters per value
    max_words: int = 0         # text: words per value
    max_run: int = 0           # text: contiguous verbatim words vs the thesis
    max_abstract_run: int = 0  # text: contiguous verbatim words vs the abstract
    lo: int | None = None      # int: inclusive bounds, when a range is meaningful
    hi: int | None = None

    def __post_init__(self) -> None:
        assert self.kind in KINDS, f"{self.name}: unknown kind {self.kind!r}"
        if self.kind == "text":
            assert self.max_chars and self.max_words and self.max_run, self.name
        if self.kind == "enum":
            assert self.max_card, self.name

    @property
    def checked(self) -> bool:
        """Is this column aligned against the thesis by test 4?"""
        return self.kind in RUN_KINDS

    @property
    def run_budget(self) -> int:
        """Contiguous words of its own thesis this column's values may reproduce."""
        if not self.checked:
            return 0
        return self.max_run or DEFAULT_RUN

    @property
    def abstract_budget(self) -> int:
        """The same, against the deposited abstract."""
        if not self.checked:
            return 0
        return self.max_abstract_run or DEFAULT_ABSTRACT_RUN


@dataclass(frozen=True)
class FileSpec:
    name: str                  # path relative to the release root
    desc: str
    columns: tuple[Column, ...]
    doc_key: str               # column naming the thesis a row derives from
    note: str = ""             # a caveat that belongs next to the file, not in it


def _c(name, kind, desc, **kw) -> Column:
    return Column(name, kind, desc, **kw)


DISCIPLINE_CARD = 30           # 16 in the data; headroom for a re-run
SUBFIELD_CARD = 200            # 98 in the data
PUBLISHER_CARD = 60            # 12 in the data

# Titles: the observed maximum after whitespace flattening is 434 characters
# and 65 words, inflated by a handful of deposits whose "title" field carries a
# copyright statement. The budgets sit just above that.
TITLE_CHARS, TITLE_WORDS = 500, 80
# A parsed citation title is capped at 8 words by citations.py, and a run only
# grows past that where two adjacent parsed titles happen to be adjacent in the
# source. Measured by aligning every citation row against every one of the
# 24,656 theses: the parsed fields reach 14 contiguous words, the raw
# reference-list strings they came from reach 32-71. The budget sits in the
# middle of that gap — ten words of headroom for a re-harvest, still eight below
# anything a raw string does.
CITE_TITLE_CHARS, CITE_TITLE_WORDS, CITE_TITLE_RUN = 320, 8, 24


def documents_columns(with_titles: bool) -> tuple[Column, ...]:
    cols = [
        _c("id", "id", "OAI identifier; the join key for every other file"),
        _c("eprint_id", "int", "numeric EPrints id, parsed from `id`", lo=1),
        _c("landing_url", "url",
           "canonical repository landing page, derived from `id` (see datasheet)"),
    ]
    if with_titles:
        cols.append(_c("title", "text",
                       "thesis title as deposited, whitespace flattened",
                       max_chars=TITLE_CHARS, max_words=TITLE_WORDS,
                       max_run=TITLE_WORDS, max_abstract_run=TITLE_WORDS))
    cols += [
        _c("repo", "enum", "source repository key", max_card=10),
        _c("publisher", "enum", "awarding institution, as deposited", max_card=PUBLISHER_CARD),
        _c("year", "int", "publication year from the OAI record; see limitations", lo=0, hi=2100),
        _c("is_education", "bool", "1 if the metadata matches the education-subject filter"),
        _c("discipline", "enum", "assigned discipline (discipline.py)", max_card=DISCIPLINE_CARD),
        _c("discipline_conf", "float", "confidence of that assignment, 0-1"),
        _c("discipline_alt", "enum", "runner-up discipline", max_card=DISCIPLINE_CARD),
        _c("discipline_method", "enum", "how it was assigned: rules | embedding | embedding-tiebreak",
           max_card=10),
        _c("subfield", "enum", "derived subject cluster within the discipline (subfields.py)",
           max_card=SUBFIELD_CARD),
        _c("pages", "int", "PDF pages extracted", lo=0),
        _c("no_text_pages", "int", "pages that yielded no extractable text (scan quality proxy)", lo=0),
        _c("sentences", "int", "sentences after segmentation", lo=0),
        _c("tokens", "int", "matching tokens retained after exclusions", lo=0),
        _c("words", "int", "words in the retained text", lo=0),
    ]
    return tuple(cols)


CITATION_COLUMNS = (
    _c("src", "id", "citing thesis"),
    _c("surname", "token", "first-author surname of the cited work, lowercased"),
    _c("year", "int", "year of the cited work", lo=1000, hi=2100),
    _c("title", "text",
       "title of the CITED work, lowercased and capped at 8 words by the parser",
       max_chars=CITE_TITLE_CHARS, max_words=CITE_TITLE_WORDS,
       max_run=CITE_TITLE_RUN, max_abstract_run=12),
)

CITATION_SOURCE_COLUMNS = (
    _c("src", "id", "thesis whose reference list was parsed"),
    _c("ref_words", "int", "words in the region identified as the reference list", lo=0),
    _c("entries", "int", "reference entries segmented", lo=0),
    _c("parsed", "int", "entries from which an author/year/title was parsed", lo=0),
)

PAIR_COLUMNS = (
    _c("a", "id", "first thesis of the pair (sorts before `b`)"),
    _c("b", "id", "second thesis of the pair"),
    _c("a_tokens", "int", "retained tokens in `a`", lo=0),
    _c("b_tokens", "int", "retained tokens in `b`", lo=0),
    _c("runs", "int", "verbatim runs of 8+ words shared by the pair", lo=0),
    _c("runs_truncated", "bool",
       "1 if run enumeration hit the 20,000-run safety cap for pathologically "
       "repetitive text; `runs` and `matched_words` are then a lower bound and "
       "the containment pass did not run"),
    _c("long_runs", "int", "of those, runs of 20+ words", lo=0),
    _c("max_run", "int", "longest shared verbatim run, in words", lo=0),
    _c("matched_words", "int", "words inside shared runs", lo=0),
    _c("share", "float", "matched_words / tokens of the shorter document"),
    _c("unique_runs", "int", "runs after dropping any run that also appears in a third thesis", lo=0),
    _c("unique_long_runs", "int", "of those, runs of 20+ words", lo=0),
    _c("unique_max_run", "int", "longest such run", lo=0),
    _c("unique_words", "int", "words inside them", lo=0),
    _c("unique_share", "float", "their share of the shorter document"),
    _c("common_runs", "int", "runs also found in an unrelated third thesis (boilerplate)", lo=0),
    _c("quoted_runs", "int", "runs sitting inside an attributed block quote", lo=0),
    _c("same_author", "bool", "1 if the deposits share a creator string"),
    _c("same_inst", "bool", "1 if the deposits share an awarding institution"),
    _c("title_sim", "int", "title similarity, 0-100 (rapidfuzz token_set_ratio)", lo=0, hi=100),
    _c("clean_runs", "int", "runs after removing extraction artefacts (rescore.py)", lo=0),
    _c("clean_long_runs", "int", "of those, runs of 20+ words", lo=0),
    _c("clean_max_run", "int", "longest such run", lo=0),
    _c("clean_words", "int", "words inside them", lo=0),
    _c("clean_share", "float", "their share of the shorter document"),
    _c("verdict", "enum",
       "triage label: fully_explained | short_only | residual | (blank = not triaged)",
       max_card=10),
)

DISCOURSE_DROP = ("year", "discipline", "publisher")     # join on documents.csv
METHOD_DROP = ("title", "creator", "year", "publisher", "education")

DISCOURSE_DESC = {
    "boosters_per1k": "boosting expressions per 1,000 words (clearly, demonstrates, must)",
    "citations_per1k": "in-text citation patterns per 1,000 words",
    "first_pl_per1k": "first-person plural pronouns per 1,000 words",
    "first_sg_per1k": "first-person singular pronouns per 1,000 words",
    "flesch": "Flesch reading-ease score",
    "hedge_boost_ratio": "hedges divided by boosters",
    "hedges_per1k": "hedging expressions per 1,000 words (may, suggests, appears)",
    "impersonal_per1k": "impersonal constructions per 1,000 words (it is argued that)",
    "lexical_density": "percentage of words that are not function words",
    "long_word_pct": "percentage of words of 7+ characters",
    "mean_sentence_len": "mean sentence length in words",
    "nominal_per1k": "nominalisations per 1,000 words (-tion, -ment, -ity)",
    "passive_per1k": "passive constructions per 1,000 words",
    "questions_per1k": "question marks per 1,000 words",
    "sentences": "sentences in the retained text",
    "words": "words in the retained text",
}

METHOD_SCALARS = {
    "method_scope": ("enum", "how the methodology chapter was located: chapter | fallback_wholedoc"),
    "method_sentences": ("int", "sentences in the located methodology scope"),
    "retained_sentences": ("int", "sentences in the retained text"),
    "sample_size_max": ("int", "largest participant count found in the methodology scope"),
    "sample_size_median": ("int", "median participant count found"),
    "sample_sizes_found": ("int", "how many participant counts were found"),
}


def _method_desc(col: str) -> str:
    """Describe a method feature column from its category::label[::declared] name."""
    if col in METHOD_SCALARS:
        return METHOD_SCALARS[col][1]
    parts = col.split("::")
    declared = parts[-1] == "declared"
    if declared:
        parts = parts[:-1]
    cat, label = parts[0], "::".join(parts[1:])
    what = {
        "collection": f"data-collection method {label!r}",
        "design": f"research design {label!r}",
        "sampling": f"sampling strategy {label!r}",
        "rigour": f"rigour or reporting practice {label!r}",
        "software": f"analysis software {label!r}",
        "open_science": f"open-science practice {label!r}",
    }.get(cat, f"{cat} feature {label!r}")
    if declared:
        return f"1 if {what} is claimed as what the author did, not merely discussed"
    return f"matches for {what} in the scope that category is scored over"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def flat(s: str | None) -> str:
    """One line, single spaces, no control characters — a CSV row stays a row."""
    if not s:
        return ""
    return _WS_RE.sub(" ", _CTRL_RE.sub(" ", str(s))).strip()


def num(v) -> str:
    """Deterministic number formatting; NULL becomes an empty field."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return ""
        return f"{v:.10g}"
    return str(v)


def eprint_id(doc_id: str) -> str:
    tail = doc_id.rsplit(":", 1)[-1]
    return tail if tail.isdigit() else ""


def landing_url(repo: str, doc_id: str) -> str:
    n = eprint_id(doc_id)
    tmpl = REPO_LANDING.get(repo or "")
    return tmpl.format(n=n) if (tmpl and n) else ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ro(path: Path) -> sqlite3.Connection:
    """Read-only connection: corpus/ is read and never written."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def open_out(path: Path, gz: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        # mtime=0: gzip stamps the clock into the header otherwise, and that
        # alone would make two identical builds differ.
        return gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"),
                             compresslevel=6, mtime=0)
    return path.open("w", encoding="utf-8", newline="")


def writer_for(path: Path, gz: bool):
    fh = open_out(path, gz)
    if gz:
        import io
        txt = io.TextIOWrapper(fh, encoding="utf-8", newline="")
        return csv.writer(txt, lineterminator="\n"), txt
    return csv.writer(fh, lineterminator="\n"), fh


def open_in(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
# The document, citation and pair schemas are declared above, by hand: they are
# this release's contract. The two study files carry ~120 machine-generated
# feature columns each, so their column NAMES come from the source CSV — but
# their KINDS do not. Every feature column is declared int or float, so a source
# CSV that grew a text column would fail the value check rather than ride along.

FEATURE_NAME_RE = re.compile(r"^[A-Za-z0-9_:'.()\- ]+$")


def _study_columns(header: list[str], drop: tuple[str, ...], kind_of, desc_of
                   ) -> tuple[Column, ...]:
    cols = [_c("id", "id", "thesis these features describe")]
    for name in header:
        if name == "id" or name in drop:
            continue
        if not FEATURE_NAME_RE.match(name):
            raise SystemExit(f"release: refusing feature column {name!r} "
                             f"— name outside the permitted character set")
        kind = kind_of(name)
        if kind == "enum":
            cols.append(_c(name, "enum", desc_of(name), max_card=20))
        else:
            cols.append(_c(name, kind, desc_of(name)))
    return tuple(cols)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def build_specs(sources: dict, with_titles: bool) -> list[FileSpec]:
    disc_hdr = read_header(sources["discourse"])
    meth_hdr = read_header(sources["method"])

    disc_cols = _study_columns(
        disc_hdr, DISCOURSE_DROP,
        kind_of=lambda n: "int" if n in ("sentences", "words") else "float",
        desc_of=lambda n: DISCOURSE_DESC.get(n, "discourse feature"))
    meth_cols = _study_columns(
        meth_hdr, METHOD_DROP,
        kind_of=lambda n: METHOD_SCALARS.get(n, ("int",))[0],
        desc_of=_method_desc)

    return [
        FileSpec("data/documents.csv",
                 "One row per thesis with usable extracted text, and the join key "
                 "for every other file.",
                 documents_columns(with_titles), "id"),
        FileSpec("data/citations.csv",
                 "Citation edge list: citing thesis -> (surname, year, title) of a "
                 "cited work. Parsed from reference lists, not resolved to DOIs.",
                 CITATION_COLUMNS, "src",
                 note="Author-date styles only; see the parse rate per thesis in "
                      "citation_sources.csv before using counts."),
        FileSpec("data/citation_sources.csv",
                 "Per-thesis reference-parsing coverage — the denominator for "
                 "anything computed from citations.csv.",
                 CITATION_SOURCE_COLUMNS, "src"),
        FileSpec("data/overlap_pairs.csv",
                 "Pairwise textual-overlap statistics for candidate pairs, with no "
                 "matched text. A ranked list of things to look at, never a finding "
                 "about a person.",
                 PAIR_COLUMNS, "a"),
        FileSpec("data/discourse_features.csv",
                 "Per-thesis writing and discourse features, computed on retained "
                 "text only (references, block quotes, boilerplate and captions "
                 "already excluded).",
                 disc_cols, "id"),
        FileSpec("data/method_features.csv",
                 "Per-thesis research-method, sampling, rigour and software "
                 "features.",
                 meth_cols, "id"),
    ]


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def write_documents(con: sqlite3.Connection, out: Path, spec: FileSpec,
                    gz: bool, limit: int) -> tuple[int, list[str]]:
    names = [c.name for c in spec.columns]
    want_title = "title" in names
    sql = ("SELECT id,title,repo,publisher,year,is_education,discipline,"
           "discipline_conf,discipline_alt,discipline_method,subfield,pages,"
           "no_text_pages,sentences,tokens,words FROM doc WHERE status='ok' "
           "ORDER BY id")
    w, fh = writer_for(out, gz)
    ids: list[str] = []
    n = 0
    try:
        w.writerow(names)
        for r in con.execute(sql):
            (did, title, repo, publisher, year, is_edu, disc, conf, alt, meth,
             sub, pages, notext, sents, toks, words) = r
            row = [did, eprint_id(did), landing_url(repo, did)]
            if want_title:
                row.append(flat(title))
            row += [repo or "", flat(publisher), num(year), num(is_edu),
                    disc or "", num(conf), alt or "", meth or "", sub or "",
                    num(pages), num(notext), num(sents), num(toks), num(words)]
            w.writerow(row)
            ids.append(did)
            n += 1
            if limit and n >= limit:
                break
    finally:
        fh.close()
    return n, ids


def write_citations(cit: sqlite3.Connection, out: Path, spec: FileSpec, gz: bool,
                    keep: set[str]) -> int:
    w, fh = writer_for(out, gz)
    n = 0
    try:
        w.writerow([c.name for c in spec.columns])
        # `raw` is deliberately not selected: see the datasheet, "Composition".
        for src, surname, year, title in cit.execute(
                "SELECT src,surname,year,title FROM cite "
                "ORDER BY src, surname, year, title"):
            if src not in keep:
                continue
            w.writerow([src, flat(surname), num(year), flat(title)])
            n += 1
    finally:
        fh.close()
    return n


def write_citation_sources(cit: sqlite3.Connection, out: Path, spec: FileSpec,
                           gz: bool, keep: set[str]) -> int:
    w, fh = writer_for(out, gz)
    n = 0
    try:
        w.writerow([c.name for c in spec.columns])
        for src, ref_words, entries, parsed in cit.execute(
                "SELECT src,ref_words,entries,parsed FROM srcstat ORDER BY src"):
            if src not in keep:
                continue
            w.writerow([src, num(ref_words), num(entries), num(parsed)])
            n += 1
    finally:
        fh.close()
    return n


def write_pairs(con: sqlite3.Connection, out: Path, spec: FileSpec, gz: bool,
                keep: set[str]) -> int:
    names = [c.name for c in spec.columns]
    w, fh = writer_for(out, gz)
    n = 0
    try:
        w.writerow(names)
        cur = con.execute(
            "SELECT a,b,a_tokens,b_tokens,runs,long_runs,max_run,matched_words,"
            "share,unique_runs,unique_long_runs,unique_max_run,unique_words,"
            "unique_share,common_runs,quoted_runs,same_author,same_inst,"
            "title_sim,clean_runs,clean_long_runs,clean_max_run,clean_words,"
            "clean_share,verdict FROM pair ORDER BY a, b")
        for r in cur:
            if r[0] not in keep or r[1] not in keep:
                continue
            # screen_corpus.py stores a truncated pair as a NEGATIVE run count.
            # An in-band sentinel is fine inside one codebase and a trap in a
            # published CSV, so it becomes a value and an explicit flag.
            runs = r[4]
            truncated = 1 if (runs is not None and runs < 0) else 0
            w.writerow([r[0], r[1], num(r[2]), num(r[3]),
                        num(abs(runs) if runs is not None else None),
                        str(truncated)]
                       + [num(v) for v in r[5:24]] + [r[24] or ""])
            n += 1
    finally:
        fh.close()
    return n


def write_study(src_csv: Path, out: Path, spec: FileSpec, gz: bool,
                keep: set[str]) -> int:
    """Copy a study CSV through, keeping only released columns and released ids."""
    names = [c.name for c in spec.columns]
    with src_csv.open("r", encoding="utf-8", newline="") as fin:
        rd = csv.DictReader(fin)
        rows = []
        for row in rd:
            if row["id"] not in keep:
                continue
            rows.append([flat(row.get(k, "")) for k in names])
    rows.sort(key=lambda r: r[0])
    w, fh = writer_for(out, gz)
    try:
        w.writerow(names)
        for r in rows:
            w.writerow(r)
    finally:
        fh.close()
    return len(rows)


def write_hierarchy(src: Path, out: Path) -> dict:
    """The subject tree, stripped of its example titles and deep PDF links.

    hierarchy.json carries eight example theses per node as {title, pdf_url}.
    The titles are covered by documents.csv and the PDF URLs are deep links to
    in-copyright files rather than to the citable landing page, so the whole
    `ex` block is dropped: the tree itself is what is reusable.
    """
    h = json.loads(src.read_text(encoding="utf-8"))
    nodes = [{"id": int(n["id"]), "label": n["label"], "group": n["grp"],
              "depth": int(n["depth"]), "n": int(n["n"]),
              "parent": (None if n.get("parent") is None else int(n["parent"]))}
             for n in h["nodes"]]
    nodes.sort(key=lambda n: n["id"])
    edges = sorted(({"s": int(e["s"]), "t": int(e["t"]), "w": int(e["w"]),
                     "j": float(e["j"])} for e in h["edges"]),
                   key=lambda e: (e["s"], e["t"]))
    kids = {str(k): sorted(int(x) for x in v) for k, v in h["kids"].items()}
    doc = {
        "description": "Discipline -> subfield -> cluster tree derived from the "
                       "corpus by subfields.py and hierarchy.py.",
        "fields": {
            "nodes": "id, label, group (top-level discipline), depth (0-2), "
                     "n (theses under the node), parent (node id or null)",
            "edges": "s, t (node ids), w (theses shared), j (Jaccard)",
            "kids": "parent node id -> child node ids",
        },
        "nodes": nodes,
        "edges": edges,
        "kids": dict(sorted(kids.items(), key=lambda kv: int(kv[0]))),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- #
# Leak check
# --------------------------------------------------------------------------- #
# Five independent tests, described at the top of this file. Each one can fail
# the build on its own; none of them trusts the others.

# Names that must never appear as a released column. The allow-list already
# rejects them; this exists so the failure says WHY rather than "unexpected
# column", and so the intent survives a future edit to the allow-list.
FORBIDDEN_COLUMNS = frozenset({
    "abstract", "raw", "text", "body", "snippet", "sentence", "sentences_text",
    "quote", "passage", "matched_text", "context", "creator", "author",
    "authors", "pdf_url", "email",
})

HIERARCHY_FILE = "data/discipline_hierarchy.json"
# 0 means every released thesis. The corpus-alignment test costs about 40 ms a
# thesis, so a whole-release check of 24,656 theses is ~15 minutes — cheap for
# the one test that stands between this project and redistributing copyright
# text, and the only setting under which "leak check clean" is a statement about
# the release rather than about a sample of it.
DEFAULT_SAMPLE = 0
MAX_REPORTED = 40


@dataclass(frozen=True, order=True)
class Problem:
    file: str
    kind: str
    column: str
    row: int
    detail: str

    def line(self) -> str:
        where = f"{self.file}"
        if self.column:
            where += f"[{self.column}]"
        if self.row:
            where += f" row {self.row}"
        return f"  {self.kind:<12} {where}: {self.detail}"


@dataclass
class ColStat:
    max_chars: int = 0
    max_words: int = 0
    distinct: set = field(default_factory=set)
    n_nonempty: int = 0


def longest_run(value_tokens: list[str], pos: dict[str, list[int]]) -> int:
    """Longest contiguous token run shared with the document behind `pos`.

    Straight longest-common-substring over token identities: `prev` maps a
    document position to the length of the run ending there for the previous
    value token, so a run only continues where the document positions are
    genuinely adjacent. This is the same notion of a "run" the overlap analysis
    uses, over the same tokenisation.
    """
    prev: dict[int, int] = {}
    best = 0
    for tok in value_tokens:
        cur: dict[int, int] = {}
        for p in pos.get(tok, ()):
            cur[p] = prev.get(p - 1, 0) + 1
        if cur:
            m = max(cur.values())
            if m > best:
                best = m
        prev = cur
    return best


def _validate(col: Column, raw: str, stat: ColStat) -> str | None:
    """Policy for one cell. Returns a problem description, or None if it is fine."""
    if raw != flat(raw):
        return "value contains a newline, a control character or padding whitespace"
    if raw:
        stat.n_nonempty += 1
        stat.max_chars = max(stat.max_chars, len(raw))

    if col.kind == "id":
        if not OAI_ID_RE.match(raw):
            return f"not an OAI identifier: {raw[:60]!r}"
        if len(raw) > ID_CHARS:
            return f"{len(raw)} characters exceeds the {ID_CHARS} allowed for an identifier"
    elif col.kind == "int":
        if raw and not re.fullmatch(r"-?\d+", raw):
            return f"not an integer: {raw[:60]!r}"
        if raw and (col.lo is not None or col.hi is not None):
            v = int(raw)
            if col.lo is not None and v < col.lo:
                return f"{v} below the declared minimum {col.lo}"
            if col.hi is not None and v > col.hi:
                return f"{v} above the declared maximum {col.hi}"
    elif col.kind == "float":
        if raw:
            try:
                float(raw)
            except ValueError:
                return f"not a number: {raw[:60]!r}"
    elif col.kind == "bool":
        if raw not in ("", "0", "1"):
            return f"not 0 or 1: {raw[:60]!r}"
    elif col.kind == "token":
        if " " in raw:
            return f"a token column must hold one whitespace-free word: {raw[:60]!r}"
        if len(raw) > TOKEN_CHARS:
            return f"{len(raw)} characters exceeds the {TOKEN_CHARS} allowed for a token"
    elif col.kind == "url":
        if raw and not LANDING_RE.match(raw):
            return f"URL is not on the landing-page allow-list: {raw[:80]!r}"
        if len(raw) > URL_CHARS:
            return f"{len(raw)} characters exceeds the {URL_CHARS} allowed for a URL"
    elif col.kind == "enum":
        if raw:
            stat.distinct.add(raw)
            if len(raw) > ENUM_VALUE_CHARS:
                return (f"{len(raw)} characters exceeds the {ENUM_VALUE_CHARS} "
                        f"allowed for a controlled value")
            if len(raw.split()) > ENUM_VALUE_WORDS:
                return (f"{len(raw.split())} words exceeds the {ENUM_VALUE_WORDS} "
                        f"allowed for a controlled value")
    elif col.kind == "text":
        w = len(raw.split())
        stat.max_words = max(stat.max_words, w)
        if len(raw) > col.max_chars:
            return (f"{len(raw)} characters exceeds the {col.max_chars} budget "
                    f"for this column")
        if w > col.max_words:
            return f"{w} words exceeds the {col.max_words}-word budget for this column"
    return None


def check_structure(root: Path, specs: list[FileSpec], gz: bool,
                    sample_ids: set[str]
                    ) -> tuple[list[Problem], dict, dict]:
    """Tests 1-3: file allow-list, column allow-list, value policy.

    Also collects, for the sampled theses, every released value of every
    word-bearing column keyed by (file, column) in file order — that is what
    test 4 aligns against the corpus text.
    """
    problems: list[Problem] = []
    stats: dict[str, dict[str, ColStat]] = {}
    samples: dict[str, dict[tuple[str, str], list[str]]] = {}

    suffix = ".gz" if gz else ""
    # The subject tree is JSON, so it is not one of the CSV specs; it is declared
    # here and checked by check_hierarchy().
    declared = {s.name + suffix for s in specs} | {HIERARCHY_FILE}
    present = {str(p.relative_to(root)).replace(os.sep, "/")
               for p in sorted((root / "data").rglob("*")) if p.is_file()}
    for extra in sorted(present - declared):
        problems.append(Problem(extra, "stray-file", "", 0,
                                "file is not declared in the release spec"))
    for missing in sorted(declared - present):
        problems.append(Problem(missing, "missing-file", "", 0,
                                "declared file was not built"))

    for spec in specs:
        path = root / (spec.name + suffix)
        if not path.exists():
            continue
        names = [c.name for c in spec.columns]
        stats[spec.name] = {c.name: ColStat() for c in spec.columns}
        # Every column that can hold a word is collected, not only the `text`
        # ones. One word per row in a `token` column, over the hundreds of rows
        # a thesis owns, reassembles into a passage exactly as prose does, and
        # no per-value cap can see that.
        run_cols = [(c.name, i) for i, c in enumerate(spec.columns) if c.checked]
        key_idx = names.index(spec.doc_key) if spec.doc_key in names else -1
        # `src` and the id columns repeat the same string on every row; interning
        # keeps a whole-release sample a list of pointers rather than of copies.
        pool: dict[str, str] = {}

        with open_in(path) as fh:
            rd = csv.reader(fh)
            try:
                header = next(rd)
            except StopIteration:
                problems.append(Problem(spec.name, "header", "", 0, "file is empty"))
                continue
            for bad in sorted(set(header) & FORBIDDEN_COLUMNS):
                problems.append(Problem(spec.name, "forbidden", bad, 0,
                                        "column name is on the never-ship list"))
            if header != names:
                problems.append(Problem(
                    spec.name, "header", "", 0,
                    f"header does not match the declared columns; "
                    f"unexpected={sorted(set(header) - set(names))} "
                    f"missing={sorted(set(names) - set(header))}"
                    + ("" if set(header) != set(names) else " (order differs)")))
                continue

            for lineno, row in enumerate(rd, start=2):
                if len(row) != len(names):
                    problems.append(Problem(spec.name, "row-width", "", lineno,
                                            f"{len(row)} fields, expected {len(names)}"))
                    continue
                for col, raw in zip(spec.columns, row):
                    bad = _validate(col, raw, stats[spec.name][col.name])
                    if bad:
                        problems.append(Problem(spec.name, "value", col.name,
                                                lineno, bad))
                if key_idx >= 0 and run_cols and row[key_idx] in sample_ids:
                    bucket = samples.setdefault(row[key_idx], {})
                    for cname, idx in run_cols:
                        v = row[idx]
                        if v:
                            bucket.setdefault((spec.name, cname), []).append(
                                pool.setdefault(v, v))

        for col in spec.columns:
            if col.kind == "enum":
                st = stats[spec.name][col.name]
                if len(st.distinct) > col.max_card:
                    problems.append(Problem(
                        spec.name, "cardinality", col.name, 0,
                        f"{len(st.distinct)} distinct values exceeds the "
                        f"{col.max_card} allowed for a controlled vocabulary — "
                        f"this column is behaving like free text"))

    return problems, stats, samples


def check_corpus_runs(samples: dict, specs: list[FileSpec], sample_ids: list[str],
                      corpus_db: Path) -> tuple[list[Problem], dict]:
    """Test 4: align every released value against the thesis it came from.

    Every column that can hold a word is aligned, whatever kind it declared —
    `text`, but also `token`, `enum` and `id`. What is aligned is the
    concatenation, in file order, of all of that column's values for one thesis:
    the per-value caps in test 3 are what make a single cell harmless, and this
    is what makes a thousand harmless cells harmless together.

    The alignment is per thesis. Words of thesis A filed under a row belonging
    to thesis B are outside what this can see; the datasheet says so.

    Uses run_analysis.match_tokens and harvest_corpus.store_path, so the notion
    of a word here is identical to the one the overlap analysis uses, and the
    text it aligns against is the FULL extracted text — including the reference
    lists and boilerplate the analysis excludes, because copyright does not care
    which part of a page a sentence sat on.
    """
    sys.path.insert(0, str(HERE))
    import run_analysis as R          # heavy; imported only when the test runs
    import harvest_corpus as H

    budgets = {(s.name, c.name): c for s in specs for c in s.columns
               if c.checked}
    problems: list[Problem] = []
    observed: dict[str, dict] = {}
    checked = missing = 0

    con = ro(corpus_db)
    try:
        abstracts = {}
        for did in sample_ids:
            row = con.execute("SELECT abstract FROM doc WHERE id=?", (did,)).fetchone()
            abstracts[did] = (row[0] if row else "") or ""
    finally:
        con.close()

    for did in sample_ids:
        bucket = samples.get(did)
        if not bucket:
            continue
        store = H.store_path(did)
        if not store.exists():
            missing += 1
            continue
        with gzip.open(store, "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
        doc_tokens: list[str] = []
        for r in rows:
            doc_tokens.extend(R.match_tokens(H.unpack_sentence(r)[0]))
        pos: dict[str, list[int]] = {}
        for i, t in enumerate(doc_tokens):
            pos.setdefault(t, []).append(i)
        abs_tokens = R.match_tokens(abstracts.get(did, ""))
        abs_pos: dict[str, list[int]] = {}
        for i, t in enumerate(abs_tokens):
            abs_pos.setdefault(t, []).append(i)
        checked += 1

        for (fname, cname), values in sorted(bucket.items()):
            col = budgets.get((fname, cname))
            if col is None:
                continue
            toks = R.match_tokens(" ".join(values))
            run = longest_run(toks, pos)
            a_run = longest_run(toks, abs_pos) if abs_tokens else 0
            o = observed.setdefault(f"{fname}[{cname}]",
                                    {"kind": col.kind, "budget": col.run_budget,
                                     "max_run": 0, "max_abstract_run": 0,
                                     "max_words_per_thesis": 0, "theses": 0})
            o["max_run"] = max(o["max_run"], run)
            o["max_abstract_run"] = max(o["max_abstract_run"], a_run)
            o["max_words_per_thesis"] = max(o["max_words_per_thesis"], len(toks))
            o["theses"] += 1
            if run > col.run_budget:
                problems.append(Problem(
                    fname, "verbatim-run", cname, 0,
                    f"{did}: the released values reproduce {run} contiguous words "
                    f"of the thesis, over the {col.run_budget}-word budget for "
                    f"this {col.kind} column"))
            if a_run > col.abstract_budget:
                problems.append(Problem(
                    fname, "abstract-run", cname, 0,
                    f"{did}: the released values reproduce {a_run} contiguous words "
                    f"of the deposited abstract, over the "
                    f"{col.abstract_budget}-word budget for this {col.kind} column"))

    return problems, {"theses_checked": checked, "theses_without_text": missing,
                      "by_column": dict(sorted(observed.items()))}


def check_hierarchy(path: Path) -> list[Problem]:
    """The tree is JSON, so the CSV policy does not reach it. Check it directly.

    Its nodes are keyed to no thesis, so test 4 cannot align them either. What
    bounds a node label is the length cap below and the number of nodes, and the
    datasheet says so rather than implying the corpus test covers this file.
    """
    problems: list[Problem] = []
    name = HIERARCHY_FILE
    if not path.exists():
        return []          # the file allow-list in check_structure reports this
    doc = json.loads(path.read_text(encoding="utf-8"))
    if set(doc) != {"description", "fields", "nodes", "edges", "kids"}:
        problems.append(Problem(name, "header", "", 0,
                                f"unexpected top-level keys: {sorted(doc)}"))
    blob = json.dumps(doc)
    for bad in sorted(FORBIDDEN_COLUMNS):
        if f'"{bad}"' in blob:
            problems.append(Problem(name, "forbidden", bad, 0,
                                    "key is on the never-ship list"))
    for n in doc.get("nodes", []):
        if set(n) != {"id", "label", "group", "depth", "n", "parent"}:
            problems.append(Problem(name, "header", "nodes", int(n.get("id", 0)),
                                    f"unexpected node keys: {sorted(n)}"))
            break
    labels = [n.get("label", "") for n in doc.get("nodes", [])]
    for lab in labels:
        if len(lab) > ENUM_VALUE_CHARS or len(lab.split()) > ENUM_VALUE_WORDS:
            problems.append(Problem(
                name, "value", "label", 0,
                f"node label is {len(lab)} characters / {len(lab.split())} words, "
                f"over the {ENUM_VALUE_CHARS}/{ENUM_VALUE_WORDS} limit: {lab[:60]!r}"))
    return problems


def verify_checksums(root: Path) -> list[Problem]:
    """Once a release is stamped, every file must still match its stamp.

    Absent during a build (the stamp is written last), so this costs nothing
    there; it is what makes `--check` on a directory somebody else handed you
    mean something.
    """
    path = root / "SHA256SUMS"
    if not path.exists():
        return []
    listed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        h, _, rel = line.partition("  ")
        if rel:
            listed[rel.strip()] = h.strip()
    present = {str(p.relative_to(root)).replace(os.sep, "/")
               for p in sorted(root.rglob("*"))
               if p.is_file() and p.name != "SHA256SUMS"}
    problems = [Problem(rel, "checksum", "", 0,
                        "listed in SHA256SUMS but not present")
                for rel in sorted(set(listed) - present)]
    problems += [Problem(rel, "checksum", "", 0,
                         "present but not covered by SHA256SUMS")
                 for rel in sorted(present - set(listed))]
    problems += [Problem(rel, "checksum", "", 0,
                         "content does not match its SHA256SUMS entry")
                 for rel in sorted(present & set(listed))
                 if sha256_file(root / rel) != listed[rel]]
    return problems


def budget_of(specs: list[FileSpec], label: str) -> int:
    """The run budget behind a "file[column]" label, for reporting."""
    fname, _, rest = label.partition("[")
    cname = rest.rstrip("]")
    for s in specs:
        if s.name == fname:
            for c in s.columns:
                if c.name == cname:
                    return c.run_budget
    return 0


def pick_sample(ids: list[str], k: int) -> list[str]:
    """A deterministic, evenly spread sample; k <= 0 means every thesis.

    No RNG reaches any artefact here. A sample is a partial check and every
    line that reports one says so — see coverage_line().
    """
    if k <= 0 or k >= len(ids):
        return list(ids)
    step = len(ids) / k
    return [ids[int(i * step)] for i in range(k)]


def coverage_line(run: dict) -> str:
    """What the verbatim-run test actually covered, in one clause.

    The success message used to read "leak check clean" whatever fraction of the
    release had been aligned. A guard that overstates its own coverage
    manufactures exactly the confidence it exists to earn, so every caller
    reports coverage through this.
    """
    if not run.get("ran"):
        return (f"the corpus-alignment test did not run "
                f"({run.get('reason') or 'reason not recorded'})")
    released = run.get("released_theses", 0)
    aligned = run.get("theses_checked", 0)
    missing = run.get("theses_without_text", 0)
    if not released:
        return "there were no released theses to align (documents.csv is missing or empty)"
    if run.get("complete"):
        return f"all {released:,} released theses aligned against their own text"
    detail = (f" ({missing:,} of them had no extracted text to align against)"
              if missing else "")
    return (f"{aligned:,} of {released:,} released theses aligned "
            f"({run.get('coverage_pct', 0.0):.1f}%); the rest were checked for "
            f"structure and value policy only{detail}")


def leak_check(root: Path, specs: list[FileSpec], gz: bool, sample: int,
               corpus_db: Path, corpus_text: bool
               ) -> tuple[list[Problem], dict]:
    suffix = ".gz" if gz else ""
    docs_path = root / ("data/documents.csv" + suffix)
    ids: list[str] = []
    if docs_path.exists():
        with open_in(docs_path) as fh:
            rd = csv.reader(fh)
            header = next(rd, [])
            if header and header[0] == "id":
                ids = [r[0] for r in rd if r]
    released = sorted(set(ids))
    chosen = pick_sample(released, sample)

    problems, stats, samples = check_structure(root, specs, gz, set(chosen))
    problems += check_hierarchy(root / HIERARCHY_FILE)
    problems += verify_checksums(root)

    run_report: dict = {"ran": False, "theses_checked": 0,
                        "reason": "skipped (--no-corpus-check)" if not corpus_text
                                  else ""}
    if corpus_text:
        run_problems, run_report = check_corpus_runs(samples, specs, chosen, corpus_db)
        run_report["ran"] = True
        problems += run_problems
        if run_report["theses_checked"] == 0:
            problems.append(Problem("data/", "no-corpus", "", 0,
                                    "no thesis text was available to align "
                                    "released values against; the verbatim-run "
                                    "test could not run"))

    # Coverage is part of the result, not a footnote: a clean check over 0.8% of
    # the release is not the same claim as a clean check over all of it.
    aligned = run_report.get("theses_checked", 0)
    run_report["released_theses"] = len(released)
    run_report["sampled_theses"] = len(chosen)
    run_report["coverage_pct"] = 100.0 * aligned / len(released) if released else 0.0
    run_report["complete"] = bool(run_report.get("ran") and released
                                  and aligned >= len(released))

    # The numeric columns are reported as a count rather than one by one, or
    # the manifest becomes a wall of zeroes. Everything else is listed, with
    # whether test 4 aligned it against the corpus.
    kind_of = {(s.name, c.name): c.kind for s in specs for c in s.columns}
    numeric = sum(1 for (f, c), k in kind_of.items()
                  if k in ("int", "float", "bool") and f in stats)
    columns: dict[str, dict] = {}
    for fname, cols in sorted(stats.items()):
        for cname, st in sorted(cols.items()):
            kind = kind_of.get((fname, cname))
            if kind in ("int", "float", "bool"):
                continue
            e = {"kind": kind, "corpus_aligned": kind in RUN_KINDS,
                 "max_chars": st.max_chars, "non_empty": st.n_nonempty}
            if kind == "text":
                e["max_words"] = st.max_words
            if kind == "enum":
                e["distinct"] = len(st.distinct)
            columns.setdefault(fname, {})[cname] = e

    report = {
        "released_theses": len(released),
        "sampled_theses": len(chosen),
        "coverage": coverage_line(run_report),
        "fit_to_publish": bool(run_report.get("complete") and not problems),
        "numeric_columns_checked": numeric,
        "columns": columns,
        "verbatim_run_test": run_report,
        "problems": len(problems),
    }
    return sorted(problems), report


# --------------------------------------------------------------------------- #
# Manifest, checksums and the datasheet
# --------------------------------------------------------------------------- #

def file_entry(root: Path, rel: str, desc: str, rows: int,
               columns: list[dict] | None, note: str = "") -> dict:
    p = root / rel
    e = {"path": rel, "description": desc, "rows": rows,
         "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    if note:
        e["note"] = note
    if columns is not None:
        e["columns"] = columns
    return e


def column_entry(c: Column) -> dict:
    """One column's manifest entry, in reading order rather than sorted order."""
    e = {"name": c.name, "kind": c.kind, "description": c.desc}
    if c.kind == "text":
        e["max_chars"] = c.max_chars
        e["max_words"] = c.max_words
    if c.kind == "enum":
        e["max_distinct"] = c.max_card
    if c.checked:
        # Every word-bearing column carries a run budget, not only the text
        # ones, so the manifest says which columns test 4 covers and how far.
        e["corpus_aligned"] = True
        e["max_verbatim_run"] = c.run_budget
    return e


def write_manifest(root: Path, payload: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_checksums(root: Path) -> int:
    """sha256sum-compatible, sorted by path — the file must not hash itself."""
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "SHA256SUMS":
            continue
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        entries.append((rel, sha256_file(p)))
    entries.sort()
    (root / "SHA256SUMS").write_text(
        "".join(f"{h}  {rel}\n" for rel, h in entries), encoding="utf-8")
    return len(entries)


def write_licence(root: Path) -> None:
    root.joinpath("LICENCE.md").write_text("""\
# Licence

Two different things are licensed here, and they are not licensed the same way.

## The code that produced this dataset

`thesisgraph` is **MIT** licensed. See the LICENSE file in the source repository.

## This dataset

**Recommended: Creative Commons Attribution 4.0 International (CC BY 4.0).**

Why this one:

* **Something explicit is needed.** In the UK a compiled dataset can attract
  *sui generis* database right independently of copyright in its contents. A
  reuser who has to reason about that will not reuse it. An explicit CC BY 4.0
  grant covers copyright and database right in one sentence and removes the
  question.
* **Attribution, not restriction.** These are derived measurements whose value
  is comparability — someone else's numbers should be traceable to the build
  that produced them. Attribution is the only condition that serves that, and
  the only one worth the friction.
* **Not share-alike (CC BY-SA / ODbL).** Share-alike would force the licence
  onto anything combined with this, which in practice keeps it out of exactly
  the aggregations it is most useful in. The goal is reuse, not a licence estate.
* **Not CC0.** Defensible — most of what is here is fact — but provenance
  matters for a dataset whose known limitations are large, and CC BY keeps the
  datasheet attached to the numbers.
* **Not a non-commercial clause.** NC is unenforceable in practice, blocks
  legitimate academic and archival work, and is not the constraint that matters
  here. The constraint that matters is that the underlying *text* is not
  redistributed, and that constraint is met by not shipping it.

## What is NOT licensed here, because it is not here

The theses themselves. They are in copyright, held by their authors, and this
dataset contains no thesis full text, no abstracts and no reference-list
strings. The UK text-and-data-mining exception (s.29A CDPA) permitted the
analysis that produced these numbers; it did not permit redistributing the
works, and nothing in this licence purports to. Each thesis carries its own
licence on its repository landing page, linked from `landing_url` in
`data/documents.csv`.
""", encoding="utf-8")


def write_readme(root: Path, counts: dict, opts_titles: bool, report: dict) -> None:
    rows = "\n".join(f"| `{k}` | {v:,} |" for k, v in sorted(counts.items()))
    root.joinpath("README.md").write_text(f"""\
# thesisgraph — derived dataset

Derived, text-free data from a corpus of {counts.get('data/documents.csv', 0):,}
openly deposited UK doctoral theses: subject labels, citation edges, textual
overlap statistics, writing and method features, and the subject tree.

**No thesis full text, abstract or reference-list string is included.** See
`datasheet.md` for what is here, how it was produced, and every limitation that
matters before you use it. See `LICENCE.md` before you redistribute it.

| file | rows |
|---|---|
{rows}

Thesis titles are {'included in this build' if opts_titles else 'omitted from this build (--no-titles)'};
the datasheet gives the argument on both sides. Join everything on the thesis `id`.

Verify the build with `sha256sum -c SHA256SUMS`. `manifest.json` describes every
column and records what the leak check measured.

**Leak check on this build:** {report.get('coverage', 'not recorded')}.
{'' if report.get('fit_to_publish') else
 'This build is therefore NOT fully verified — re-run `tg release --check <dir> --sample 0` before treating it as such.'}
""", encoding="utf-8")


def datasheet(root: Path, manifest: dict, report: dict, counts: dict,
              specs: list[FileSpec], stats: dict, opts) -> None:
    """A datasheet on the Gebru et al. (2018) headings, with measured numbers."""
    ndocs = counts.get("data/documents.csv", 0)
    run = report.get("verbatim_run_test", {})
    by_col = run.get("by_column", {})
    worst_col, worst = max(((k, v["max_run"]) for k, v in by_col.items()),
                           key=lambda kv: kv[1], default=("", 0))
    cite_run = by_col.get("data/citations.csv[title]", {}).get("max_run", 0)
    run_rows = "\n".join(
        f"| `{k}` | {v.get('kind', '')} | {v['max_run']} | {v.get('budget', 0)} | "
        f"{v['max_abstract_run']} | {v['max_words_per_thesis']} |"
        for k, v in sorted(by_col.items())) or "| _(not run)_ | | | | | |"
    file_rows = "\n".join(
        f"| `{f['path']}` | {f['rows']:,} | {f.get('description','')} |"
        for f in manifest["files"])

    d = manifest["dataset"]
    L = [
        f"# Datasheet — {d['name']}",
        "",
        "Structured on the headings of Gebru et al., *Datasheets for Datasets*",
        "(2018/2021). Every number in this document was measured during the build",
        "that produced these files, not asserted.",
        "",
        "---",
        "",
        "## Motivation",
        "",
        "**Why was the dataset created?** A corpus of UK doctoral theses can be",
        "measured but not republished. The theses are in copyright; the UK",
        "text-and-data-mining exception, s.29A of the Copyright, Designs and",
        "Patents Act 1988, permits computational analysis of works one has lawful",
        "access to and does not permit redistributing them. Everything the",
        "analysis produced *about* those works — counts, shares, labels, features,",
        "bibliographic edges — is not the works, and is publishable. This dataset",
        "is that derived layer, packaged so somebody else can use it: to study",
        "how disciplines write, what doctoral researchers cite, how research",
        "methods diffuse between fields, and how textual overlap is actually",
        "distributed across a real corpus rather than in the anecdotes.",
        "",
        "**Who created it and who funded it?** Built with the `thesisgraph`",
        "toolkit (MIT licensed) as an unfunded personal research project. No",
        "institution commissioned it and no institution has endorsed it.",
        "",
        "## Composition",
        "",
        f"**What do the instances represent?** {ndocs:,} doctoral theses",
        "deposited in White Rose eTheses Online (the shared repository of the",
        "Universities of Leeds, Sheffield and York), one row per thesis in",
        "`data/documents.csv`, plus derived rows keyed on that id.",
        "",
        "| file | rows | what it is |",
        "|---|---:|---|",
        file_rows,
        "",
        "**What is deliberately NOT here, and why.**",
        "",
        "* **Thesis full text.** The whole point. Not included in any form.",
        "* **Abstracts.** They are the authors' own prose, they are long enough to",
        "  be a substantial part of a work, and nothing here needs them.",
        "* **`cite.raw`, the reference-list string each citation was parsed from.**",
        "  This is the interesting call, and it was settled by measurement rather",
        "  than by taste. A raw string is a 300-character window of contiguous text",
        "  lifted from a thesis page; measured on a sample of this corpus, the raw",
        "  strings for a single thesis reproduce 32-71 contiguous words of it, and",
        "  `tg release --self-test` re-demonstrates that by feeding them to the leak",
        "  check and watching it refuse the build. The *parsed* fields",
        f"  reproduce at most {cite_run}, measured over the theses this build",
        "  aligned, because the parser caps a title at eight words and the run breaks",
        "  at almost every row boundary. Eight words of a cited work's",
        "  title is bibliographic fact of the kind Crossref and OpenCitations",
        "  already publish openly; a 45-word verbatim window of a thesis page is",
        "  redistribution of the thesis. So the parsed fields ship and `raw` does",
        "  not, and the leak check enforces the distinction rather than trusting it.",
        "* **Author names (`creator`).** The project's framing is that it never",
        "  makes findings about people. Names are on the public landing page, which",
        "  is linked; they are not in these files, so nothing here can be trivially",
        "  pivoted into a per-person dossier. `same_author` in the overlap file is",
        "  a derived flag, not a name.",
        "* **Direct PDF URLs.** `landing_url` points at the citable repository",
        "  record, which carries the thesis's own licence and embargo status. A",
        "  deep link to the file bypasses both.",
        "* **The example theses inside the subject tree.** `hierarchy.json` carries",
        "  eight sample titles and PDF links per node; the whole block is dropped.",
        "",
        "**Titles: included, and why.** This build "
        + ("**includes**" if opts.titles else "**omits**")
        + " `documents.title`.",
        "Titles are the authors' own words, so the decision is not automatic. They",
        "are included by default because they are already public, in full, on the",
        "repository landing page that `landing_url` links to and in the OAI-PMH",
        "feed the corpus was harvested from — this republishes nothing that is not",
        "already openly available from the source of record. They are also what",
        "makes the dataset usable: without a title, a row cannot be recognised,",
        "linked to a DOI, deduplicated against another corpus, or eyeballed for a",
        "labelling error, and a subject-classification dataset whose labels cannot",
        "be checked is not a research asset. The counter-argument — that a title is",
        "still expressive text — is why it is a flag: `--no-titles` produces an",
        "otherwise identical build with the column dropped, for anyone whose risk",
        "assessment differs from ours.",
        "",
        "**Is any of it confidential or sensitive?** No. Every source record is",
        "openly published by the repository. No personal data beyond what is on a",
        "public bibliographic record is derived, and author names are not carried.",
        "",
        "**Are there errors, noise or redundancies?** Yes, materially — see",
        "*Limitations* below. Treat every label and every count as approximate and",
        "read the confidence and coverage columns that ship alongside them.",
        "",
        "## Collection process",
        "",
        "**How was the data acquired?** Metadata was harvested over OAI-PMH from",
        "White Rose eTheses Online. For each doctoral record with a PDF, the PDF",
        "was downloaded once, text-extracted, and **deleted**: no PDF is retained",
        "anywhere in this project. Crawling respected `robots.txt` and the",
        "published `Crawl-delay: 10`, identified itself with a contact string, and",
        "never ran in parallel against the host.",
        "",
        "**Over what timeframe?** The corpus is a single snapshot of the",
        "repository. Deposits span the range recorded in `year`; a handful of rows",
        "carry implausible years (values below 1900) because the source Dublin Core",
        "date field was malformed, and they are left as deposited rather than",
        "silently corrected.",
        "",
        "**Was anyone told, or asked?** No. The legal basis is s.29A CDPA, which",
        "requires lawful access — the repository is openly accessible — and permits",
        "computational analysis without further permission. Nothing in this dataset",
        "is a communication of the works to the public.",
        "",
        "**Sampling.** Not a sample: every doctoral record in the repository whose",
        "PDF yielded usable text is included. Records that failed extraction, or",
        "that were never fetched, are absent — which biases the corpus mildly",
        "towards born-digital and non-embargoed deposits.",
        "",
        "## Preprocessing / cleaning / labelling",
        "",
        "* **Extraction** is word-level with x/y geometry (`pdfplumber`, `pypdf`",
        "  fallback). Paragraphs are rebuilt across line breaks; a hyphenated",
        "  line-end split is rejoined only when the continuation begins lowercase.",
        "  Sentence splitting is abbreviation-aware.",
        "* **Exclusions.** Reference lists, attributed indented block quotes,",
        "  boilerplate and captions are identified and excluded from the retained",
        "  text; `discourse_features` and `method_features` are computed on the",
        "  retained text only, so they describe the author's prose rather than",
        "  their bibliography.",
        "* **Discipline labels** come from a rule set with an embedding tie-break",
        f"  ({stats['disc_rules']:,} rules, {stats['disc_embed']:,} embedding,",
        f"  {stats['disc_tie']:,} tie-break). `discipline_conf` is the margin of the",
        f"  winning label; {stats['disc_low']:,} rows ({stats['disc_low_pct']:.1f}%)",
        "  score below 0.4 and should be treated as unreliable.",
        "* **Subfields** are unsupervised clusters of title terms within a",
        "  discipline, named by their top terms. They are descriptive labels, not a",
        "  taxonomy anyone has agreed.",
        "* **Overlap statistics** come from winnowed-fingerprint screening",
        "  (Schleimer, Wilkerson & Aiken 2003: 8-gram hashes, window 13, which",
        "  guarantees detection of any shared passage of 20+ words) followed by",
        "  exact alignment on candidate pairs only. All-pairs was not run and is not",
        "  feasible; a pair absent from `overlap_pairs.csv` was screened out, not",
        "  measured at zero.",
        "* **Whitespace** in every text field is flattened to single spaces so that",
        "  one CSV row is one line.",
        "* **One in-band sentinel was unpacked.** The screening code stores a",
        "  pathologically repetitive pair — glossaries, word lists, numeric tables —",
        "  as a *negative* run count, meaning \"enumeration stopped at the 20,000-run",
        "  safety cap\". That is fine inside one codebase and a trap in a published",
        "  CSV, so `runs` here is always a count and the flag is its own column,",
        "  `runs_truncated`. For those rows `runs` and `matched_words` are lower",
        "  bounds and the containment pass (`unique_*`) did not run.",
        "* **The raw corpus is untouched.** This build reads `corpus/` read-only and",
        "  writes nothing to it.",
        "",
        "## Uses",
        "",
        "**What it has been used for.** Studies of writing and discourse features",
        "across disciplines and decades, citation practice, research-method",
        "diffusion between fields, and the base-rate distribution of textual",
        "overlap in a real corpus.",
        "",
        "**What it should not be used for.**",
        "",
        "* **`overlap_pairs.csv` is not a list of suspected misconduct.** It is a",
        "  ranked list of things a human might look at. Shared text has many",
        "  innocent causes and in this corpus most of it is exactly that: on a",
        "  400-thesis pilot, 95.5% of shared runs also appear in an unrelated third",
        "  thesis, and only 1.34% of non-zero pairs survive filtering. Textual",
        "  overlap is not evidence of intent and is not a finding of misconduct;",
        "  only an institution can make that determination. The `verdict` column",
        "  records the triage a filter reached, not a judgement about a person.",
        "* **Not a ranking of institutions, supervisors or authors.** The dataset",
        "  deliberately carries no author names, and per-institution comparisons",
        "  would be confounded by deposit policy, embargo practice and scanning",
        "  quality rather than measuring anything about the research.",
        "* **Not a citation index.** References are parsed heuristically and not",
        "  resolved to DOIs; see the parse coverage in `citation_sources.csv`.",
        "",
        "**Limitations that will bite you.**",
        "",
        "* **Reference parsing is author-date only.** No Vancouver style, and a",
        f"  minority of parenthesised-year entries. Measured on this build:",
        f"  {stats['cite_parsed']:,} entries parsed from {stats['cite_entries']:,}",
        f"  segmented ({stats['cite_rate']:.1f}%), and only",
        f"  {stats['cite_srcs']:,} of {ndocs:,} theses yielded any reference list at",
        "  all. This biases the citation graph towards the social sciences and",
        "  humanities. GROBID or AnyStyle is the proper fix and has not been done.",
        "* **Chapter and heading detection fails on some documents.** Affected rows",
        "  are flagged rather than silently reported: `method_scope` is",
        "  `fallback_wholedoc` where the methodology chapter could not be located,",
        "  and method features for those rows are scored over the whole document.",
        "* **The discourse time series has a digitisation confound at about 2010**,",
        "  where retro-digitised theses give way to born-digital ones. Any trend",
        "  crossing that boundary is partly a trend in OCR quality.",
        "* **Discipline labels are noisy** — see the confidence column above.",
        "* **One repository, one country.** Everything here is White Rose. Nothing",
        "  in it should be read as a fact about UK doctoral research in general.",
        "",
        "## Distribution",
        "",
        f"**Licence.** {DATA_LICENCE} for the data, {CODE_LICENCE} for the code",
        "that produced it. `LICENCE.md` gives the reasoning, including why not CC0,",
        "why not share-alike and why not a non-commercial clause.",
        "",
        "**Third-party rights.** The theses remain in copyright, held by their",
        "authors, and each carries its own licence on the repository landing page",
        "linked from `landing_url`. This dataset grants nothing over them because",
        "it contains none of them.",
        "",
        "**How the no-text claim is enforced.** Not by review. The build assembles",
        "into a staging directory and is only moved into place if four automated",
        "tests pass, and the release cannot be produced at all if they do not:",
        "",
        "1. the set of data files must be exactly the declared set;",
        "2. every column header must equal its declared column list, in order, so a",
        "   schema change cannot silently add a text column;",
        "3. every column declares a kind, and a kind is a pattern a value must",
        "   match in full. Four kinds cannot express a word at all: `int` and",
        "   `bool` are digits, `float` is a number, `url` is one landing-page",
        "   template. The four that can — `id`, `token`, `enum`, `text` — are",
        "   capped on length per value, an `enum` is capped on how many distinct",
        "   values it may ever take, and all four are then handed to test 4. Only",
        "   two columns in the release are declared `text`;",
        "4. every value of every one of those word-bearing columns is",
        "   concatenated per thesis in file order and aligned against that",
        "   thesis's own extracted text, with the same tokeniser the overlap",
        "   analysis uses. The longest contiguous verbatim run must fit the",
        "   column's budget. This is the test the per-value caps cannot do: it",
        "   catches text reassembled across many short rows, and it does not care",
        "   what kind the column declared — 120 single words in a `token` column",
        "   are 120 contiguous words of a thesis if they align like one.",
        "",
        "A fifth test runs on any later `tg release --check`, once the release has",
        "been stamped: every file must still match its `SHA256SUMS` entry and every",
        "file must be covered by one, so a release edited after the fact fails the",
        "same command that verifies a fresh one.",
        "",
        "**What test 4 does not cover.** It aligns a row's values against the",
        "thesis that row is keyed to, and two things sit outside that. Words taken",
        "from one thesis and filed under a row belonging to a different thesis:",
        "catching that would mean aligning every value against all "
        f"{ndocs:,} theses,",
        "which is the all-pairs problem this project does not solve at build time.",
        "And `data/discipline_hierarchy.json`, whose nodes are keyed to no thesis",
        "at all — its labels are capped at the length an `enum` value gets, and the",
        "tree has a bounded number of nodes, but nothing aligns them against",
        "anything. What stands in place of test 4 in both cases is the value policy",
        "on its own: a bounded length on every value, a bounded number of distinct",
        "values in an `enum`, a bounded number of nodes in the tree. That bounds",
        "what a misfiled column or a relabelled node can hold to short fragments",
        "rather than to a document. It is a weaker guarantee than the per-thesis",
        "one, and it is written down here rather than glossed.",
        "",
    ]
    if not run.get("ran"):
        L += [
            "**Test 4 did not run in this build** "
            f"({run.get('reason', 'reason not recorded')}), so the corpus-alignment",
            "measurement below is absent. Tests 1-3 passed. Do not publish a build",
            "in this state: re-run `tg release --check` with the corpus available.",
            "",
        ]
    else:
        L += [
            f"**Coverage of test 4 in this build: {report.get('coverage', '')}.**",
            "",
        ]
        if not run.get("complete"):
            L += [
                "That is a PARTIAL check. It says nothing about the theses it did",
                "not align, and this build should not be described as verified",
                "until `tg release --check <dir> --sample 0` has aligned all of",
                "them. `manifest.json` records the same fact as",
                "`leak_check.fit_to_publish: false`.",
                "",
            ]
        L += [
            "Measured on what was aligned:",
            "",
            "| column | kind | longest verbatim run vs the thesis (words) | budget "
            "| vs its abstract | total released words per thesis |",
            "|---|---|---:|---:|---:|---:|",
            run_rows,
            "",
            "The largest contiguous verbatim run of any thesis appearing anywhere in",
            f"this release is **{worst} words**, in `{worst_col}`.",
        ]
        if worst_col == "data/documents.csv[title]":
            L += [
                "That column holds one deposited title per thesis, and a title",
                "matches the title page it was printed on by definition, so its run",
                "budget is its per-value word cap and what actually constrains the",
                "column is that cap rather than the run test. Over the whole corpus",
                "the longest values in it are not titles as a reader pictures them:",
                "a handful of deposits carry the repository's conformance statement",
                "in the Dublin Core title field — *\"...The results, discussions and",
                "conclusions presented herein are identical to those in the printed",
                "version. This electronic version of the thesis has been edited",
                "solely to ensure conformance with copyright legislation...\"* — and",
                "those deposit statements, which are public landing-page metadata",
                "rather than thesis prose, are what the maximum on this column is",
                "made of.",
            ]
        L += [""]
    L += [
        "**Errata and verification.** `SHA256SUMS` covers every other file;",
        "`manifest.json` records row counts, per-column descriptions and what the",
        "leak check measured. Two builds from the same corpus snapshot are",
        "byte-identical (`tg release --determinism-check`).",
        "",
        "## Maintenance",
        "",
        "**Who maintains it?** The author of the `thesisgraph` repository, via the",
        "repository's issue tracker.",
        "",
        "**Will it be updated?** Only by re-running the pipeline over a fresh",
        "harvest. There is no rolling update and no service level. Because the",
        "build is deterministic, a rebuild that differs is a real change in the",
        "corpus or the code, and the manifest's row counts identify which.",
        "",
        "**How can somebody extend or contribute?** The code is MIT and every",
        "column here is reproducible from it. A better reference parser and a",
        "second repository are the two changes that would improve this dataset most.",
        "",
        "**If a depositor objects.** Ask, and their rows will be removed on the",
        "next build. Nothing here is load-bearing on any single thesis.",
        "",
    ]
    root.joinpath("datasheet.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def own_files(specs: list[FileSpec]) -> set[str]:
    """Every path a release build writes, under either --gzip setting."""
    return ({s.name + suf for s in specs for suf in ("", ".gz")}
            | {HIERARCHY_FILE, "manifest.json", "datasheet.md", "README.md",
               "LICENCE.md", "SHA256SUMS"})


def why_not_writable(path: Path, specs: list[FileSpec], staging: bool) -> str | None:
    """Why `path` must not be deleted and rebuilt, or None if it may be.

    A build replaces its output directory wholesale, and it used to do that to
    whatever it was pointed at: one mistyped `--out` and unrelated files were
    gone, silently and unrecoverably. So the directory has to be absent, empty,
    or recognisably ours — every file in it one this tool writes, and, for a
    finished release rather than a staging directory, carrying the
    manifest.json/SHA256SUMS pair a release is stamped with. Anything else is
    somebody's data and needs an explicit --force.
    """
    if not path.exists():
        return None
    if not path.is_dir():
        return f"{path} exists and is not a directory"
    rel = {str(f.relative_to(path)).replace(os.sep, "/")
           for f in path.rglob("*") if f.is_file()}
    if not rel:
        return None
    strays = sorted(rel - own_files(specs))
    if strays:
        return (f"{path} contains {len(strays)} file(s) no release build wrote "
                f"(e.g. {strays[0]}), so it is not a release directory")
    if not staging and not {"manifest.json", "SHA256SUMS"} <= rel:
        return (f"{path} has release-shaped files but no manifest.json and "
                f"SHA256SUMS, so it is not a finished release")
    return None


def source_paths(args) -> dict:
    c = Path(args.corpus)
    return {"corpus": Path(args.db) if args.db else c / "corpus.db",
            "citations": Path(args.citations) if args.citations else c / "citations.db",
            "discourse": c / "studies" / "discourse_features.csv",
            "method": c / "studies" / "method_features.csv",
            "hierarchy": c / "hierarchy.json"}


def provenance_stats(con: sqlite3.Connection, cit: sqlite3.Connection,
                     keep: set[str]) -> dict:
    """The handful of numbers the datasheet quotes, measured on what shipped."""
    by_method = {m: 0 for m in ("rules", "embedding", "embedding-tiebreak")}
    low = 0
    for did, meth, conf in con.execute(
            "SELECT id,discipline_method,discipline_conf FROM doc WHERE status='ok'"):
        if did not in keep:
            continue
        by_method[meth] = by_method.get(meth, 0) + 1
        if (conf or 0) < 0.4:
            low += 1
    n = max(len(keep), 1)
    entries = parsed = srcs = 0
    for src, e, p in cit.execute("SELECT src,entries,parsed FROM srcstat"):
        if src not in keep:
            continue
        entries += e or 0
        parsed += p or 0
        srcs += 1 if (e or 0) > 0 else 0
    return {"disc_rules": by_method.get("rules", 0),
            "disc_embed": by_method.get("embedding", 0),
            "disc_tie": by_method.get("embedding-tiebreak", 0),
            "disc_low": low, "disc_low_pct": 100.0 * low / n,
            "cite_entries": entries, "cite_parsed": parsed,
            "cite_rate": 100.0 * parsed / max(entries, 1), "cite_srcs": srcs}


def build(args, out_root: Path) -> tuple[int, dict]:
    src = source_paths(args)
    for key in ("corpus", "citations", "discourse", "method", "hierarchy"):
        if not src[key].exists():
            log(f"release: missing input {src[key]}")
            return 1, {}

    specs = build_specs(src, args.titles)
    suffix = ".gz" if args.gzip else ""
    staging = out_root.with_name(out_root.name + ".staging")

    # Both directories are deleted and rebuilt by this function. Ask first.
    for target, is_staging in ((out_root, False), (staging, True)):
        why = why_not_writable(target, specs, is_staging)
        if why and not getattr(args, "force", False):
            log(f"release: refusing to overwrite — {why}")
            log("release: point --out at a new directory, or pass --force to "
                "delete that one and everything under it")
            return 1, {}

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    by_name = {s.name: s for s in specs}
    counts: dict[str, int] = {}
    con, cit = ro(src["corpus"]), ro(src["citations"])
    try:
        log("release: documents")
        n, ids = write_documents(con, staging / ("data/documents.csv" + suffix),
                                 by_name["data/documents.csv"], args.gzip, args.limit)
        counts["data/documents.csv"] = n
        keep = set(ids)

        log("release: citations")
        counts["data/citations.csv"] = write_citations(
            cit, staging / ("data/citations.csv" + suffix),
            by_name["data/citations.csv"], args.gzip, keep)
        counts["data/citation_sources.csv"] = write_citation_sources(
            cit, staging / ("data/citation_sources.csv" + suffix),
            by_name["data/citation_sources.csv"], args.gzip, keep)

        log("release: overlap pairs")
        counts["data/overlap_pairs.csv"] = write_pairs(
            con, staging / ("data/overlap_pairs.csv" + suffix),
            by_name["data/overlap_pairs.csv"], args.gzip, keep)

        log("release: study features")
        counts["data/discourse_features.csv"] = write_study(
            src["discourse"], staging / ("data/discourse_features.csv" + suffix),
            by_name["data/discourse_features.csv"], args.gzip, keep)
        counts["data/method_features.csv"] = write_study(
            src["method"], staging / ("data/method_features.csv" + suffix),
            by_name["data/method_features.csv"], args.gzip, keep)

        hier = write_hierarchy(src["hierarchy"], staging / HIERARCHY_FILE)
        counts[HIERARCHY_FILE] = len(hier["nodes"])
        stats = provenance_stats(con, cit, keep)
    finally:
        con.close()
        cit.close()

    log("release: leak check "
        + ("(every released thesis aligned against the corpus)" if args.sample <= 0
           else f"(a PARTIAL check: {args.sample} theses sampled)"))
    problems, report = leak_check(staging, specs, args.gzip, args.sample,
                                  src["corpus"], not args.no_corpus_check)
    if problems:
        log(f"release: LEAK CHECK FAILED — {len(problems)} problem(s); "
            f"nothing was published")
        for p in problems[:MAX_REPORTED]:
            log(p.line())
        if len(problems) > MAX_REPORTED:
            log(f"  ... and {len(problems) - MAX_REPORTED} more")
        log(f"release: the failed build is left at {staging} for inspection")
        return 1, {"problems": problems, "report": report, "specs": specs}

    version = args.version or (f"docs{counts['data/documents.csv']}"
                               f"-pairs{counts['data/overlap_pairs.csv']}"
                               f"-cites{counts['data/citations.csv']}")
    files = [file_entry(
        staging, s.name + suffix, s.desc, counts[s.name],
        [column_entry(c) for c in s.columns], s.note) for s in specs]
    files.append(file_entry(staging, HIERARCHY_FILE, hier["description"],
                            counts[HIERARCHY_FILE], None))
    manifest = {
        "dataset": {
            "name": "thesisgraph derived dataset",
            "version": version,
            "description": "Text-free derived data from a corpus of openly "
                           "deposited UK doctoral theses.",
            "licence": DATA_LICENCE,
            "code_licence": CODE_LICENCE,
            "generator": "tg_release.py (thesisgraph)",
            "contains_thesis_text": False,
        },
        "source": {
            "repositories": dict(sorted(REPO_NAME.items())),
            "protocol": "OAI-PMH",
            "legal_basis": "s.29A CDPA 1988 (text and data mining) — analysis "
                           "only; no work is redistributed",
            "excluded": sorted(["full text", "abstracts",
                                "reference-list raw strings (cite.raw)",
                                "author names (creator)", "direct PDF URLs",
                                "hierarchy example titles"]),
        },
        "options": {"titles": bool(args.titles), "gzip": bool(args.gzip),
                    "limit": int(args.limit or 0), "sample": int(args.sample)},
        "files": files,
        "leak_check": report,
    }
    write_manifest(staging, manifest)
    datasheet(staging, manifest, report, counts, specs, stats, args)
    write_readme(staging, counts, args.titles, report)
    write_licence(staging)
    nsum = write_checksums(staging)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(out_root)

    total = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    log(f"release: {out_root}  ({nsum + 1} files, {total / 1e6:.1f} MB)")
    for k, v in sorted(counts.items()):
        log(f"  {k:<38} {v:>10,}")
    run = report["verbatim_run_test"]
    worst = max((v["max_run"] for v in run.get("by_column", {}).values()), default=0)
    if run.get("complete"):
        log(f"  leak check: clean — {coverage_line(run)}; the longest verbatim "
            f"run of any thesis anywhere in the release is {worst} words")
    elif run.get("ran"):
        log(f"  leak check: PARTIAL — {coverage_line(run)}")
        log(f"  the longest verbatim run in what WAS aligned is {worst} words; "
            f"this build is not fit to publish until "
            f"`tg release --check {out_root} --sample 0` has aligned the rest")
    else:
        log("  leak check: tests 1-3 passed, but the verbatim-run test was "
            "SKIPPED — this build is not fit to publish until "
            "`tg release --check` has run it")
    return 0, {"manifest": manifest, "report": report, "specs": specs,
               "root": out_root, "counts": counts}


# --------------------------------------------------------------------------- #
# Self-test — prove the leak check fails on data it should reject
# --------------------------------------------------------------------------- #

class _Skip(Exception):
    """A case this corpus snapshot cannot express — reported, never passed off
    as a case the guard caught."""


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        return next(rd), [r for r in rd]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _poison_reference_strings(root: Path, ctx: dict) -> None:
    """Put the real reference-list strings back into the parsed title column."""
    p = root / "data/citations.csv"
    header, rows = _read_csv(p)
    ti = header.index("title")
    raws = ctx["raws"]
    n = 0
    for r in rows:
        key = (r[0], r[1], r[2])
        if key in raws:
            r[ti] = flat(raws[key])
            n += 1
    assert n, "self-test could not find a reference string to inject"
    _write_csv(p, header, rows)


def _poison_abstract(root: Path, ctx: dict) -> None:
    p = root / "data/documents.csv"
    header, rows = _read_csv(p)
    ti = header.index("title")
    for r in rows:
        if r[0] == ctx["abstract_doc"]:
            r[ti] = flat(ctx["abstract"])
            _write_csv(p, header, rows)
            return
    raise AssertionError("self-test could not find the abstract's document")


def _poison_chunked_passage(root: Path, ctx: dict) -> None:
    """A body passage cut into per-row chunks that each pass every value cap.

    This is the case the per-value budgets cannot catch: eight words is exactly
    what a parsed citation title is allowed to be, so nothing about any single
    row is out of policy. Only aligning the concatenation against the thesis
    finds the 40-word passage.
    """
    p = root / "data/citations.csv"
    header, rows = _read_csv(p)
    ti, si = header.index("title"), header.index("src")
    chunks = ctx["chunks"]
    hits = [i for i, r in enumerate(rows) if r[si] == ctx["passage_doc"]]
    assert len(hits) >= len(chunks), "self-test needs more citation rows"
    for i, chunk in zip(hits, chunks):
        rows[i][ti] = chunk
    _write_csv(p, header, rows)


def _poison_token_column(root: Path, ctx: dict) -> None:
    """The same body passage, one word per row, through a `token` column.

    Nothing about any single value here is out of policy: a surname IS one
    whitespace-free word, and every one of these is. Until test 4 covered every
    word-bearing kind rather than only the `text` ones, a contiguous passage of
    a thesis rode out through this column and the check called it clean.
    """
    p = root / "data/citations.csv"
    header, rows = _read_csv(p)
    ni, si = header.index("surname"), header.index("src")
    words = ctx["passage"]
    hits = [i for i, r in enumerate(rows) if r[si] == ctx["passage_doc"]]
    assert len(hits) >= len(words), "self-test needs more citation rows"
    for i, w in zip(hits, words):
        rows[i][ni] = w
    _write_csv(p, header, rows)


def _poison_enum_column(root: Path, ctx: dict) -> None:
    """A sentence of the thesis inside a controlled vocabulary.

    One value, inside the enum's length cap and adding one to its cardinality —
    so tests 1-3 see a subfield label like any other.
    """
    p = root / "data/documents.csv"
    header, rows = _read_csv(p)
    fi = header.index("subfield")
    value = " ".join(ctx["passage"][:ENUM_VALUE_WORDS])
    for r in rows:
        if r[0] == ctx["passage_doc"]:
            r[fi] = value[:ENUM_VALUE_CHARS].strip()
            _write_csv(p, header, rows)
            return
    raise AssertionError("self-test could not find the passage's document")


def _poison_id_column(root: Path, ctx: dict) -> None:
    r"""Words dotted into an identifier that still matches the id pattern.

    `oai:[a-z0-9.-]+:\d+` has a word-shaped middle, so an `id` column that is
    not the row's own key — `b` in the pair file — can spell out prose and pass
    the value policy unchanged.
    """
    if not ctx.get("pair_doc"):
        raise _Skip("no thesis in this build has two or more overlap pairs")
    p = root / "data/overlap_pairs.csv"
    header, rows = _read_csv(p)
    ai, bi = header.index("a"), header.index("b")
    words = list(ctx["pair_words"])
    for r in rows:
        if r[ai] != ctx["pair_doc"] or not words:
            continue
        chunk, n = [], len("oai::1")
        while words and n + len(words[0]) + 1 <= ID_CHARS:
            n += len(words[0]) + 1
            chunk.append(words.pop(0))
        r[bi] = "oai:" + ".".join(chunk) + ":1"
    _write_csv(p, header, rows)


def _poison_numeric(root: Path, ctx: dict) -> None:
    p = root / "data/documents.csv"
    header, rows = _read_csv(p)
    wi = header.index("words")
    rows[0][wi] = "the analysis proceeded in three stages as set out below"
    _write_csv(p, header, rows)


def _poison_extra_column(root: Path, ctx: dict) -> None:
    p = root / "data/citations.csv"
    header, rows = _read_csv(p)
    header.append("raw")
    for r in rows:
        r.append("")
    _write_csv(p, header, rows)


def _poison_stray_file(root: Path, ctx: dict) -> None:
    (root / "data/extra_notes.csv").write_text("id,note\n1,hello\n", encoding="utf-8")


def _poison_tamper(root: Path, ctx: dict) -> None:
    """Edit a published file without rebuilding — the only case left unstamped."""
    p = root / "data/citation_sources.csv"
    p.write_text(p.read_text(encoding="utf-8") + "oai:etheses.whiterose.ac.uk:1,0,0,0\n",
                 encoding="utf-8")


# (label, how to break it, kinds the check MUST report, re-stamp SHA256SUMS?)
# Re-stamping makes a case look like a BUILD that produced bad data rather than
# a tampered release, so the leak tests are what fires. The last case does not
# re-stamp, because tampering is exactly what it is testing.
CASES = [
    ("reference-list strings in a title column", _poison_reference_strings,
     {"value", "verbatim-run"}, True),
    ("a deposited abstract in the title column", _poison_abstract, {"value"}, True),
    ("a body passage cut into per-row chunks that each pass every cap",
     _poison_chunked_passage, {"verbatim-run"}, True),
    ("a body passage one word per row through a `token` column",
     _poison_token_column, {"verbatim-run"}, True),
    ("a sentence of a thesis inside an `enum` column",
     _poison_enum_column, {"verbatim-run"}, True),
    ("a body passage dotted into an `id` column",
     _poison_id_column, {"verbatim-run"}, True),
    ("prose in a numeric column", _poison_numeric, {"value"}, True),
    ("an undeclared column", _poison_extra_column, {"forbidden", "header"}, True),
    ("a stray file in data/", _poison_stray_file, {"stray-file"}, True),
    ("a published file edited after the build", _poison_tamper, {"checksum"}, False),
]


def _selftest_context(root: Path, corpus_db: Path, citations_db: Path) -> dict:
    """Real bad data, taken from the real corpus — nothing synthetic."""
    sys.path.insert(0, str(HERE))
    import run_analysis as R
    import harvest_corpus as H

    header, rows = _read_csv(root / "data/citations.csv")
    si = header.index("src")
    per_src: dict[str, int] = {}
    for r in rows:
        per_src[r[si]] = per_src.get(r[si], 0) + 1

    def doc_tokens(did: str) -> list[str]:
        """The thesis's own extracted text, tokenised as the analysis tokenises it."""
        store = H.store_path(did)
        if not store.exists():
            return []
        with gzip.open(store, "rt", encoding="utf-8") as fh:
            sents = json.load(fh)["sentences"]
        toks: list[str] = []
        for r in sents:
            toks.extend(R.match_tokens(H.unpack_sentence(r)[0]))
        return toks

    cit = ro(citations_db)
    con = ro(corpus_db)
    try:
        # A reference-list string for a thesis that is in this build.
        raws: dict[tuple[str, str, str], str] = {}
        for src in sorted(per_src)[:5]:
            for s, y, t, raw in cit.execute(
                    "SELECT surname,year,title,raw FROM cite WHERE src=? "
                    "ORDER BY surname, year, title LIMIT 30", (src,)):
                if raw:
                    raws[(src, s, str(y))] = raw

        # A thesis with an abstract, and one with enough body text to cut up.
        # Forty rows of one word each is the token-column case; the same forty
        # words in eight-word chunks is the text-column one.
        abstract_doc = abstract = ""
        passage_doc, passage, chunks = "", [], []
        for did in sorted(per_src, key=lambda d: (-per_src[d], d)):
            if not abstract:
                row = con.execute("SELECT abstract FROM doc WHERE id=?", (did,)).fetchone()
                if row and row[0] and len(row[0].split()) > 100:
                    abstract_doc, abstract = did, row[0]
            if not passage and per_src[did] >= 40:
                toks = doc_tokens(did)
                if len(toks) > 6000:
                    passage_doc = did
                    passage = toks[5000:5040]
                    chunks = [" ".join(passage[i:i + 8]) for i in range(0, 40, 8)]
            if abstract and passage:
                break

        # A thesis with more than one pair row, for the `id`-column case: `b` is
        # the one identifier column that is not its own row's key.
        per_a: dict[str, int] = {}
        ph, prows = _read_csv(root / "data/overlap_pairs.csv")
        pa = ph.index("a")
        for r in prows:
            per_a[r[pa]] = per_a.get(r[pa], 0) + 1
        pair_doc, pair_words = "", []
        for did in sorted(per_a, key=lambda d: (-per_a[d], d)):
            if per_a[did] < 2:
                break
            toks = doc_tokens(did)
            if len(toks) > 6000:
                pair_doc, pair_words = did, toks[5000:5000 + 14 * per_a[did]]
                break
    finally:
        cit.close()
        con.close()
    assert raws and abstract and chunks, "self-test could not assemble real bad data"
    return {"raws": raws, "abstract_doc": abstract_doc, "abstract": abstract,
            "passage_doc": passage_doc, "chunks": chunks, "passage": passage,
            "pair_doc": pair_doc, "pair_words": pair_words}


def self_test(args) -> int:
    from types import SimpleNamespace
    src = source_paths(args)
    tmp = Path(tempfile.mkdtemp(prefix="tg_release_selftest_"))
    log(f"self-test: workspace {tmp}")
    try:
        opts = SimpleNamespace(**vars(args))
        opts.limit, opts.gzip, opts.version = args.selftest_docs, False, "selftest"
        opts.no_corpus_check = False
        clean = tmp / "clean"
        rc, info = build(opts, clean)
        if rc != 0:
            log("self-test: FAILED — the clean build did not pass its own leak check")
            return 1
        specs = info["specs"]
        ok = True
        log("")
        log(f"  {'case':<62} {'result':<8} problems")
        log(f"  {'clean release':<62} {'PASS':<8} 0")

        ctx = _selftest_context(clean, src["corpus"], src["citations"])
        skipped = 0
        for label, poison, expect, restamp in CASES:
            variant = tmp / ("bad_" + re.sub(r"[^a-z0-9]+", "_", label.lower())[:40])
            shutil.copytree(clean, variant)
            try:
                poison(variant, ctx)
            except _Skip as why:
                # Not a pass. This corpus snapshot cannot express the case, and
                # the summary says so rather than counting it as caught.
                log(f"  {label:<62} {'N/A':<8} {why}")
                skipped += 1
                continue
            if restamp:
                write_checksums(variant)
            problems, _ = leak_check(variant, specs, False, args.sample,
                                     src["corpus"], True)
            kinds = {p.kind for p in problems}
            good = bool(problems) and expect <= kinds
            ok &= good
            log(f"  {label:<62} {('CAUGHT' if good else 'MISSED'):<8} "
                f"{len(problems)} {sorted(kinds)}")
            if not good:
                log(f"      expected kinds {sorted(expect)}, saw {sorted(kinds)}")
            for p in problems[:2]:
                log(f"      e.g. {p.line().strip()}")
        log("")
        log(f"self-test: {'PASS' if ok else 'FAIL'} — "
            f"{len(CASES) - skipped} of {len(CASES)} bad releases exercised, "
            f"{'all' if ok else 'not all'} rejected"
            + (f"; {skipped} not expressible in this build" if skipped else ""))
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def determinism_check(args) -> int:
    from types import SimpleNamespace
    tmp = Path(tempfile.mkdtemp(prefix="tg_release_determinism_"))
    try:
        hashes = []
        for i in (1, 2):
            opts = SimpleNamespace(**vars(args))
            opts.limit = args.limit or args.selftest_docs
            root = tmp / f"build{i}"
            rc, _ = build(opts, root)
            if rc != 0:
                return rc
            hashes.append({str(p.relative_to(root)): sha256_file(p)
                           for p in sorted(root.rglob("*")) if p.is_file()})
        a, b = hashes
        diff = sorted(set(a) ^ set(b)) + sorted(
            k for k in set(a) & set(b) if a[k] != b[k])
        if diff:
            log(f"release: NOT deterministic — {len(diff)} file(s) differ:")
            for k in diff[:20]:
                log(f"  {k}")
            return 1
        log(f"release: deterministic — {len(a)} files identical across two builds")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def nonneg(v: str) -> int:
    """A count argument. `--limit -5` used to build and stamp a one-row release."""
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{v!r} is not an integer")
    if n < 0:
        raise argparse.ArgumentTypeError(f"{v!r} is negative; expected 0 or more")
    return n


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=HERE / "out" / "release",
                   help="release directory to build (default: %(default)s)")
    p.add_argument("--corpus", type=Path, default=HERE / "corpus",
                   help="corpus directory to read (default: %(default)s)")
    p.add_argument("--db", type=Path, default=None,
                   help="corpus database (default: <corpus>/corpus.db)")
    p.add_argument("--citations", type=Path, default=None,
                   help="citations database (default: <corpus>/citations.db)")
    p.add_argument("--no-titles", dest="titles", action="store_false",
                   help="omit thesis titles; see the datasheet for the argument "
                        "on both sides")
    p.add_argument("--gzip", action="store_true",
                   help="write .csv.gz instead of .csv (deterministic: mtime=0)")
    p.add_argument("--limit", type=nonneg, default=0,
                   help="release only the first N theses (a smoke-test build)")
    p.add_argument("--sample", type=nonneg, default=DEFAULT_SAMPLE,
                   help="align only N theses against the corpus in the leak "
                        "check, instead of every released thesis (the default). "
                        "That is a PARTIAL check and the log, the manifest and "
                        "the datasheet all say so and name the fraction")
    p.add_argument("--force", action="store_true",
                   help="delete the --out directory even if it is not an empty "
                        "directory or a previous release")
    p.add_argument("--no-corpus-check", action="store_true",
                   help="skip the verbatim-run test (it needs corpus/text)")
    p.add_argument("--version", default="",
                   help="version string for the manifest (default: derived from "
                        "row counts, so it stays deterministic)")
    p.add_argument("--check", type=Path, default=None,
                   help="run the leak check over an already-built release and exit")
    p.add_argument("--self-test", action="store_true",
                   help="build a small release, then feed the leak check "
                        "deliberately bad releases and assert it rejects them")
    p.add_argument("--selftest-docs", type=nonneg, default=120,
                   help="theses in a self-test or determinism build (default: %(default)s)")
    p.add_argument("--determinism-check", action="store_true",
                   help="build twice and diff byte for byte")
    p.add_argument("--keep", action="store_true",
                   help="keep the self-test workspace for inspection")


def run(args: argparse.Namespace) -> int:
    if args.self_test:
        return self_test(args)
    if args.determinism_check:
        return determinism_check(args)

    if args.check:
        root = Path(args.check)
        mpath = root / "manifest.json"
        titles = args.titles
        gz = args.gzip
        if mpath.exists():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            titles = bool(m.get("options", {}).get("titles", titles))
            gz = bool(m.get("options", {}).get("gzip", gz))
        specs = build_specs(source_paths(args), titles)
        problems, report = leak_check(root, specs, gz, args.sample,
                                      source_paths(args)["corpus"],
                                      not args.no_corpus_check)
        run_r = report["verbatim_run_test"]
        log(f"release: checked {root} — {coverage_line(run_r)}")
        if problems:
            log(f"release: LEAK CHECK FAILED — {len(problems)} problem(s)")
            for p in problems[:MAX_REPORTED]:
                log(p.line())
            if len(problems) > MAX_REPORTED:
                log(f"  ... and {len(problems) - MAX_REPORTED} more")
            return 1
        for cname, v in sorted(run_r.get("by_column", {}).items()):
            log(f"  {cname:<38} {v.get('kind', ''):<5} longest run "
                f"{v['max_run']:>4} words (budget {budget_of(specs, cname)}), "
                f"vs abstract {v['max_abstract_run']:>4}, {v['theses']:,} theses")
        worst = max((v["max_run"] for v in run_r.get("by_column", {}).values()),
                    default=0)
        if run_r.get("complete"):
            log(f"release: leak check clean — {coverage_line(run_r)}; the longest "
                f"verbatim run anywhere in it is {worst} words")
        elif run_r.get("ran"):
            log(f"release: leak check found nothing wrong, but it was a PARTIAL "
                f"check — {coverage_line(run_r)}")
            log(f"release: the longest verbatim run in the part that was aligned "
                f"is {worst} words; re-run with --sample 0 before treating this "
                f"release as verified")
        else:
            log(f"release: tests 1-3 passed, but {coverage_line(run_r)} — nothing "
                f"here says this release is free of thesis text")
        return 0

    rc, _ = build(args, Path(args.out))
    return rc


if __name__ == "__main__":
    # tg.py turns a raised exception into one line on stderr. This module has to
    # work without tg.py, so it does the same thing here rather than printing a
    # traceback at somebody who pointed --db at the wrong file.
    ap = argparse.ArgumentParser(description=HELP)
    add_args(ap)
    try:
        raise SystemExit(run(ap.parse_args()))
    except KeyboardInterrupt:
        log("release: interrupted")
        raise SystemExit(130)
    except Exception as exc:                # noqa: BLE001 — a message, not a stack
        if os.environ.get("TG_TRACEBACK"):
            raise
        log(f"release: failed — {type(exc).__name__}: {exc}")
        log("release: re-run with TG_TRACEBACK=1 for the full traceback")
        raise SystemExit(1)
