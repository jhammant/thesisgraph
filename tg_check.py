#!/usr/bin/env python3
"""
tg_check.py — run YOUR OWN thesis against the corpus, before anybody else does.

Every other capability here looks at other people's documents in bulk. This one
looks at exactly one document: yours. That inverts who the tool is for. A
research group running a corpus screen wants a ranked list of pairs to
investigate; a candidate three weeks from submission wants to know whether the
paragraph they wrote up from two-year-old notes is actually somebody else's
paragraph, while there is still time to do something about it.

The value is in the filters, not in the matching. Matching is easy, and on its
own it is worse than useless for this audience: 20% of same-discipline pairs in
this corpus share an eight-word run, and 95.5% of all shared runs also occur in
an unrelated third thesis — which makes them field-standard phrasing and
evidence of nothing at all. A tool that hands an anxious PhD student a raw
overlap number will frighten them with a number that is almost always normal.

So this applies the same filters the corpus screen applies — PDF
character-spacing artefacts, reference lists, attributed block quotes,
boilerplate and ethics templates, quoted-on-both-sides passages, and above all
the third-document test — and it prints the corpus baseline beside every
headline number, so "you share a run with 40 theses" can be read as what it
usually is.

Each stage reuses the module that already implements it:

    extract   run_analysis.extract_pages / build_document — the same extractor,
              exclusions and tokeniser that built the corpus, so your PDF is
              measured on exactly the terms the corpus was measured on
    screen    screen_corpus.winnow against every stored fingerprint file;
              winnowing guarantees any shared passage of 20+ words is found
    exact     run_analysis.verbatim_runs (and optionally the near-verbatim and
              embedding passes) against candidate documents rebuilt from the
              stored corpus text
    filter    rescore's prose test, harvest_corpus's quotation test, and the
              third-document test from harvest_corpus.verify
    report    out/selfcheck/<stem>/{report.md, matches.csv, summary.json}

WHAT THIS CANNOT SEE. The third-document test reads corpus/tokens/*.npy, and
screen_corpus.doc_tokens built those arrays from each thesis's RETAINED text —
every sentence that thesis excluded (references, declarations, ethics and other
boilerplate) was dropped before the array was written, and that is about one
word in seven of the stored corpus. So the filter cannot recognise boilerplate
as common if the other theses excluded it. Where your own document's boilerplate
was excluded too this costs nothing, because excluded text is never matched; it
bites when the extractor did not classify a page of yours as boilerplate, and
then no third document can be found to explain it. The report says so where the
third-document test is described. The proper fix is a token index built over
full text, which is a corpus-build change and not something this command can do.

COPYRIGHT. report.md quotes matched fragments from your document and from other
people's theses, so it is a local working document and must not be published;
it says so in its own header. matches.csv carries locations and headings but no
body text unless you ask for it with --csv-text, which makes it as sensitive as
the report. summary.json is counts and catalogue metadata only. The report
states this for each file rather than only for itself.

COST. A full run over the 24,656-document corpus takes minutes, not seconds,
and peak RSS ranged from 0.3 GB to 2.4 GB across the four PDFs in pdfs/ — see
the measurements beside DEFAULT_MAX_CANDIDATES. This is not a tool that runs
comfortably beside a full IDE on an 8 GB laptop.

Determinism: no wall clock, no RNG, every collection sorted before it is
written. Two runs over the same PDF produce byte-identical artefacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "check"
HELP = "screen your own thesis PDF against the corpus, with the baseline beside it"

# Screening. The corpus screen uses >= 3 shared fingerprints because it is
# looking at 303 million pairs and cannot afford recall; one document against
# the corpus can, so the default here is 1 and the corpus threshold is kept
# only for the like-for-like baseline comparison.
DEFAULT_MIN_FP = 1
BASELINE_MIN_FP = 3
# No cap by default. The headline count — "this many theses share a run with
# you" — is the first number the reader sees, and a cap silently turns it into
# the cap's own value: the tool would be reporting a CLI default as a fact about
# somebody's thesis. Scoring one more candidate costs about 0.15 s, so the
# honest number is affordable; a cap stays available for smoke tests and for a
# pathological neighbourhood, and every count in the report is then labelled a
# lower bound. Measured: control_alhumaidan_2025.pdf, 322 pages, 269 documents
# in its neighbourhood — 86 s wall and 306 MB peak RSS with no cap, against 75 s
# capped at 200. Peak RSS is dominated by the largest candidate rebuilt, not by
# the number of them: later_later_2015.pdf, 162 candidates, peaks at 2.4 GB.
DEFAULT_MAX_CANDIDATES = 0   # 0 = score every document over the threshold
DF_MAX_FRAC = 0.004          # same document-frequency pruning as screen_corpus

# Prose test (rescore.py): a "run" of spaced-out characters is an extraction
# artefact, not shared writing. rescore only ever applied this to substantial
# pairs (matched_words >= 800, max_run >= 40, long_runs >= 5), so its "at least
# 5 tokens of 4+ characters" was calibrated on long runs; applied unchanged to
# an eight-word run it rejects ordinary English — "the main purpose of this
# study was to" has four. The constant is therefore pro-rated below the length
# it was calibrated at and never exceeds rescore's own value.
MAX_SINGLE_FRAC = 0.30
MIN_LONG_TOKENS = 5
PROSE_CALIBRATION_WORDS = 20     # == run_analysis.LONG_RUN_WORDS

# Published pilot figures, quoted in README.md. Everything else in the baseline
# table is recomputed from corpus.db at run time.
PILOT_SHARE_8_WORD = 0.20
PILOT_SHARE_20_WORD = 0.059
PILOT_SIZE = 400

# Share of the stored corpus text that each thesis's own exclusions removed
# before corpus/tokens/*.npy was written, and which the third-document test
# therefore cannot see. Measured over a deterministic 1-in-200 sample (124
# documents, 8.1M words): 1,160,970 excluded of 8,148,392 stored. Quoted in the
# report as a limitation, not used in any calculation.
EXCLUDED_TEXT_SHARE = 0.142

CONTEXT_WORDS = 22           # display words shown either side of a match
FRAGMENT_MAX_WORDS = 70      # long runs are elided in the middle
REPORT_NOT_PUBLISHABLE = (
    "**Local working document — do not publish, circulate or attach it to "
    "anything.** It quotes text from your thesis and from other people's "
    "theses. Theses are in copyright: the UK text-and-data-mining exception "
    "(s.29A CDPA) permits this analysis, not redistribution of what it quotes."
)


# --------------------------------------------------------------------------- #
# Per-candidate result
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    doc_id: str
    title: str
    creator: str
    year: int
    publisher: str
    discipline: str
    landing: str
    shared_fp: int
    runs: list = field(default_factory=list)        # run_analysis.Run, a=theirs b=yours
    decisions: dict = field(default_factory=dict)   # run index -> (kept, reason, third)
    same_author: int = 0
    title_sim: int = 0
    verdict: str = "fully_explained"
    records: list = field(default_factory=list)     # run_analysis.MatchRecord
    near_pairs: int | None = None
    embed_pairs: int | None = None

    @property
    def kept_runs(self) -> list[int]:
        return sorted(k for k, d in self.decisions.items() if d[0])

    def counts(self) -> dict:
        c = {"total": len(self.runs), "artefact": 0, "common": 0, "quoted": 0,
             "kept": 0}
        for kept, reason, _ in self.decisions.values():
            c["kept" if kept else reason] += 1
        return c


# --------------------------------------------------------------------------- #
# Small helpers that mirror logic living in modules whose globals we cannot fill
# --------------------------------------------------------------------------- #

_QUOTE_RE = re.compile(r"[\"'‘’“”]")


def covered_sentences(doc, t0: int, t1: int) -> set[int]:
    """Sentence indices a token span touches (barrier tokens carry -1)."""
    ts = doc.tok_sent
    return {ts[k] for k in range(t0, min(t1, len(ts))) if ts[k] >= 0}


def run_is_quoted(doc, t0: int, t1: int, attrib_window: int) -> bool:
    """True if the span sits inside quoted or attributed text.

    This mirrors harvest_corpus._run_is_quoted exactly. It is restated rather
    than imported because that function reads corpus-wide module globals keyed
    by document id, and a PDF sitting on somebody's laptop has no such id.
    """
    idxs = covered_sentences(doc, t0, t1)
    if not idxs:
        return False
    sents = doc.sentences
    lo, hi = min(idxs), max(idxs)
    for j in range(max(0, lo - attrib_window), min(len(sents), hi + attrib_window + 1)):
        text = sents[j].text
        if j in idxs and len(_QUOTE_RE.findall(text)) >= 2:
            return True
        if _ATTRIBUTION_RE.search(text):
            return True
    return False


_ATTRIBUTION_RE = None       # bound to run_analysis.ATTRIBUTION_RE inside run()


def your_attribution(R, doc, run) -> str:
    """Does YOUR side of this match already say where the words came from?

    The single most useful thing to put next to a fragment. A passage in
    quotation marks with an author and year beside it is a quotation, and the
    reader can stop reading there; one with neither is the one worth a minute.
    It is a description of the text, not a judgement of it: an author-date
    pattern near the passage is not proof the passage is properly attributed,
    and its absence is not proof of anything at all.
    """
    idxs = covered_sentences(doc, run.b0, run.b1)
    if not idxs:
        return "location unknown"
    lo, hi = min(idxs), max(idxs)
    inside = [doc.sentences[j].text for j in range(lo, hi + 1)]
    window = [doc.sentences[j].text for j in
              range(max(0, lo - R.ATTRIB_WINDOW),
                    min(len(doc.sentences), hi + R.ATTRIB_WINDOW + 1))]
    quoted = any(len(_QUOTE_RE.findall(t)) >= 2 for t in inside)
    attributed = any(_ATTRIBUTION_RE.search(t) for t in window)
    if quoted and attributed:
        return "in quotation marks and attributed on your side"
    if attributed:
        return "attributed nearby on your side, not in quotation marks"
    if quoted:
        return "in quotation marks on your side, no source named nearby"
    return "no quotation marks and no source named nearby on your side"


def is_prose(tokens: list[str]) -> bool:
    """rescore.py's test: does this run read like prose, or like an artefact?"""
    n = len(tokens)
    if not n:
        return False
    if sum(1 for t in tokens if len(t) == 1) / n > MAX_SINGLE_FRAC:
        return False
    need = min(MIN_LONG_TOKENS,
               max(2, round(n * MIN_LONG_TOKENS / PROSE_CALIBRATION_WORDS)))
    return sum(1 for t in tokens if len(t) >= 4) >= need


def landing_url(doc_id: str, landing: str, pdf_url: str) -> str:
    """A page a human can open. The stored landing column is often a file URL."""
    m = re.match(r"^oai:([^:]+):(\d+)$", doc_id or "")
    if m:
        return f"https://{m.group(1)}/id/eprint/{m.group(2)}/"
    m = re.match(r"^(https?://[^/]+/id/eprint/\d+)/", pdf_url or "")
    if m:
        return m.group(1) + "/"
    return landing or pdf_url or ""


def surname_of(creator: str) -> str:
    if not creator:
        return ""
    first = creator.split(";")[0].strip()
    sur = first.split(",")[0].strip() if "," in first else (first.split() or [""])[-1]
    return re.sub(r"[^A-Za-z'’\-]", "", sur)


def percentile_of(sorted_values, x: float) -> float:
    """Share of the corpus at or below x, as a percentage."""
    import bisect
    if not sorted_values:
        return 0.0
    return 100.0 * bisect.bisect_right(sorted_values, x) / len(sorted_values)


def md_escape(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*") \
            .replace("_", "\\_").replace("`", "\\`").replace("[", "\\[")


def md_code(s: str) -> str:
    """Inside a code span markdown escapes are literal, so only strip hazards."""
    return re.sub(r"[`|\s]+", " ", s or "").strip()


def fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def plural(n: int, word: str, suffix: str = "s") -> str:
    return f"{fmt_int(n)} {word}{'' if n == 1 else suffix}"


def ordinal(n: float) -> str:
    k = int(round(n))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("pdf", type=Path, help="the PDF to check (your own thesis)")
    p.add_argument("--out", type=Path, default=HERE / "out" / "selfcheck",
                   help="output directory (default: %(default)s)")
    p.add_argument("--db", type=Path, default=HERE / "corpus" / "corpus.db",
                   help="corpus database (default: %(default)s)")
    p.add_argument("--title", default="",
                   help="your thesis title, for the duplicate-deposit check "
                        "(default: the PDF's own metadata)")
    p.add_argument("--author", default="",
                   help="your name as 'Surname, Forename', for the same-author "
                        "check (default: the PDF's own metadata)")
    p.add_argument("--min-shared-fp", type=int, default=DEFAULT_MIN_FP,
                   help="shared fingerprints needed to score a document exactly "
                        "(default: %(default)s; the corpus screen uses 3)")
    p.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
                   help="score exactly only the N documents sharing the most "
                        "fingerprints, 0 for no cap (default: %(default)s). A cap "
                        "makes every count in the report a lower bound, and the "
                        "report says so on the rows it affects")
    p.add_argument("--common-scan", choices=("corpus", "neighbourhood", "off"),
                   default="corpus",
                   help="documents scanned for the third-document test: the whole "
                        "corpus (exact, ~1 min), only documents sharing a "
                        "fingerprint with you (exact for runs of 20+ words, which "
                        "is what winnowing guarantees), or off (default: %(default)s)")
    p.add_argument("--near-top", type=int, default=3,
                   help="candidates also scored with the near-verbatim pass, 0 to "
                        "skip it (default: %(default)s)")
    p.add_argument("--embeddings", action="store_true",
                   help="also run the embedding pass on those candidates; degrades "
                        "to 'not run' if no offline model is available")
    p.add_argument("--max-docs", type=int, default=25,
                   help="documents detailed in report.md (default: %(default)s)")
    p.add_argument("--max-fragments", type=int, default=6,
                   help="matched fragments quoted per document (default: %(default)s)")
    p.add_argument("--csv-text", action="store_true",
                   help="add the matched text to matches.csv (makes the CSV "
                        "unpublishable too)")
    p.add_argument("--screen-limit", type=int, default=None,
                   help="screen only the first N corpus documents (smoke test)")
    p.add_argument("--no-cache", action="store_true",
                   help="re-extract the PDF instead of reusing the cached extraction")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def extract_query(R, pdf: Path, cache_dir: Path, use_cache: bool):
    """Extract the PDF with the project's own extraction path, unmodified."""
    sha = R.sha256_file(pdf)
    spec = R.DocSpec(
        key="SELF", short="SELF", label=f"your document ({pdf.name})",
        surname="", surname_regex=r"(?!x)x", year=0, filename=pdf.name,
        landing_url="", pdf_url="", expected_pages=None, role="focal-later")
    pages = R.load_or_extract(spec, pdf, sha, use_cache=use_cache,
                              cache_dir=cache_dir)
    R.detect_printed_folios(pages)
    doc = R.build_document(spec, pages)
    return doc, sha


def pdf_metadata(pdf: Path) -> tuple[str, str]:
    """(title, author) as the PDF declares them — deterministic, from the file."""
    try:
        from pypdf import PdfReader
        md = PdfReader(str(pdf)).metadata or {}
        return (str(md.get("/Title", "") or "").strip(),
                str(md.get("/Author", "") or "").strip())
    except Exception:
        return "", ""


def rebuild_document(R, H, doc_id: str, meta: dict):
    """Rebuild a corpus document from stored text — no PDF, no re-extraction.

    The store holds exactly what build_document produced when the corpus was
    harvested (text, page, printed folio, bucket, heading, exclusions), so the
    reconstruction is the same object the matching passes were written against.
    """
    with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    sur = surname_of(meta.get("creator") or "")
    spec = R.DocSpec(
        key=meta["key"], short=meta["key"], label=meta.get("label", ""),
        surname=sur, surname_regex=(rf"\b{re.escape(sur)}\b" if sur else r"(?!x)x"),
        year=int(meta.get("year") or 0), filename="", landing_url="", pdf_url="",
        expected_pages=None, role="control")
    sentences = []
    for i, row in enumerate(payload["sentences"]):
        text, page, printed, bucket, heading, excl = H.unpack_sentence(row)
        sentences.append(R.Sentence(
            idx=i, text=text, pdf_page_start=page, pdf_page_end=page,
            printed_page=printed or None, heading=heading, chapter="",
            bucket=bucket, x0=0.0, indented=False,
            n_tokens=len(R.match_tokens(text)),
            exclusions=tuple(sorted(e for e in (excl or "").split(";") if e))))
    doc = R.Document(
        spec=spec, pages=[], sentences=sentences,
        modal_size=float(payload.get("modal_size") or 0.0),
        modal_x0=float(payload.get("modal_x0") or 0.0),
        no_text_pages=[], pypdf_pages=[], folio_band="",
        folio_pages=int(payload.get("folio_pages") or 0))
    R._build_token_streams(doc)
    return doc


def screen(S, np, doc_ids: list[str], query_fps, log) -> tuple[dict, dict]:
    """Shared winnowed fingerprints per corpus document.

    Returns (shared after document-frequency pruning, shared before it). The
    pruning is screen_corpus's: a fingerprint occurring in more than 0.4% of
    documents is field-wide phrasing and counting it only inflates candidates.
    The unpruned counts define the neighbourhood used by the third-document
    test, where recall rather than precision is what matters.
    """
    q = np.sort(query_fps)
    hits: list[tuple[str, object]] = []
    for k, d in enumerate(doc_ids, 1):
        try:
            a = np.load(S.fp_path(d))
        except Exception:
            continue
        if not a.size:
            continue
        i = np.searchsorted(q, a)
        i[i >= q.size] = 0
        m = q[i] == a
        if m.any():
            hits.append((d, i[m]))
        if k % 5000 == 0:
            log(f"    screened {k:,}/{len(doc_ids):,}")
    df = np.zeros(q.size, dtype=np.int32)
    for _, idx in hits:
        df[idx] += 1
    df_max = max(3, int(len(doc_ids) * DF_MAX_FRAC))
    keep = df <= df_max
    pruned = {d: int(keep[idx].sum()) for d, idx in hits}
    raw = {d: int(idx.size) for d, idx in hits}
    return {d: n for d, n in pruned.items() if n}, raw


def common_run_scan(S, np, scan_ids: list[str], spans: list[tuple[int, int]],
                    query_grams, exclude: set[str], log) -> dict:
    """Which unrelated documents contain a whole matched run?

    This is the filter that matters. harvest_corpus.verify calls a run "common"
    when some third document contains every one of its n-grams, and 95.5% of
    all shared runs in this corpus are exactly that: a standard definition, a
    quoted instrument, a sentence the field has written the same way for thirty
    years. Such a run cannot be evidence of transfer between two documents,
    because a third document evidently got it from somewhere else.

    Runs are keyed by their span in YOUR token stream, so a span matched by
    several candidates is scanned once.
    """
    out: dict[tuple[int, int], list[str]] = {s: [] for s in spans}
    if not spans:
        return out
    n = S.NGRAM
    spans = [s for s in spans if query_grams[s[0]:s[1] - n + 1].size]
    if not spans:
        return out
    per_span = [query_grams[s0:s1 - n + 1] for s0, s1 in spans]
    wanted = np.unique(np.concatenate(per_span))
    gram_idx = np.concatenate(
        [np.searchsorted(wanted, g) for g in per_span]).astype(np.int64)
    offsets = np.zeros(len(spans), dtype=np.int64)
    acc = 0
    for k, g in enumerate(per_span):
        offsets[k] = acc
        acc += g.size
    present = np.zeros(wanted.size, dtype=bool)
    log(f"    {wanted.size:,} distinct n-grams across {len(spans):,} matched spans; "
        f"scanning {len(scan_ids):,} documents")
    for k, d in enumerate(scan_ids, 1):
        if d in exclude:
            continue
        try:
            G = S.gram_hashes(np.load(S.tok_path(d)))
        except Exception:
            continue
        if not G.size:
            continue
        i = np.searchsorted(wanted, G)
        i[i >= wanted.size] = 0
        m = wanted[i] == G
        if not m.any():
            continue
        sel = i[m]
        present[sel] = True
        whole = np.logical_and.reduceat(present[gram_idx], offsets)
        present[sel] = False
        for j in np.flatnonzero(whole).tolist():
            if len(out[spans[j]]) < 3:
                out[spans[j]].append(d)
        if k % 5000 == 0:
            log(f"    scanned {k:,}/{len(scan_ids):,}")
    for v in out.values():
        v.sort()
    return out


def corpus_baseline(con, discipline: str) -> dict:
    """Recompute the context numbers from corpus.db rather than quoting them.

    Two distributions matter to somebody reading their own result: how many
    other theses a thesis typically shares a run with at all, and how many it
    shares a LONG run with. Both are computed over every document with
    status='ok', so a thesis sharing nothing counts as a zero and the medians
    are not conditioned on having been matched.
    """
    docs = {r[0]: (r[1] or "") for r in
            con.execute("SELECT id, discipline FROM doc WHERE status='ok'")}
    any_run: dict[str, int] = {}
    long_run: dict[str, int] = {}
    for a, b, long_runs in con.execute("SELECT a, b, long_runs FROM pair"):
        for x in (a, b):
            if x in docs:
                any_run[x] = any_run.get(x, 0) + 1
                if long_runs:
                    long_run[x] = long_run.get(x, 0) + 1

    def dist(counter, subset):
        vals = sorted(counter.get(d, 0) for d in subset)
        if not vals:
            return {"n": 0}
        def pc(p):
            return vals[min(len(vals) - 1, int(p / 100 * len(vals)))]
        return {"n": len(vals), "median": pc(50), "p75": pc(75), "p90": pc(90),
                "p95": pc(95), "p99": pc(99), "max": vals[-1],
                "mean": round(sum(vals) / len(vals), 2), "_sorted": vals}

    all_ids = sorted(docs)
    same = sorted(d for d, disc in docs.items() if discipline and disc == discipline)
    row = con.execute(
        "SELECT COUNT(*), SUM(runs), SUM(common_runs), SUM(quoted_runs), "
        "SUM(unique_runs) FROM pair WHERE verdict IS NOT NULL").fetchone()
    verdicts = {v: n for v, n in con.execute(
        "SELECT verdict, COUNT(*) FROM pair WHERE verdict IS NOT NULL "
        "GROUP BY verdict")}
    verified, runs, common, quoted, unique = (row or (0, 0, 0, 0, 0))
    return {
        "corpus_docs": len(docs),
        "pairs_with_a_shared_run": con.execute(
            "SELECT COUNT(*) FROM pair").fetchone()[0],
        "any_run_partners": dist(any_run, all_ids),
        "long_run_partners": dist(long_run, all_ids),
        "same_discipline": discipline,
        "same_discipline_docs": len(same),
        "any_run_partners_same_discipline": dist(any_run, same) if same else {"n": 0},
        "long_run_partners_same_discipline": dist(long_run, same) if same else {"n": 0},
        "verified_pairs": verified or 0,
        "verified_runs": runs or 0,
        "verified_common_runs": common or 0,
        "verified_quoted_runs": quoted or 0,
        "verified_unique_runs": unique or 0,
        "common_share": (common or 0) / max(runs or 0, 1),
        "verdicts": {k: verdicts.get(k, 0) for k in sorted(verdicts)},
        "residual_share": verdicts.get("residual", 0) / max(verified or 0, 1),
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def sentence_disp_range(doc, s0: int, s1: int) -> tuple[int, int]:
    lo = hi = None
    for i in range(max(0, s0), min(s1, len(doc.sentences) - 1) + 1):
        r = doc.sent_tok_range.get(i)
        if not r:
            continue
        d0, d1 = doc.tok_disp[r[0]], doc.tok_disp[r[1] - 1]
        lo = d0 if lo is None else min(lo, d0)
        hi = d1 if hi is None else max(hi, d1)
    return (lo if lo is not None else 0, hi if hi is not None else 0)


def fragment_in_context(doc, rec) -> str:
    """The matched run, marked, inside the sentence(s) of yours that carry it."""
    d0, d1 = rec.b_disp_start, rec.b_disp_end
    if d0 < 0 or d1 < d0:
        return md_escape(rec.b_text)
    slo, shi = sentence_disp_range(doc, rec.b_sent_start, rec.b_sent_end)
    lo = max(slo, d0 - CONTEXT_WORDS, 0)
    hi = min(shi, d1 + CONTEXT_WORDS, len(doc.disp_words) - 1)
    words = doc.disp_words
    matched = words[d0:d1 + 1]
    if len(matched) > FRAGMENT_MAX_WORDS:
        half = FRAGMENT_MAX_WORDS // 2
        matched = (matched[:half] + [f"[... {len(matched) - FRAGMENT_MAX_WORDS} "
                                     f"more words ...]"] + matched[-half:])
    before = md_escape(" ".join(words[lo:d0]))
    after = md_escape(" ".join(words[d1 + 1:hi + 1]))
    core = md_escape(" ".join(matched))
    parts = []
    if lo > slo:
        parts.append("...")
    if before:
        parts.append(before)
    parts.append(f"**{core}**")
    if after:
        parts.append(after)
    if hi < shi:
        parts.append("...")
    return " ".join(parts)


def write_csv(path: Path, cands: list[Candidate], with_text: bool) -> int:
    fields = [
        "other_doc_id", "other_title", "other_creator", "other_year",
        "other_landing", "other_discipline", "document_verdict",
        "kept", "filter_reason", "also_appears_in",
        "pass", "match_id", "run_length_words", "similarity",
        "your_pdf_page_start", "your_pdf_page_end", "your_printed_page",
        "your_chapter_bucket", "your_heading", "your_sent_start", "your_sent_end",
        "other_pdf_page_start", "other_pdf_page_end", "other_printed_page",
        "other_chapter_bucket", "other_heading", "other_sent_start",
        "other_sent_end", "citation_class", "direction_flag",
        "direction_categories", "error_carryover", "error_carryover_detail",
    ]
    if with_text:
        fields.append("matched_text")
    rows = []
    for c in cands:
        for rec in c.records:
            k = rec_run_index(rec)
            kept, reason, third = c.decisions.get(k, (1, "kept", []))
            if rec.pass_name != "verbatim":
                kept, reason, third = 1, "not_filtered", []
            row = [
                c.doc_id, c.title, c.creator, c.year, c.landing, c.discipline,
                c.verdict, int(bool(kept)), reason, ";".join(third),
                rec.pass_name, rec.match_id, rec.run_length_words,
                f"{rec.similarity:.4f}",
                rec.b_pdf_page_start, rec.b_pdf_page_end, rec.b_printed_page,
                rec.b_bucket, rec.b_heading, rec.b_sent_start, rec.b_sent_end,
                rec.a_pdf_page_start, rec.a_pdf_page_end, rec.a_printed_page,
                rec.a_bucket, rec.a_heading, rec.a_sent_start, rec.a_sent_end,
                rec.citation_class, rec.direction_flag, rec.direction_categories,
                rec.error_carryover, rec.error_carryover_detail,
            ]
            if with_text:
                row.append(rec.b_text)
            rows.append(row)
    # Positions, in `fields` order: 0 other_doc_id, 10 pass, 12 run_length_words,
    # 19 your_sent_start, 11 match_id — a total order, so the file is byte-stable.
    rows.sort(key=lambda r: (r[0], r[10], -int(r[12]), int(r[19]), r[11]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(fields)
        w.writerows(rows)
    return len(rows)


def rec_run_index(rec) -> int:
    """MatchRecord ids are '<pair>:<pass initial><6-digit run index>'."""
    try:
        return int(rec.match_id.rsplit(":", 1)[1][1:])
    except (IndexError, ValueError):
        return -1


# --------------------------------------------------------------------------- #
# report.md
# --------------------------------------------------------------------------- #

PREAMBLE = """## What this is, and what it is not

This is a screen, not a verdict. It measures one thing: passages that your
document and a deposited thesis have in common, word for word. It cannot detect
fabricated data, ghostwriting or a purchased thesis — those leave no textual
trace, and no amount of this will surface them.

Shared text has many innocent causes, and in this corpus nearly all of it is
exactly that: a standard definition, a published instrument quoted properly, a
methods sentence the field has worded the same way for thirty years, an ethics
paragraph the university supplied, a reference list, or a PDF that spaces its
characters so widely the extractor reads every letter as a word. Each of those
has a filter here, and the table below shows what each of them removed — with
one gap, stated in full where the third-document test is described below.

Above all, a run of words that also appears in an unrelated third thesis is not
evidence of anything, because a third author evidently reached it without
copying either of you. **{common_share:.1f}% of all shared runs in this corpus
are exactly that**, and they are removed before anything is shown to you.

**Textual overlap is not evidence of intent, and this report is not a finding of
misconduct.** Only your institution can make that determination. What survives
here is a list of things to look at — usually so that you can check a quotation
mark is where you meant it to be.
"""

HOW_TO_READ = """## How to read the numbers

Every headline number has the corpus beside it, because the number alone means
nothing. The comparison corpus is {corpus_docs:,} openly deposited UK doctoral
theses. In it, the *median* thesis shares an eight-word run with
{median_any} other theses and a twenty-word run with {median_long}; a pilot over
{pilot_size} theses found {pilot8:.0f}% of same-discipline pairs sharing an
eight-word run and {pilot20:.1f}% sharing one of twenty or more. Sharing runs is
the normal condition of a thesis, not an unusual one.

Of the pairs this project screened and then verified, {residual_share:.2f}%
survived filtering — {residual:,} out of {verified:,}. If your document is in
that small group it still does not mean anything has gone wrong; it means the
overlap has no innocent explanation that this tool can find on its own, and it
is worth your own five minutes.
"""


def build_report(ctx: dict) -> str:
    b = ctx["baseline"]
    q = ctx["query"]
    L: list[str] = []
    add = L.append

    add(f"# Self-check — {md_escape(q['name'])}")
    add("")
    add(f"> {REPORT_NOT_PUBLISHABLE}")
    add("")
    add(PREAMBLE.format(common_share=100.0 * b["common_share"]))
    add(HOW_TO_READ.format(
        corpus_docs=b["corpus_docs"],
        median_any=b["any_run_partners"].get("median", 0),
        median_long=b["long_run_partners"].get("median", 0),
        pilot_size=PILOT_SIZE, pilot8=100 * PILOT_SHARE_8_WORD,
        pilot20=100 * PILOT_SHARE_20_WORD,
        residual_share=100 * b["residual_share"],
        residual=b["verdicts"].get("residual", 0),
        verified=b["verified_pairs"]))

    # ---- your document -----------------------------------------------------
    add("## Your document")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| file | `{md_code(q['name'])}` |")
    add(f"| sha256 | `{md_code(q['sha256'])}` |")
    from_pdf = " *(from the PDF's own metadata)*"
    add(f"| title (as given) | {md_escape(q['title']) or '—'}"
        f"{from_pdf if q['title_from_pdf'] else ''} |")
    add(f"| author (as given) | {md_escape(q['author']) or '—'}"
        f"{from_pdf if q['author_from_pdf'] else ''} |")
    add(f"| pages | {fmt_int(q['pages'])} |")
    add(f"| sentences | {fmt_int(q['sentences'])} "
        f"({fmt_int(q['sentences_included'])} retained after exclusions) |")
    add(f"| words | {fmt_int(q['words'])} "
        f"({fmt_int(q['words_included'])} retained) |")
    add(f"| subject (rules only) | {md_escape(q['discipline'])} "
        f"(confidence {q['discipline_conf']:.2f}, {md_escape(q['discipline_method'])}) |")
    add("")
    if q["author_from_pdf"] or q["title_from_pdf"]:
        add("> **You did not give a name or a title, so the PDF's own metadata "
            "was used.** That name is used for one thing: deciding whether a "
            "deposited document is your own earlier work, which is then set "
            "aside rather than ranked. PDF metadata is often a producer's "
            "default — `Owner`, `user`, a template's author — and a wrong name "
            "either sets aside a document that is not yours, or fails to set "
            "aside one that is. Pass `--author 'Surname, Forename'` and "
            "`--title` to settle it.")
        add("")
    if q["no_text_pages"]:
        add(f"> **{len(q['no_text_pages'])} of {q['pages']} pages yielded no "
            f"extractable text.** Those pages were not checked at all. If your "
            f"PDF is a scan, this whole report is measuring only the part of "
            f"your thesis the extractor could read.")
        add("")
    if q["excluded"]:
        add("Excluded from matching, and counted separately (this is what stops "
            "a reference list or a declaration page being reported as overlap):")
        add("")
        add("| exclusion | sentences |")
        add("|---|---:|")
        for k, n in q["excluded"]:
            add(f"| {k} | {fmt_int(n)} |")
        add("")
    if q["self_doc"]:
        s = q["self_doc"]
        add(f"> **This exact file is already in the corpus** as "
            f"[{md_escape(s['title'])[:110]}]({s['landing']}) "
            f"({md_escape(s['creator'])}, {s['year']}) — the sha256 matches "
            f"byte for byte. It has been excluded from the comparison, along "
            f"with its own text in the third-document test; otherwise you would "
            f"be shown your whole thesis matching itself.")
        add("")

    # ---- headline ----------------------------------------------------------
    h = ctx["headline"]
    cap = ctx["cap"]

    def bound(n) -> str:
        """A capped run measures part of the neighbourhood, so say so, here."""
        if not cap["capped"]:
            return fmt_int(n)
        return (f"at least {fmt_int(n)}" if n else
                f"none in the {cap['limit']:,} scanned")

    add("## The headline numbers, with the corpus beside them")
    add("")
    add("| measure | your document | this corpus |")
    add("|---|---:|---|")
    ap = b["any_run_partners"]
    lp = b["long_run_partners"]
    add(f"| theses that share any 8+ word run with you | "
        f"{bound(h['partners_any_fp'])} | — |")
    add(f"| the same count at the corpus screen's own threshold, which is what "
        f"the baseline column measures | {bound(h['partners_baseline_fp'])} | "
        f"median {ap.get('median', 0)} · 75th pct {ap.get('p75', 0)} · "
        f"90th pct {ap.get('p90', 0)} · 95th pct {ap.get('p95', 0)} |")
    add(f"| — where that puts you | {ordinal(h['pct_any'])} percentile"
        f"{' or higher' if cap['capped'] else ''} | "
        f"a thesis at the 50th shares with {ap.get('median', 0)} |")
    add(f"| theses sharing a 20+ word run with you, same threshold | "
        f"{bound(h['partners_long'])} | "
        f"median {lp.get('median', 0)} · 75th pct {lp.get('p75', 0)} · "
        f"90th pct {lp.get('p90', 0)} · 95th pct {lp.get('p95', 0)} |")
    add(f"| — where that puts you | {ordinal(h['pct_long'])} percentile"
        f"{' or higher' if cap['capped'] else ''} | "
        f"a thesis at the 50th shares with {lp.get('median', 0)} |")
    if b["any_run_partners_same_discipline"].get("n"):
        sd = b["any_run_partners_same_discipline"]
        add(f"| the same, against your own subject only "
            f"({md_escape(b['same_discipline'])}, "
            f"{fmt_int(b['same_discipline_docs'])} theses) | "
            f"{bound(h['partners_baseline_fp'])} | median {sd.get('median', 0)} · "
            f"90th pct {sd.get('p90', 0)} |")
    add(f"| shared runs found, before filtering | {bound(h['runs_total'])} | — |")
    add(f"| of those, also in an unrelated third thesis | "
        f"{bound(h['runs_common'])} ({h['common_pct']:.1f}%) | "
        f"{100 * b['common_share']:.1f}% |")
    add(f"| documents with a surviving run of 20+ words | "
        f"{bound(h['docs_residual'])} | "
        f"{100 * b['residual_share']:.2f}% of verified pairs reach this state |")
    add(f"| documents with only short surviving runs (8–19 words) | "
        f"{bound(h['docs_short'])} | "
        f"{100 * h['baseline_short_share']:.2f}% of verified pairs |")
    add("")
    if cap["capped"]:
        add(f"> **Every number in that table is truncated, and none of them is a "
            f"measurement of your whole document.** `--max-candidates "
            f"{cap['limit']:,}` stopped the exact scoring at the "
            f"{cap['limit']:,} documents that share the most fingerprints with "
            f"you, out of {cap['eligible']:,} that passed the screening "
            f"threshold ({cap['neighbourhood']:,} share a fingerprint at all). "
            f"The {cap['eligible'] - cap['limit']:,} documents below the cut "
            f"were never scored, so every count here is a lower bound and both "
            f"percentiles are floors. Re-run with `--max-candidates 0` for the "
            f"true figures.")
        add("")
    else:
        add(f"No cap was applied: all "
            f"{plural(cap['eligible'], 'document')} over the screening "
            f"threshold {'was' if cap['eligible'] == 1 else 'were'} scored "
            f"exactly, so these are counts of your whole document rather than "
            f"of a sample of it.")
        add("")
    add("The first two rows differ because this tool screens one document and "
        "can afford to look at every candidate, while the corpus screen had 303 "
        "million pairs to get through and only kept documents sharing three or "
        "more fingerprints. The baseline column was measured at that threshold, "
        "so the second row is the one to compare.")
    add("")
    add(h["verdict_sentence"])
    add("")

    # ---- your own work, already deposited ----------------------------------
    dups = ctx["duplicates"]
    if dups:
        add("### One thing to read before the rest")
        add("")
        add(f"{len(dups)} deposited document(s) match yours so heavily, or carry "
            f"your name or your title, that they look like **your own work "
            f"already in the repository** — an earlier deposit, a corrected "
            f"version, or a thesis you published a chapter from. Overlap with "
            f"your own work is expected and is not what this tool is for, so "
            f"these are set aside rather than ranked.")
        add("")
        add("| document | year | shared runs | longest run | why |")
        add("|---|---:|---:|---:|---|")
        for c in dups:
            why = []
            if c["same_author"]:
                why.append("author matches the name you gave")
            if c["title_sim"] >= 95:
                why.append(f"title {c['title_sim']}% similar to yours")
            add(f"| [{md_escape(c['title'])[:80]}]({c['landing']}) | {c['year']} | "
                f"{c['counts']['total']} | {c['max_run']} words | "
                f"{'; '.join(why) or 'matched on identity'} |")
        add("")
        add("**If any of those is not yours, that changes what this report "
            "means** — re-run with `--author` and `--title` set to your own, so "
            "the check stops treating it as you.")
        add("")

    # ---- filters -----------------------------------------------------------
    f = ctx["filters"]
    add("## What each filter removed")
    add("")
    add("| filter | runs removed | why it exists |")
    add("|---|---:|---|")
    add(f"| excluded before matching | — | reference lists, attributed block "
        f"quotes, captions, declarations, ethics and other boilerplate are "
        f"dropped from both documents at extraction, on both sides |")
    add(f"| character-spacing artefacts and other non-prose runs | "
        f"{bound(f['artefact'])} | some PDF producers space letters so widely "
        f"that every letter extracts as a word; two such theses share tens of "
        f"thousands of meaningless runs |")
    add(f"| also in an unrelated third thesis | {bound(f['common'])} | the "
        f"single most important filter: a third author reached the same words "
        f"independently, so the run is field-standard phrasing |")
    add(f"| quoted or attributed on both sides | {bound(f['quoted'])} | both "
        f"documents are quoting the same source, and both say so |")
    add(f"| **surviving** | **{bound(f['kept'])}** | worth your own eyes |")
    add("")
    # The tally counts runs from every candidate, duplicates included, while the
    # verdict above counts only documents that are not your own deposit. Saying
    # so here is what stops the two contradicting each other.
    if ctx["duplicates"]:
        dk = ctx["dup_kept"]
        add(f"{fmt_int(dk)} of those {fmt_int(f['kept'])} surviving runs are "
            f"shared with the {plural(len(ctx['duplicates']), 'document')} set "
            f"aside above as your own work already deposited, which is why "
            f"they are not counted as a finding; "
            + (f"the remaining {fmt_int(f['kept'] - dk)} are with other "
               f"people's theses." if f["kept"] > dk else
               "no surviving run is shared with anybody else's thesis."))
        add("")
    if ctx["scan_mode"] == "off":
        add("The third-document test was switched off, so nothing below has "
            "been checked against a third document and the list is not "
            "filtered.")
    elif not ctx["spans"]:
        add("No shared run was found at all, so the third-document test had "
            "nothing to scan.")
    else:
        add(f"The third-document test scanned {fmt_int(ctx['scan_docs'])} "
            f"documents ({md_escape(ctx['scan_mode'])}). "
            + ("Winnowing guarantees that any shared passage of 20 or more "
               "words is found, so for long runs this test is exact; below 20 "
               "words it is a lower bound and some field-standard phrasing may "
               "still be shown to you." if ctx["scan_mode"] == "neighbourhood"
               else ("Every document in the corpus was scanned, so within the "
                     "text it can see this test is exact at every run length."
                     if ctx["scan_docs"] >= b["corpus_docs"] else
                     "That is a screening subset, not the whole corpus, so this "
                     "test is a lower bound: some field-standard phrasing will "
                     "still be shown to you.")))
        add("")
        add(f"> **Limitation: this test can only see each corpus thesis's "
            f"retained text.** The index it reads was built with every "
            f"sentence those theses excluded — references, declarations, "
            f"acknowledgements, ethics and other boilerplate — already removed, "
            f"and that is about {100 * EXCLUDED_TEXT_SHARE:.0f}% of the stored "
            f"corpus. So the one filter that matters most is blind to exactly "
            f"the text that is most often shared innocently. It costs you "
            f"nothing where your own boilerplate was excluded too, because "
            f"excluded text is never matched on either side; it bites where the "
            f"extractor did not recognise a page of yours as boilerplate — an "
            f"acknowledgements page it read as ordinary prose, say. Such a "
            f"passage can be word-for-word standard and still appear below as "
            f"surviving, because no third document is visible to explain it. "
            f"Read the page and heading printed beside each fragment with that "
            f"in mind. The proper fix is a token index built over full text, "
            f"which is a change to how the corpus is built, not something this "
            f"command can do.")
    add("")

    # ---- surviving ---------------------------------------------------------
    add("## What survived")
    add("")
    surv = ctx["surviving"]
    short = ctx["short_only"]
    # Where the limitation above actually bites: a passage of yours the
    # extractor could not classify is often front matter — acknowledgements, a
    # declaration — and every other thesis's version of it was removed from the
    # index before the third-document test could find it there. Emitted once,
    # above whichever list is the first to contain such a row.
    hint = ("Some rows below sit in a part of your document the extractor could "
            "not place in a chapter (`unclassified`, often front matter). Read "
            "those against the limitation stated above: if the passage is "
            "acknowledgements or declaration boilerplate, other theses' "
            "versions of the same words were removed from the index before the "
            "third-document test could see them, so it had no way to explain "
            "the overlap and shows it to you instead."
            if any(fr["your_bucket"] == "unclassified"
                   for c in surv + short for fr in c["fragments"]) else "")
    if not surv and not short and dups:
        add(f"**Nothing survived the filters except your own deposited work.** "
            f"The {plural(len(dups), 'document')} listed above as your own — "
            f"where {plural(ctx['dup_kept'], 'shared run')} did survive the "
            f"filters — {'is' if len(dups) == 1 else 'are'} the whole of what "
            f"was found. Every passage you share with anybody else's thesis is "
            f"explained by one of the causes above, most often that the same "
            f"words also appear in an unrelated third thesis. If that really "
            f"is your own deposit there is nothing here to act on; if it is "
            f"not, re-run with `--author` and `--title` set to your own, "
            f"because it is being kept out of this section on the strength of "
            f"that name.")
        add("")
    elif not surv and not short and not ctx["spans"]:
        add("**Your document shares no run of eight words or more with any "
            "thesis in this corpus.** Nothing was found, so nothing had to be "
            "filtered. That is unusual — the median thesis shares a run with "
            "several others — and it is worth checking that the pages you "
            "expected to be read were read: the table above says how many "
            "words came out of the PDF and how many were retained.")
        add("")
    elif not surv and not short:
        add("**Nothing survived the filters. This is the ordinary result.** "
            "Every passage your document shares with this corpus is explained "
            "by one of the causes above — most often that the same words also "
            "appear in an unrelated third thesis. There is nothing here to act "
            "on.")
        add("")
    elif not surv:
        add("**No document shares a surviving run of 20 words or more with "
            "yours.** That is the threshold this project treats as a long run, "
            "and it is the ordinary result. What is left is short overlaps "
            "only, listed below.")
        add("")
    else:
        add(f"{len(surv)} document(s) share a run of 20 words or more with you "
            f"that no filter explains, ranked by the amount of surviving text. "
            f"This is a list of things to look at. Nothing here is a finding "
            f"about anybody, and the innocent explanations do not stop at the "
            f"four this tool can test for.")
        add("")
        if hint:
            add(hint)
            add("")
            hint = ""
        for i, c in enumerate(surv[:ctx['max_docs']], 1):
            cc = c["counts"]
            add(f"### {i}. {md_escape(c['title'])[:150]}")
            add("")
            add(f"*{md_escape(c['creator'])}, {c['year']}"
                + (f", {md_escape(c['publisher'])}" if c["publisher"] else "")
                + f"* — [{c['landing']}]({c['landing']})")
            add("")
            add(f"- verdict: **{c['verdict']}**")
            # Runs and passages are not the same count: one sentence of yours
            # matched at four places in their thesis is four runs and one thing
            # to read. Both are printed so neither can mislead.
            add(f"- surviving runs: **{cc['kept']}** of {cc['total']}, covering "
                f"**{plural(c['distinct_passages'], 'distinct passage')}** of "
                f"your text (longest {c['max_kept_run']} words, "
                f"{c['kept_words']} words of yours in total, "
                f"{100 * c['kept_share']:.3f}% of your retained text)")
            add(f"- removed: {cc['common']} as also-in-a-third-thesis, "
                f"{cc['quoted']} as quoted on both sides, "
                f"{cc['artefact']} as extraction artefacts")
            if c["citation"]:
                add(f"- do you name {md_escape(c['surname']) or 'this author'} "
                    f"within {c['citation_window']} sentences of the shared "
                    f"text? " + ", ".join(f"{k}: {v}" for k, v in c["citation"]))
            if c["near_pairs"] is not None:
                add(f"- near-verbatim pass (rapidfuzz >= {c['fuzz_cutoff']}): "
                    f"{c['near_pairs']} sentence pairs"
                    + (f"; embedding pass (cosine >= {c['embed_cutoff']}): "
                       f"{c['embed_pairs']} pairs"
                       if c["embed_pairs"] is not None else ""))
            add("")
            for frag in c["fragments"]:
                add(f"**{frag['words']} words** — your page {frag['your_page']}"
                    + (f" (printed {frag['your_printed']})" if frag["your_printed"] else "")
                    + f", {frag['your_bucket']}"
                    + (f", under “{md_escape(frag['your_heading'])[:70]}”"
                       if frag["your_heading"] else "")
                    + f" · their page {frag['their_page']}, {frag['their_bucket']}"
                    + f" · **{frag['your_attribution']}**")
                add("")
                add(f"> {frag['context']}")
                add("")
            rest = c["distinct_passages"] - len(c["fragments"])
            if rest > 0:
                add(f"*({plural(rest, 'further passage')} of yours survived; "
                    f"{'it is' if rest == 1 else 'they are'} in "
                    f"`matches.csv`.)*")
                add("")
        if len(surv) > ctx["max_docs"]:
            add(f"*({plural(len(surv) - ctx['max_docs'], 'further document')} "
                f"in `matches.csv` rather than listed here.)*")
            add("")

    if short:
        add("### Short overlaps only (8–19 words)")
        add("")
        add(f"{len(short)} document(s) share runs with you that survived the "
            f"filters but are all shorter than 20 words. An eight-word "
            f"coincidence between two theses in the same field is unremarkable "
            f"— the corpus screen files these as `short_only` and does not "
            f"treat them as anything to act on. They are listed for "
            f"completeness.")
        add("")
        if hint:
            add(hint)
            add("")
            hint = ""
        add("| document | year | longest | where in your thesis | the shared words |")
        add("|---|---:|---:|---|---|")
        for c in short[:ctx["max_docs"]]:
            f0 = c["fragments"][0] if c["fragments"] else None
            where = (f"p. {f0['your_page']}, {f0['your_bucket']}" if f0 else "—")
            add(f"| [{md_escape(c['title'])[:70]}]({c['landing']}) | {c['year']} | "
                f"{c['max_kept_run']} | {where} | "
                f"{md_escape(f0['plain']) if f0 else '—'} |")
        add("")
        if len(short) > ctx["max_docs"]:
            add(f"*({plural(len(short) - ctx['max_docs'], 'further document')} "
                f"in `matches.csv` rather than listed here.)*")
            add("")

    # ---- set aside ---------------------------------------------------------
    aside = ctx["set_aside"]
    if aside:
        add("## What was found and set aside")
        add("")
        add("These documents shared text with yours, and every shared run was "
            "explained. They are listed so the filtering is visible rather than "
            "silent.")
        add("")
        add("| document | year | shared runs | longest | why it was set aside |")
        add("|---|---:|---:|---:|---|")
        for c in aside[:ctx["max_docs"]]:
            cc = c["counts"]
            why = []
            if c["verdict"] == "same_author_or_duplicate":
                why.append("same author or duplicate deposit")
            if cc["common"]:
                why.append(f"{cc['common']} also in a third thesis")
            if cc["quoted"]:
                why.append(f"{cc['quoted']} quoted on both sides")
            if cc["artefact"]:
                why.append(f"{cc['artefact']} extraction artefacts")
            add(f"| [{md_escape(c['title'])[:80]}]({c['landing']}) | {c['year']} | "
                f"{cc['total']} | {c['max_run']} | {'; '.join(why) or 'no run survived'} |")
        add("")
        if len(aside) > ctx["max_docs"]:
            add(f"*({plural(len(aside) - ctx['max_docs'], 'further document')} "
                f"omitted; all of them are in `matches.csv`.)*")
            add("")

    # ---- closing -----------------------------------------------------------
    add("## What to do with this")
    add("")
    add("Read the surviving fragments, if there are any. For each one ask the "
        "ordinary questions: is this a quotation that lost its quotation marks "
        "in a draft; is it a definition that should be attributed; is it a "
        "sentence you wrote from a note you no longer recognise as somebody "
        "else's; or is it simply how the field says this. Most of the time it "
        "is the last one. Fix what needs fixing and re-run.")
    add("")
    add("**This report is not a finding of misconduct and cannot be one.** It "
        "measures text, and text is not intent. Overlap has innocent causes far "
        "more often than not, and only your institution — with the whole "
        "context, and with you in the room — can reach any conclusion at all.")
    add("")

    # ---- what may be shared ------------------------------------------------
    add("## Which of these three files you may share")
    add("")
    add("| file | what is in it | may it leave your machine? |")
    add("|---|---|---|")
    add("| `report.md` (this file) | passages of your text and of other "
        "people's theses, quoted verbatim | **No.** |")
    add("| `matches.csv` | "
        + ("**the matched text itself** (you passed `--csv-text`), beside page "
           "numbers, headings and catalogue metadata | **No — this run made it "
           "exactly as sensitive as the report.** Nothing inside the file says "
           "so; that is what this row is for. |"
           if ctx["csv_text"] else
           "page numbers, printed folios, headings, chapter buckets and "
           "catalogue metadata — locations, no thesis text | Yes, as far as "
           "copyright goes. It still names documents against yours, so treat it "
           "as private working material. |"))
    add("| `summary.json` | counts, thresholds and catalogue metadata, no "
        "thesis text | Yes. |")
    add("")
    add("This is a copyright question, not a courtesy: theses are in copyright, "
        "and the exception that permits this analysis (s.29A CDPA) does not "
        "permit redistributing what it quotes.")
    add("")

    # ---- method ------------------------------------------------------------
    m = ctx["method"]
    add("## How this was produced")
    add("")
    add(f"- extraction: `run_analysis.extract_pages` / `build_document` — the "
        f"same code, thresholds and exclusions that built the corpus")
    add(f"- screening: winnowed fingerprints (n={m['ngram']}, w={m['window']}), "
        f"guaranteeing detection of any shared passage of "
        f"{m['ngram'] + m['window'] - 1}+ words; "
        f"{fmt_int(m['query_fingerprints'])} fingerprints from your document "
        f"against {fmt_int(m['screened_docs'])} corpus documents")
    add(f"- {plural(m['neighbourhood'], 'document')} shared at least one "
        f"fingerprint, {fmt_int(m['eligible'])} of them at the threshold of "
        f"{plural(m['min_fp'], 'shared fingerprint')} after document-frequency "
        f"pruning; {fmt_int(m['candidates'])} scored exactly"
        + (f" — **capped at {m['max_candidates']:,} by `--max-candidates`, so "
           f"the counts above are lower bounds**"
           if m["capped"] else " (no cap)"))
    add(f"- matching: `run_analysis.verbatim_runs`, runs of "
        f"{m['ngram']}+ words extended maximally, contained runs dropped"
        + (f"; near-verbatim pass (`rapidfuzz token_set_ratio`) on "
           f"{m['near_top']} candidate(s)" if m["near_top"] else "")
        + (f"; embedding pass: {md_escape(m['embed_status'])}"
           if m["embed_status"] else ""))
    add(f"- the near-verbatim and embedding passes compare whole sentences, not "
        f"runs, so the third-document test does not apply to them: their counts "
        f"above are unfiltered and their pairs are in `matches.csv` marked "
        f"`not_filtered`")
    add(f"- filters: `rescore.py` prose test, `harvest_corpus.verify`'s "
        f"third-document and quotation tests; the third-document test reads "
        f"`corpus/tokens/`, which holds each thesis's retained text only "
        f"(the limitation stated above)")
    add(f"- baseline: recomputed from `corpus/corpus.db` at run time "
        f"({fmt_int(b['corpus_docs'])} documents, "
        f"{fmt_int(b['pairs_with_a_shared_run'])} screened pairs, "
        f"{fmt_int(b['verified_pairs'])} of them verified); the two pilot "
        f"percentages are quoted from README.md")
    add(f"- determinism: no wall clock, no RNG, every collection sorted before "
        f"writing — two runs over this PDF produce byte-identical artefacts")
    add("")
    add(f"> {REPORT_NOT_PUBLISHABLE}")
    add("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def doc_key(doc_id: str) -> str:
    """A short, stable, filesystem-safe key for a corpus document."""
    m = re.search(r"(\d+)$", doc_id or "")
    if m:
        return "WR" + m.group(1)
    return "D" + hashlib.sha256((doc_id or "").encode()).hexdigest()[:10]


def run(args: argparse.Namespace) -> int:
    global _ATTRIBUTION_RE
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import numpy as np
    import run_analysis as R
    import harvest_corpus as H
    import screen_corpus as S

    _ATTRIBUTION_RE = R.ATTRIBUTION_RE
    log = R.log

    # Argparse takes any int, and a negative or zero one silently degrades a
    # stage instead of failing: --near-top -1 skips the pass but reports it as
    # run on "-1 candidates", --screen-limit 0 is falsy and screens everything.
    for flag, value, lo in (("--max-candidates", args.max_candidates, 0),
                            ("--max-docs", args.max_docs, 0),
                            ("--max-fragments", args.max_fragments, 0),
                            ("--near-top", args.near_top, 0),
                            ("--min-shared-fp", args.min_shared_fp, 1)):
        if value < lo:
            print(f"check: {flag} must be {lo} or more, not {value}",
                  file=sys.stderr)
            return 2
    if args.screen_limit is not None and args.screen_limit < 1:
        print(f"check: --screen-limit must be 1 or more, not "
              f"{args.screen_limit} (omit it to screen the whole corpus)",
              file=sys.stderr)
        return 2

    pdf = Path(args.pdf).expanduser()
    if not pdf.is_file():
        print(f"check: no such file: {pdf}", file=sys.stderr)
        return 1
    with pdf.open("rb") as fh:                     # same guard as ensure_pdfs
        if fh.read(5) != b"%PDF-":
            print(f"check: {pdf} is not a PDF", file=sys.stderr)
            return 1
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"check: no corpus database at {db_path}", file=sys.stderr)
        return 1
    if not S.FP_DIR.exists() or not S.TOK_DIR.exists():
        print(f"check: the fingerprint or token index is missing "
              f"({S.FP_DIR}, {S.TOK_DIR}) — run screen_corpus.py first",
              file=sys.stderr)
        return 1

    out_root = Path(args.out)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf.stem)[:80] or "document"
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "_cache"
    # run_pipeline sets this the same way; it points run_analysis's extraction
    # and embedding caches somewhere we own, because cache/ belongs to the
    # focal-pair analysis and this must not write into it.
    R._ACTIVE_CACHE_DIR[0] = cache_dir

    # ---- 1. extract --------------------------------------------------------
    log(f"check: extracting {pdf.name}")
    doc, sha = extract_query(R, pdf, cache_dir, use_cache=not args.no_cache)
    words_all = sum(s.n_tokens for s in doc.sentences)
    words_kept = sum(s.n_tokens for s in doc.sentences if s.included)
    if len(doc.tokens) < S.NGRAM + S.WINDOW:
        # Too little text has two quite different causes, and sending somebody
        # to look for an OCR problem they do not have wastes their afternoon.
        blank = len(doc.no_text_pages)
        if blank >= max(1, len(doc.pages) // 2):
            print(f"check: only {len(doc.tokens)} matchable words came out of "
                  f"this PDF, and {blank} of its {len(doc.pages)} pages yielded "
                  f"no text at all — it is probably a scan with no text layer, "
                  f"and nothing can be checked", file=sys.stderr)
        else:
            print(f"check: this PDF's text layer reads fine, but only "
                  f"{len(doc.tokens)} matchable words are left after exclusions "
                  f"— fewer than the {S.NGRAM + S.WINDOW} the fingerprint "
                  f"window needs, so there is nothing to screen. This is a "
                  f"document too short to check, not a broken one",
                  file=sys.stderr)
        return 1
    log(f"  {len(doc.pages)} pages, {len(doc.sentences):,} sentences, "
        f"{words_kept:,} retained words")

    # Where the name came from matters: the author string decides whether a
    # deposited document is treated as your own earlier work and suppressed
    # from the verdict, and PDF metadata is frequently a producer's default.
    title = args.title.strip()
    author = args.author.strip()
    title_from_pdf = author_from_pdf = False
    if not title or not author:
        mt, ma = pdf_metadata(pdf)
        if not title and mt:
            title, title_from_pdf = mt, True
        if not author and ma:
            author, author_from_pdf = ma, True

    # ---- 2. subject, for a same-subject baseline ---------------------------
    disc = {"discipline": "Unknown", "conf": 0.0, "method": "none"}
    try:
        import discipline as D
        body = " ".join(s.text for s in doc.sentences[:400] if s.included)[:4000]
        disc = D.classify_one(title, "", body, allow_embeddings=False)
    except Exception as exc:                       # a missing optional module
        log(f"  subject classification unavailable ({type(exc).__name__})")

    # ---- 3. screen ---------------------------------------------------------
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT id, title, creator, year, publisher, discipline, landing, "
        "pdf_url, sha256 FROM doc WHERE status='ok' ORDER BY id").fetchall()
    if args.screen_limit:
        rows = rows[:args.screen_limit]
    meta = {r[0]: r for r in rows}
    ids = [r[0] for r in rows]

    self_doc = None
    for r in rows:
        if r[8] and r[8] == sha:
            self_doc = r
            break

    # Fingerprints are computed over the retained token stream WITHOUT the
    # inter-sentence barriers, because that is exactly how every stored corpus
    # fingerprint was computed; a different stream is not comparable with them.
    bare = [t for t in doc.tokens if not t.startswith("\x00")]
    query_fps = S.winnow(bare)
    if not query_fps.size:
        print(f"check: {len(bare)} retained words is too little to fingerprint "
              f"(winnowing needs {S.NGRAM + S.WINDOW}) — nothing to screen",
              file=sys.stderr)
        return 1
    log(f"  {query_fps.size:,} fingerprints; screening {len(ids):,} documents")
    shared, shared_raw = screen(S, np, ids, query_fps, log)
    if self_doc is not None:
        shared.pop(self_doc[0], None)
        shared_raw.pop(self_doc[0], None)
    log(f"  {len(shared_raw):,} documents share a fingerprint; "
        f"{sum(1 for v in shared.values() if v >= args.min_shared_fp):,} pass "
        f"the >= {args.min_shared_fp} threshold")

    eligible = sorted(((n, d) for d, n in shared.items() if n >= args.min_shared_fp),
                      key=lambda t: (-t[0], t[1]))
    capped = bool(args.max_candidates) and len(eligible) > args.max_candidates
    ranked = eligible[:args.max_candidates] if args.max_candidates else eligible
    if capped:
        log(f"  scoring the top {len(ranked):,} of {len(eligible):,} — every "
            f"count in the report will be a lower bound (--max-candidates 0 "
            f"for the true figures)")
    elif ranked:
        log(f"  scoring all {len(ranked):,} exactly; this is the slow stage "
            f"(pass --max-candidates N to cap it)")

    # ---- 4. exact matching -------------------------------------------------
    query_hashes = np.fromiter((S._tok_hash(t) for t in doc.tokens),
                               dtype=np.uint64, count=len(doc.tokens))
    query_grams = S.gram_hashes(query_hashes)

    cands: list[Candidate] = []
    for k, (n, d) in enumerate(ranked, 1):
        if k % 100 == 0:
            log(f"    scored {k:,}/{len(ranked):,}")
        m = meta[d]
        try:
            other = rebuild_document(R, H, d, {"key": doc_key(d), "label": m[1] or "",
                                               "creator": m[2] or "", "year": m[3]})
        except Exception as exc:
            log(f"  skipping {d}: {type(exc).__name__}: {exc}")
            continue
        runs, _ = R.verbatim_runs(other, doc)          # a = theirs, b = yours
        if not runs:
            continue
        cands.append(Candidate(
            doc_id=d, title=m[1] or "", creator=m[2] or "", year=m[3] or 0,
            publisher=m[4] or "", discipline=m[5] or "",
            landing=landing_url(d, m[6] or "", m[7] or ""),
            shared_fp=n, runs=runs))
    log(f"  {len(cands):,} documents share at least one {S.NGRAM}-word run")

    # ---- 5. the third-document test ----------------------------------------
    spans = sorted({(r.b0, r.b1) for c in cands for r in c.runs})
    exclude = {self_doc[0]} if self_doc is not None else set()
    if args.common_scan == "off" or not spans:
        third: dict[tuple[int, int], list[str]] = {s: [] for s in spans}
        scan_ids: list[str] = []
    else:
        scan_ids = ids if args.common_scan == "corpus" else sorted(shared_raw)
        log(f"  third-document test over {len(scan_ids):,} documents "
            f"({args.common_scan})")
        third = common_run_scan(S, np, scan_ids, spans, query_grams, exclude, log)

    # ---- 6. filter ---------------------------------------------------------
    for c in cands:
        for k, r in enumerate(c.runs):
            toks = doc.tokens[r.b0:r.b1]
            if not is_prose(toks):
                c.decisions[k] = (0, "artefact", [])
                continue
            others = [x for x in third.get((r.b0, r.b1), []) if x != c.doc_id]
            if others:
                c.decisions[k] = (0, "common", others)
                continue
            c.decisions[k] = (1, "kept", [])

    # Rank on the union of the surviving spans, not the sum of their lengths: a
    # document that matched one sentence of yours ten times has one passage to
    # read, and summing would rank it above a document with five.
    order = sorted(cands, key=lambda c: (
        -sum(1 for k, r in enumerate(c.runs)
             if c.decisions[k][0] and r.length >= R.LONG_RUN_WORDS),
        -S._merge_len([(r.b0, r.b1) for k, r in enumerate(c.runs)
                       if c.decisions[k][0]]),
        c.doc_id))
    near_set = {c.doc_id for c in order[:max(0, args.near_top)]}

    # ---- 7. quotation test, match records, optional passes 2 and 3 ---------
    from rapidfuzz import fuzz
    my_author = H._norm_author(author) if author else ""
    embed_status = ""
    for c in cands:
        other = rebuild_document(R, H, c.doc_id,
                                 {"key": doc_key(c.doc_id), "label": c.title,
                                  "creator": c.creator, "year": c.year})
        for k, r in enumerate(c.runs):
            if not c.decisions[k][0]:
                continue
            if run_is_quoted(doc, r.b0, r.b1, R.ATTRIB_WINDOW) and \
                    run_is_quoted(other, r.a0, r.a1, R.ATTRIB_WINDOW):
                c.decisions[k] = (0, "quoted", [])

        near = emb = None
        if c.doc_id in near_set and args.near_top:
            near, _stats = R.near_verbatim_pairs(other, doc)
            c.near_pairs = len(near)
            if args.embeddings:
                emb, emb_stats = R.embedding_pairs(
                    other, doc,
                    hashlib.sha256(c.doc_id.encode()).hexdigest(), sha)
                c.embed_pairs = len(emb) if emb is not None else None
                embed_status = str(emb_stats.get("status", "")) + (
                    f" — {emb_stats.get('reason', '')}"
                    if emb_stats.get("status") == "not run" else "")
        c.records = R.build_match_records(
            f"{doc_key(c.doc_id)}~SELF", "self-check", other, doc, c.runs, near, emb)

        c.same_author = int(bool(my_author and H._norm_author(c.creator) == my_author))
        c.title_sim = int(fuzz.token_set_ratio(title or "", c.title or ""))
        kept = [c.runs[k] for k in c.kept_runs]
        lens = [r.length for r in kept]
        if c.same_author or c.title_sim >= 95:
            c.verdict = "same_author_or_duplicate"
        elif not kept:
            c.verdict = "fully_explained"
        elif max(lens) < R.LONG_RUN_WORDS:
            c.verdict = "short_only"
        else:
            c.verdict = "residual"

    # ---- 8. assemble -------------------------------------------------------
    baseline = corpus_baseline(con, disc.get("discipline", ""))
    con.close()

    partners_fp3 = sorted(c.doc_id for c in cands if c.shared_fp >= BASELINE_MIN_FP)
    partners_long = sorted(c.doc_id for c in cands
                           if c.shared_fp >= BASELINE_MIN_FP
                           and any(r.length >= R.LONG_RUN_WORDS for r in c.runs))
    runs_total = sum(len(c.runs) for c in cands)
    tally = {"artefact": 0, "common": 0, "quoted": 0, "kept": 0}
    for c in cands:
        for kept_flag, reason, _ in c.decisions.values():
            tally["kept" if kept_flag else reason] += 1

    # The corpus screen's own taxonomy: a document whose only surviving runs are
    # shorter than 20 words is 'short_only' and is not treated as a finding.
    # Keeping the two apart is the difference between a report that alarms
    # somebody about an eight-word coincidence and one that does not.
    surviving = [c for c in order if c.verdict == "residual"]
    short_only = [c for c in order if c.verdict == "short_only"]
    duplicates = [c for c in order if c.verdict == "same_author_or_duplicate"]
    reported = {id(c) for c in surviving + short_only + duplicates}
    set_aside = [c for c in order if id(c) not in reported]

    def cand_ctx(c: Candidate) -> dict:
        kept = [c.runs[k] for k in c.kept_runs]
        cov = S._merge_len([(r.b0, r.b1) for r in kept])
        by_id = {rec_run_index(rec): rec for rec in c.records
                 if rec.pass_name == "verbatim"}
        # Distinct passages OF YOURS, not runs: the same sentence of yours
        # matched at four places in their thesis is one thing to read, and
        # quoting it four times would spend the whole display budget on it.
        distinct, end = 0, -1
        for b0, b1 in sorted((c.runs[k].b0, c.runs[k].b1) for k in c.kept_runs):
            if b0 >= end:
                distinct += 1
            end = max(end, b1)
        frags = []
        shown: list[tuple[int, int]] = []
        for k in sorted(c.kept_runs,
                        key=lambda k: (-c.runs[k].length, c.runs[k].b0)):
            if len(frags) >= args.max_fragments:
                break
            r = c.runs[k]
            if any(r.b0 < e and s < r.b1 for s, e in shown):
                continue
            rec = by_id.get(k)
            if rec is None:
                continue
            shown.append((r.b0, r.b1))
            frags.append({
                "words": rec.run_length_words,
                "your_page": rec.b_pdf_page_start,
                "your_printed": rec.b_printed_page,
                "your_bucket": rec.b_bucket,
                "your_heading": rec.b_heading,
                "their_page": rec.a_pdf_page_start,
                "their_bucket": rec.a_bucket,
                "citation_class": rec.citation_class,
                "your_attribution": your_attribution(R, doc, c.runs[k]),
                "context": fragment_in_context(doc, rec),
                "plain": rec.b_text,
            })
        kept_ids = set(c.kept_runs)
        kept_recs = [rec for rec in c.records if rec.pass_name == "verbatim"
                     and rec_run_index(rec) in kept_ids]
        cite_counts = [(k, sum(1 for rec in kept_recs if rec.citation_class == k))
                       for k in sorted({rec.citation_class for rec in kept_recs})]
        return {
            "doc_id": c.doc_id, "title": c.title, "creator": c.creator,
            "year": c.year, "publisher": c.publisher, "landing": c.landing,
            "discipline": c.discipline, "shared_fp": c.shared_fp,
            "verdict": c.verdict, "counts": c.counts(),
            "max_kept_run": max((r.length for r in kept), default=0),
            "max_run": max((r.length for r in c.runs), default=0),
            "kept_words": cov, "kept_share": cov / max(words_kept, 1),
            "distinct_passages": distinct,
            "citation": cite_counts, "fragments": frags,
            "surname": surname_of(c.creator),
            "citation_window": R.CITATION_WINDOW,
            "same_author": c.same_author, "title_sim": c.title_sim,
            "near_pairs": c.near_pairs, "embed_pairs": c.embed_pairs,
            "fuzz_cutoff": R.FUZZ_CUTOFF, "embed_cutoff": R.EMBED_CUTOFF,
        }

    pct_any = percentile_of(baseline["any_run_partners"].get("_sorted", []),
                            len(partners_fp3))
    pct_long = percentile_of(baseline["long_run_partners"].get("_sorted", []),
                             len(partners_long))
    # Runs shared with a document set aside as your own deposit are kept by the
    # filters and counted in the tally, but they are not a finding about
    # anybody. Saying "nothing survived" while the tally reads 86 is the one
    # thing this report must never do.
    dup_kept = sum(c.counts()["kept"] for c in duplicates)
    if not surviving and not short_only and duplicates:
        verdict_sentence = (
            f"**Nothing survived filtering except your own deposited work.** "
            f"{plural(dup_kept, 'run')} did survive, and all of them are "
            f"shared with the {plural(len(duplicates), 'document')} listed "
            f"below as your own earlier deposit; everything you share with "
            f"anybody else's thesis has an explanation the tool can name. "
            f"Check that identification before reading this as good news.")
    elif not surviving and not short_only and not runs_total:
        verdict_sentence = (
            "**No thesis in this corpus shares a run of eight words or more "
            "with your document.** Nothing was found, so nothing had to be "
            "filtered.")
    elif not surviving and not short_only:
        verdict_sentence = (
            "**Nothing survived filtering.** Every passage you share with this "
            "corpus has an explanation the tool can name, which is the ordinary "
            "outcome for a thesis.")
    elif not surviving:
        verdict_sentence = (
            f"**No long run survived.** {len(short_only)} document(s) share "
            f"short runs of 8 to 19 words with you that the filters do not "
            f"explain, which is an ordinary amount of coincidence between "
            f"theses in one field and is not something to act on.")
    else:
        verdict_sentence = (
            f"**{len(surviving)} document(s) share a run of 20 words or more "
            f"with yours that none of the filters explain.** That is a list to "
            f"look at, not a finding: {100 * baseline['residual_share']:.2f}% of "
            f"verified pairs in this corpus reach the same state, and being in "
            f"that group is not evidence of anything on its own.")

    excluded_counts = {}
    for s in doc.sentences:
        for e in s.exclusions:
            excluded_counts[e] = excluded_counts.get(e, 0) + 1

    ctx = {
        "query": {
            "name": pdf.name, "sha256": sha, "title": title, "author": author,
            "title_from_pdf": title_from_pdf, "author_from_pdf": author_from_pdf,
            "pages": len(doc.pages), "sentences": len(doc.sentences),
            "sentences_included": sum(1 for s in doc.sentences if s.included),
            "words": words_all, "words_included": words_kept,
            "no_text_pages": doc.no_text_pages,
            "excluded": sorted(excluded_counts.items()),
            "discipline": disc.get("discipline", "Unknown"),
            "discipline_conf": float(disc.get("conf", 0.0)),
            "discipline_method": disc.get("method", "none"),
            "self_doc": ({"id": self_doc[0], "title": self_doc[1] or "",
                          "creator": self_doc[2] or "", "year": self_doc[3] or 0,
                          "landing": landing_url(self_doc[0], self_doc[6] or "",
                                                 self_doc[7] or "")}
                         if self_doc is not None else None),
        },
        "headline": {
            "partners_baseline_fp": len(partners_fp3),
            "partners_any_fp": len(cands),
            "partners_long": len(partners_long),
            "pct_any": pct_any, "pct_long": pct_long,
            "runs_total": runs_total,
            "runs_common": tally["common"],
            "common_pct": 100.0 * tally["common"] / max(runs_total, 1),
            "docs_residual": len(surviving),
            "docs_duplicate": len(duplicates),
            "docs_short": len(short_only),
            "baseline_short_share": (baseline["verdicts"].get("short_only", 0)
                                     / max(baseline["verified_pairs"], 1)),
            "verdict_sentence": verdict_sentence,
        },
        "filters": tally,
        "baseline": baseline,
        "surviving": [cand_ctx(c) for c in surviving],
        "short_only": [cand_ctx(c) for c in short_only],
        "duplicates": [cand_ctx(c) for c in duplicates],
        "set_aside": [cand_ctx(c) for c in set_aside],
        "dup_kept": dup_kept,
        "scan_docs": len(scan_ids),
        "scan_mode": args.common_scan,
        "spans": len(spans),
        "csv_text": bool(args.csv_text),
        "max_docs": args.max_docs,
        "cap": {"capped": capped, "limit": args.max_candidates,
                "eligible": len(eligible), "neighbourhood": len(shared_raw)},
        "method": {
            "ngram": S.NGRAM, "window": S.WINDOW,
            "query_fingerprints": int(query_fps.size),
            "screened_docs": len(ids),
            "neighbourhood": len(shared_raw),
            "eligible": len(eligible),
            "candidates": len(cands),
            "capped": capped,
            "min_fp": args.min_shared_fp,
            "max_candidates": args.max_candidates,
            "near_top": min(args.near_top, len(cands)) if args.near_top else 0,
            "embed_status": embed_status,
        },
    }

    # ---- 9. write ----------------------------------------------------------
    report_path = out_dir / "report.md"
    csv_path = out_dir / "matches.csv"
    summary_path = out_dir / "summary.json"
    report_path.write_text(build_report(ctx), encoding="utf-8")
    n_rows = write_csv(csv_path, sorted(cands, key=lambda c: c.doc_id),
                       args.csv_text)

    summary = {
        "input": {"filename": pdf.name, "sha256": sha, "pages": len(doc.pages),
                  "sentences": len(doc.sentences), "words": words_all,
                  "words_retained": words_kept,
                  "no_text_pages": len(doc.no_text_pages),
                  "discipline": ctx["query"]["discipline"],
                  "discipline_conf": ctx["query"]["discipline_conf"],
                  "title_from_pdf_metadata": title_from_pdf,
                  "author_from_pdf_metadata": author_from_pdf,
                  "already_in_corpus": bool(self_doc)},
        "screen": ctx["method"],
        "filters": tally,
        # Which artefact may leave the machine, stated in the artefacts
        # themselves rather than only in --help.
        "outputs": {
            "report.md": {"contains": "verbatim fragments of your text and of "
                                      "other people's theses",
                          "publishable": False},
            "matches.csv": {"contains": ("locations, headings and catalogue "
                                         "metadata, plus the matched text "
                                         "itself (--csv-text)")
                            if args.csv_text else
                            "locations, headings and catalogue metadata, no "
                            "thesis text",
                            "publishable": not args.csv_text},
            "summary.json": {"contains": "counts and catalogue metadata only",
                             "publishable": True},
        },
        "headline": {k: v for k, v in ctx["headline"].items()
                     if k != "verdict_sentence"},
        "baseline": {k: (v if not isinstance(v, dict)
                         else {kk: vv for kk, vv in v.items() if kk != "_sorted"})
                     for k, v in baseline.items()},
        "documents": sorted(
            ({"doc_id": c.doc_id, "title": c.title, "creator": c.creator,
              "year": c.year, "landing": c.landing, "discipline": c.discipline,
              "shared_fingerprints": c.shared_fp, "verdict": c.verdict,
              "runs": c.counts(),
              "max_run_words": max((r.length for r in c.runs), default=0),
              "max_surviving_run_words": max(
                  (c.runs[k].length for k in c.kept_runs), default=0)}
             for c in cands),
            key=lambda d: d["doc_id"]),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    log(f"  wrote {report_path}")
    log(f"  wrote {csv_path} ({n_rows:,} rows)")
    log(f"  wrote {summary_path}")
    print(f"{plural(len(surviving), 'document')} with a surviving 20+ word "
          f"run, {len(short_only)} with short runs only"
          + (f" (and {plural(len(duplicates), 'document')} set aside as your "
             f"own deposit, carrying {plural(dup_kept, 'surviving run')})"
             if duplicates else "")
          + f"; {tally['kept']:,} of {runs_total:,} shared runs remain after "
            f"filtering"
          + (f", counted over the top {len(ranked):,} of {len(eligible):,} "
             f"candidates only" if capped else "")
          + f". See {report_path}")
    return 0


def _main() -> int:
    """Standalone entry point.

    tg.py turns an unexpected exception into one line on stderr; the plugin
    contract says this file must also work on its own, so it has to do the same
    rather than printing a traceback at somebody whose PDF is simply truncated.
    """
    ap = argparse.ArgumentParser(description=HELP)
    add_args(ap)
    ap.add_argument("--traceback", action="store_true",
                    help="print the full traceback if this command fails")
    args = ap.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("check: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:                # noqa: BLE001 — one line, not a dump
        if args.traceback:
            import traceback
            traceback.print_exc()
        print(f"check: failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.traceback:
            print("check: re-run with --traceback for the full traceback",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
