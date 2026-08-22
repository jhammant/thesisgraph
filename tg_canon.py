#!/usr/bin/env python3
"""
tg_canon.py — what a field actually reads, read off 1.5M citation edges.

`citations.py` built the edges; nothing so far has asked what they say about the
fields that produced them. Three questions are worth asking, and each one has a
methodological trap sitting in front of it that has to be dealt with before the
number means anything:

  1. CANON — the works a discipline leans on. The trap is the key. An edge is
     identified by (surname, year), and `wang 2014` is not a work: it is 354
     citations spread over roughly 300 different papers. Presenting that as one
     row would be a fabrication. So every key is split into works by clustering
     the parsed titles inside it, entries whose title is too thin to cluster are
     never attributed to anybody, and every row carries the diagnostics needed to
     see how much of that key was resolvable.

  2. CITATION AGE — how old is the literature a field stands on, and is it
     getting older? The trap is that "half-life" and "median age" are the same
     statistic measured two ways, and that a corpus-wide time series crosses the
     digitisation boundary the README flags at ~2010, where retro-digitised
     theses give way to born-digital ones. Both are stated, and the trend table
     carries the extraction-quality columns beside the age columns so a reader
     can see whether a movement in age is tracking a movement in OCR.

  3. FIELD FLOW — which discipline imports from which. The trap is that a cited
     work has no field in the data; its field has to be inferred from its
     citers, which makes the diagonal true by construction. Three things push
     back. The home field of a work must survive the deletion of any one citing
     thesis, so no single thesis can hand a work to a field. That test is
     applied identically to every citer of a work, home or away — an earlier
     version of this module tested each citer against the field of the *others*,
     which sounds even-handed and is not: it made a home read harder to place
     than a cross-field read of the same work and inflated the off-diagonal
     roughly fourfold. Both the strict rule and a permissive one are run and
     both are printed, so the residual circularity has a size and not just a
     direction. And the specificity policy that decides whether a work belongs
     to anybody at all is `bridges.py`'s, imported rather than re-derived — that
     module already worked out that a work cited by 4% of the corpus across
     every discipline is universal methods literature and belongs to no field.

Everything here reads two small SQLite files and never opens corpus/text, so a
full rebuild is seconds and no index is cached. Output is derived data only —
counts, medians, matrices and normalised bibliographic titles of *third-party*
published works. The `cite.raw` column, which holds thesis reference-list text,
is never read and never emitted, so every artefact this writes is publishable.

    tg canon                       # all three analyses
    tg canon --section age         # one of them
    tg canon --sample 8            # eyeball how the worst colliding keys resolved
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "canon"
HELP = "canonical works, citation age and field-to-field flow from the citation graph"

# --- title disambiguation -------------------------------------------------
# cite.title is already normalised by citations.norm_title: lowercased, stripped
# of punctuation, first eight words of >2 characters. Two entries are the same
# work when one title's tokens are essentially contained in the other's — that,
# not equality, is the relation that matters, because the parser truncates a
# title at the first sentence break and so emits "the discovery grounded theory"
# and "the discovery grounded theory strategies for qualitative research" for the
# same book. token_sort_ratio picks up the rest: OCR damage and spelling variants
# ("naturalistic enquiry"). 90 is stricter than the 85 run_analysis.py uses for
# near-verbatim sentence matching, because a title is short and a false merge
# here invents a work.
MIN_TITLE_TOKENS = 2      # below this a title cannot identify anything
CONTAINMENT = 0.80        # |A n B| / min(|A|,|B|)
SORT_RATIO = 90           # rapidfuzz token_sort_ratio fallback

# A (surname, year) key is only worth splitting if enough theses cite it; a key
# cited by one thesis is not a canon candidate and has no field to belong to.
MIN_KEY_CITERS = 3

# A lift is a ratio and goes wild on a cell of three. Cells below this are still
# ranked — the confidence bound is what disciplines them now — but the headline
# marks them, so a reader can see when a claim rests on a handful of pairs.
MIN_FLOW_CELL = 30

# z for the lower end of a 95% interval on a count, used to stop a small cell
# with a large ratio from taking the headline off a large cell with a real one.
Z95 = 1.96

# The top import is recomputed under these (min_citers, home_share) offsets and
# the agreement is reported. A headline that moves under its own parameters is a
# property of the parameters, and the reader is entitled to know which it is.
STABILITY_GRID = ((0, -0.10), (0, 0.10), (1, 0.0), (2, 0.0))

# Crude words-per-reference-entry, used only to turn ref_words into an estimated
# denominator for parser recall. Humanities entries run longer than this, so
# their estimated recall is if anything a floor.
WORDS_PER_ENTRY = 30

# A parsed "title" that carries no word is page numbers or a DOI fragment that
# the segmenter mistook for a title. It identifies nothing, so say so on the row.
MIN_WORDY_TOKENS = 1      # alphabetic tokens of 3+ characters a title must have

MAX_AGE = 120             # a citation older than this is a parse artefact
MIN_DOC_YEAR, MAX_DOC_YEAR = 1900, 2026
DIGITISATION_YEAR = 2010  # README: the corpus flips retro-digitised -> born-digital


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--citations-db", type=Path,
                   default=HERE / "corpus" / "citations.db",
                   help="citation edges (default: %(default)s)")
    p.add_argument("--corpus-db", type=Path,
                   default=HERE / "corpus" / "corpus.db",
                   help="thesis metadata (default: %(default)s)")
    p.add_argument("--out", type=Path, default=HERE / "out" / "analytics",
                   help="directory for the CSV/JSON artefacts (default: %(default)s)")
    p.add_argument("--section", choices=("all", "canon", "age", "flow"),
                   default="all", help="run one analysis instead of all three")
    p.add_argument("--top", type=int, default=25,
                   help="works listed per discipline (default: %(default)s)")
    p.add_argument("--top-subfield", type=int, default=10,
                   help="works listed per subfield (default: %(default)s)")
    p.add_argument("--min-citers", type=int, default=3,
                   help="distinct citing theses, corpus-wide, before a work is "
                        "reported (default: %(default)s)")
    p.add_argument("--min-field-citers", type=int, default=3,
                   help="citing theses from within the field itself before a "
                        "work appears in that field's canon; --min-citers is a "
                        "corpus-wide floor and does not imply this one "
                        "(default: %(default)s)")
    p.add_argument("--min-theses", type=int, default=40,
                   help="theses in a field, or in a time-series cell, before it "
                        "is reported (default: %(default)s)")
    p.add_argument("--home-share", type=float, default=0.60,
                   help="leave-one-out share of citers needed before a work is "
                        "said to belong to a discipline (default: %(default)s)")
    p.add_argument("--band", type=int, default=3,
                   help="years per band in the citation-age time series "
                        "(default: %(default)s)")
    p.add_argument("--sample", type=int, default=0,
                   help="print how the N most-cited (surname, year) keys were "
                        "split into works, to check the disambiguation by eye")


def validate(args: argparse.Namespace) -> str | None:
    """Reject arguments that would produce output rather than an error.

    A --band of 0 used to raise ZeroDivisionError and a --band of -3 used to
    succeed and emit reversed band labels; --home-share 5 used to write a
    profile of empty cells. Nonsense that runs to completion is worse than
    nonsense that stops, so every numeric bound is checked here.
    """
    checks = [
        ("--top", args.top, 1, None),
        ("--top-subfield", args.top_subfield, 1, None),
        ("--min-citers", args.min_citers, 1, None),
        ("--min-field-citers", args.min_field_citers, 1, None),
        ("--min-theses", args.min_theses, 1, None),
        ("--band", args.band, 1, None),
        ("--sample", args.sample, 0, None),
    ]
    for flag, value, lo, hi in checks:
        if value < lo or (hi is not None and value > hi):
            bound = f"at least {lo}" if hi is None else f"between {lo} and {hi}"
            return f"{flag} must be {bound}, got {value}"
    if not 0.0 < args.home_share <= 1.0:
        return (f"--home-share is a share of citers and must be in (0, 1], got "
                f"{args.home_share}")
    return None


# ---------------------------------------------------------------- loading

def _ro(path: Path) -> sqlite3.Connection:
    """Read-only connection. corpus/ is data this toolkit reads and never writes."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


class Docs:
    """Thesis metadata, plus the reference-parser coverage that qualifies it."""

    def __init__(self, corpus_db: Path, citations_db: Path):
        con = _ro(corpus_db)
        try:
            rows = con.execute(
                "SELECT id, year, discipline, subfield, no_text_pages "
                "FROM doc WHERE status='ok' ORDER BY id").fetchall()
        finally:
            con.close()
        self.year: dict[str, int | None] = {}
        self.disc: dict[str, str] = {}
        self.sub: dict[str, str] = {}
        self.blank: dict[str, int] = {}
        for did, yr, disc, sub, blank in rows:
            if not disc:
                continue
            self.year[did] = yr if yr and MIN_DOC_YEAR <= yr <= MAX_DOC_YEAR else None
            self.disc[did] = disc
            self.sub[did] = sub or ""
            self.blank[did] = blank or 0

        con = _ro(citations_db)
        try:
            self.stat = {r[0]: (r[1] or 0, r[2] or 0, r[3] or 0) for r in con.execute(
                "SELECT src, ref_words, entries, parsed FROM srcstat")}
        finally:
            con.close()

    def field(self, did: str, field_of) -> str:
        return field_of(self.disc.get(did, ""), self.sub.get(did, ""))

    def disciplines(self) -> list[str]:
        return sorted(set(self.disc.values()))


def distinct_citer_key_pairs(citations_db: Path) -> int:
    """(thesis, surname, year) pairs the parser produced — the flow denominator.

    The field-flow matrix places a subset of these, and the honest statement of
    how much of the citation graph it covers is a share of this number, not a
    share of works. Most works have too few citers to place, which sounds like a
    remark about obscure works; this is the remark about the graph.
    """
    con = _ro(citations_db)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT src, surname, year FROM cite)"
        ).fetchone()[0]
    finally:
        con.close()


def bridge_policy():
    """Reuse bridges.py's tested answer to "is this work specific to a field?".

    bridges.py already argued this out: a work cited by 150+ theses across every
    discipline is universal methods literature, and "Education and Medicine both
    use thematic analysis" is not a crossover. Its constants and its field grain
    are imported rather than restated so the two capabilities cannot drift apart.
    """
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import bridges  # noqa: PLC0415 — deliberately lazy; tg.py imports eagerly
        return bridges.MAX_TOTAL_CITING, bridges.MAX_FIELDS, bridges.field_of, True
    except Exception as exc:  # noqa: BLE001 — a missing bridges.py must not be fatal
        print(f"canon: bridges.py unavailable ({type(exc).__name__}: {exc}); "
              f"using its published constants inline", file=sys.stderr)
        return 150, 6, (lambda d, s: s or d), False


# ------------------------------------------------- work disambiguation

def cluster_titles(titles: list[str]) -> list[int]:
    """Split one (surname, year) key's titles into works. -1 means unresolvable.

    Single-link agglomerative over a list already sorted by (-citers, title), so
    the most-cited spelling of a title is the seed and the result does not depend
    on anything but the data. Comparing a candidate against every earlier title
    rather than against cluster representatives is what lets a full title, its
    truncation and its truncation-plus-publisher end up in one work.
    """
    from rapidfuzz import fuzz  # noqa: PLC0415 — keep module import cheap

    seen: list[tuple[str, set[str], int]] = []
    out: list[int] = []
    n_clusters = 0
    for title in titles:
        toks = set(title.split())
        if len(toks) < MIN_TITLE_TOKENS:
            out.append(-1)
            continue
        hit = -1
        for prev_t, prev_toks, prev_c in seen:
            inter = len(toks & prev_toks)
            if inter >= MIN_TITLE_TOKENS and \
                    inter / min(len(toks), len(prev_toks)) >= CONTAINMENT:
                hit = prev_c
                break
            if fuzz.token_sort_ratio(title, prev_t) >= SORT_RATIO:
                hit = prev_c
                break
        if hit < 0:
            hit, n_clusters = n_clusters, n_clusters + 1
        seen.append((title, toks, hit))
        out.append(hit)
    return out


class Work:
    """One resolved cited work, and the evidence that it is in fact one work."""

    __slots__ = ("surname", "year", "title", "srcs",
                 "key_citers", "key_works", "key_unresolved")

    def __init__(self, surname, year, title, srcs,
                 key_citers, key_works, key_unresolved):
        self.surname = surname
        self.year = year
        self.title = title
        self.srcs = srcs
        self.key_citers = key_citers
        self.key_works = key_works
        self.key_unresolved = key_unresolved

    @property
    def n(self) -> int:
        return len(self.srcs)

    @property
    def key(self) -> str:
        return f"{self.surname} {self.year}"

    @property
    def share_of_key(self) -> float:
        return self.n / max(self.key_citers, 1)

    @property
    def resolution(self) -> str:
        """Which mechanism produced this row, and how much of its key it holds.

        "disambiguated" describes a mechanism, not a confidence, and on its own
        it was doing duty as a quality mark it had not earned: it was applied to
        works holding anywhere from 1% to 100% of their key. A work that is a
        minority of its own key says so in the label as well as in pct_of_key.
        """
        if sum(1 for t in self.title.split()
               if len(t) >= 3 and t.isalpha()) < MIN_WORDY_TOKENS:
            return "not-a-title"          # page numbers, a DOI fragment, digits
        if self.key_unresolved > self.key_citers / 2:
            return "mostly-unresolved-key"
        if len(self.title.split()) < 3:
            return "thin-title"
        if self.key_works == 1 and self.key_unresolved == 0:
            return "clean"
        if self.share_of_key < 0.5:
            return "minority-of-key"
        return "disambiguated"


def resolve_works(citations_db: Path, min_key_citers: int = MIN_KEY_CITERS):
    """Every (surname, year) key with enough citers, split into works.

    Returns (works, keystats). keystats is the honest accounting: how many keys
    were seen, how many split into more than one work, and how many citing
    theses could not be attributed to any work at all.
    """
    con = _ro(citations_db)
    works: list[Work] = []
    keys_seen = keys_kept = keys_split = 0
    citers_kept = citers_unresolved = 0
    worst: list[tuple[int, str, int, int, int]] = []
    try:
        cur = con.execute(
            "SELECT surname, year, title, src FROM cite ORDER BY surname, year")
        for (surname, year), grp in groupby(cur, key=lambda r: (r[0], r[1])):
            keys_seen += 1
            by_title: dict[str, set[str]] = defaultdict(set)
            all_srcs: set[str] = set()
            for _, _, title, src in grp:
                by_title[title or ""].add(src)
                all_srcs.add(src)
            if len(all_srcs) < min_key_citers:
                continue
            keys_kept += 1

            titles = sorted(by_title, key=lambda t: (-len(by_title[t]), t))
            labels = cluster_titles(titles)
            groups: dict[int, set[str]] = defaultdict(set)
            reps: dict[int, str] = {}
            unresolved: set[str] = set()
            for title, lab in zip(titles, labels):
                if lab < 0:
                    unresolved |= by_title[title]
                    continue
                groups[lab] |= by_title[title]
                reps.setdefault(lab, title)   # titles are sorted, so this is the
                                              # most-cited spelling in the cluster
            # a thesis that reached a work by one spelling is attributed, even if
            # another of its entries for the same key was unresolvable
            attributed = set().union(*groups.values()) if groups else set()
            unresolved -= attributed
            n_works = len(groups)
            if n_works > 1:
                keys_split += 1
            citers_kept += len(attributed)
            citers_unresolved += len(unresolved)
            worst.append((len(all_srcs), surname, year, n_works, len(unresolved)))

            for lab in sorted(groups):
                works.append(Work(surname, year, reps[lab],
                                  frozenset(groups[lab]),
                                  len(all_srcs), n_works, len(unresolved)))
    finally:
        con.close()

    works.sort(key=lambda w: (-w.n, w.surname, w.year, w.title))
    worst.sort(key=lambda r: (-r[0], r[1], r[2]))
    keystats = {
        "keys_total": keys_seen,
        "keys_with_min_citers": keys_kept,
        "min_key_citers": min_key_citers,
        "keys_resolving_to_one_work": keys_kept - keys_split,
        "keys_split_into_several_works": keys_split,
        "works_resolved": len(works),
        "citing_theses_attributed": citers_kept,
        "citing_theses_unattributable": citers_unresolved,
        "attributable_pct": round(100 * citers_kept /
                                  max(citers_kept + citers_unresolved, 1), 2),
        "worst_keys": worst[:200],
    }
    return works, keystats


# ---------------------------------------------------------------- stats

def median(xs: list[int] | list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    m = n // 2
    return float(s[m]) if n % 2 else (s[m - 1] + s[m]) / 2.0


def half_life(counts: Counter) -> float:
    """Age at which the cumulative citation distribution crosses 50%.

    The bibliometric cited half-life. It is the same statistic as the
    edge-weighted median age, measured continuously instead of in whole years;
    both are reported because the discrete one is what people check by hand and
    the continuous one is what moves visibly over time.
    """
    total = sum(counts.values())
    if not total:
        return float("nan")
    target, run = total / 2.0, 0
    for age in sorted(counts):
        prev, run = run, run + counts[age]
        if run >= target:
            if counts[age] == 0:
                return float(age)
            return age + (target - prev) / counts[age] - 0.5
    return float(max(counts))


def slope_per_decade(points: list[tuple[float, float, int]]) -> float | None:
    """Weighted least-squares slope, in years-of-age per decade of thesis year."""
    if len(points) < 3:
        return None
    import numpy as np  # noqa: PLC0415

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    w = np.sqrt(np.array([p[2] for p in points], dtype=float))
    if float(x.max() - x.min()) < 1e-9:
        return None
    coef = np.polyfit(x, y, 1, w=w)
    return round(float(coef[0]) * 10, 3)


# ---------------------------------------------------------------- writing

def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Deterministic CSV: LF endings, caller-sorted rows, no locale formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def caveats(cov: dict, keystats: dict, bridges_ok: bool,
            flow: dict | None = None) -> list[dict]:
    """Every caveat that a number in these files needs standing next to it.

    These are written into canon_caveats.csv and into canon.json, and printed,
    because a table of medians detached from them would be misleading. The flow
    caveats carry this run's own figures when the flow section ran, since a
    caveat is quoted more often than the table it qualifies and a caveat that
    says "a low single-digit percentage" is doing half the job.
    """
    rate = cov["overall"]
    no_edge_pct = 100 * cov["theses_without_edges"] / max(cov["theses_total"], 1)
    if flow:
        rch, est = flow["reach"], flow["estimators"]
        off_ratio = (est["superseded-asymmetric"]["pct_off_diagonal"] /
                     max(est["strict"]["pct_off_diagonal"], 0.01))
        reach_txt = (
            f"It rests on {rch['placed_thesis_work_pairs']:,} thesis-work pairs "
            f"out of {rch['distinct_citer_key_pairs_in_cite']:,} distinct "
            f"(thesis, surname, year) pairs the parser produced — "
            f"{rch['pct_of_parsed_citer_key_pairs']}% — and the parser sees an "
            f"estimated {rch['est_parser_recall_pct']}% of the reference entries "
            f"present, so it describes roughly "
            f"{rch['est_pct_of_all_reference_edges']}% of the reference edges "
            f"actually in the corpus. The same figures are in "
            f"field_flow_matrix.json under 'reach'.")
        bracket_txt = (
            f"On this run the strict rule puts "
            f"{est['strict']['pct_diagonal']}% of "
            f"{est['strict']['placed_pairs']:,} placed pairs on the diagonal and "
            f"the permissive rule {est['permissive']['pct_diagonal']}% of "
            f"{est['permissive']['placed_pairs']:,}; the superseded rule reported "
            f"{est['superseded-asymmetric']['pct_diagonal']}%, outside that range "
            f"and below both, on an off-diagonal {off_ratio:.1f} times the "
            f"strict one's.")
    else:
        reach_txt = ("It rests on a low single-digit percentage of the "
                     "(thesis, surname, year) pairs the parser produced, and the "
                     "parser sees an estimated quarter of the reference entries "
                     "present, so it describes on the order of one per cent of "
                     "the reference edges actually in the corpus. Run the flow "
                     "section for this corpus's exact figures.")
        bracket_txt = ("Run the flow section for the size of the gap on this "
                       "corpus.")
    return [
        {"id": "author-date-only", "applies_to": "all",
         "caveat": "Reference parsing is author-date (Harvard) only. "
                   f"{cov['theses_without_edges']:,} of {cov['theses_total']:,} "
                   f"theses ({no_edge_pct:.0f}%) yield no parsed citation at all "
                   "and are absent from every table here, not merely "
                   "under-counted. It matches "
                   f"{rate['parsed']:,} entries out of {rate['entries']:,} "
                   f"parenthesised-year candidates ({rate['candidate_rate']}%), and "
                   f"an estimated {rate['est_recall']}% of all reference entries "
                   "present. Vancouver and numbered styles are not parsed at all. "
                   "Every count here is a count of what the parser could see."},
        {"id": "discipline-bias", "applies_to": "all",
         "caveat": "That parser bias is not uniform across fields: estimated "
                   f"recall runs from {rate['min_recall']}% ({rate['min_recall_disc']}) "
                   f"to {rate['max_recall']}% ({rate['max_recall_disc']}). Social "
                   "sciences, psychology and education are over-represented in "
                   "every table below and the physical sciences and engineering "
                   "are under-represented. Cross-discipline comparisons of "
                   "absolute volume are not meaningful; comparisons of shape "
                   "(age distribution, import profile) are weaker but survive."},
        {"id": "est-recall-crude", "applies_to": "citation_coverage.csv",
         "caveat": f"Estimated recall divides parsed entries by ref_words / "
                   f"{WORDS_PER_ENTRY}. {WORDS_PER_ENTRY} words per reference entry "
                   "is a stated assumption, not a measurement. Humanities entries "
                   "run longer, so their estimated recall is a floor."},
        {"id": "key-collision", "applies_to": "canon_*.csv",
         "caveat": "A citation edge identifies a work only by (surname, year), "
                   "which collides badly: 'wang 2014' is roughly 300 different "
                   "papers. Keys are split into works by clustering their parsed "
                   f"titles. {keystats['keys_split_into_several_works']:,} of "
                   f"{keystats['keys_with_min_citers']:,} keys held more than one "
                   f"work. {keystats['citing_theses_unattributable']:,} citing "
                   "theses had no title good enough to attribute and are counted "
                   "in no work. Read every row with its resolution column, which "
                   "names the mechanism that produced the row: 'clean' is a key "
                   "that held one work, 'disambiguated' a work that is the majority "
                   "of a split key, 'minority-of-key' one that is not, and "
                   "'thin-title', 'not-a-title' and 'mostly-unresolved-key' are "
                   "warnings. pct_of_key is the number behind that word."},
        {"id": "canon-denominator", "applies_to": "canon_discipline.csv, "
                                                  "canon_subfield.csv",
         "caveat": "pct_of_field_theses_with_parsed_refs is a percentage of the "
                   "field's theses that yielded a parsed reference list, not of the "
                   "field's theses. The two differ by about 2.4x in the sciences, "
                   "so the row also carries field_theses_with_parsed_refs and "
                   "field_theses_total and the reader can compute either. The "
                   "restricted denominator is the defensible one — a thesis with no "
                   "parsed references could not have cited anything — but it is not "
                   "the one a heading like 'percentage of the field' suggests. "
                   "Separately, --min-citers is a corpus-wide floor and says "
                   "nothing about support inside a field, so --min-field-citers "
                   "gates these two tables on in-field citing theses; without it "
                   "the subfield table published rows resting on a single thesis. "
                   "Rows whose title the segmenter could not identify at all "
                   "(resolution 'not-a-title', usually a page range) are excluded "
                   "from these two tables and kept, flagged, in canon_works.csv."},
        {"id": "title-clustering-errs", "applies_to": "canon_*.csv",
         "caveat": "Title clustering can under-split (two works by one author in "
                   "one year with near-identical titles merge) and can under-merge "
                   "(a title mangled by OCR stays separate and undercounts). "
                   "Under-merging is the commoner failure, so a work's citer count "
                   "is a floor. --sample prints how the worst keys resolved."},
        {"id": "half-life-identity", "applies_to": "citation_age_*.csv",
         "caveat": "half_life_years and median_age_edges are the same statistic, "
                   "continuous and discrete. They are not two independent findings. "
                   "median_age_theses is the different one: it gives each thesis "
                   "one vote instead of each reference, so a thesis with a 900-item "
                   "bibliography cannot set a field's number by itself. Where the "
                   "two diverge sharply the divergence is the finding: it means the "
                   "field's reference-heavy theses read differently from its "
                   "reference-light ones."},
        {"id": "age-weighting", "applies_to": "citation_age_*.csv",
         "caveat": "Every age column names its weighting, because the choice moves "
                   "the number. price_index_pct_edges and pct_over_20y_edges give "
                   "every parsed reference a vote; price_index_pct_theses gives "
                   "every thesis one. On this corpus the two Price indices differ "
                   "by up to 5.9 points on the same field (History, philosophy and "
                   "religion: 13.1 thesis-weighted against 19.0 edge-weighted) and "
                   "in both directions (Mathematics 27.8 against 24.5). An earlier "
                   "version printed the thesis-weighted Price index beside "
                   "edge-weighted neighbours in the same row without saying so. "
                   "Compare like with like, and do not read the two Price indices "
                   "as two findings."},
        {"id": "digitisation-confound",
         "applies_to": "citation_age_trend.csv",
         "caveat": f"The README flags a digitisation confound at ~{DIGITISATION_YEAR}, "
                   "where retro-digitised theses give way to born-digital ones. "
                   "Extraction and therefore reference parsing are worse on the "
                   "older scans. The trend table carries mean_no_text_pages and "
                   "parse_rate beside the age columns so a movement in age can be "
                   "checked against a movement in extraction quality, and reports "
                   f"a separate slope for {DIGITISATION_YEAR}+ only."},
        {"id": "age-truncation", "applies_to": "citation_age_*.csv",
         "caveat": "Ages outside 0..%d years are dropped as parse artefacts, and "
                   "negative ages (a cited year after the thesis year) are dropped "
                   "rather than clamped; both counts are reported. Theses before "
                   "%d are too few to support a per-discipline number and are "
                   "excluded from the trend." % (MAX_AGE, DIGITISATION_YEAR - 20)},
        {"id": "flow-circularity", "applies_to": "field_flow*",
         "caveat": "A cited work has no field in the data, so its field is "
                   "inferred from its citers, which makes the diagonal partly true "
                   "by construction and cannot be got round. What can be got round "
                   "is a single thesis deciding a work's field: a work is placed "
                   "only if its plurality field still holds the required share "
                   "after any one citing thesis is deleted, and that identical test "
                   "is then applied to every citer of that work. The same numbers "
                   "on the permissive rule (plain plurality share, a citer's own "
                   "vote allowed to place the work it cites) are in "
                   "field_flow_estimator.csv. The two do NOT bracket the truth: a work "
                   "is placed only when one field holds at least the required "
                   "share of its citers, so every placed work contributes at "
                   "least that share to the diagonal by construction, and both "
                   "rules inherit that floor. The gap between them measures "
                   "sensitivity to the placement rule, not the size of the "
                   "circularity, and the true diagonal is below both. "
                   + bracket_txt + " Do not read "
                   "either as a measurement of how much a field reads itself."},
        {"id": "flow-estimator-superseded", "applies_to": "field_flow*",
         "caveat": "Earlier output from this module used a leave-one-out rule that "
                   "tested each citer against the field of the OTHER citers. It is "
                   "not even-handed: for a work with k citers of which m are from "
                   "field d, a d-citer faced (m-1)/(k-1) while a non-d citer faced "
                   "m/(k-1), so the marginal home reads were dropped and the "
                   "marginal cross-field reads kept, and the winner could flip — a "
                   "work cited by two Law and two Social theses produced four "
                   "cross-field reads and no home reads at all. It deflated the "
                   "diagonal and inflated the off-diagonal — the quantity that "
                   "section is about. Its numbers are still computed and printed "
                   "in field_flow_estimator.csv, labelled superseded, so the "
                   "correction can be checked; they are not a bound and must not "
                   "be quoted."},
        {"id": "flow-coverage", "applies_to": "field_flow*",
         "caveat": "The matrix's coverage is small, and stating it in works "
                   "understates that: most works have too few citers to place, "
                   "which reads as a remark about obscure works when it is a "
                   "remark about most of the graph. " + reach_txt + " It is a "
                   "thin and non-random slice, not a census."},
        {"id": "flow-import-stability", "applies_to": "field_flow_profile.csv",
         "caveat": "An import is listed only where the lower end of a 95% interval "
                   "on its lift clears 1.0, and cells are ranked on that bound "
                   "rather than on the raw ratio, because ranking on the ratio "
                   "handed the headline to the smallest admissible cell every time "
                   "and once printed an association weaker than chance under the "
                   "heading 'strongest imports'. The interval is taken on the "
                   "number of distinct works behind a cell rather than the pair "
                   "count, since several pairs can come from one work; it still "
                   "assumes those works are independent of each other, so it is a "
                   "floor on the uncertainty and not a ceiling. "
                   "top_import_agreeing_settings reports how many of the "
                   "alternative --min-citers and --home-share settings return the "
                   "same top import: below full agreement, the headline is as much "
                   "a property of the parameters as of the data. Most fields have "
                   "no import that clears the bound, and that is the finding for "
                   "those fields."},
        {"id": "flow-specificity", "applies_to": "field_flow*",
         "caveat": "Whether a work belongs to any field at all is decided by "
                   "bridges.py's policy, %s: a work cited by more theses than "
                   "MAX_TOTAL_CITING is universal methods literature and is owned "
                   "by nobody, and a work spread across more than MAX_FIELDS "
                   "subfields is a bridge. Both are counted separately rather than "
                   "silently folded into the matrix."
                   % ("imported directly" if bridges_ok else
                      "restated inline because the import failed")},
        {"id": "not-about-people", "applies_to": "all",
         "caveat": "These are counts of what reference lists contain. A most-cited "
                   "list is a description of a literature, not a ranking of "
                   "scholars, and a surname in it is a bibliographic key produced "
                   "by a lossy parser, not an identified person."},
        {"id": "publishable", "applies_to": "all",
         "caveat": "Every artefact here is derived data: counts, medians, "
                   "matrices, and normalised bibliographic titles of third-party "
                   "published works taken from reference lists. No thesis text is "
                   "read or emitted; the cite.raw column is never touched. Safe to "
                   "publish."},
    ]


# ---------------------------------------------------------------- coverage

def coverage(docs: Docs) -> dict:
    """Per-discipline reference-parser coverage — the qualifier on everything."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for did, disc in sorted(docs.disc.items()):
        rw, ent, par = docs.stat.get(did, (0, 0, 0))
        a = agg[disc]
        a[0] += 1
        a[1] += 1 if par else 0
        a[2] += rw
        a[3] += ent
        a[4] += par

    per = {}
    for disc in sorted(agg):
        n, with_refs, rw, ent, par = agg[disc]
        est = rw / WORDS_PER_ENTRY
        per[disc] = {
            "theses": n,
            "theses_with_parsed_refs": with_refs,
            "ref_words": rw,
            "candidate_entries": ent,
            "parsed_entries": par,
            "pct_theses_with_parsed_refs": round(100 * with_refs / max(n, 1), 1),
            "parsed_per_thesis": round(par / max(with_refs, 1), 1),
            "candidate_rate": round(100 * par / max(ent, 1), 1),
            "est_recall": round(100 * par / max(est, 1.0), 1),
        }
    tot_ent = sum(v["candidate_entries"] for v in per.values())
    tot_par = sum(v["parsed_entries"] for v in per.values())
    tot_rw = sum(v["ref_words"] for v in per.values())
    n_all = sum(v["theses"] for v in per.values())
    n_with = sum(v["theses_with_parsed_refs"] for v in per.values())
    lo = min(per, key=lambda d: (per[d]["est_recall"], d))
    hi = max(per, key=lambda d: (per[d]["est_recall"], d))
    per_overall = {
        "entries": tot_ent, "parsed": tot_par,
        "candidate_rate": round(100 * tot_par / max(tot_ent, 1), 1),
        "est_recall": round(100 * tot_par / max(tot_rw / WORDS_PER_ENTRY, 1.0), 1),
        "min_recall": per[lo]["est_recall"], "min_recall_disc": lo,
        "max_recall": per[hi]["est_recall"], "max_recall_disc": hi,
    }
    return {"per_discipline": per, "overall": per_overall,
            "theses_total": n_all, "theses_without_edges": n_all - n_with}


# ---------------------------------------------------------------- 1. canon

def canon(works, docs, keystats, out: Path, top: int, top_sub: int,
          min_citers: int, min_field_citers: int, min_theses: int,
          field_of) -> dict:
    """Most-cited works per discipline and per subfield, with the collisions shown."""
    keep = [w for w in works if w.n >= min_citers]

    # Two denominators, because they differ by a factor of ~2.4 in the sciences
    # and a percentage quoted against the wrong one overstates a work's reach.
    # The rate is computed against theses that parsed at all, since a thesis with
    # no parsed reference list could not have cited anything; the field's full
    # thesis count travels beside it so the gap is visible rather than implied.
    disc_den = Counter()
    sub_den = Counter()
    disc_all = Counter()
    sub_all = Counter()
    for did, disc in sorted(docs.disc.items()):
        sub = field_of(disc, docs.sub.get(did, ""))
        disc_all[disc] += 1
        sub_all[sub] += 1
        if docs.stat.get(did, (0, 0, 0))[2] <= 0:
            continue                        # nothing parsed: it can cite nothing
        disc_den[disc] += 1
        sub_den[sub] += 1

    per_disc: dict[str, Counter] = defaultdict(Counter)
    per_sub: dict[str, Counter] = defaultdict(Counter)
    for i, w in enumerate(keep):
        for src in w.srcs:
            d = docs.disc.get(src)
            if not d:
                continue
            per_disc[d][i] += 1
            per_sub[field_of(d, docs.sub.get(src, ""))][i] += 1

    dropped = {"not_a_title": 0, "below_min_field_citers": 0}

    def table(buckets, denom, total, n_top, min_n):
        rows = []
        for field in sorted(buckets):
            den = denom.get(field, 0)
            if den < min_n:
                continue
            ranked = sorted(buckets[field].items(),
                            key=lambda kv: (-kv[1], keep[kv[0]].surname,
                                            keep[kv[0]].year, keep[kv[0]].title))
            rank = 0
            for i, n in ranked:
                w = keep[i]
                # A page range the segmenter mistook for a title identifies no
                # work, so it cannot be a field's most-cited one. It stays in
                # canon_works.csv, flagged, but it does not head a table.
                if w.resolution == "not-a-title":
                    dropped["not_a_title"] += 1
                    continue
                if n < min_field_citers:
                    dropped["below_min_field_citers"] += 1
                    continue
                rank += 1
                if rank > n_top:
                    break
                rows.append([field, rank, w.surname, w.year, w.title, n,
                             round(100 * n / den, 2), den, total.get(field, 0),
                             w.n, round(100 * n / w.n, 1), w.key_works,
                             w.key_citers, w.key_unresolved,
                             round(100 * w.share_of_key, 1), w.resolution])
        return rows

    header = ["field", "rank", "cited_surname", "cited_year", "cited_title",
              "citing_theses_in_field", "pct_of_field_theses_with_parsed_refs",
              "field_theses_with_parsed_refs", "field_theses_total",
              "citing_theses_total", "pct_of_citers_from_field",
              "works_in_key", "citers_of_key", "unattributable_citers_of_key",
              "pct_of_key", "resolution"]
    d_rows = table(per_disc, disc_den, disc_all, top, min_theses)
    s_rows = table(per_sub, sub_den, sub_all, top_sub, min_theses)
    write_csv(out / "canon_discipline.csv", header, d_rows)
    write_csv(out / "canon_subfield.csv", header, s_rows)

    write_csv(out / "canon_works.csv",
              ["cited_surname", "cited_year", "cited_title", "citing_theses",
               "works_in_key", "citers_of_key", "unattributable_citers_of_key",
               "pct_of_key", "top_discipline", "top_discipline_theses",
               "pct_from_top_discipline", "disciplines_citing", "resolution"],
              [[w.surname, w.year, w.title, w.n, w.key_works, w.key_citers,
                w.key_unresolved, round(100 * w.share_of_key, 1),
                *_top_disc(w, docs), w.resolution] for w in keep])

    write_csv(out / "canon_resolution.csv",
              ["citers_of_key", "cited_surname", "cited_year", "works_in_key",
               "unattributable_citers"],
              [list(r) for r in keystats["worst_keys"]])
    return {"works_reported": len(keep),
            "disciplines_reported": len({r[0] for r in d_rows}),
            "subfields_reported": len({r[0] for r in s_rows}),
            "min_field_citers": min_field_citers,
            "rows_dropped_not_a_title": dropped["not_a_title"],
            "rows_dropped_below_min_field_citers":
                dropped["below_min_field_citers"]}


def _top_disc(w: Work, docs: Docs):
    c = Counter(docs.disc[s] for s in w.srcs if s in docs.disc)
    if not c:
        return ["", 0, 0.0, 0]
    top = min(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return [top[0], top[1], round(100 * top[1] / sum(c.values()), 1), len(c)]


# ------------------------------------------------------------- 2. citation age

def citation_age(citations_db: Path, docs: Docs, out: Path,
                 min_theses: int, band: int) -> dict:
    """Per-thesis citation-age distributions, then per-discipline age and half-life."""
    per_src: dict[str, Counter] = defaultdict(Counter)
    dropped_future = dropped_range = dropped_nodoc = kept = 0
    con = _ro(citations_db)
    try:
        for src, cyear in con.execute("SELECT src, year FROM cite ORDER BY src"):
            dyear = docs.year.get(src)
            if dyear is None:
                dropped_nodoc += 1
                continue
            age = dyear - (cyear or 0)
            if age < 0:
                dropped_future += 1
                continue
            if age > MAX_AGE:
                dropped_range += 1
                continue
            per_src[src][age] += 1
            kept += 1
    finally:
        con.close()

    # per thesis
    rows = []
    thesis: dict[str, tuple[int, float, float]] = {}
    for src in sorted(per_src):
        c = per_src[src]
        n = sum(c.values())
        ages = []
        for a in sorted(c):
            ages.extend([a] * c[a])
        med = median(ages)
        price = 100 * sum(v for a, v in c.items() if a <= 5) / n
        thesis[src] = (n, med, price)
        rows.append([src, docs.disc.get(src, ""), docs.year.get(src, ""), n,
                     round(med, 1), round(price, 1),
                     round(100 * sum(v for a, v in c.items() if a > 20) / n, 1)])
    write_csv(out / "citation_age_thesis.csv",
              ["thesis_id", "discipline", "thesis_year", "parsed_references",
               "median_citation_age", "pct_refs_5y_or_newer", "pct_refs_over_20y"],
              rows)

    # per discipline
    by_disc: dict[str, Counter] = defaultdict(Counter)
    med_disc: dict[str, list[float]] = defaultdict(list)
    price_disc: dict[str, list[float]] = defaultdict(list)
    for src, c in per_src.items():
        d = docs.disc.get(src)
        if not d:
            continue
        by_disc[d].update(c)
        n, med, price = thesis[src]
        med_disc[d].append(med)
        price_disc[d].append(price)

    drows, summary = [], {}
    for d in sorted(by_disc):
        c = by_disc[d]
        n = sum(c.values())
        if len(med_disc[d]) < min_theses:
            continue
        ages = []
        for a in sorted(c):
            ages.extend([a] * c[a])
        # The Price index used to be thesis-weighted while pct_over_20y beside it
        # was edge-weighted, in the same row and under adjacent headings. The two
        # weightings differ by up to 5.9 points on the same field, so both are
        # now emitted and both are named for their weighting.
        rec = {
            "theses": len(med_disc[d]),
            "edges": n,
            "median_age_edges": round(median(ages), 1),
            "half_life_years": round(half_life(c), 2),
            "median_age_theses": round(median(med_disc[d]), 1),
            "p25_age": round(_pct(c, 25), 1),
            "p75_age": round(_pct(c, 75), 1),
            "p90_age": round(_pct(c, 90), 1),
            "price_index_pct_edges":
                round(100 * sum(v for a, v in c.items() if a <= 5) / n, 1),
            "price_index_pct_theses":
                round(sum(price_disc[d]) / len(price_disc[d]), 1),
            "pct_over_20y_edges":
                round(100 * sum(v for a, v in c.items() if a > 20) / n, 1),
        }
        summary[d] = rec
        drows.append([d] + [rec[k] for k in ("theses", "edges", "median_age_edges",
                                             "half_life_years", "median_age_theses",
                                             "p25_age", "p75_age", "p90_age",
                                             "price_index_pct_edges",
                                             "price_index_pct_theses",
                                             "pct_over_20y_edges")])
    write_csv(out / "citation_age_discipline.csv",
              ["discipline", "theses", "citation_edges", "median_age_edges",
               "half_life_years", "median_age_theses", "p25_age", "p75_age",
               "p90_age", "price_index_pct_edges", "price_index_pct_theses",
               "pct_over_20y_edges"], drows)

    # trend
    cells: dict[tuple[str, int], list] = defaultdict(
        lambda: [[], [], Counter(), 0, 0, 0])
    for src, c in per_src.items():
        d, y = docs.disc.get(src), docs.year.get(src)
        if not d or y is None or y < DIGITISATION_YEAR - 20:
            continue
        b = (y // band) * band
        cell = cells[(d, b)]
        n, med, price = thesis[src]
        cell[0].append(med)
        cell[1].append(price)
        cell[2].update(c)
        cell[3] += docs.blank.get(src, 0)
        rw, ent, par = docs.stat.get(src, (0, 0, 0))
        cell[4] += ent
        cell[5] += par

    trows, series = [], defaultdict(list)
    for (d, b) in sorted(cells):
        meds, prices, c, blanks, ent, par = cells[(d, b)]
        if len(meds) < min_theses:
            continue
        mt = median(meds)
        n_edges = sum(c.values())
        trows.append([d, b, f"{b}-{b + band - 1}", len(meds), n_edges,
                      round(mt, 1), round(half_life(c), 2),
                      round(100 * sum(v for a, v in c.items() if a <= 5) /
                            max(n_edges, 1), 1),
                      round(sum(prices) / len(prices), 1),
                      round(blanks / len(meds), 2),
                      round(100 * par / max(ent, 1), 1),
                      1 if b < DIGITISATION_YEAR else 0])
        series[d].append((b + (band - 1) / 2, mt, len(meds), b))
    write_csv(out / "citation_age_trend.csv",
              ["discipline", "band_start", "band", "theses", "citation_edges",
               "median_age_theses", "half_life_years", "price_index_pct_edges",
               "price_index_pct_theses", "mean_no_text_pages", "parse_rate_pct",
               "pre_digitisation_confound"], trows)

    trend = {}
    for d in sorted(series):
        pts = series[d]
        post = [p for p in pts if p[3] >= DIGITISATION_YEAR]
        # With no pre-digitisation band the two slopes are the same fit on the
        # same points, so their agreement is arithmetic and says nothing about
        # the confound. Only a field with bands on both sides can be checked.
        trend[d] = {
            "bands": len(pts),
            "bands_before_%d" % DIGITISATION_YEAR: len(pts) - len(post),
            "first_band": pts[0][3], "last_band": pts[-1][3],
            "first_median_age": round(pts[0][1], 1),
            "last_median_age": round(pts[-1][1], 1),
            "slope_years_per_decade": slope_per_decade([(p[0], p[1], p[2]) for p in pts]),
            "slope_years_per_decade_post_%d" % DIGITISATION_YEAR:
                slope_per_decade([(p[0], p[1], p[2]) for p in post]),
        }
    return {"summary": summary, "trend": trend,
            "edges_kept": kept,
            "edges_dropped_cited_year_after_thesis": dropped_future,
            "edges_dropped_age_over_%d" % MAX_AGE: dropped_range,
            "edges_dropped_no_usable_thesis_year": dropped_nodoc}


def _pct(counts: Counter, pct: float) -> float:
    total = sum(counts.values())
    target, run = total * pct / 100.0, 0
    for age in sorted(counts):
        run += counts[age]
        if run >= target:
            return float(age)
    return float(max(counts)) if counts else float("nan")


# ------------------------------------------------------------- 3. field flow

def poisson_lower_95(obs: int) -> float:
    """Byar's closed form for the lower end of a 95% interval on a count.

    A lift computed on a cell of thirty is not comparable with one computed on a
    cell of two thousand, and ranking on the raw ratio hands the headline to the
    smallest admissible cell every time. Ranking on the lower bound does not.
    """
    if obs <= 0:
        return 0.0
    return obs * (1 - 1 / (9 * obs) - Z95 / (3 * obs ** 0.5)) ** 3


def _prepare_citers(works, docs, field_of):
    """Per work: its citers' disciplines and its subfield spread, computed once.

    The placement rule is run seven times — twice for the estimator bracket, once
    for the superseded rule it replaced, four times for the stability grid — so
    the per-citer lookups are hoisted out of it.
    """
    prepared = []
    for w in works:
        discs, subs = [], set()
        for s in sorted(w.srcs):
            d = docs.disc.get(s)
            if not d:
                continue
            discs.append(d)
            subs.add(field_of(d, docs.sub.get(s, "")))
        prepared.append((w.n, tuple(discs), len(subs)))
    return prepared


def _place(prepared, min_citers, home_share, max_total_citing, max_fields,
           discount: int) -> dict:
    """Give each work one home field, then count every citer's edge to it.

    `discount` carries the whole methodological argument in one integer. For a
    work with k citers of which m are from the plurality field h:

      1 — strict. h keeps the work only if it still holds `home_share` of the
          citers once any single citing thesis is deleted: (m-1)/(k-1). No one
          thesis can hand a work to a field, which is what the superseded
          leave-one-out rule was reaching for.
      0 — permissive. The plain plurality share m/k. A citer's own vote helps
          place the work it is citing: the circularity undiluted.

    Both apply the *same* test to every citer of a given work, which is the
    property the superseded rule lacked and the reason it was replaced.

    Every work lands in exactly one bucket, so the accounting closes.
    """
    flow: Counter = Counter()
    cell_works: Counter = Counter()
    unowned: Counter = Counter()
    universal: Counter = Counter()
    bridging: Counter = Counter()
    n_thin = n_universal = n_bridge = n_placed = n_unplaced = 0

    for n_srcs, discs, spread in prepared:
        k = len(discs)
        if n_srcs < min_citers or k < min_citers:
            n_thin += 1
            continue
        if n_srcs > max_total_citing:
            n_universal += 1
            for d in discs:
                universal[d] += 1
            continue
        if spread > max_fields:
            n_bridge += 1
            for d in discs:
                bridging[d] += 1
            continue
        counts = Counter(discs)
        home, m = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        den = k - discount
        if den > 0 and (m - discount) / den >= home_share:
            n_placed += 1
            for d in discs:
                flow[(d, home)] += 1
            for d in set(discs):
                cell_works[(d, home)] += 1
        else:
            n_unplaced += 1
            for d in discs:
                unowned[d] += 1

    return {"flow": flow, "cell_works": cell_works, "unowned": unowned,
            "universal": universal, "bridging": bridging,
            "works_too_thin": n_thin, "works_universal_excluded": n_universal,
            "works_bridging_excluded": n_bridge, "works_placed": n_placed,
            "works_evaluated_but_not_placed": n_unplaced,
            "works_seen": len(prepared)}


def _place_superseded(prepared, min_citers, home_share, max_total_citing,
                      max_fields) -> dict:
    """The rule this module used to ship, kept only to report the size of its bias.

    It decided the home field for each citer from the *other* citers. That reads
    as even-handed and is not: a home citer was tested on (m-1)/(k-1) while a
    cross-field citer was tested on m/(k-1) at the same work, so the marginal
    home reads were dropped and the marginal cross-field reads were kept, and the
    argmax could flip so that a work cited by two Law and two Social theses
    produced four cross-field reads and no home reads at all. A correction whose
    size the reader cannot see is not much of a correction, so it is measured
    rather than described.
    """
    flow: Counter = Counter()
    for n_srcs, discs, spread in prepared:
        k = len(discs)
        if n_srcs < min_citers or k < min_citers or n_srcs > max_total_citing:
            continue
        if spread > max_fields:
            continue
        ranked = sorted(Counter(discs).items(), key=lambda kv: (-kv[1], kv[0]))
        for d in discs:
            if k - 1 <= 0:
                continue
            best_name, best_n = None, -1
            for name, c in ranked:
                cc = c - (1 if name == d else 0)
                if cc > best_n or (cc == best_n and best_name is not None
                                   and name < best_name):
                    best_name, best_n = name, cc
            if best_n / (k - 1) >= home_share:
                flow[(d, best_name)] += 1
    return flow


def _lifts(flow: Counter, cell_works: Counter, discs: list[str]):
    """observed / expected-if-independent for every cell, with a lower bound."""
    grand = sum(flow.values())
    row_tot: Counter = Counter()
    col_tot: Counter = Counter()
    for (a, b), v in flow.items():
        row_tot[a] += v
        col_tot[b] += v
    cells = {}
    for a in discs:
        for b in discs:
            obs = flow.get((a, b), 0)
            nw = cell_works.get((a, b), 0)
            exp = row_tot[a] * col_tot[b] / grand if grand else 0.0
            lift = obs / exp if exp > 0 else None
            # The interval is taken on the number of distinct works behind the
            # cell rather than on the pair count, because several pairs can come
            # from one work and are not independent observations of anything.
            lo = lift * poisson_lower_95(nw) / nw if lift and nw else None
            cells[(a, b)] = (obs, nw, exp, lift, lo)
    return cells, grand, row_tot, col_tot


def _imports(cells, discs: list[str], a: str):
    """A field's external cells that are above chance, best-supported first.

    A cell whose lower bound does not clear 1.0 is not evidence of an import and
    is not listed as one. The old ranking took the largest raw ratio whatever it
    was, which put 'Social sciences 0.98x' — an association *weaker* than chance
    — under the heading "strongest imports".
    """
    out = []
    for b in discs:
        if b == a:
            continue
        obs, nw, _, lift, lo = cells[(a, b)]
        if lo is not None and lo > 1.0:
            out.append((b, obs, nw, lift, lo))
    return sorted(out, key=lambda t: (-t[4], t[0]))


def _diagonal(flow: Counter) -> int:
    return sum(v for (a, b), v in flow.items() if a == b)


def field_flow(works, docs, out: Path, min_citers: int, home_share: float,
               max_total_citing: int, max_fields: int, field_of,
               cite_key_pairs: int, est_recall: float) -> dict:
    """Discipline x discipline citation flow, on a rule that treats every citer
    of a work alike, with the residual circularity bracketed rather than asserted."""
    discs = sorted(set(docs.disc.values()))
    prepared = _prepare_citers(works, docs, field_of)

    strict = _place(prepared, min_citers, home_share, max_total_citing,
                    max_fields, 1)
    loose = _place(prepared, min_citers, home_share, max_total_citing,
                   max_fields, 0)
    old = _place_superseded(prepared, min_citers, home_share, max_total_citing,
                            max_fields)
    flow = strict["flow"]
    unowned, universal, bridging = (strict["unowned"], strict["universal"],
                                    strict["bridging"])
    cells, grand, row_tot, col_tot = _lifts(flow, strict["cell_works"], discs)
    ranked_imports = {a: _imports(cells, discs, a) for a in discs}

    # How much of the headline is the data and how much is the parameter choice.
    agree: Counter = Counter()
    for d_mc, d_hs in STABILITY_GRID:
        alt = _place(prepared, max(1, min_citers + d_mc),
                     min(1.0, max(0.05, home_share + d_hs)),
                     max_total_citing, max_fields, 1)
        acells, _, _, _ = _lifts(alt["flow"], alt["cell_works"], discs)
        for a in discs:
            alt_top = _imports(acells, discs, a)
            here = ranked_imports[a]
            if (alt_top[0][0] if alt_top else None) == \
                    (here[0][0] if here else None):
                agree[a] += 1

    rows = []
    for a in discs:
        for b in discs:
            obs, nw, exp, lift, lo = cells[(a, b)]
            rows.append([a, b, obs, nw,
                         round(100 * obs / max(row_tot[a], 1), 2),
                         round(exp, 1),
                         round(lift, 3) if lift is not None else "",
                         round(lo, 3) if lo is not None else "",
                         1 if a == b else 0])
    write_csv(out / "field_flow.csv",
              ["citing_discipline", "home_discipline_of_cited_work",
               "citing_theses_x_works", "works_behind_cell",
               "pct_of_citing_discipline", "expected_if_independent", "lift",
               "lift_lower_95", "is_diagonal"], rows)

    profile = []
    for a in discs:
        tot = row_tot[a] + unowned[a] + universal[a] + bridging[a]
        if not tot:
            continue
        imp = ranked_imports[a]
        self_obs, _, _, self_lift, self_lo = cells[(a, a)]
        row = [a, tot, row_tot[a],
               round(100 * self_obs / max(row_tot[a], 1), 1),
               round(self_lift, 2) if self_lift is not None else "",
               round(self_lo, 2) if self_lo is not None else "",
               round(100 * unowned[a] / tot, 1),
               round(100 * universal[a] / tot, 1),
               round(100 * bridging[a] / tot, 1),
               len(imp), agree[a], len(STABILITY_GRID)]
        for slot in (0, 1):
            e = imp[slot] if len(imp) > slot else None
            row += [e[0] if e else "", e[1] if e else 0, e[2] if e else 0,
                    round(e[3], 2) if e else "", round(e[4], 2) if e else ""]
        profile.append(row)
    write_csv(out / "field_flow_profile.csv",
              ["discipline", "attributable_citations", "citations_to_owned_works",
               "pct_owned_citations_to_self", "self_lift", "self_lift_lower_95",
               "pct_to_unowned_works", "pct_to_universal_works",
               "pct_to_bridging_works", "imports_above_chance",
               "top_import_agreeing_settings", "settings_tried",
               "top_import", "top_import_pairs", "top_import_works",
               "top_import_lift", "top_import_lift_lower_95",
               "second_import", "second_import_pairs", "second_import_works",
               "second_import_lift", "second_import_lift_lower_95"], profile)

    # The bracket. The superseded rule sits outside it, which is the point: it
    # was never a bound on anything, and its diagonal cannot be read as one.
    est_rows = []
    for label, f, note in (
        ("strict", flow,
         "primary. (m-1)/(k-1) >= home_share; a work's field must survive the "
         "deletion of any one citing thesis. Same test for every citer."),
        ("permissive", loose["flow"],
         "comparator. m/k >= home_share; a citer's own vote helps place the work "
         "it cites. Same test for every citer. Read as the loose end of the "
         "diagonal, not as a second finding."),
        ("superseded-asymmetric", old,
         "NOT A BOUND, DO NOT QUOTE. The rule this module used to ship: each "
         "citer tested against the other citers, so a home read faced a stricter "
         "test than a cross-field read of the same work. Reported only so the "
         "size of the correction is visible."),
    ):
        n = sum(f.values())
        dg = _diagonal(f)
        est_rows.append([label, n, dg, round(100 * dg / max(n, 1), 1),
                         round(100 * (n - dg) / max(n, 1), 1), note])
    write_csv(out / "field_flow_estimator.csv",
              ["estimator", "placed_pairs", "diagonal_pairs", "pct_diagonal",
               "pct_off_diagonal", "note"], est_rows)

    # Coverage, in citation terms rather than in works. Stating only that most
    # works have too few citers to place reads as a remark about obscure works;
    # it is really a remark about most of the graph.
    pct_of_key_pairs = 100 * grand / max(cite_key_pairs, 1)
    reach = {
        "placed_thesis_work_pairs": grand,
        "distinct_citer_key_pairs_in_cite": cite_key_pairs,
        "pct_of_parsed_citer_key_pairs": round(pct_of_key_pairs, 2),
        "est_pct_of_all_reference_edges":
            round(pct_of_key_pairs * est_recall / 100, 2),
        "est_parser_recall_pct": est_recall,
    }

    write_json(out / "field_flow_matrix.json", {
        "disciplines": discs,
        "counts": {a: {b: cells[(a, b)][0] for b in discs} for a in discs},
        "row_share_pct": {a: {b: round(100 * cells[(a, b)][0] /
                                       max(row_tot[a], 1), 2) for b in discs}
                          for a in discs},
        "lift": {a: {b: (round(cells[(a, b)][3], 3)
                         if cells[(a, b)][3] is not None else None)
                     for b in discs} for a in discs},
        "lift_lower_95": {a: {b: (round(cells[(a, b)][4], 3)
                                  if cells[(a, b)][4] is not None else None)
                              for b in discs} for a in discs},
        "reach": reach,
        "note": "lift = observed / expected-if-independent, so a large field "
                "does not dominate by size alone, and lift_lower_95 is the end "
                "of a 95% interval taken on the number of distinct works behind "
                "the cell. A cell whose lower bound does not clear 1.0 is not "
                "evidence of an import. The diagonal is partly true by "
                "construction; see caveat flow-circularity.",
    })

    return {"works_seen": strict["works_seen"],
            "works_too_thin": strict["works_too_thin"],
            "works_universal_excluded": strict["works_universal_excluded"],
            "works_bridging_excluded": strict["works_bridging_excluded"],
            "works_with_a_home_field": strict["works_placed"],
            "works_evaluated_but_not_placed":
                strict["works_evaluated_but_not_placed"],
            "attributed_citations": grand,
            "reach": reach,
            "estimators": {r[0]: {"placed_pairs": r[1], "diagonal_pairs": r[2],
                                  "pct_diagonal": r[3], "pct_off_diagonal": r[4]}
                           for r in est_rows},
            "stability_settings": len(STABILITY_GRID),
            "flow": {a: {b: cells[(a, b)][0] for b in discs} for a in discs},
            "row_totals": dict(row_tot), "col_totals": dict(col_tot),
            "unowned": dict(unowned), "universal": dict(universal),
            "bridging": dict(bridging)}

# ---------------------------------------------------------------- reporting

def rule(title: str, width: int = 96) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def run(args: argparse.Namespace) -> int:
    bad = validate(args)
    if bad:
        print(f"canon: {bad}", file=sys.stderr)
        return 2
    for path in (args.citations_db, args.corpus_db):
        if not Path(path).exists():
            print(f"canon: no database at {path}", file=sys.stderr)
            return 1
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        print(f"canon: --out {out} exists and is not a directory", file=sys.stderr)
        return 2
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"canon: cannot create --out {out}: {exc}", file=sys.stderr)
        return 2

    max_total, max_fields, field_of, bridges_ok = bridge_policy()

    print("canon: loading thesis metadata ...", file=sys.stderr)
    docs = Docs(Path(args.corpus_db), Path(args.citations_db))
    cov = coverage(docs)

    print("canon: resolving (surname, year) keys into works ...", file=sys.stderr)
    works, keystats = resolve_works(Path(args.citations_db))
    cite_key_pairs = distinct_citer_key_pairs(Path(args.citations_db))

    payload: dict = {
        "parameters": {
            "min_citers": args.min_citers,
            "min_field_citers": args.min_field_citers,
            "min_theses": args.min_theses,
            "home_share": args.home_share, "band_years": args.band,
            "min_key_citers": MIN_KEY_CITERS, "top": args.top,
            "title_containment": CONTAINMENT, "title_sort_ratio": SORT_RATIO,
            "min_title_tokens": MIN_TITLE_TOKENS,
            "bridges_max_total_citing": max_total,
            "bridges_max_fields": max_fields,
            "bridges_imported": bridges_ok,
        },
        "corpus": {
            "theses": len(docs.disc),
            "theses_with_parsed_references":
                sum(1 for d in docs.disc if docs.stat.get(d, (0, 0, 0))[2] > 0),
            "citation_edges": sum(s[2] for s in docs.stat.values()),
            "distinct_citer_key_pairs": cite_key_pairs,
        },
        "coverage": cov,
        "key_resolution": {k: v for k, v in keystats.items() if k != "worst_keys"},
    }

    write_csv(out / "citation_coverage.csv",
              ["discipline", "theses", "theses_with_parsed_refs",
               "pct_theses_with_parsed_refs", "ref_words", "candidate_entries",
               "parsed_entries", "parsed_per_thesis", "pct_of_candidates_parsed",
               "est_pct_of_all_entries_parsed"],
              [[d] + [cov["per_discipline"][d][k] for k in
                      ("theses", "theses_with_parsed_refs",
                       "pct_theses_with_parsed_refs", "ref_words",
                       "candidate_entries", "parsed_entries", "parsed_per_thesis",
                       "candidate_rate", "est_recall")]
               for d in sorted(cov["per_discipline"])])

    rule("tg canon — what a field actually reads")
    n_with = payload["corpus"]["theses_with_parsed_references"]
    n_all = payload["corpus"]["theses"]
    print(f"  {payload['corpus']['citation_edges']:,} parsed citation edges from "
          f"{n_with:,} of {n_all:,} theses "
          f"({100 * (n_all - n_with) / max(n_all, 1):.0f}% of the corpus yields no "
          f"edge at all)")
    print(f"  parser sees {cov['overall']['candidate_rate']}% of parenthesised-year "
          f"candidates, an estimated {cov['overall']['est_recall']}% of all "
          f"reference entries")
    print(f"  estimated recall by field: {cov['overall']['min_recall']}% "
          f"({cov['overall']['min_recall_disc']}) .. "
          f"{cov['overall']['max_recall']}% ({cov['overall']['max_recall_disc']})")
    print(f"  {keystats['keys_with_min_citers']:,} (surname, year) keys with "
          f"{MIN_KEY_CITERS}+ citing theses resolved into "
          f"{keystats['works_resolved']:,} works; "
          f"{keystats['keys_split_into_several_works']:,} keys held more than one; "
          f"{keystats['attributable_pct']}% of citing theses attributable")

    if args.sample:
        print("\n  HOW THE MOST-CITED KEYS RESOLVED (check by eye)")
        idx: dict[tuple[str, int], list[Work]] = defaultdict(list)
        for w in works:
            idx[(w.surname, w.year)].append(w)
        for n_src, surname, year, n_works, unres in keystats["worst_keys"][:args.sample]:
            print(f"    {surname} {year}: {n_src:,} citing theses -> {n_works} work(s)"
                  f", {unres} unattributable")
            for w in sorted(idx[(surname, year)], key=lambda w: (-w.n, w.title))[:4]:
                print(f"        {w.n:>5,}  {w.title[:66]}")

    if args.section in ("all", "canon"):
        print("canon: ranking works per field ...", file=sys.stderr)
        payload["canon"] = canon(works, docs, keystats, out, args.top,
                                 args.top_subfield, args.min_citers,
                                 args.min_field_citers, args.min_theses, field_of)
        rule("1. CANON — most-cited works, per discipline")
        cn = payload["canon"]
        print(f"  a work needs {args.min_citers}+ citing theses corpus-wide and "
              f"{args.min_field_citers}+ from the field itself to appear in that "
              f"field's list;\n  {cn['rows_dropped_not_a_title']:,} candidate rows "
              f"were dropped as not-a-title (a page range the segmenter took for a "
              f"title)\n  and {cn['rows_dropped_below_min_field_citers']:,} for "
              f"thin in-field support. The percentage below is of the field's "
              f"theses that\n  parsed a reference list, which is well short of the "
              f"field's theses — both denominators are in the CSV.")
        with (out / "canon_discipline.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for disc in sorted({r["field"] for r in rows}):
            top = [r for r in rows if r["field"] == disc][:3]
            print(f"\n  {disc}  ({int(top[0]['field_theses_with_parsed_refs']):,} of "
                  f"{int(top[0]['field_theses_total']):,} theses parsed a "
                  f"reference list)")
            for r in top:
                print(f"    {int(r['citing_theses_in_field']):>5,} theses "
                      f"({float(r['pct_of_field_theses_with_parsed_refs']):>4.1f}% "
                      f"of those)  "
                      f"{r['cited_surname'].title()} ({r['cited_year']}) "
                      f"{r['cited_title'][:44]}")
                print(f"           key holds {r['works_in_key']} work(s), "
                      f"{r['pct_of_key']}% of its citers are this one, "
                      f"{r['resolution']}")

    if args.section in ("all", "age"):
        print("canon: measuring citation age ...", file=sys.stderr)
        age = citation_age(Path(args.citations_db), docs, out,
                           args.min_theses, args.band)
        payload["citation_age"] = age
        rule("2. CITATION AGE — how old is the literature a field stands on")
        print(f"  {age['edges_kept']:,} edges used; dropped "
              f"{age['edges_dropped_cited_year_after_thesis']:,} with a cited year "
              f"after the thesis year, "
              f"{age['edges_dropped_age_over_%d' % MAX_AGE]:,} over {MAX_AGE} years, "
              f"{age['edges_dropped_no_usable_thesis_year']:,} with no usable "
              f"thesis year")
        print(f"\n  {'discipline':<34}{'theses':>7}{'half-life':>10}"
              f"{'med/thesis':>11}{'p90':>5}{'<=5y%/e':>9}{'<=5y%/t':>9}"
              f"{'>20y%/e':>9}{'yr/dec':>8}{'yr/dec ' + str(DIGITISATION_YEAR) + '+':>14}")
        n_checkable = 0
        for d in sorted(age["summary"]):
            st = age["summary"][d]
            t = age["trend"].get(d, {})
            sl = t.get("slope_years_per_decade")
            sp = t.get("slope_years_per_decade_post_%d" % DIGITISATION_YEAR)
            # No pre-digitisation band means the two fits use the same points,
            # so their agreement is arithmetic, not evidence of anything.
            same = t.get("bands_before_%d" % DIGITISATION_YEAR, 0) == 0
            n_checkable += 0 if same else 1
            post = ("n/a" if sp is None else
                    f"{sp:+.2f}{'=' if same else ' '}")
            print(f"  {d:<34}{st['theses']:>7,}{st['half_life_years']:>10.2f}"
                  f"{st['median_age_theses']:>11.1f}{st['p90_age']:>5.0f}"
                  f"{st['price_index_pct_edges']:>9.1f}"
                  f"{st['price_index_pct_theses']:>9.1f}"
                  f"{st['pct_over_20y_edges']:>9.1f}"
                  f"{(f'{sl:+.2f}' if sl is not None else 'n/a'):>8}"
                  f"{post:>14}")
        print(f"\n  <=5y%/e and >20y%/e are edge-weighted (every parsed reference "
              f"votes); <=5y%/t is the\n  same Price index thesis-weighted (every "
              f"thesis votes once). They differ by up to six\n  points on the same "
              f"field, so both are printed and both are named in the CSV — an "
              f"earlier\n  version put the thesis-weighted one next to "
              f"edge-weighted neighbours under one heading.")
        print(f"\n  yr/dec is the weighted slope of the thesis-weighted median age "
              f"against thesis\n  year — positive means the field is standing on "
              f"older literature over time. A '=' on\n  the "
              f"{DIGITISATION_YEAR}+ figure means the field has no pre-"
              f"{DIGITISATION_YEAR} band at all, so the two columns\n  are the "
              f"same fit on the same points and their agreement is arithmetic. "
              f"Only {n_checkable} of\n  {len(age['summary'])} fields can be "
              f"checked against the digitisation confound that way; for\n  the "
              f"rest, read citation_age_trend.csv, which carries mean_no_text_pages "
              f"and parse_rate\n  beside every age it reports.")

    if args.section in ("all", "flow"):
        print("canon: deriving field flow ...", file=sys.stderr)
        flow = field_flow(works, docs, out, args.min_citers, args.home_share,
                          max_total, max_fields, field_of, cite_key_pairs,
                          cov["overall"]["est_recall"])
        payload["field_flow"] = flow
        rule("3. FIELD FLOW — which field's literature a field imports")
        print(f"  {flow['works_seen']:,} works evaluated: "
              f"{flow['works_too_thin']:,} have fewer than {args.min_citers} "
              f"citing theses and cannot be placed at all;\n  "
              f"{flow['works_universal_excluded']:,} were universal "
              f"(>{max_total} citers, bridges.py policy); "
              f"{flow['works_bridging_excluded']:,} were bridging "
              f"(>{max_fields} subfields);\n  "
              f"{flow['works_evaluated_but_not_placed']:,} had no field holding "
              f">={args.home_share:.0%} of their citers after deleting any one of "
              f"them; {flow['works_with_a_home_field']:,} were placed.")
        rch = flow["reach"]
        print(f"\n  WHAT THE MATRIX COVERS. It rests on "
              f"{rch['placed_thesis_work_pairs']:,} thesis-work pairs out of "
              f"{rch['distinct_citer_key_pairs_in_cite']:,} distinct\n  "
              f"(thesis, surname, year) pairs the parser produced — "
              f"{rch['pct_of_parsed_citer_key_pairs']}% — and the parser itself "
              f"sees an estimated\n  {rch['est_parser_recall_pct']}% of reference "
              f"entries, so the matrix describes roughly "
              f"{rch['est_pct_of_all_reference_edges']}% of the reference\n  edges "
              f"actually in the corpus. It is a thin, non-random slice, not a "
              f"census.")
        est = flow["estimators"]
        print(f"\n  THE DIAGONAL IS BRACKETED, NOT MEASURED. A cited work has no "
              f"field in the data, so the\n  strict rule "
              f"({est['strict']['pct_diagonal']}% of "
              f"{est['strict']['placed_pairs']:,} placed pairs on the diagonal) "
              f"and the permissive one "
              f"({est['permissive']['pct_diagonal']}% of\n  "
              f"{est['permissive']['placed_pairs']:,}) differ only in whether a "
              f"citer's own vote may place the work it cites. Read the\n  "
              f"diagonal as that range. The asymmetric rule this module used to "
              f"ship reported "
              f"{est['superseded-asymmetric']['pct_diagonal']}%,\n  which is "
              f"outside the range and below both: see field_flow_estimator.csv.")
        with (out / "field_flow_profile.csv").open(encoding="utf-8") as fh:
            prof = list(csv.DictReader(fh))
        print(f"\n  {'discipline':<34}{'self %':>8}{'self x':>8}{'unowned':>9}"
              f"{'univ':>6}{'bridge':>8}   imports above chance (lift, pairs, "
              f"agreement)")
        for r in sorted(prof, key=lambda r: r["discipline"]):
            imports = ", ".join(
                f"{r[a]} {r[c]}x/{int(r[b]):,}"
                f"{'?' if int(r[b]) < MIN_FLOW_CELL else ''}"
                for a, b, c in
                (("top_import", "top_import_pairs", "top_import_lift"),
                 ("second_import", "second_import_pairs",
                  "second_import_lift")) if r[a])
            # "none above chance" needs the agreement count as much as a named
            # import does: for some fields it is the answer under one setting of
            # four, which is not the same claim at all.
            if not imports:
                imports = "none above chance"
            imports += (f" [{r['top_import_agreeing_settings']}/"
                        f"{r['settings_tried']}]")
            print(f"  {r['discipline']:<34}{float(r['pct_owned_citations_to_self']):>7.1f}%"
                  f"{float(r['self_lift'] or 0):>8.1f}"
                  f"{float(r['pct_to_unowned_works']):>8.1f}%"
                  f"{float(r['pct_to_universal_works']):>5.1f}%"
                  f"{float(r['pct_to_bridging_works']):>7.1f}%   {imports}")
        print(f"\n  An import is listed only where the lower end of a 95% interval "
              f"on its lift clears 1.0,\n  so a cell weaker than chance is not "
              f"reported as a strength; the interval is taken on the\n  number of "
              f"distinct works behind the cell, not the pair count. [k/"
              f"{flow['stability_settings']}] is how many of "
              f"{flow['stability_settings']}\n  alternative "
              f"(--min-citers, --home-share) settings return the same answer "
              f"— including\n  'none above chance', which is a claim too. Anything "
              f"below {flow['stability_settings']}/{flow['stability_settings']} is "
              f"as much a property of the\n  parameters as of the data. A '?' "
              f"marks a cell of fewer than {MIN_FLOW_CELL} thesis-work pairs.")
        print(f"\n  self % is the share of a field's placed citations that stay "
              f"home. self x is that against a\n  null of reassigning every placed "
              f"pair at random across the whole matrix; since the matrix\n  is "
              f"close to diagonal, a small field's expected diagonal is tiny and "
              f"its self x is huge in\n  consequence. The two columns do not rank "
              f"fields the same way and self x is not\n  comparable between fields "
              f"of very different size — read it only as 'far more than\n  chance'. "
              f"Both are inflated by the circularity below.")

    cv = caveats(cov, keystats, bridges_ok, payload.get("field_flow"))
    payload["caveats"] = cv
    write_csv(out / "canon_caveats.csv", ["id", "applies_to", "caveat"],
              [[c["id"], c["applies_to"], c["caveat"]] for c in cv])
    write_json(out / "canon.json", payload)

    rule("CAVEATS — every number above needs these standing next to it")
    for c in cv:
        print(f"\n  [{c['id']}]  ({c['applies_to']})")
        body = c["caveat"].split()
        line = "   "
        for word in body:
            if len(line) + len(word) + 1 > 92:
                print(line)
                line = "   "
            line += " " + word
        print(line)

    print()
    for f in sorted(p.name for p in out.glob("*") if p.is_file()):
        print(f"  wrote {out / f}")
    print("\n  All of the above is derived data — counts, medians, matrices and "
          "bibliographic\n  titles of third-party works. No thesis text is read or "
          "emitted. Safe to publish.")
    return 0


if __name__ == "__main__":
    # tg.py turns an exception into one clear line for the plugin path; run
    # standalone, nothing does, and a malformed --db should not end in a trace.
    ap = argparse.ArgumentParser(description=HELP)
    add_args(ap)
    try:
        raise SystemExit(run(ap.parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (sqlite3.Error, OSError, ValueError) as exc:
        print("%s: %s: %s" % (NAME, type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
