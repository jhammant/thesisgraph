#!/usr/bin/env python3
"""
tg_search.py — find a thesis in the corpus without writing SQL.

Everything else in this toolkit takes an OAI identifier as its starting point,
and until now the only way to get one was to open corpus.db and write a LIKE
query by hand. That is a bad front door: `LIKE '%wind turbine%'` cannot rank,
cannot stem, scans 28,647 rows for every attempt, and silently misses the
thesis that says "wind turbines" or "aeolian".

So this builds a proper index instead — SQLite FTS5 over title, creator and
abstract, ranked by BM25 with the title weighted well above the abstract,
alongside the structured filters the corpus actually has (discipline, sub-field,
year, author, length). A query answers in a few milliseconds because the index
is built once and cached; nothing here streams the 3.8 GB of extracted text.

`--stats` answers the other question a newcomer has, which is "what am I
holding?": documents by status and discipline and decade, how much text was
extracted, how far screening and citation parsing got. Its labels say exactly
what its numbers count, because nobody reading that screen is in a position to
check: a reference entry *detected* is not one *parsed* (17,159 theses against
13,262 here), and a (surname, year) citation key is not a work — tg_canon.py
resolves keys into works by clustering their titles, and nothing here pretends
to have done that.

COPYRIGHT. The index at corpus/search.db embeds titles, author names and
abstracts. Those are repository metadata rather than thesis body text, but they
are still other people's words, so the index lives under corpus/ with the rest
of the gitignored data and is not something to commit or redistribute. Search
output quotes a short fragment of an abstract to show why a hit matched; pass
--no-snippet for output containing nothing but metadata and links. The full-text
body of a thesis is never read, indexed or printed by this module.

    tg search --rebuild                       # build corpus/search.db (once)
    tg search "wind turbine wake"             # ranked, readable
    tg search fatigue --discipline engineering --from 2015
    tg search --author "smith" --min-words 60000 --json
    tg search --stats                         # corpus shape at a glance

Determinism: every ranking breaks ties on the document id, every listed
collection is sorted, and nothing recorded in the index or printed by it comes
from the clock or an RNG.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DEFAULT_DB = CORPUS / "corpus.db"
DEFAULT_INDEX = CORPUS / "search.db"

NAME = "search"
HELP = "full-text search over titles, abstracts and authors, plus --stats"

SCHEMA_VERSION = "1"

# A title match is worth far more than an abstract match, and an author match
# sits between the two: searching a surname should surface that person's thesis
# ahead of every thesis that merely cites them in its abstract.
BM25_WEIGHTS = (10.0, 5.0, 1.0)

FIELDS = ("title", "creator", "abstract")
OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})

# Structured columns are searched with SQL, not FTS, so they stay exact.
SCHEMA = """
CREATE TABLE docs (
  docid           INTEGER PRIMARY KEY,
  id              TEXT UNIQUE NOT NULL,
  repo            TEXT,
  title           TEXT,
  creator         TEXT,
  abstract        TEXT,
  year            INTEGER,
  publisher       TEXT,
  discipline      TEXT,
  discipline_conf REAL,
  subfield        TEXT,
  status          TEXT,
  words           INTEGER,
  pages           INTEGER,
  url             TEXT
);
CREATE INDEX docs_discipline ON docs(discipline);
CREATE INDEX docs_subfield   ON docs(subfield);
CREATE INDEX docs_year       ON docs(year);
CREATE INDEX docs_status     ON docs(status);
CREATE INDEX docs_words      ON docs(words);
CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
-- External content: the FTS index points back at docs rather than holding a
-- second copy of every abstract. No prefix= index — measured on this corpus it
-- adds 37 MB (31%) and saves 0.03 ms on a prefix query, which is no trade at all.
CREATE VIRTUAL TABLE fts USING fts5(
  title, creator, abstract,
  content='docs', content_rowid='docid',
  tokenize="porter unicode61 remove_diacritics 2"
);
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def open_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection. corpus/ is data this toolkit reads, not writes."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def norm(s: str | None) -> str:
    """Collapse whitespace. Harvested titles and abstracts carry the PDF's own
    CRLFs, which wreck both a table layout and an FTS snippet."""
    return re.sub(r"\s+", " ", s or "").strip()


def like_literal(s: str) -> str:
    """Escape LIKE metacharacters so a name is matched as itself.

    `--author '%'` is a person searching for a name, not a request for the whole
    corpus; `_` sits inside plenty of mangled harvested creator strings. The
    query is parameterised either way, so this is semantics, not injection."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def landing_url(doc_id: str, landing: str | None, pdf_url: str | None) -> str:
    """The page a human should open for a document.

    `doc.landing` is misnamed: harvest stores the first non-PDF dc:identifier
    there, which for most records is a supplementary .zip or .flac rather than
    the record page. The record page is implied by the OAI identifier, and that
    derivation was checked against all 26,513 stored URLs in this corpus — every
    one of them sits under it — so derive it, and fall back only when the id has
    an unexpected shape.
    """
    m = re.match(r"^oai:([^:]+):(\d+)$", doc_id or "")
    if m:
        return f"https://{m.group(1)}/id/eprint/{m.group(2)}/"
    return norm(landing) or norm(pdf_url) or ""


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clip(s: str, n: int) -> str:
    """Truncate to n visible characters — ANSI emphasis is invisible and must
    not be counted, or a highlighted line ends up shorter than an unhighlighted
    one for no reason a reader can see."""
    if len(ANSI.sub("", s)) <= n:
        return s
    out, seen, i = [], 0, 0                    # walk it, keeping escapes whole
    while i < len(s) and seen < n - 1:
        m = ANSI.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        out.append(s[i])
        seen += 1
        i += 1
    return "".join(out).rstrip() + "…" + ("\x1b[0m" if ANSI.search(s) else "")


def resolve(values: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    """Map user fragments onto the corpus's own vocabulary.

    Discipline and sub-field names are long ("History, philosophy and religion");
    nobody should have to type one exactly. Exact match beats prefix beats
    substring, and every survivor is kept, so `--discipline sci` deliberately
    means all four sciences rather than an arbitrary one of them.
    """
    hits: set[str] = set()
    unknown: list[str] = []
    for want in values:
        w = want.strip().lower()
        if not w:
            continue
        for rule in (lambda v: v.lower() == w,
                     lambda v: v.lower().startswith(w),
                     lambda v: w in v.lower()):
            found = [v for v in available if rule(v)]
            if found:
                hits.update(found)
                break
        else:
            unknown.append(want)
    return sorted(hits), unknown


def fts_expr(text: str, any_of: bool = False) -> str:
    """Turn what a person typed into an FTS5 MATCH expression that cannot throw.

    Raw input reaches FTS5's parser as syntax: a hyphen, a colon or a stray
    bracket in a perfectly reasonable query ("covid-19", "Smith: a study")
    raises rather than searching. So quote every term, while keeping the three
    things a person may reasonably mean to use: "a quoted phrase", an explicit
    AND/OR/NOT, and a field prefix such as title:learning.
    """
    parts: list[str] = []
    joiner = "OR" if any_of else "AND"
    for tok in re.findall(r'"[^"]*"|\S+', text or ""):
        if tok in OPERATORS:
            if parts and parts[-1] not in OPERATORS:
                parts.append(tok)
            continue
        neg = False
        if tok.startswith("-") and len(tok) > 1:
            neg, tok = True, tok[1:]
        field = ""
        m = re.match(r"^(%s):(.+)$" % "|".join(FIELDS), tok, re.I)
        if m:
            field, tok = m.group(1).lower(), m.group(2)
        star = "*" if tok.endswith("*") else ""
        body = tok[:-1] if star else tok
        body = body.strip('"')
        body = re.sub(r'"', "", body).strip()
        # A token with nothing alphanumeric in it ("???", "(((") tokenises to
        # nothing, so it would silently match nothing at all rather than being
        # ignored as the noise it is.
        if not re.search(r"\w", body):
            continue
        term = f'"{body}"{star}'
        if field:
            term = f"{field} : {term}"
        if parts and parts[-1] not in OPERATORS:
            parts.append("NOT" if neg else joiner)
        elif neg and not parts:
            # FTS5 has no unary NOT; a query that is only an exclusion has no
            # candidate set to exclude from.
            raise ValueError(f"cannot start a query with an exclusion (-{body})")
        elif neg:
            parts[-1] = "NOT"
        parts.append(term)
    while parts and parts[-1] in OPERATORS:
        parts.pop()
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# index construction
# --------------------------------------------------------------------------- #

def build(src: Path, dst: Path, quiet: bool = False) -> int:
    if not src.exists():
        print(f"search: no corpus database at {src}", file=sys.stderr)
        return 1

    # Build beside the target and swap it in atomically, so an interrupted
    # rebuild can never leave a half-built index where a working one was. The
    # pid keeps two concurrent rebuilds out of each other's way.
    tmp = dst.with_name(f"{dst.name}.building-{os.getpid()}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    con = open_ro(src)
    out = sqlite3.connect(tmp)
    done = False
    try:
        out.execute("PRAGMA journal_mode=OFF")
        out.execute("PRAGMA synchronous=OFF")
        out.executescript(SCHEMA)

        # Metadata only — a few tens of MB, not the 3.8 GB of extracted text,
        # which this module never opens.
        try:
            rows = con.execute(
                "SELECT id, repo, title, creator, abstract, year, publisher, "
                "discipline, discipline_conf, subfield, status, words, pages, "
                "landing, pdf_url FROM doc").fetchall()
        except sqlite3.DatabaseError as exc:
            # An existing path that is not the corpus database — search.db and
            # corpus.db live in the same directory, so this is one typo away.
            print(f"search: cannot read {src} as a corpus database ({exc})",
                  file=sys.stderr)
            return 1

        # Rowids follow the corpus's own order, not SQLite's scan order, so
        # every tie-break downstream is a property of the data.
        def sort_key(r):
            m = re.match(r"^oai:([^:]+):(\d+)$", r[0] or "")
            return (r[1] or "", m.group(1) if m else "", int(m.group(2)) if m else 0, r[0])

        payload = []
        for n, r in enumerate(sorted(rows, key=sort_key), 1):
            payload.append((
                n, r[0], r[1], norm(r[2]), norm(r[3]), norm(r[4]),
                r[5], norm(r[6]), r[7], r[8], r[9], r[10], r[11], r[12],
                landing_url(r[0], r[13], r[14])))
        out.executemany(
            "INSERT INTO docs(docid,id,repo,title,creator,abstract,year,publisher,"
            "discipline,discipline_conf,subfield,status,words,pages,url) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", payload)

        if not quiet:
            print(f"  {len(payload):,} documents copied; building FTS index ...",
                  file=sys.stderr)
        out.execute("INSERT INTO fts(fts) VALUES('rebuild')")
        out.execute("INSERT INTO fts(fts) VALUES('optimize')")

        # No build timestamp: an artefact of this project records what the data
        # was, never when the command happened to run.
        out.executemany("INSERT INTO meta(k,v) VALUES(?,?)", sorted([
            ("schema_version", SCHEMA_VERSION),
            ("source", src.name),
            ("source_docs", str(len(payload))),
            ("bm25_weights", ",".join(f"{w:g}" for w in BM25_WEIGHTS)),
            ("columns", ",".join(FIELDS)),
            ("tokenizer", "porter unicode61 remove_diacritics 2"),
        ]))
        out.commit()
        out.execute("VACUUM")
        out.commit()
        done = True
    finally:
        out.close()
        con.close()
        if not done:
            tmp.unlink(missing_ok=True)

    os.replace(tmp, dst)
    if not quiet:
        print(f"  wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return 0


def check_fresh(index: sqlite3.Connection, src: Path) -> None:
    """Warn — never fail — when the corpus has moved on since the index was built."""
    try:
        indexed = int(dict(index.execute("SELECT k,v FROM meta")).get("source_docs", "0"))
        con = open_ro(src)
        live = con.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
        con.close()
    except Exception:
        return
    if live != indexed:
        print(f"search: index holds {indexed:,} documents but {src.name} now has "
              f"{live:,} — re-run with --rebuild", file=sys.stderr)


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #

COLUMNS = ("d.id, d.title, d.creator, d.year, d.publisher, d.discipline, "
           "d.discipline_conf, d.subfield, d.status, d.words, d.pages, d.url")

SORTS = {
    "relevance": None,                  # only meaningful with a query
    "year": "d.year DESC, d.docid",
    "oldest": "d.year, d.docid",
    "words": "d.words DESC, d.docid",
    "title": "d.title, d.docid",
    "id": "d.docid",
}


def build_where(args, disciplines, subfields) -> tuple[list[str], list]:
    where: list[str] = []
    params: list = []
    if disciplines:
        where.append("d.discipline IN (%s)" % ",".join("?" * len(disciplines)))
        params += disciplines
    if subfields:
        where.append("d.subfield IN (%s)" % ",".join("?" * len(subfields)))
        params += subfields
    if args.author:
        where.append("(%s)" % " OR ".join(
            [r"LOWER(d.creator) LIKE ? ESCAPE '\'"] * len(args.author)))
        params += [f"%{like_literal(a.strip().lower())}%" for a in args.author]
    if args.year is not None:
        where.append("d.year = ?")
        params.append(args.year)
    if args.year_from is not None:
        where.append("d.year >= ?")
        params.append(args.year_from)
    if args.year_to is not None:
        where.append("d.year <= ?")
        params.append(args.year_to)
    if args.min_words is not None:
        where.append("d.words >= ?")
        params.append(args.min_words)
    return where, params


def run_query(con, expr, where, params, order, limit, markers):
    """One page of results plus the total, matched and counted the same way."""
    if expr:
        frm = "FROM fts JOIN docs d ON d.docid = fts.rowid"
        cond = ["fts MATCH ?"] + where
        head = [expr] + params
        score = "bm25(fts, %s) AS score" % ", ".join(f"{w:g}" for w in BM25_WEIGHTS)
        snip = "snippet(fts, 2, ?, ?, '…', 14) AS snip"
        select = f"SELECT {COLUMNS}, {score}, {snip} {frm}"
        sel_params = [markers[0], markers[1], expr] + params
        default_order = "score, d.docid"
    else:
        frm = "FROM docs d"
        cond = list(where)
        head = list(params)
        select = (f"SELECT {COLUMNS}, 0.0 AS score, "
                  f"substr(d.abstract, 1, 220) AS snip {frm}")
        sel_params = list(params)
        default_order = SORTS["year"]

    tail = (" WHERE " + " AND ".join(cond)) if cond else ""
    total = con.execute(f"SELECT COUNT(*) {frm}{tail}", head).fetchone()[0]
    rows = con.execute(f"{select}{tail} ORDER BY {order or default_order} LIMIT ?",
                       sel_params + [limit]).fetchall()
    return rows, total


def to_record(row, want_snippet: bool) -> dict:
    rec = {
        "id": row[0], "title": row[1], "creator": row[2], "year": row[3],
        "publisher": row[4], "discipline": row[5], "discipline_conf": row[6],
        "subfield": row[7], "status": row[8], "words": row[9], "pages": row[10],
        "url": row[11], "score": round(row[12], 4),
    }
    if want_snippet:
        rec["snippet"] = norm(row[13])
    return rec


def render(rows, total, args, expr, hidden) -> None:
    width = max(60, args.width)
    id_w = max((len(r[0]) for r in rows), default=2)
    disc_w = 20
    title_w = max(24, width - id_w - disc_w - 15)

    if rows:
        print(f"{'#':>3}  {'ID':<{id_w}}  {'YEAR':>4}  {'DISCIPLINE':<{disc_w}}  TITLE")
    for n, r in enumerate(rows, 1):
        year = str(r[3]) if r[3] else "?"
        print(f"{n:>3}  {r[0]:<{id_w}}  {year:>4}  "
              f"{clip(r[5] or '-', disc_w):<{disc_w}}  {clip(r[1] or '(untitled)', title_w)}")
        meta = [r[11] or "(no url)"]
        if r[2]:
            meta.append(clip(r[2], 40))
        if r[9]:
            meta.append(f"{r[9]:,} words")
        if r[8] and r[8] != "ok":
            meta.append(f"status={r[8]}")
        print("     " + clip(" · ".join(meta), width - 5))
        if not args.no_snippet:
            snip = norm(r[13])
            if snip:
                print("     " + clip(snip, width - 5))
    shown = len(rows)
    if rows:
        print()
    if total == 0:
        print("no matches" + (f" for {expr}" if expr else ""))
    else:
        print(f"{shown:,} of {total:,} match" + ("es" if total != 1 else ""))
    if hidden:
        print(f"({hidden:,} further match{'es' if hidden != 1 else ''} excluded by "
              f"--status {args.status}; use --status any to include them)")


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def bar(n: int, mx: int, cells: int = 24) -> str:
    return "█" * ((1 + round((cells - 1) * n / mx)) if n and mx else 0)


def collect_stats(db: Path, index: Path) -> dict:
    con = open_ro(db)
    q = con.execute
    out: dict = {"database": str(db)}
    out["documents"] = q("SELECT COUNT(*) FROM doc").fetchone()[0]
    out["by_status"] = sorted(q("SELECT status, COUNT(*) FROM doc GROUP BY 1"))
    out["by_discipline"] = sorted(
        q("SELECT discipline, COUNT(*) FROM doc WHERE status='ok' "
          "AND discipline<>'' GROUP BY 1"), key=lambda r: (-r[1], r[0]))
    out["subfields"] = q("SELECT COUNT(DISTINCT subfield) FROM doc "
                         "WHERE subfield IS NOT NULL AND subfield<>''").fetchone()[0]

    # Years below 1900 in this corpus are metadata errors (2, 13, 204), not
    # Victorian theses; bucket them honestly rather than inventing a decade.
    dec = sorted(q("SELECT (year/10)*10, COUNT(*) FROM doc WHERE status='ok' "
                   "AND year >= 1900 GROUP BY 1"))
    out["by_decade"] = dec
    out["year_unknown"] = q("SELECT COUNT(*) FROM doc WHERE status='ok' "
                            "AND (year IS NULL OR year < 1900)").fetchone()[0]
    out["year_range"] = q("SELECT MIN(year), MAX(year) FROM doc "
                          "WHERE status='ok' AND year >= 1900").fetchone()

    tot = q("SELECT SUM(words), SUM(tokens), SUM(sentences), SUM(pages), "
            "SUM(no_text_pages) FROM doc WHERE status='ok'").fetchone()
    out["text"] = {"words": tot[0] or 0, "tokens": tot[1] or 0,
                   "sentences": tot[2] or 0, "pages": tot[3] or 0,
                   "no_text_pages": tot[4] or 0}
    out["abstracts"] = q("SELECT COUNT(*) FROM doc WHERE status='ok' "
                         "AND abstract IS NOT NULL AND abstract<>''").fetchone()[0]
    out["doctoral"] = q("SELECT COUNT(*) FROM doc WHERE is_doctoral=1").fetchone()[0]
    out["low_confidence_discipline"] = q(
        "SELECT COUNT(*) FROM doc WHERE status='ok' AND discipline_conf < 0.5"
    ).fetchone()[0]
    out["by_discipline_method"] = sorted(
        q("SELECT discipline_method, COUNT(*) FROM doc WHERE status='ok' "
          "AND discipline_method IS NOT NULL GROUP BY 1"))
    out["pairs"] = q("SELECT COUNT(*) FROM pair").fetchone()[0]
    out["by_verdict"] = sorted(
        (("(unjudged)" if v is None else v, n) for v, n in
         q("SELECT verdict, COUNT(*) FROM pair GROUP BY 1")))
    con.close()

    cites = CORPUS / "citations.db"
    if cites.exists():
        c = open_ro(cites)
        n, ent, par, refw = c.execute(
            "SELECT COUNT(*), SUM(entries), SUM(parsed), SUM(ref_words) "
            "FROM srcstat").fetchone()
        out["citations"] = {
            "rows": c.execute("SELECT COUNT(*) FROM cite").fetchone()[0],
            "sources": n or 0, "entries": ent or 0, "parsed": par or 0,
            "reference_words": refw or 0,
            # Detected and parsed are different things and the gap is large:
            # entries>0 is "a reference list was found and segmented", parsed>0
            # is "something came out of it". Reporting the first as the second
            # flatters the parser by 3,897 theses.
            "sources_with_entries": c.execute(
                "SELECT COUNT(*) FROM srcstat WHERE entries > 0").fetchone()[0],
            "sources_with_parsed": c.execute(
                "SELECT COUNT(*) FROM srcstat WHERE parsed > 0").fetchone()[0],
            # NOT a count of works. tg_canon.py splits these keys into works by
            # clustering their titles, because `wang 2014` is one key and about
            # 300 papers; naming this "works" here would contradict that module.
            "distinct_surname_year_keys": c.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT surname, year FROM cite)"
            ).fetchone()[0],
        }
        c.close()

    ctx = CORPUS / "context.db"
    if ctx.exists():
        c = open_ro(ctx)
        out["contexts"] = {
            "rows": c.execute("SELECT COUNT(*) FROM ctx").fetchone()[0],
            "theses": c.execute("SELECT COUNT(DISTINCT src) FROM ctx").fetchone()[0],
        }
        c.close()

    files = []
    for p in sorted(CORPUS.glob("*.db")) + sorted(CORPUS.glob("*.json")):
        files.append((p.name, p.stat().st_size))
    out["files"] = files
    out["index"] = {"path": str(index), "present": index.exists(),
                    "bytes": index.stat().st_size if index.exists() else 0,
                    "documents": 0}
    if index.exists():
        try:
            c = open_ro(index)
            out["index"]["documents"] = c.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
            c.close()
        except sqlite3.DatabaseError:
            pass
    return out


def render_stats(s: dict) -> None:
    def pc(n, d):
        return f"{100.0 * n / d:5.1f}%" if d else "    - "

    docs = s["documents"]
    print(s["database"])
    print(f"  documents  {docs:>12,}")
    for status, n in sorted(s["by_status"], key=lambda r: (-r[1], r[0])):
        print(f"    {status:<10} {n:>12,}  {pc(n, docs)}")
    print(f"  of which flagged doctoral {s['doctoral']:>7,}  {pc(s['doctoral'], docs)}")

    t = s["text"]
    print("\n  extracted text (status='ok')")
    for label in ("words", "tokens", "sentences", "pages"):
        print(f"    {label:<14} {t[label]:>14,}")
    print(f"    {'pages w/o text':<14} {t['no_text_pages']:>14,}")
    ok = dict(s["by_status"]).get("ok", 0)
    print(f"    {'abstracts':<14} {s['abstracts']:>14,}  {pc(s['abstracts'], ok)} of ok")

    print(f"\n  by discipline (status='ok', {s['subfields']} sub-fields)")
    mx = max((n for _, n in s["by_discipline"]), default=0)
    for name, n in s["by_discipline"]:
        print(f"    {name:<34} {n:>7,}  {pc(n, ok)}  {bar(n, mx)}")
    print(f"    {'(low-confidence labels)':<34} {s['low_confidence_discipline']:>7,}"
          f"  {pc(s['low_confidence_discipline'], ok)}")
    for meth, n in s["by_discipline_method"]:
        print(f"      via {meth:<30} {n:>7,}")

    lo, hi = s["year_range"]
    print(f"\n  by decade (status='ok', {lo}-{hi})")
    mx = max((n for _, n in s["by_decade"]), default=0)
    for dec, n in s["by_decade"]:
        print(f"    {dec}s{'':<29} {n:>7,}  {pc(n, ok)}  {bar(n, mx)}")
    if s["year_unknown"]:
        print(f"    {'unknown / out of range':<34} {s['year_unknown']:>7,}")

    print(f"\n  screened pairs {s['pairs']:>10,}")
    for verdict, n in sorted(s["by_verdict"], key=lambda r: (-r[1], r[0])):
        print(f"    {verdict:<34} {n:>7,}  {pc(n, s['pairs'])}")

    c = s.get("citations")
    if c:
        print("\n  citations (corpus/citations.db)")
        print(f"    {'parsed reference entries':<34} {c['rows']:>10,}")
        print(f"    {'distinct (surname, year) keys':<34} "
              f"{c['distinct_surname_year_keys']:>10,}")
        print(f"    {'reference entries seen':<34} {c['entries']:>10,}"
              f"  {pc(c['parsed'], c['entries'])} parsed")
        print(f"    {'theses scanned for references':<34} {c['sources']:>10,}")
        print(f"    {'  with a reference entry detected':<34} "
              f"{c['sources_with_entries']:>10,}"
              f"  {pc(c['sources_with_entries'], c['sources'])}")
        print(f"    {'  with at least one entry parsed':<34} "
              f"{c['sources_with_parsed']:>10,}"
              f"  {pc(c['sources_with_parsed'], c['sources'])}")
        print("    a (surname, year) key is not a work: `wang 2014` is one key")
        print("    and roughly 300 papers — `tg canon` splits keys into works.")
    x = s.get("contexts")
    if x:
        print(f"\n  citation contexts {x['rows']:,} from {x['theses']:,} theses")

    print("\n  on disk")
    for name, size in s["files"]:
        print(f"    {name:<34} {size / 1e6:>9.1f} MB")
    i = s["index"]
    if i["present"]:
        note = f"{i['documents']:,} documents indexed"
        if i["documents"] != docs:
            note += " — stale, run --rebuild"
        print(f"\n  search index  {i['path']}\n    {note}")
    else:
        print(f"\n  search index  absent — run `tg search --rebuild`")


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #

def _capture(argv: list[str]) -> tuple[int, str]:
    ap = argparse.ArgumentParser()
    add_args(ap)
    buf = io.StringIO()
    # stderr too: a warning is part of what a run produced, so comparing two
    # runs compares the whole of it.
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = run(ap.parse_args(argv))
    return rc, buf.getvalue()


def selftest(index: Path) -> int:
    """Assert the behaviour this module claims, rather than describing it.

    The unit half needs no corpus at all. The integration half asserts
    invariants (filters actually filter, ranking is ordered, two identical runs
    are identical) rather than fixed result lists, so it stays true as the
    corpus grows.
    """
    checks: list[tuple[str, bool]] = []

    def ck(label: str, got, want=True) -> None:
        checks.append((label, got == want))

    ck("plain terms are quoted and ANDed",
       fts_expr("wind turbine"), '"wind" AND "turbine"')
    ck("punctuation cannot reach the FTS parser",
       fts_expr("covid-19 nursing"), '"covid-19" AND "nursing"')
    ck("phrases survive", fts_expr('"machine learning"'), '"machine learning"')
    ck("field prefixes survive", fts_expr("title:diagnosis"), 'title : "diagnosis"')
    ck("--any switches the joiner", fts_expr("a b", any_of=True), '"a" OR "b"')
    ck("a leading dash is an exclusion", fts_expr("a -b"), '"a" NOT "b"')
    ck("explicit operators are honoured", fts_expr("a OR b"), '"a" OR "b"')
    ck("a trailing operator is dropped", fts_expr("a AND"), '"a"')
    ck("prefix stars survive", fts_expr("zebra*"), '"zebra"*')
    ck("noise tokens are ignored", fts_expr("((( ???"), "")
    try:
        fts_expr("-only")
        ck("a query that is only an exclusion is refused", False)
    except ValueError:
        ck("a query that is only an exclusion is refused", True)

    ck("landing url comes from the id",
       landing_url("oai:etheses.whiterose.ac.uk:15", "", "x.pdf"),
       "https://etheses.whiterose.ac.uk/id/eprint/15/")
    ck("an odd id falls back to the stored url",
       landing_url("weird", "", "http://x/y.pdf"), "http://x/y.pdf")
    ck("whitespace is collapsed", norm("a\r\n b\t c"), "a b c")
    ck("clip counts visible characters only",
       len(ANSI.sub("", clip("\x1b[1mabcdef\x1b[0m", 4))), 4)
    ck("exact match beats substring",
       resolve(["law"], ["Law", "Lawful studies"])[0], ["Law"])
    ck("an unknown value is reported",
       resolve(["zzz"], ["Law"])[1], ["zzz"])
    ck("a LIKE wildcard in a name is escaped", like_literal("a%b_c"), r"a\%b\_c")
    ck("the escape character escapes itself", like_literal("a\\b"), r"a\\b")

    # A file that exists but is not this index: the commonest way to hit it is
    # --index corpus/corpus.db, which sits next door.
    with tempfile.NamedTemporaryFile("w", suffix=".db", delete=False) as fh:
        fh.write("not a database\n")
        junk = Path(fh.name)
    try:
        rc, msg = _capture(["x", "--index", str(junk)])
        ck("a file that is not an index is an error, not a crash", rc, 1)
        ck("and the message names the file", str(junk) in msg)
    finally:
        junk.unlink(missing_ok=True)

    if not index.exists():
        print(f"selftest: no index at {index}; unit checks only", file=sys.stderr)
    else:
        rc, a = _capture(["zebrafish", "-n", "5", "--index", str(index), "--no-color"])
        ck("a query runs", rc, 0)
        rc, b = _capture(["zebrafish", "-n", "5", "--index", str(index), "--no-color"])
        ck("two identical runs are byte-identical", a, b)

        rc, wide = _capture(["cell", "-n", "1", "--index", str(index), "--json"])
        rc2, narrow = _capture(["cell", "-n", "1", "--index", str(index), "--json",
                                "--discipline", "Biological sciences"])
        wide_n = json.loads(wide)["total"]
        narrow_j = json.loads(narrow)
        ck("a filter can only narrow", narrow_j["total"] <= wide_n)
        ck("the filter is actually applied",
           all(r["discipline"] == "Biological sciences" for r in narrow_j["results"]))

        rc, page = _capture(["water", "-n", "10", "--index", str(index), "--json"])
        rows = json.loads(page)["results"]
        ck("bm25 order is best first",
           all(rows[i]["score"] <= rows[i + 1]["score"] for i in range(len(rows) - 1)))
        ck("every row carries a landing url", all(r["url"] for r in rows))

        rc, byyear = _capture(["water", "--from", "2015", "--to", "2016",
                               "--index", str(index), "--json", "-n", "50"])
        ck("year bounds hold",
           all(2015 <= r["year"] <= 2016 for r in json.loads(byyear)["results"]))

        rc, hidden = _capture(["--author", "smith", "--min-words", "80000",
                               "--index", str(index), "--json", "-n", "50"])
        recs = json.loads(hidden)["results"]
        ck("author and length filters hold",
           all("smith" in r["creator"].lower() and r["words"] >= 80000 for r in recs))
        ck("a filter-only search still returns rows", bool(recs))

        one = json.loads(_capture(["oai:etheses.whiterose.ac.uk:15", "--json",
                                   "--index", str(index)])[1])
        ck("an id is looked up, not searched",
           [r["id"] for r in one["results"]], ["oai:etheses.whiterose.ac.uk:15"])

        rc, missing = _capture(["x", "--index", str(index.parent / "nope.db")])
        ck("a missing index is an error, not a crash", rc, 1)

        zero = json.loads(_capture(["water", "-n", "0", "--index", str(index),
                                    "--json"])[1])
        ck("-n 0 asks for the count alone",
           (len(zero["results"]), zero["total"] > 0), (0, True))

        rc, _ = _capture(["--from", "2020", "--to", "2010", "--index", str(index)])
        ck("an impossible year range is refused, not answered", rc, 2)
        rc, _ = _capture(["water", "--year", "2019", "--from", "2021",
                          "--index", str(index)])
        ck("--year outside --from/--to is refused too", rc, 2)

        pct = json.loads(_capture(["--author", "%", "-n", "1", "--json",
                                   "--index", str(index)])[1])
        ck("a LIKE wildcard in --author is a literal, not everybody",
           pct["total"], 0)

    bad = [label for label, ok in checks if not ok]
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def nonneg_int(value: str) -> int:
    """`-n 0` is a fair request for the total alone; `-n -5` is a typo, and
    silently clamping either one to a single result corrupts any script that
    computes a page size."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a whole number") from None
    if n < 0:
        raise argparse.ArgumentTypeError(
            "must be 0 or more (0 counts, shows nothing)")
    return n


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", nargs="*", default=[],
                   help='words to search for; "quoted phrases", AND/OR/NOT and '
                        'title:term / creator:term / abstract:term are honoured')
    p.add_argument("--rebuild", action="store_true",
                   help="build the search index from the corpus, then exit")
    p.add_argument("--stats", action="store_true",
                   help="print corpus shape instead of searching")
    p.add_argument("--selftest", action="store_true",
                   help="assert this module's own behaviour and exit")
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help="corpus database to index / describe (default: %(default)s)")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                   help="search index to use or build (default: %(default)s)")

    f = p.add_argument_group("filters")
    f.add_argument("--discipline", action="append", default=[], metavar="NAME",
                   help="discipline; a fragment is enough, repeatable")
    f.add_argument("--subfield", action="append", default=[], metavar="NAME",
                   help="sub-field; a fragment is enough, repeatable")
    f.add_argument("--author", action="append", default=[], metavar="NAME",
                   help="substring of the creator field, repeatable")
    f.add_argument("--year", type=int, help="exact year of award")
    f.add_argument("--from", dest="year_from", type=int, metavar="YEAR",
                   help="earliest year of award")
    f.add_argument("--to", dest="year_to", type=int, metavar="YEAR",
                   help="latest year of award")
    f.add_argument("--min-words", type=int, metavar="N",
                   help="only theses with at least N extracted words")
    f.add_argument("--status", default="ok",
                   choices=("ok", "pending", "failed", "any"),
                   help="extraction status; 'ok' means the text was extracted "
                        "(default: %(default)s)")

    o = p.add_argument_group("output")
    o.add_argument("-n", "--limit", type=nonneg_int, default=20,
                   help="results to show; 0 reports the total and lists nothing "
                        "(default: %(default)s)")
    o.add_argument("--sort", default="relevance", choices=sorted(SORTS),
                   help="ranking (default: %(default)s)")
    o.add_argument("--any", action="store_true",
                   help="match any word rather than all of them")
    o.add_argument("--json", action="store_true", help="machine-readable output")
    o.add_argument("--no-snippet", action="store_true",
                   help="omit abstract fragments — output is then metadata only")
    o.add_argument("--width", type=int, default=120,
                   help="output width (default: %(default)s; fixed, not the "
                        "terminal's, so runs are comparable)")
    o.add_argument("--no-color", action="store_true",
                   help="never emphasise matched words with ANSI codes")


def run(args: argparse.Namespace) -> int:
    index = Path(args.index)
    db = Path(args.db)

    if getattr(args, "selftest", False):
        return selftest(index)

    if args.stats:
        if not db.exists():
            print(f"search: no corpus database at {db}", file=sys.stderr)
            return 1
        try:
            s = collect_stats(db, index)
        except sqlite3.DatabaseError as exc:
            print(f"search: cannot read {db} as a corpus database ({exc})",
                  file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(s, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            render_stats(s)
        return 0

    if args.rebuild:
        return build(db, index, quiet=args.json)

    if not index.exists():
        print(f"search: no index at {index} — run `tg search --rebuild` first",
              file=sys.stderr)
        return 1

    # An unsatisfiable range is a mistake in the command, not a fact about the
    # corpus, and "no matches" would report it as if it were one.
    clash = []
    if (args.year_from is not None and args.year_to is not None
            and args.year_from > args.year_to):
        clash.append(f"--from {args.year_from} is later than --to {args.year_to}")
    if args.year is not None:
        if args.year_from is not None and args.year < args.year_from:
            clash.append(f"--year {args.year} is earlier than "
                         f"--from {args.year_from}")
        if args.year_to is not None and args.year > args.year_to:
            clash.append(f"--year {args.year} is later than --to {args.year_to}")
    if clash:
        print("search: no year can satisfy these filters — " + "; ".join(clash),
              file=sys.stderr)
        return 2

    query = " ".join(args.query).strip()
    filters_given = bool(args.discipline or args.subfield or args.author
                         or args.year is not None or args.year_from is not None
                         or args.year_to is not None or args.min_words is not None)
    if not query and not filters_given:
        print("search: give some words to search for, or at least one filter "
              "(see --help)", file=sys.stderr)
        return 2

    # The path exists, but existing is not the same as being this index:
    # corpus.db sits beside search.db, and a half-written or unrelated file
    # would otherwise reach the vocabulary query below and raise there.
    con = None
    try:
        con = open_ro(index)
        con.execute("SELECT 1 FROM docs LIMIT 1")
        con.execute("SELECT 1 FROM fts LIMIT 1")
    except sqlite3.DatabaseError as exc:
        if con is not None:
            con.close()
        print(f"search: {index} is not a search index ({exc}) — run "
              f"`tg search --rebuild` to build one", file=sys.stderr)
        return 1

    try:
        check_fresh(con, db)
        vocab = {
            "discipline": sorted(r[0] for r in con.execute(
                "SELECT DISTINCT discipline FROM docs WHERE discipline<>''")),
            "subfield": sorted(r[0] for r in con.execute(
                "SELECT DISTINCT subfield FROM docs "
                "WHERE subfield IS NOT NULL AND subfield<>''")),
        }
        disciplines, bad_d = resolve(args.discipline, vocab["discipline"])
        subfields, bad_s = resolve(args.subfield, vocab["subfield"])
        for label, bad in (("discipline", bad_d), ("sub-field", bad_s)):
            if bad:
                print(f"search: unknown {label}: {', '.join(sorted(bad))}",
                      file=sys.stderr)
                key = "discipline" if label == "discipline" else "subfield"
                print(f"search: known {label}s:", file=sys.stderr)
                for v in vocab[key][:40]:
                    print(f"  {v}", file=sys.stderr)
                if len(vocab[key]) > 40:
                    print(f"  ... and {len(vocab[key]) - 40} more", file=sys.stderr)
                return 2

        # A person holding an id from another tg command will paste it in
        # here; tokenising it as free text would find nothing useful.
        by_id = bool(re.match(r"^oai:[^:\s]+:\d+$", query))
        try:
            expr = "" if by_id else fts_expr(query, any_of=args.any)
        except ValueError as exc:
            print(f"search: {exc}", file=sys.stderr)
            return 2
        if query and not expr and not by_id:
            print("search: nothing searchable in that query", file=sys.stderr)
            return 2

        where, params = build_where(args, disciplines, subfields)
        if by_id:
            where.append("d.id = ?")
            params.append(query)
            args.status = "any"         # you asked for this document by name
        status_clause = None
        if args.status != "any":
            status_clause = len(where)
            where.append("d.status = ?")
            params.append(args.status)

        order = SORTS[args.sort]
        if args.sort == "relevance" and not expr:
            order = None                    # nothing to rank by; fall back to year
        markers = ("", "")
        if not args.no_color and not args.json and sys.stdout.isatty():
            markers = ("\x1b[1m", "\x1b[0m")

        try:
            rows, total = run_query(con, expr, where, params, order,
                                    max(0, args.limit), markers)
        except sqlite3.OperationalError as exc:
            print(f"search: {exc} (query was: {expr})", file=sys.stderr)
            return 2

        hidden = 0
        if status_clause is not None and total >= 0:
            bare_where = [w for i, w in enumerate(where) if i != status_clause]
            bare_params = [p for i, p in enumerate(params) if i != status_clause]
            _, all_total = run_query(con, expr, bare_where, bare_params, order, 1,
                                     markers)
            hidden = max(0, all_total - total)

        if args.json:
            payload = {
                "query": query,
                "fts": expr,
                "filters": {
                    "author": sorted(args.author),
                    "discipline": disciplines,
                    "max_year": args.year_to,
                    "min_words": args.min_words,
                    "min_year": args.year_from,
                    "status": args.status,
                    "subfield": subfields,
                    "year": args.year,
                },
                "sort": args.sort,
                "total": total,
                "excluded_by_status": hidden,
                "results": [to_record(r, not args.no_snippet) for r in rows],
            }
            print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            render(rows, total, args, expr, hidden)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="tg_search.py", description=HELP)
    add_args(ap)
    raise SystemExit(run(ap.parse_args()))
