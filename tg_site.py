#!/usr/bin/env python3
"""
tg_site.py — build one publishable static site for the corpus, in out/site/.

The toolkit currently emits four unrelated HTML blobs. Two of them (out/app.html,
out/viewer.html) embed thesis full text and can therefore never be published at
all; the other two are separate single-purpose apps with separate looks and no
way in from anywhere. There is no page that answers the first question a
stranger asks — what is this corpus, how big is it, and what is it not good for.

This builds that page, and three more around it, under one shell:

    index.html      what the corpus is, how it was built, limitations first
    search.html     client-side search over every thesis in the corpus
    methods.html    winnowing, the detection guarantee, the filters, the caveats
    findings.html   whatever tg_canon.py left in out/analytics/, or an honest
                    "not built yet" — never an invented number
    graph.html      the existing graph explorer, copied in unchanged

COPYRIGHT is the constraint that decides what may be in here. Everything this
module writes is derived data: counts, titles, years, subject labels and links
back to the repository record. Titles and bibliographic metadata are public
facts and are already published by make_graph_app.py --public. NO THESIS BODY
TEXT is emitted — not sentences, not abstracts, not matched fragments — so
out/site/ can be copied to a web host as it stands.

The two things that could break that guarantee are guarded rather than trusted:

  * out/analytics/ is written by another module. Its values are rendered
    verbatim but EVERY string is truncated at --max-string characters, with no
    exemption for any key, and each truncation is reported on stderr and on the
    page, so a stray passage cannot be published whole and cannot pass
    unnoticed either. An earlier version exempted a list of prose-sounding key
    names (caveat, note, method, source ...) from the cap and let 1,600
    characters through under them; that exemption keyed on the field's NAME and
    never on its content, so anything written under a name on the list was
    published unread. The cap is the guard, so the cap has no exceptions.
  * out/public/bridges.html quotes the sentence in which a thesis cites a work.
    That is a separate publishing decision, so it is linked but NOT copied in
    unless --include-bridges says so, and asking for it prints what it means.

Self-contained: one file per page, no CDN, no web fonts, no fetch. The search
index is gzipped and base64'd into the page and inflated with
DecompressionStream, the same trick make_bridge_app.py uses, so every page also
opens straight from file:// with no server.

Deterministic: rows sorted, JSON keys sorted, gzip mtime pinned to 0, no
wall-clock anywhere. Two builds over the same corpus are byte-identical, which
`--verify-determinism` checks rather than claims.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "site"
HELP = "build out/site/ — one publishable static site: search, methods, findings"

REPO_URL = "https://github.com/jhammant/thesisgraph"
RECORD_URL = "https://etheses.whiterose.ac.uk/id/eprint/"
EPRINT_ID = re.compile(r":(\d+)$")

# A title is bibliographic metadata, not thesis text, but a handful of records
# carry a 800-character "title" that is really an abstract pasted into the
# field. Cap it: long enough for any real title, short enough that the cap can
# never smuggle a paragraph of a thesis into a published page.
TITLE_CAP = 240

# Analytics values are rendered verbatim from a file this module did not write.
# 200 characters holds a work title, an author-date key or a short label; it
# does not hold a passage of a thesis.
MAX_STRING = 200
MAX_ROWS = 60
MAX_ANALYTICS_BYTES = 262144   # bigger than this is a data file, not a web page

PAGES = [
    ("index.html", "Overview"),
    ("search.html", "Search"),
    ("methods.html", "Methods"),
    ("findings.html", "Findings"),
    ("graph.html", "Graph"),
    ("bridges.html", "Bridges"),
]


# --------------------------------------------------------------------------
# corpus figures
# --------------------------------------------------------------------------

def _ro(path: Path) -> sqlite3.Connection:
    """Open read-only. corpus/ is data the toolkit reads and never writes."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def corpus_stats(db: Path, citations: Path | None) -> dict:
    """Every number the site quotes, computed once from the corpus itself.

    Nothing here is copied out of the README. The README's headline figures
    (95.5% of shared runs explained by a third thesis, 1.34% of verified pairs
    left as residual) fall out of these queries, which is the point: the prose
    and the data agree because the prose is generated from the data.
    """
    con = _ro(db)
    try:
        one = lambda q, *a: con.execute(q, a).fetchone()[0]      # noqa: E731
        ok = "WHERE status='ok'"
        s: dict = {
            "harvested": one("SELECT COUNT(*) FROM doc"),
            "docs": one(f"SELECT COUNT(*) FROM doc {ok}"),
            "words": one(f"SELECT SUM(words) FROM doc {ok}") or 0,
            "pages": one(f"SELECT SUM(pages) FROM doc {ok}") or 0,
            "sentences": one(f"SELECT SUM(sentences) FROM doc {ok}") or 0,
            "tokens": one(f"SELECT SUM(tokens) FROM doc {ok}") or 0,
            "disciplines": one(f"SELECT COUNT(DISTINCT discipline) FROM doc {ok}"),
            "subfields": one(f"SELECT COUNT(DISTINCT subfield) FROM doc {ok}"),
            "year_min": one(f"SELECT MIN(year) FROM doc {ok} AND year BETWEEN 1900 AND 2100"),
            "year_max": one(f"SELECT MAX(year) FROM doc {ok} AND year BETWEEN 1900 AND 2100"),
            "year_bad": one(f"SELECT COUNT(*) FROM doc {ok} AND "
                            "(year IS NULL OR year<1900 OR year>2100)"),
            "low_conf": one(f"SELECT COUNT(*) FROM doc {ok} AND discipline_conf<0.5"),
            "pairs": one("SELECT COUNT(*) FROM pair"),
            "verified": one("SELECT COUNT(*) FROM pair WHERE verdict IS NOT NULL"),
        }
        s["by_discipline"] = sorted(
            con.execute(f"SELECT discipline, COUNT(*) FROM doc {ok} GROUP BY 1"),
            key=lambda r: (-r[1], r[0]))
        s["by_decade"] = sorted(
            con.execute(f"SELECT (year/10)*10, COUNT(*) FROM doc {ok} "
                        "AND year BETWEEN 1900 AND 2100 GROUP BY 1"))
        s["by_method"] = sorted(
            con.execute(f"SELECT discipline_method, COUNT(*) FROM doc {ok} GROUP BY 1"),
            key=lambda r: (-r[1], r[0]))
        s["verdicts"] = sorted(
            con.execute("SELECT verdict, COUNT(*) FROM pair "
                        "WHERE verdict IS NOT NULL GROUP BY 1"),
            key=lambda r: (-r[1], r[0]))
        common, quoted, uniq = con.execute(
            "SELECT SUM(common_runs), SUM(quoted_runs), SUM(unique_runs) "
            "FROM pair WHERE verdict IS NOT NULL").fetchone()
        s["runs_common"], s["runs_quoted"], s["runs_unique"] = (
            common or 0, quoted or 0, uniq or 0)
    finally:
        con.close()

    n = s["docs"]
    s["all_pairs"] = n * (n - 1) // 2
    s["residual"] = dict(s["verdicts"]).get("residual", 0)
    total_runs = s["runs_common"] + s["runs_quoted"] + s["runs_unique"]
    s["common_pct"] = 100.0 * s["runs_common"] / total_runs if total_runs else 0.0
    s["residual_pct"] = 100.0 * s["residual"] / s["verified"] if s["verified"] else 0.0
    # What share of the screened candidates was actually adjudicated. Every
    # run-level figure below is computed over those pairs and only those, so
    # this is the number that says how much of the corpus they speak for.
    s["verified_pct"] = 100.0 * s["verified"] / s["pairs"] if s["pairs"] else 0.0
    s["reduction"] = s["all_pairs"] / s["pairs"] if s["pairs"] else 0.0
    s["low_conf_pct"] = 100.0 * s["low_conf"] / n if n else 0.0

    s["cites"] = s["ref_entries"] = s["ref_sources"] = 0
    if citations and citations.exists():
        c = _ro(citations)
        try:
            s["cites"] = c.execute("SELECT COUNT(*) FROM cite").fetchone()[0]
            s["ref_sources"], s["ref_entries"] = c.execute(
                "SELECT COUNT(*), SUM(entries) FROM srcstat").fetchone()
            s["ref_entries"] = s["ref_entries"] or 0
        finally:
            c.close()
    s["parse_pct"] = (100.0 * s["cites"] / s["ref_entries"]) if s["ref_entries"] else 0.0
    return s


def search_index(db: Path) -> tuple[dict, int]:
    """Columnar metadata for every usable thesis: the whole searchable corpus.

    Columnar rather than row-of-objects because the keys then appear once
    instead of 24,656 times, and gzip has a much easier job on six runs of
    similar values than on one interleaved stream — it is ~5% smaller before
    compression and noticeably smaller after.

    The record URL is not stored. Every identifier in this corpus is an EPrints
    id, so the landing page is that integer appended to a constant, and storing
    the integer instead of a 45-character URL removes ~1 MB from the payload.
    """
    con = _ro(db)
    try:
        rows = con.execute(
            "SELECT id, title, year, discipline, subfield, discipline_conf "
            "FROM doc WHERE status='ok'").fetchall()
    finally:
        con.close()

    disciplines = sorted({r[3] or "" for r in rows})
    subfields = sorted({r[4] or "" for r in rows})
    di = {d: i for i, d in enumerate(disciplines)}
    si = {sf: i for i, sf in enumerate(subfields)}

    recs = []
    skipped = 0
    for doc_id, title, year, disc, sub, conf in rows:
        m = EPRINT_ID.search(doc_id or "")
        if not m:
            skipped += 1                    # not an EPrints id: no derivable URL
            continue
        # \s matches U+00A0 too, so this also normalises the non-breaking
        # spaces EPrints titles arrive with.
        clean = re.sub(r"\s+", " ", title or "").strip()
        recs.append((int(m.group(1)), clean[:TITLE_CAP], int(year or 0),
                     di[disc or ""], si[sub or ""], int(round((conf or 0) * 100))))
    recs.sort()

    payload = {
        "d": disciplines,
        "s": subfields,
        "id": [r[0] for r in recs],
        "t": [r[1] for r in recs],
        "y": [r[2] for r in recs],
        "di": [r[3] for r in recs],
        "sf": [r[4] for r in recs],
        "c": [r[5] for r in recs],
        "base": RECORD_URL,
    }
    return payload, skipped


# --------------------------------------------------------------------------
# analytics: render whatever tg_canon.py left behind, and nothing it did not
# --------------------------------------------------------------------------

ANALYTICS_SECTIONS = [
    ("canon", "The canon",
     "Works cited often enough, and widely enough across theses, to function as "
     "shared reference points."),
    ("citation", "Citation age and coverage",
     "How old the literature a thesis cites tends to be, how that differs by "
     "field, and how much of each field's referencing the parser could see."),
    ("field_flow", "Field flow",
     "Where cited literature travels between subject areas."),
]

class Guard:
    """The cap on any string this module did not compute, and its audit trail.

    out/analytics/ is written by another module. Rendering it verbatim is the
    honest thing to do — this page must not paraphrase somebody else's figures —
    but "verbatim" and "publishable" are only compatible if there is a bound on
    what can arrive. So there is one bound and it applies to every string,
    whatever key it arrived under: a cap that a field name can opt out of is not
    a cap, it is a naming convention, and the thing it is guarding against is a
    passage of an in-copyright thesis landing in the one artefact here that is
    meant to be published. A caveat cut at 200 characters is a real cost, paid
    deliberately; the page says which values were cut and prints the name and
    sha256 of the file the full text is in.
    """

    def __init__(self, max_string: int):
        self.max_string = max_string
        self.truncated: list[str] = []

    def text(self, value: str, path: str) -> str:
        text = re.sub(r"\s+", " ", value).strip()
        if len(text) > self.max_string:
            self.truncated.append(path)
            text = text[:self.max_string].rstrip() + "\u2026"
        return esc(text)


def _fmt_scalar(v, guard: "Guard", path: str) -> str:
    if v is None or v == "":
        return "&mdash;"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.4g}"
    return guard.text(str(v), path)


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def render_value(value, path: str, guard: "Guard", depth: int = 0) -> str:
    """Render arbitrary JSON as HTML tables, without knowing its schema.

    findings.html presents figures; it does not compute them. Anything
    tg_canon.py chooses to emit therefore has to render without this module
    being taught its shape first, or the two are coupled and the site breaks
    every time the analytics change.
    """
    if _is_scalar(value):
        return f"<p class=v>{_fmt_scalar(value, guard, path)}</p>"

    if isinstance(value, list):
        if not value:
            return "<p class=mut>(empty)</p>"
        if all(_is_scalar(v) for v in value):
            items = ", ".join(_fmt_scalar(v, guard, f"{path}[]") for v in value[:MAX_ROWS])
            more = f" &hellip; ({len(value):,} in total)" if len(value) > MAX_ROWS else ""
            return f"<p class=v>{items}{more}</p>"
        if all(isinstance(v, dict) for v in value):
            cols: list[str] = []
            for row in value:                       # first-seen order, deterministic
                for k in row:
                    if k not in cols and _is_scalar(row[k]):
                        cols.append(k)
            return _table(cols, [[_fmt_scalar(r.get(c), guard, f"{path}.{c}")
                                  for c in cols] for r in value[:MAX_ROWS]],
                          len(value))
        return "".join(render_value(v, f"{path}[{i}]", guard, depth + 1)
                       for i, v in enumerate(value[:MAX_ROWS]))

    if isinstance(value, dict):
        flat = {k: v for k, v in value.items() if _is_scalar(v)}
        deep = {k: v for k, v in value.items() if not _is_scalar(v)}
        out = []
        if flat:
            rows = "".join(
                f"<tr><th>{esc(k)}</th>"
                f"<td>{_fmt_scalar(v, guard, f'{path}.{k}')}</td></tr>"
                for k, v in flat.items())
            out.append(f"<div class=tw><table class=kv>{rows}</table></div>")
        for k, v in deep.items():
            tag = "h4" if depth == 0 else "h5"
            out.append(f"<{tag}>{esc(k)}</{tag}>")
            out.append(render_value(v, f"{path}.{k}", guard, depth + 1))
        return "".join(out) or "<p class=mut>(empty)</p>"

    return f"<p class=v>{_fmt_scalar(str(value), guard, path)}</p>"


def _table(cols: list[str], rows: list[list[str]], total: int) -> str:
    # Table cells do not wrap by default — a column of numbers reads far better
    # unwrapped — but a caveat is a paragraph, and a paragraph that only exists
    # inside a horizontal scrollbar has not been shown to anybody. Anything long
    # gets to wrap.
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f'<td class="wrap">{c}</td>' if len(c) > 140
                         else f"<td>{c}</td>" for c in r) + "</tr>"
        for r in rows)
    note = (f"<p class=mut>showing the first {MAX_ROWS} of {total:,} rows</p>"
            if total > MAX_ROWS else "")
    return (f"<div class=tw><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>{note}")


def render_csv(text: str, path: str, guard: "Guard") -> str:
    """A CSV is a table already; render it as one and cap the rows.

    tg_canon.py puts its headline numbers in JSON and its full tables in CSV.
    Ignoring the CSVs would drop citation age entirely, which lives only there.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return "<p class=mut>(empty)</p>"
    header, data = rows[0], rows[1:]

    def cell(v: str, col: str) -> str:
        v = v.strip()
        if re.fullmatch(r"-?\d{1,3}(,\d{3})*(\.\d+)?|-?\d+(\.\d+)?", v):
            return esc(v)               # already formatted by whoever wrote it
        return _fmt_scalar(v, guard, f"{path}.{col}")

    return _table(header,
                  [[cell(v, header[i] if i < len(header) else str(i))
                    for i, v in enumerate(r)] for r in data[:MAX_ROWS]],
                  len(data))


def section_for(name: str) -> tuple[int, str, str]:
    """Group a filename onto a findings section, longest declared prefix first."""
    for i, (prefix, title, blurb) in enumerate(ANALYTICS_SECTIONS):
        if name.startswith(prefix):
            return i, title, blurb
    return len(ANALYTICS_SECTIONS), "Other analytics", ""


def load_analytics(directory: Path, guard: "Guard",
                   max_bytes: int) -> tuple[list[dict], list[dict]]:
    """Read out/analytics/. Missing is a state, not an error.

    Returns the rendered sections and the files that were too large to render,
    which are listed rather than silently dropped — a reader should be able to
    see that canon_works.csv exists even though five megabytes of it does not
    belong in a web page.
    """
    rendered: list[dict] = []
    oversize: list[dict] = []
    if not directory.is_dir():
        return rendered, oversize

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".json", ".csv") or not path.is_file():
            continue
        size = path.stat().st_size
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        order, title, blurb = section_for(path.name)
        if size > max_bytes:
            oversize.append({"file": path.name, "size": size, "sha256": digest})
            continue
        entry = {"file": path.name, "order": order, "title": title, "blurb": blurb,
                 "sha256": digest, "size": size, "error": "", "html": ""}
        try:
            text = raw.decode("utf-8")
            if path.suffix.lower() == ".json":
                entry["html"] = render_value(json.loads(text), path.stem, guard)
            else:
                entry["html"] = render_csv(text, path.stem, guard)
        except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        rendered.append(entry)

    rendered.sort(key=lambda e: (e["order"], e["file"]))
    oversize.sort(key=lambda e: e["file"])
    return rendered, oversize

# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------

def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


CSS = """
:root{
 --bg:#fbfbfd;--panel:#fff;--card:#f4f5f8;--fg:#151821;--mut:#5d6577;
 --line:#e0e3ea;--acc:#1b5fd0;--acc-fg:#fff;--warn:#8a5a00;--warn-bg:#fff6e2;
 --warn-line:#f0dcae;--ok:#1a7f4e;--shadow:0 1px 2px rgba(16,20,30,.06)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --bg:#0f1115;--panel:#171a21;--card:#1c202a;--fg:#e8eaf0;--mut:#8b93a7;
 --line:#262b36;--acc:#5b9cff;--acc-fg:#08101f;--warn:#ffc866;--warn-bg:#241d0c;
 --warn-line:#4a3a13;--ok:#4ecf8f;--shadow:none}}
:root[data-theme=dark]{
 --bg:#0f1115;--panel:#171a21;--card:#1c202a;--fg:#e8eaf0;--mut:#8b93a7;
 --line:#262b36;--acc:#5b9cff;--acc-fg:#08101f;--warn:#ffc866;--warn-bg:#241d0c;
 --warn-line:#4a3a13;--ok:#4ecf8f;--shadow:none}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);overflow-wrap:break-word;
 font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--acc)}
header.site{background:var(--panel);border-bottom:1px solid var(--line);position:sticky;
 top:0;z-index:10}
.hin{max-width:1080px;margin:0 auto;padding:10px 20px;display:flex;gap:14px;
 align-items:center;flex-wrap:wrap}
.mark{font-weight:680;letter-spacing:-.01em;text-decoration:none;color:var(--fg);
 white-space:nowrap}
.mark small{display:block;font-weight:400;color:var(--mut);font-size:11px;
 letter-spacing:0}
nav.site{display:flex;gap:2px;flex-wrap:wrap;margin-left:auto;align-items:center}
nav.site a,nav.site span.off{font-size:13px;text-decoration:none;color:var(--mut);
 padding:5px 10px;border-radius:7px;white-space:nowrap}
nav.site a:hover{background:var(--card);color:var(--fg)}
nav.site a[aria-current=page]{background:var(--acc);color:var(--acc-fg);font-weight:620}
nav.site span.off{opacity:.45;cursor:not-allowed}
#theme{background:none;border:1px solid var(--line);color:var(--mut);border-radius:7px;
 padding:4px 9px;font:inherit;font-size:12px;cursor:pointer;margin-left:4px}
#theme:hover{color:var(--fg);border-color:var(--acc)}
main{max-width:1080px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:29px;line-height:1.22;margin:.2em 0 .35em;letter-spacing:-.02em}
h2{font-size:20px;margin:2.1em 0 .5em;letter-spacing:-.01em}
h3{font-size:15.5px;margin:1.7em 0 .4em}
h4{font-size:14px;margin:1.3em 0 .3em;color:var(--mut)}
h5{font-size:13px;margin:1.1em 0 .3em;color:var(--mut);font-weight:600}
p{margin:.65em 0}
.lede{font-size:17px;color:var(--mut);max-width:62ch}
.mut{color:var(--mut)}
small.note{color:var(--mut);font-size:12.5px;display:block;margin-top:6px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
 margin:18px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:13px 15px;box-shadow:var(--shadow)}
.stat b{display:block;font-size:24px;font-weight:660;letter-spacing:-.02em;
 font-variant-numeric:tabular-nums}
.stat span{color:var(--mut);font-size:12.5px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:16px 18px;margin:18px 0;box-shadow:var(--shadow)}
.warn{background:var(--warn-bg);border:1px solid var(--warn-line);border-radius:12px;
 padding:15px 18px;margin:18px 0}
.warn h2,.warn h3{margin-top:0;color:var(--warn)}
.warn li{margin:.35em 0}
ul.tight{margin:.5em 0;padding-left:20px}
ul.tight li{margin:.3em 0}
.steps{counter-reset:s;list-style:none;padding:0;margin:14px 0}
.steps li{counter-increment:s;position:relative;padding:2px 0 12px 40px;
 border-left:1px solid var(--line);margin-left:12px}
.steps li:last-child{border-left-color:transparent}
.steps li::before{content:counter(s);position:absolute;left:-12px;top:0;width:24px;
 height:24px;border-radius:50%;background:var(--acc);color:var(--acc-fg);
 display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:660}
.steps b{display:block}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:12px 0}
table{border-collapse:collapse;font-size:13.5px;min-width:100%}
th,td{text-align:left;padding:6px 12px 6px 0;border-bottom:1px solid var(--line);
 vertical-align:top;white-space:nowrap}
td:first-child,th:first-child{white-space:normal}
td.wrap{white-space:normal;min-width:32ch;max-width:74ch}
/* Data tables read better unwrapped; a table whose cells are sentences does
   not, and left to itself it hands all the width to the widest column. */
table.prose th,table.prose td{white-space:normal}
table.prose td:first-child,table.prose th:first-child{width:32%}
thead th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;
 letter-spacing:.04em}
table.kv th{color:var(--mut);font-weight:500;padding-right:20px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--acc);
 vertical-align:middle;min-width:2px}
/* A long path in <code> has no break opportunity and will push a phone-width
   page sideways, so break it anywhere rather than let the layout overflow. */
code,kbd{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;
 background:var(--card);border:1px solid var(--line);border-radius:5px;padding:.5px 5px;
 overflow-wrap:anywhere}
pre{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:12px 14px;
 overflow-x:auto;font-size:12.5px;line-height:1.5}
pre code{background:none;border:0;padding:0}
blockquote{margin:14px 0;padding:2px 0 2px 15px;border-left:3px solid var(--acc);
 color:var(--fg)}
footer.site{border-top:1px solid var(--line);margin-top:40px;padding-top:16px;
 color:var(--mut);font-size:12.5px}
/* search */
.tools{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
 align-items:end;margin:16px 0 6px}
.tools .wide{grid-column:1/-1}
label.f{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
input,select,button{font:inherit}
input[type=search],input[type=number],select{width:100%;background:var(--panel);
 border:1px solid var(--line);color:var(--fg);border-radius:9px;padding:9px 11px}
input[type=search]{font-size:16px}
input:focus,select:focus{outline:2px solid var(--acc);outline-offset:-1px}
.yr{display:flex;gap:8px;align-items:center}
.yr input{padding-left:7px;padding-right:7px;text-align:center;min-width:0}
button.act{background:var(--acc);color:var(--acc-fg);border:0;border-radius:9px;
 padding:9px 15px;cursor:pointer;font-weight:600}
button.ghost{background:var(--panel);color:var(--fg);border:1px solid var(--line);
 border-radius:9px;padding:9px 15px;cursor:pointer}
button.ghost:hover{border-color:var(--acc)}
#count{color:var(--mut);font-size:13px;margin:14px 0 8px}
#hits{list-style:none;padding:0;margin:0}
#hits li{border-bottom:1px solid var(--line);padding:11px 0}
#hits a.t{font-weight:600;text-decoration:none;display:block;line-height:1.45}
#hits a.t:hover{text-decoration:underline}
#hits .m{color:var(--mut);font-size:12.5px;margin-top:3px;
 display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.tag{background:var(--card);border:1px solid var(--line);border-radius:999px;
 padding:1px 9px;font-size:11.5px;white-space:nowrap}
.tag.low{color:var(--warn);border-color:var(--warn-line);background:var(--warn-bg)}
mark{background:rgba(91,156,255,.28);color:inherit;border-radius:3px;padding:0 1px}
.empty{color:var(--mut);padding:28px 0;text-align:center}
@media (max-width:620px){
 h1{font-size:24px}main{padding:20px 15px 44px}.hin{padding:9px 15px}
 nav.site{width:100%;margin-left:0}
}
"""

THEME_JS = """
(function(){
 var K='tg-site-theme',r=document.documentElement,b=document.getElementById('theme');
 function get(){try{return localStorage.getItem(K)||''}catch(e){return ''}}
 function put(v){try{localStorage.setItem(K,v)}catch(e){}}
 function paint(){var v=get();if(v)r.setAttribute('data-theme',v);
  else r.removeAttribute('data-theme');
  if(b)b.textContent=v==='dark'?'light':v==='light'?'auto':'dark';}
 if(b)b.addEventListener('click',function(){
  var v=get();put(v==='dark'?'light':v==='light'?'':'dark');paint();});
 paint();
})();
"""


def page(current: str, title: str, body: str, available: set[str],
         extra_head: str = "", extra_js: str = "") -> str:
    """One HTML file: nav, body, footer, inlined CSS/JS. No external anything."""
    links = []
    for href, label in PAGES:
        if href not in available:
            links.append(f'<span class="off" title="not built">{esc(label)}</span>')
        elif href == current:
            links.append(f'<a href="{href}" aria-current="page">{esc(label)}</a>')
        else:
            links.append(f'<a href="{href}">{esc(label)}</a>')
    nav = "".join(links)
    return f"""<!doctype html>
<html lang="en-GB"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{esc(title)}</title>
<style>{CSS}</style>{extra_head}
</head><body>
<header class="site"><div class="hin">
 <a class="mark" href="index.html">thesisgraph<small>White Rose eTheses corpus</small></a>
 <nav class="site">{nav}<button id="theme" type="button" title="light / dark / auto">dark</button></nav>
</div></header>
<main>
{body}
<footer class="site">
 <p>Built from openly deposited doctoral theses in White Rose eTheses Online
 (Leeds, Sheffield, York). Titles, subject labels and links are public
 bibliographic facts; <b>no thesis text is reproduced on this site</b>.
 Theses remain in copyright &mdash; follow a link to read one at its repository
 record. Method and source:
 <a href="{REPO_URL}" rel="noopener">{REPO_URL}</a>.</p>
</footer>
</main>
<script>{THEME_JS}</script>{extra_js}
</body></html>
"""


def stat(value: str, label: str) -> str:
    return f'<div class="stat"><b>{value}</b><span>{esc(label)}</span></div>'


def bar_table(rows, total: int, head: tuple[str, str]) -> str:
    """A count table with an inline proportional bar — no chart library needed."""
    top = max((c for _, c in rows), default=1) or 1
    out = [f"<div class=tw><table><thead><tr><th>{esc(head[0])}</th>"
           f"<th class=num>{esc(head[1])}</th><th class=num>%</th><th></th>"
           "</tr></thead><tbody>"]
    for label, count in rows:
        pct = 100.0 * count / total if total else 0.0
        w = max(2, round(112 * count / top))
        out.append(f"<tr><td>{esc(label or '(unlabelled)')}</td>"
                   f"<td class=num>{count:,}</td><td class=num>{pct:.1f}%</td>"
                   f'<td><i class="bar" style="width:{w}px"></i></td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

def build_index(s: dict, available: set[str]) -> str:
    bn = s["words"] / 1e9
    stats = "".join([
        stat(f"{s['docs']:,}", "theses with usable extracted text"),
        stat(f"{bn:.2f}bn", "words analysed"),
        stat(f"{s['pages'] / 1e6:.2f}M", "PDF pages"),
        stat(f"{s['sentences'] / 1e6:.1f}M", "sentences"),
        stat(f"{s['disciplines']}", "disciplines"),
        stat(f"{s['subfields']}", "sub-fields"),
        stat(f"{s['year_min']}&ndash;{s['year_max']}", "years of deposit"),
        stat(f"{s['cites']:,}", "parsed citation edges"),
    ])
    body = f"""
<h1>A map of {s['docs']:,} UK doctoral theses</h1>
<p class="lede">Every openly deposited thesis in the White Rose repository
(Leeds, Sheffield and York), harvested, extracted, classified by subject,
screened for textual reuse, and parsed into a citation graph &mdash;
{bn:.2f} billion words of it. This site is the derived data. The theses
themselves stay where they were deposited.</p>

<div class="warn">
<h2>Read this before you read anything else</h2>
<ul class="tight">
<li><b>It measures textual reuse, and only that.</b> It cannot detect fabricated
 data, ghostwriting or a purchased thesis. Those leave no textual trace.</li>
<li><b>Overlap output is a ranked list of things to look at, never a finding
 about a person.</b> Shared text has many innocent causes, and in this corpus
 most of it is exactly that: {s['common_pct']:.1f}% of all shared runs between
 the {s['verified']:,} adjudicated pairs also turn up in an unrelated third
 thesis. Textual overlap is
 not evidence of intent and is not a finding of misconduct. Only an institution
 can make that determination &mdash; which is why no pair is named here.</li>
<li><b>The subject labels are noisy.</b> {s['low_conf']:,} of {s['docs']:,}
 theses ({s['low_conf_pct']:.0f}%) carry a low-confidence discipline label. They
 are flagged in search rather than quietly presented as fact.</li>
<li><b>Reference parsing is author&ndash;date only.</b> {s['cites']:,} of
 {s['ref_entries']:,} detected reference entries ({s['parse_pct']:.0f}%) became a
 citation edge, which biases the citation graph towards the social sciences and
 humanities and away from anything using Vancouver numbering.</li>
<li><b>One repository, not the UK.</b> Three universities, mostly recent
 deposits. Nothing here generalises to British doctoral research as a whole.</li>
</ul>
<p class="mut" style="margin-bottom:0">The <a href="methods.html">methods
page</a> gives the full version, including what each filter removes.</p>
</div>

<h2>How big it is</h2>
<div class="grid">{stats}</div>
<p class="mut">{s['harvested']:,} records were harvested; {s['docs']:,} of them
yielded text good enough to analyse. The rest are metadata-only deposits,
withheld files, or PDFs with no extractable text layer.
{s['year_bad']} records have an unusable date and are excluded from the year
range above.</p>

<h2>How it was built</h2>
<ol class="steps">
<li><b>Harvest</b> Repository metadata over OAI-PMH, then a polite fetch of each
 open PDF &mdash; <code>robots.txt</code> respected, the published
 <code>Crawl-delay</code> observed, a real User-Agent. No PDF is retained: each
 is downloaded, extracted and deleted, so the corpus costs about 10&nbsp;GB of
 text rather than 72&nbsp;GB of PDFs.</li>
<li><b>Extract</b> <code>pdfplumber</code> at word level with x/y geometry
 (needed to see block-quote indentation and headings), with <code>pypdf</code> as
 a per-page and whole-document fallback. Paragraphs are rebuilt across line
 breaks; a hyphenated line-end split is rejoined only when the continuation
 begins lowercase. Sentence splitting is abbreviation-aware. Reference lists,
 attributed indented block quotes, boilerplate and captions are excluded and
 counted separately.</li>
<li><b>Classify</b> A rule pass over title, abstract and subject terms, with an
 embedding tie-break where the rules are ambiguous. Every label carries its
 confidence, and the low ones say so.</li>
<li><b>Screen</b> Winnowed fingerprints over the whole corpus, then exact
 alignment on candidates only &mdash; {s['all_pairs']:,} possible pairs reduced
 to {s['pairs']:,} worth opening. See <a href="methods.html">methods</a>.</li>
<li><b>Read</b> Reference lists parsed into a citation graph
 ({s['cites']:,} edges from {s['ref_sources']:,} theses), method and reporting
 practice extracted, discourse features measured, and the whole thing laid out
 as a <a href="graph.html">graph</a>.</li>
</ol>

<h2>What is here</h2>
<div class="tw"><table class="prose">
<thead><tr><th>Page</th><th>What it gives you</th></tr></thead><tbody>
<tr><td><a href="search.html">Search</a></td><td>Every thesis in the corpus by
 title, year, discipline and sub-field, searched in your browser &mdash; no
 server, no query logged, no request leaves the page.</td></tr>
<tr><td><a href="methods.html">Methods</a></td><td>Winnowing and its detection
 guarantee, the reduction it buys, every filter applied to the ranking, and what
 the measurement does not cover.</td></tr>
<tr><td><a href="findings.html">Findings</a></td><td>Canon, citation age and
 field flow, as computed by <code>tg canon</code>.</td></tr>
<tr><td><a href="graph.html">Graph</a></td><td>Subject tree, co-citation,
 discipline bridges and methods, as one explorable map.</td></tr>
<tr><td>{'<a href="bridges.html">Bridges</a>' if 'bridges.html' in available
 else '<span class="mut">Bridges</span>'}</td><td>The specific works two fields
 share and the theses on each side of them.
 {'' if 'bridges.html' in available else
 '<b>Not included in this build</b> &mdash; the bridge reader quotes the '
 'sentence in which a thesis cites a work, so publishing it is a separate '
 'decision. Build it with <code>make_bridge_app.py --public</code>.'}</td></tr>
</tbody></table></div>

<h2>Reusing this</h2>
<p>Everything on this site is derived data and may be reused under the terms of
the repository (MIT). The underlying theses are in copyright: the UK
text-and-data-mining exception (s.29A CDPA) permits computational analysis of
lawfully accessed works, it does not permit redistributing them. That is why you
will find counts, labels and links here and never a sentence of a thesis.</p>
"""
    return page("index.html", "thesisgraph — a map of UK doctoral theses",
                body, available)


def build_methods(s: dict, available: set[str]) -> str:
    verdict_labels = {
        "fully_explained": "fully explained &mdash; every shared run also appears "
                           "in an unrelated third thesis, or is an attributed quotation",
        "short_only": "short only &mdash; overlap exists but no run reaches 20 words",
        "residual": "residual &mdash; a long run survives every filter; worth a look",
        "same_author_or_duplicate": "same author or duplicate deposit",
    }
    vrows = "".join(
        f"<tr><td>{verdict_labels.get(v, esc(v))}</td><td class=num>{c:,}</td>"
        f"<td class=num>{100.0 * c / s['verified']:.2f}%</td></tr>"
        for v, c in s["verdicts"])
    body = f"""
<h1>Methods</h1>
<p class="lede">How {s['docs']:,} theses are screened against each other without
running {s['all_pairs']:,} comparisons, and what the resulting numbers are and
are not evidence of.</p>

<h2>Screening at corpus scale</h2>
<p>All-pairs is impossible. {s['docs']:,} theses is
<b>{s['all_pairs']:,}</b> pairs; at any honest per-pair cost that is months of
CPU. The standard near-duplicate architecture applies instead: fingerprint once,
index, and run exact alignment only on candidates.</p>

<h3>Winnowing, and the guarantee that makes it safe</h3>
<p><b>Winnowing</b> (Schleimer, Wilkerson &amp; Aiken, 2003). Hash every 8-gram
of words, slide a window of 13 hashes over the sequence, and keep the minimum in
each window. That retains roughly one fingerprint in thirteen &mdash; but it is
not a sample, it is a selection with a proof attached:</p>
<blockquote><p>any shared passage of <b>n + w &minus; 1 = 8 + 13 &minus; 1 =
20</b> words or more is <b>guaranteed</b> to share at least one
fingerprint.</p></blockquote>
<p>Twenty words is exactly the threshold this project treats as a long run, so
the index is <b>lossless for everything the project cares about</b> and lossy
only below it. Verified empirically rather than assumed: 300/300 planted
passages detected at 20 words, 210/300 at 14 words &mdash; below the bound,
which is where theory says detection should start to fail.</p>

<h3>What that buys</h3>
<div class="tw"><table class="prose">
<thead><tr><th>Stage</th><th class=num>Pairs</th><th>Note</th></tr></thead><tbody>
<tr><td>All pairs</td><td class=num>{s['all_pairs']:,}</td>
 <td>{s['docs']:,} theses, every combination</td></tr>
<tr><td>Candidates after winnowing</td><td class=num>{s['pairs']:,}</td>
 <td><b>{s['reduction']:,.0f}&times;</b> reduction in this build</td></tr>
<tr><td>Exactly aligned and adjudicated</td><td class=num>{s['verified']:,}</td>
 <td>candidates carried through the full verbatim pass</td></tr>
</tbody></table></div>
<p class="mut">The project's published figure is 303,946,840 pairs &rarr;
222,881 candidates, a 1,363&times; reduction. The candidate count moves as the
corpus grows and as document-frequency pruning is retuned; the table above is
what this build actually contains.</p>

<h2>Filters, and why the unfiltered ranking is useless</h2>
<p>The top of an unfiltered overlap ranking is not plagiarism. It is
infrastructure. Every item below was found at the top of a real ranking, and
every one of them is now a filter in the code.</p>
<div class="tw"><table class="prose">
<thead><tr><th>What it is</th><th>How it is removed</th></tr></thead><tbody>
<tr><td><b>PDF character-spacing artefacts</b> &mdash; some producers space
 letters widely enough that the extractor emits every letter as a word, so
 <code>i n f o r m a t i o n s y s t e m s</code> becomes a 20-word run</td>
 <td>a run counts only if it looks like prose: at most 30% single-character
 tokens and at least 5 tokens of 4+ characters. Both the filtered and unfiltered
 scores are kept so the effect is visible rather than silently applied</td></tr>
<tr><td><b>Duplicate deposits</b> &mdash; the same thesis deposited twice, or a
 thesis and its own corrected version</td>
 <td>same normalised author, or title similarity &ge; 95</td></tr>
<tr><td><b>Software licences, ethics templates, published instruments,
 repository deposit forms</b> bound into the PDF</td>
 <td>a run that also appears in an <em>unrelated third thesis</em> is common
 text, not reuse. This alone accounts for
 <b>{s['common_pct']:.1f}%</b> of all shared runs</td></tr>
<tr><td><b>Attributed block quotations</b></td>
 <td>indented passages with nearby attribution are excluded at extraction; an
 indented passage with <em>no</em> attribution is deliberately retained</td></tr>
<tr><td><b>Reference lists, boilerplate, captions</b></td>
 <td>excluded during extraction and counted separately, so the exclusion is
 auditable</td></tr>
</tbody></table></div>

<h3>What survives</h3>
<div class="tw"><table>
<thead><tr><th>Verdict</th><th class=num>Pairs</th><th class=num>% of adjudicated</th>
</tr></thead><tbody>{vrows}</tbody></table></div>
<p>Of {s['verified']:,} adjudicated pairs, <b>{s['residual']:,}</b>
({s['residual_pct']:.2f}%) still carry a 20-word run after every filter. That
is the list a human would open. It is not published here and it is not a list of
findings; it is a list of things a reviewer might look at, most of which will
also turn out to have an innocent explanation that this pipeline could not see.</p>

<h2>What this measures &mdash; and what it does not</h2>
<p>It measures <b>textual reuse</b>. It cannot detect fabricated data,
ghostwriting or a purchased thesis; those leave no textual trace.</p>
<p>Output is <b>a ranked list of things to look at</b>, never a finding about a
person. Shared text has many innocent causes, and in this corpus most of it is
exactly that. Measured on a 400-thesis pilot:</p>
<ul class="tight">
<li><b>20%</b> of same-discipline pairs share an 8-word run; <b>5.9%</b> share a
 20+ word run</li>
<li><b>95.5%</b> of all shared runs also appear in an unrelated third thesis
 &mdash; this build reproduces that as <b>{s['common_pct']:.1f}%</b></li>
<li>Only <b>1.34%</b> of non-zero pairs survive filtering &mdash; this build
 reproduces that as <b>{s['residual_pct']:.2f}%</b></li>
</ul>
<p class="mut">Both of this build's figures are measured over the
{s['verified']:,} adjudicated pairs &mdash; the candidates carried through the
full verbatim pass, {s['verified_pct']:.1f}% of the {s['pairs']:,} the screen
produced. Run counts exist for those pairs and no others, so neither is a
full-corpus figure: the rest of the candidate list is unadjudicated and could
move both.</p>
<blockquote><p><b>Textual overlap is not evidence of intent and is not a finding
of misconduct.</b> Only an institution can make that determination.</p>
</blockquote>

<h2>Determinism</h2>
<p>Determinism is a project value that is verified, not asserted. No wall-clock,
no RNG, no reliance on set or dict ordering; every collection is sorted before
it is written; artefacts are produced twice and diffed byte-for-byte. That
applies to this site too &mdash; <code>tg site --verify-determinism</code>
builds it twice and compares every file.</p>

<h2>Known limitations</h2>
<ul class="tight">
<li><b>Reference parsing is author&ndash;date only.</b> {s['cites']:,} of
 {s['ref_entries']:,} detected entries ({s['parse_pct']:.0f}%) parsed; no
 Vancouver style at all. This biases the citation graph towards the social
 sciences and humanities. GROBID or AnyStyle is the proper fix.</li>
<li><b>Chapter and heading detection fails on some documents.</b> Affected rows
 are flagged rather than silently reported.</li>
<li><b>The discourse time series has a digitisation confound</b> at around 2010,
 where retro-digitised theses give way to born-digital ones.</li>
<li><b>{s['low_conf_pct']:.0f}% of discipline labels are low-confidence</b>
 ({s['low_conf']:,} of {s['docs']:,}) and are flagged as such in search.</li>
<li><b>One repository.</b> Three universities, deposits skewed towards recent
 years. Not a sample of UK doctoral research.</li>
</ul>

<h2>Corpus composition</h2>
<h3>By discipline</h3>
{bar_table(s['by_discipline'], s['docs'], ('Discipline', 'Theses'))}
<h3>By decade of deposit</h3>
{bar_table([(f"{d}s", c) for d, c in s['by_decade']], s['docs'], ('Decade', 'Theses'))}
<h3>How the discipline label was decided</h3>
{bar_table(s['by_method'], s['docs'], ('Method', 'Theses'))}
"""
    return page("methods.html", "Methods — thesisgraph", body, available)


def build_findings(rendered: list[dict], oversize: list[dict], analytics_dir: Path,
                   available: set[str], guard: "Guard") -> str:
    if not rendered and not oversize:
        body = f"""
<h1>Findings</h1>
<p class="lede">Canon, citation age and field flow &mdash; the three questions
the citation graph is built to answer.</p>
<div class="warn">
<h3>Not built yet</h3>
<p>These figures are produced by <code>tg canon</code>, which writes them to
<code>{esc(analytics_dir)}</code>. That directory is empty or absent in this
build, so there is nothing to show.</p>
<p style="margin-bottom:0">This page deliberately shows nothing rather than
plausible placeholders. Run the analysis and rebuild:</p>
<pre><code>.venv/bin/python tg.py canon
.venv/bin/python tg.py site</code></pre>
</div>
<h2>What will appear here</h2>
<div class="tw"><table class="prose"><thead><tr><th>Files</th><th>Section</th><th>Question</th>
</tr></thead><tbody>
{''.join(f'<tr><td><code>{esc(pre)}*</code></td><td>{esc(ti)}</td>'
         f'<td>{esc(bl)}</td></tr>' for pre, ti, bl in ANALYTICS_SECTIONS)}
</tbody></table></div>
<p class="mut">Any other <code>.json</code> or <code>.csv</code> dropped into
that directory is rendered too, as tables derived from its own structure &mdash;
this page is a presenter, so it does not need to be taught each new file.</p>
"""
        return page("findings.html", "Findings — thesisgraph", body, available)

    sections, last = [], None
    for f in rendered:
        if f["title"] != last:
            last = f["title"]
            sections.append(f'<h2 id="{esc(f["title"].lower().replace(" ", "-"))}">'
                            f'{esc(f["title"])}</h2>')
            if f["blurb"]:
                sections.append(f'<p class="lede">{esc(f["blurb"])}</p>')
        sections.append(f'<h3>{esc(f["file"])}</h3>')
        if f["error"]:
            sections.append(f'<div class=warn><p>Could not read this file: '
                            f'{esc(f["error"])}</p></div>')
        else:
            sections.append(f["html"])
        sections.append(f'<small class=note>{f["size"]:,} bytes '
                        f'&middot; sha256 <code>{esc(f["sha256"])}</code></small>')

    big = ""
    if oversize:
        rows = "".join(
            f'<tr><td><code>{esc(o["file"])}</code></td>'
            f'<td class=num>{o["size"] / 1e6:.1f} MB</td>'
            f'<td><code>{esc(o["sha256"])}</code></td></tr>' for o in oversize)
        big = f"""
<h2 id="not-rendered">Produced, but not rendered here</h2>
<p>These files are part of the same analysis and are too large for a web page.
They are listed so that what exists is visible; they are not published as part
of this site.</p>
<div class="tw"><table><thead><tr><th>File</th><th class=num>Size</th>
<th>sha256</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    guard_note = ""
    if guard.truncated:
        guard_note = (f'<p class="mut" style="margin-bottom:0">This page caps '
                      f'every string it did not compute itself at '
                      f'{guard.max_string} characters, whatever field it arrived '
                      f'under: {len(guard.truncated)} value(s) were truncated and '
                      f'end in an ellipsis. The full text is in the source file '
                      f'named and checksummed beneath the table it belongs to.</p>')

    body = f"""
<h1>Findings</h1>
<p class="lede">Read verbatim from <code>{esc(analytics_dir.name)}/</code>. This
page presents figures; it does not compute them and it does not interpret them.
The size and checksum of every source file are printed beneath it.</p>
<div class="panel">
<p style="margin-top:0">These are corpus-level measurements over a single
repository, subject to every limitation on the
<a href="methods.html">methods page</a> &mdash; in particular that reference
parsing is author&ndash;date only, so anything derived from citations
under-represents fields that number their references. Where the analysis states
its own caveats they are reproduced below, cut at
{guard.max_string} characters like every other string read from that directory,
and should be read before the tables they qualify.</p>
<p style="margin-bottom:0">{len(rendered)} file(s) rendered from
<code>{esc(analytics_dir)}</code>.</p>
{guard_note}
</div>
{''.join(sections)}
{big}
"""
    return page("findings.html", "Findings — thesisgraph", body, available)


# --------------------------------------------------------------------------
# search page
# --------------------------------------------------------------------------

SEARCH_JS = r"""
(function(){
var D=null, hits=[], shown=0, PAGE=100;
var q=document.getElementById('q'), fd=document.getElementById('fd'),
    fs=document.getElementById('fs'), y0=document.getElementById('y0'),
    y1=document.getElementById('y1'), so=document.getElementById('so'),
    list=document.getElementById('hits'), cnt=document.getElementById('count'),
    more=document.getElementById('more'), clr=document.getElementById('clr');

function esc(s){return String(s).replace(/[&<>"]/g,function(c){
  return c==='&'?'&amp;':c==='<'?'&lt;':c==='>'?'&gt;':'&quot;';});}

/* A sub-field label is stored as "Discipline: a / b / c"; the discipline is
   already shown beside it, so only the tail is worth repeating. */
function tail(s){var i=s.indexOf(': ');return i<0?s:s.slice(i+2);}

function terms(){
  return q.value.toLowerCase().split(/\s+/).filter(function(t){return t.length>0;});
}

function search(){
  var ts=terms(), d=fd.value, sf=fs.value,
      a=parseInt(y0.value,10), b=parseInt(y1.value,10);
  if(isNaN(a)) a=-1e9; if(isNaN(b)) b=1e9;
  var out=[], lc=D.lc, t=D.t, y=D.y, di=D.di, sfi=D.sf;
  var wantD = d===''?-1:D.dix[d], wantS = sf===''?-1:D.six[sf];
  for(var i=0;i<lc.length;i++){
    if(wantD>=0 && di[i]!==wantD) continue;
    if(wantS>=0 && sfi[i]!==wantS) continue;
    if(y[i]<a || y[i]>b) continue;
    var score=0, ok=true;
    for(var k=0;k<ts.length;k++){
      var p=lc[i].indexOf(ts[k]);
      if(p<0){ok=false;break;}
      /* a term that starts a word beats one buried inside one, and earlier
         beats later — enough ranking for a title-only index, and cheap */
      score += (p===0||lc[i].charCodeAt(p-1)===32?0:60) + Math.min(p,180);
    }
    if(ok) out.push([score,i]);
  }
  var mode=so.value;
  out.sort(function(m,n){
    if(mode==='new'){ if(y[n[1]]!==y[m[1]]) return y[n[1]]-y[m[1]]; }
    else if(mode==='old'){ if(y[m[1]]!==y[n[1]]) return y[m[1]]-y[n[1]]; }
    else if(m[0]!==n[0]) return m[0]-n[0];
    if(y[n[1]]!==y[m[1]]) return y[n[1]]-y[m[1]];
    return t[m[1]]<t[n[1]]?-1:t[m[1]]>t[n[1]]?1:0;
  });
  hits=out; shown=0; list.innerHTML=''; render();
}

function mark(title, ts){
  var h=esc(title);
  if(!ts.length) return h;
  var low=h.toLowerCase(), spans=[];
  ts.forEach(function(t){
    var from=0,p;
    while((p=low.indexOf(t,from))>=0){ spans.push([p,p+t.length]); from=p+t.length; }
  });
  if(!spans.length) return h;
  spans.sort(function(a,b){return a[0]-b[0];});
  var out='', at=0;
  spans.forEach(function(s){
    if(s[0]<at) return;
    out+=h.slice(at,s[0])+'<mark>'+h.slice(s[0],s[1])+'</mark>'; at=s[1];
  });
  return out+h.slice(at);
}

function render(){
  var ts=terms(), frag=document.createDocumentFragment();
  var end=Math.min(hits.length, shown+PAGE);
  for(var k=shown;k<end;k++){
    var i=hits[k][1], li=document.createElement('li');
    var low = D.c[i]<50;
    li.innerHTML='<a class="t" href="'+D.base+D.id[i]+'/" rel="noopener">'+
      mark(D.t[i],ts)+'</a><div class="m">'+
      '<span class="tag">'+esc(D.d[D.di[i]])+'</span>'+
      '<span class="tag">'+esc(tail(D.s[D.sf[i]]))+'</span>'+
      '<span>'+(D.y[i]||'year unknown')+'</span>'+
      (low?'<span class="tag low" title="the subject label for this thesis is '+
        'low-confidence">label '+D.c[i]+'% confident</span>':'')+
      '</div>';
    frag.appendChild(li);
  }
  list.appendChild(frag); shown=end;
  cnt.textContent = hits.length
    ? ('showing '+shown.toLocaleString()+' of '+hits.length.toLocaleString()+
       ' matching theses ('+D.total.toLocaleString()+' in the corpus)')
    : 'no theses match';
  if(!hits.length && !list.querySelector('.empty')){
    list.innerHTML='<li class="empty">Nothing matches. Try fewer words, or '+
      'clear the filters.</li>';
  }
  more.style.display = shown<hits.length ? '' : 'none';
  more.textContent = 'show '+Math.min(PAGE,hits.length-shown)+' more';
}

function fillSub(){
  var d=fd.value, keep=fs.value;
  fs.innerHTML='<option value="">all sub-fields</option>';
  D.s.forEach(function(s){
    if(d && s.indexOf(d+':')!==0) return;
    var o=document.createElement('option'); o.value=s; o.textContent=tail(s);
    fs.appendChild(o);
  });
  if(keep && fs.querySelector('option[value="'+keep.replace(/"/g,'\\"')+'"]'))
    fs.value=keep;
}

function boot(data){
  D=data;
  D.lc=D.t.map(function(s){return s.toLowerCase();});
  D.dix={}; D.d.forEach(function(v,i){D.dix[v]=i;});
  D.six={}; D.s.forEach(function(v,i){D.six[v]=i;});
  D.total=D.id.length;
  D.d.forEach(function(v){
    var o=document.createElement('option'); o.value=v; o.textContent=v; fd.appendChild(o);
  });
  fillSub();
  var lo=1e9, hi=-1e9;
  for(var i=0;i<D.y.length;i++){ if(D.y[i]>1900){ if(D.y[i]<lo)lo=D.y[i];
    if(D.y[i]>hi)hi=D.y[i]; } }
  y0.placeholder=lo; y1.placeholder=hi;
  fd.addEventListener('change',function(){fillSub();search();});
  [fs,so].forEach(function(e){e.addEventListener('change',search);});
  [q,y0,y1].forEach(function(e){e.addEventListener('input',search);});
  more.addEventListener('click',render);
  clr.addEventListener('click',function(){
    q.value=''; fd.value=''; y0.value=''; y1.value=''; so.value='rel';
    fillSub(); search(); q.focus();
  });
  document.getElementById('loading').style.display='none';
  document.getElementById('ui').style.display='';
  search();
}

(async function(){
  var b64=document.getElementById('ix').textContent.trim();
  var bin=Uint8Array.from(atob(b64),function(c){return c.charCodeAt(0);});
  if(typeof DecompressionStream==='undefined'){
    document.getElementById('loading').innerHTML=
      'This browser lacks DecompressionStream; please use a current '+
      'Chrome, Safari or Firefox.';
    return;
  }
  var s=new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
  boot(JSON.parse(await new Response(s).text()));
})();
})();
"""


def build_search(payload: dict, s: dict, available: set[str],
                 skipped: int) -> tuple[str, int, int]:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    # mtime=0: gzip stamps the clock into its header by default, which would
    # make every build differ from the last for no reason at all.
    gz = gzip.compress(raw, 9, mtime=0)
    b64 = base64.b64encode(gz).decode("ascii")

    n = len(payload["id"])
    skip_note = (f" {skipped} record(s) without a derivable repository URL are "
                 "omitted." if skipped else "")
    body = f"""
<h1>Search the corpus</h1>
<p class="lede">All {n:,} theses, searched entirely in your browser. The index
is {len(raw) / 1e6:.1f}&nbsp;MB of titles and subject labels, shipped compressed
inside this page &mdash; nothing is requested from anywhere while you type, and
no query is logged.{esc(skip_note)}</p>

<div id="loading" class="empty">Loading the index&hellip;</div>
<div id="ui" style="display:none">
<div class="tools">
 <div class="wide"><label class="f" for="q">Title contains</label>
  <input type="search" id="q" placeholder="e.g. thermal imaging, medieval, catalysis"
   autocomplete="off" spellcheck="false"></div>
 <div><label class="f" for="fd">Discipline</label>
  <select id="fd"><option value="">all disciplines</option></select></div>
 <div><label class="f" for="fs">Sub-field</label>
  <select id="fs"><option value="">all sub-fields</option></select></div>
 <div><label class="f" for="y0">Year</label>
  <div class="yr"><input type="number" id="y0" inputmode="numeric" aria-label="from year">
   <span class="mut">to</span>
   <input type="number" id="y1" inputmode="numeric" aria-label="to year"></div></div>
 <div><label class="f" for="so">Order</label>
  <select id="so"><option value="rel">best match</option>
   <option value="new">newest first</option>
   <option value="old">oldest first</option></select></div>
 <div><label class="f">&nbsp;</label>
  <button class="ghost" id="clr" type="button">clear</button></div>
</div>
<p id="count"></p>
<ul id="hits"></ul>
<p><button class="act" id="more" type="button" style="display:none">show more</button></p>
</div>

<div class="panel">
<h3 style="margin-top:0">What a result is, and what it is not</h3>
<p>Each row is a repository record: a title, the year of deposit, an assigned
discipline and sub-field, and a link to the record itself. The subject labels
are <em>assigned by this pipeline</em>, not by the author or the library, and
{s['low_conf']:,} of {s['docs']:,} are low-confidence &mdash; those carry a
confidence tag so you can discount them.</p>
<p><b>You cannot search by author here.</b> The index carries titles, years and
subject labels and deliberately carries no author names, so a search for a
person returns nothing. Search the
<a href="https://etheses.whiterose.ac.uk/" rel="noopener">repository</a> itself
for that.</p>
<p style="margin-bottom:0">Follow a link to read the thesis where it was
deposited. No thesis text is held on this site, and no overlap or screening
result is shown against any thesis here.</p>
</div>
"""
    extra_js = ('<script type="text/plain" id="ix">' + b64 + "</script>\n"
                "<script>" + SEARCH_JS + "</script>")
    html = page("search.html", "Search — thesisgraph", body, available,
                extra_js=extra_js)
    return html, len(raw), len(gz)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, default=HERE / "corpus" / "corpus.db",
                   help="corpus database (default: %(default)s)")
    p.add_argument("--citations", type=Path, default=HERE / "corpus" / "citations.db",
                   help="citation database, for the reference-parsing figures")
    p.add_argument("--out", type=Path, default=HERE / "out" / "site",
                   help="site directory to write (default: %(default)s)")
    p.add_argument("--analytics", type=Path, default=HERE / "out" / "analytics",
                   help="directory tg_canon.py writes its JSON to; a missing "
                        "directory degrades to a 'not built' findings page")
    p.add_argument("--graph", type=Path, default=HERE / "out" / "public" / "index.html",
                   help="publishable graph explorer to copy in as graph.html")
    p.add_argument("--bridges", type=Path, default=HERE / "out" / "public" / "bridges.html",
                   help="bridge reader to copy in, only with --include-bridges")
    p.add_argument("--include-bridges", action="store_true",
                   help="also copy the bridge reader into the site. It quotes the "
                        "sentence in which a thesis cites a work, so the result is "
                        "no longer free of thesis text and publishing it is a "
                        "separate decision")
    p.add_argument("--no-apps", action="store_true",
                   help="do not copy the graph explorer in; link to it only")
    p.add_argument("--max-string", type=int, default=MAX_STRING,
                   help="truncate every string read from --analytics at this many "
                        "characters, whatever key it arrived under. This is the "
                        "guard that keeps a stray passage of an in-copyright "
                        "thesis out of a publishable page, so raise it only "
                        "after reading what it will let through "
                        "(default: %(default)s)")
    p.add_argument("--max-analytics-bytes", type=int, default=MAX_ANALYTICS_BYTES,
                   help="analytics files larger than this are listed rather than "
                        "rendered (default: %(default)s)")
    p.add_argument("--verify-determinism", action="store_true",
                   help="build twice into temporary directories and diff every "
                        "file byte-for-byte")


def build(args: argparse.Namespace, out: Path) -> tuple[dict, list[str]]:
    """Write the whole site into `out`. Returns figures and a stderr log."""
    log: list[str] = []
    out.mkdir(parents=True, exist_ok=True)

    stats = corpus_stats(Path(args.db), Path(args.citations))
    payload, skipped = search_index(Path(args.db))
    guard = Guard(args.max_string)
    rendered, oversize = load_analytics(Path(args.analytics), guard,
                                        args.max_analytics_bytes)

    available = {"index.html", "search.html", "methods.html", "findings.html"}
    copies: list[tuple[Path, str]] = []
    if not args.no_apps and Path(args.graph).is_file():
        copies.append((Path(args.graph), "graph.html"))
        available.add("graph.html")
    elif not args.no_apps:
        log.append(f"site: no graph explorer at {args.graph} — "
                   "build it with `make_graph_app.py --public`")
    if args.include_bridges:
        if Path(args.bridges).is_file():
            copies.append((Path(args.bridges), "bridges.html"))
            available.add("bridges.html")
            log.append("site: --include-bridges copied the bridge reader in. It "
                       "quotes thesis sentences, so out/site/ is NO LONGER free "
                       "of thesis text — publishing it is a separate decision.")
        else:
            log.append(f"site: no bridge reader at {args.bridges}")

    files = {
        "index.html": build_index(stats, available),
        "methods.html": build_methods(stats, available),
        "findings.html": build_findings(rendered, oversize, Path(args.analytics),
                                        available, guard),
    }
    search_html, raw_n, gz_n = build_search(payload, stats, available, skipped)
    files["search.html"] = search_html
    # GitHub Pages runs Jekyll over an uploaded directory unless told not to.
    files[".nojekyll"] = ""

    written: list[tuple[str, int, str]] = []
    for name in sorted(files):
        path = out / name
        path.write_text(files[name], encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        written.append((name, path.stat().st_size, digest))
    for src, name in sorted(copies, key=lambda c: c[1]):
        shutil.copyfile(src, out / name)
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()[:12]
        written.append((name, (out / name).stat().st_size, digest))

    if guard.truncated:
        log.append(f"site: {len(guard.truncated)} analytics value(s) truncated at "
                   f"{args.max_string} chars: "
                   + ", ".join(sorted(set(guard.truncated))[:6])
                   + (" …" if len(set(guard.truncated)) > 6 else ""))
    # A file that could not be parsed is reported on the page, so it must be
    # reported here too: the summary below counts files, and a count alone
    # cannot distinguish "rendered" from "rendered as an error message".
    for f in rendered:
        if f["error"]:
            log.append(f"site: {f['file']} could not be read — {f['error']}")
    for o in oversize:
        log.append(f"site: {o['file']} is {o['size'] / 1e6:.1f} MB — listed on "
                   "findings.html, not rendered into it")

    return {"stats": stats, "written": sorted(written), "index_raw": raw_n,
            "index_gz": gz_n, "analytics": rendered, "oversize": oversize,
            "skipped": skipped}, log


def run(args: argparse.Namespace) -> int:
    db = Path(args.db)
    if not db.exists():
        print(f"site: no corpus database at {db}", file=sys.stderr)
        return 1
    if args.max_string < 1:
        print(f"site: --max-string must be at least 1, got {args.max_string}",
              file=sys.stderr)
        return 1
    if args.max_analytics_bytes < 1:
        print(f"site: --max-analytics-bytes must be at least 1, got "
              f"{args.max_analytics_bytes}", file=sys.stderr)
        return 1

    # mkdir(exist_ok=True) still raises when the path exists and is not a
    # directory, and this module is required to run standalone, where nothing
    # turns that traceback into a sentence.
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        print(f"site: --out {out} exists and is not a directory", file=sys.stderr)
        return 1
    try:
        result, log = build(args, out)
    except OSError as exc:
        print(f"site: could not write the site into {out} — {exc}", file=sys.stderr)
        return 1

    for line in log:
        print(line, file=sys.stderr)

    total = sum(size for _, size, _ in result["written"])
    print(f"{out}")
    width = max(len(n) for n, _, _ in result["written"])
    for name, size, digest in result["written"]:
        print(f"  {name:<{width}}  {size / 1024:>9,.0f} KB  {digest}")
    print(f"  {'total':<{width}}  {total / 1024:>9,.0f} KB")
    print(f"  search index: {result['index_raw'] / 1e6:.2f} MB raw -> "
          f"{result['index_gz'] / 1e6:.2f} MB gzip -> "
          f"{result['index_gz'] * 4 / 3 / 1e6:.2f} MB base64 in the page")
    if result["analytics"] or result["oversize"]:
        # "rendered" has to mean rendered. A file that failed to parse reaches
        # the page as an error message, and counting it as rendered here is the
        # one place this build could tell a reader something that is not so.
        ok = sorted(f["file"] for f in result["analytics"] if not f["error"])
        bad = sorted(f["file"] for f in result["analytics"] if f["error"])
        parts = [f"{len(ok)} rendered"]
        if bad:
            parts.append(f"{len(bad)} unreadable")
        parts.append(f"{len(result['oversize'])} listed only")
        print(f"  analytics: {', '.join(parts)} — "
              + ", ".join(ok + [f"{n} (unreadable)" for n in bad]))
    else:
        print(f"  analytics: none in {args.analytics} — findings.html says so")

    if args.verify_determinism:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            build(args, a)
            build(args, b)
            names = sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()})
            bad = [n for n in names
                   if not (a / n).exists() or not (b / n).exists()
                   or (a / n).read_bytes() != (b / n).read_bytes()]
            print(f"  determinism: {len(names) - len(bad)}/{len(names)} files "
                  f"byte-identical across two builds")
            if bad:
                for n in bad:
                    print(f"    DIFFERS  {n}", file=sys.stderr)
                return 1
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
