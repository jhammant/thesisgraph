#!/usr/bin/env python3
"""check_site.py — render every page of out/site/ headless and assert it behaves.

Follows check_graph.py / check_bridges.py: one Chromium, CDP over websocket,
navigate from file:// and interrogate the live DOM. Three things are asserted
that the other checks do not, because they are what makes this site publishable:

  * NO REQUEST LEAVES THE PAGE. The whole point of a 1.4 MB embedded index is
    that search never talks to a server, so a single off-file:// request is a
    failure, not a warning.
  * NO THESIS TEXT. Every string the search payload carries is checked against
    the title cap, and the payload's keys are checked against the declared set,
    so a future field cannot smuggle a sentence in unnoticed. Then the outcome
    itself is checked: real sentences are taken out of the extraction store and
    their verbatim 8-word runs looked for in EVERY file the site ships —
    graph.html included, and inside gzip+base64 payloads, which is how this
    codebase embeds its own data and therefore how a leak would most likely
    travel.
  * THE ANALYTICS PATH BOTH WAYS. The site is built once with out/analytics/
    absent (it must say so, and must not invent figures) and once against a
    synthetic analytics directory (it must render it), because "degrades
    gracefully" is a claim that only means something if both halves run.

    python check_site.py [path/to/out/site]
"""
from __future__ import annotations

import base64
import gzip
import html
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from check_viewer import CDP, find_chrome, free_port

HERE = Path(__file__).resolve().parent
TITLE_CAP = 240                     # tg_site.TITLE_CAP
PAYLOAD_KEYS = {"d", "s", "id", "t", "y", "di", "sf", "c", "base"}

# The copyright scan, below, is measured in the project's own units. A verbatim
# run is 8 words (screen_corpus.py fingerprints 8-grams), and a run counts only
# if it looks like prose — at most 30% single-character tokens and at least 5
# tokens of 4+ characters — which is rescore.py's filter, restated here rather
# than imported so this check does not drag in torch to read some HTML.
RUN_WORDS = 8
MAX_SINGLE_FRAC = 0.30
MIN_LONG_TOKENS = 5
MIN_SENTENCE_CHARS = 60

SYNTHETIC = {
    "canon.json": {
        "note": "synthetic fixture for check_site.py",
        "works": [
            {"work": "Braun & Clarke (2006)", "theses": 1234, "disciplines": 14},
            {"work": "Creswell (2014)", "theses": 987, "disciplines": 11},
        ],
        "totals": {"works": 2, "citations": 2221},
    },
    "citation_age.json": {
        "median_age_years": 9.4,
        "by_discipline": [
            {"discipline": "Mathematics", "median_age": 14.2},
            {"discipline": "Computing", "median_age": 6.1},
        ],
    },
    "field_flow.json": {
        "edges": [{"from": "Psychology", "to": "Education", "weight": 0.31}],
    },
}


# --------------------------------------------------------------------------
# the copyright scan
#
# This is the assertion the whole site rests on, so it is written to be hard to
# fool rather than easy to pass. Two rules follow from that:
#
#   * SCAN WHAT IS SHIPPED, not what is convenient to read. Every file in the
#     directory counts, including graph.html, and a payload that is gzipped and
#     base64'd into a <script> tag counts as much as the prose around it — that
#     is exactly how this codebase itself embeds 24,656 records, and a scanner
#     that cannot see through it certifies nothing. An earlier version of this
#     function read raw .html only and skipped graph.html outright.
#   * ANSWER IN THE PROJECT'S OWN UNITS. The unit of reuse here is an 8-word
#     verbatim run that looks like prose, so that is what is searched for,
#     rather than whole sentences: a leak does not have to arrive as a whole
#     sentence, and after --max-string it usually would not.
#
# Published thesis TITLES are excluded, because they are the metadata this site
# exists to publish. A thesis's own title recurs inside its text (title page,
# declaration, its appearance in another thesis's reference list), so without
# that exemption the scan reports the site doing its job. The exemption is
# reported, not silent.
# --------------------------------------------------------------------------

B64_RUN = re.compile(r"[A-Za-z0-9+/=\s]{512,}")
SCRIPT_BLOCK = re.compile(r"<script[^>]*>(.*?)</script>", re.I | re.S)
HASH_P = np.uint64(1000003)


def _decompress(raw: bytes) -> bytes | None:
    """gzip, zlib or raw deflate — whichever the payload turns out to be."""
    for attempt in (lambda: gzip.decompress(raw),
                    lambda: zlib.decompress(raw),
                    lambda: zlib.decompress(raw, -15)):
        try:
            return attempt()
        except Exception:                                        # noqa: BLE001
            continue
    return None


def _json_strings(text: str) -> str | None:
    """Every string value in a JSON document, unescaped by the parser.

    A sentence embedded in JSON arrives with its quotes and backslashes
    escaped, so it is not present verbatim in the raw bytes even though it is
    plainly present in the page. Decoding gives the scan the text a reader
    would see.
    """
    if text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        obj = json.loads(text)
    except Exception:                                            # noqa: BLE001
        return None
    out, stack = [], [obj]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            stack.extend(v.keys())
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
    return "\n".join(out) if out else None


def _expand(label: str, text: str, out: list[tuple[str, str]], depth: int = 0) -> None:
    """Add `text` and everything hiding inside it to the haystack.

    Whitespace is collapsed on the way in, because the published form of a
    sentence may be wrapped differently from the extracted one and the claim
    being tested is about the words, not the line breaks.
    """
    out.append((label, " ".join(text.split())))
    unescaped = html.unescape(text)
    if unescaped != text:
        out.append((f"{label} (entities decoded)", " ".join(unescaped.split())))
    strings = _json_strings(text)
    if strings:
        out.append((f"{label} (JSON strings)", " ".join(strings.split())))
    if depth >= 3:
        return
    inner: list[tuple[str, str]] = []
    for m in SCRIPT_BLOCK.finditer(text):
        inner.append((f"{label} <script> @{m.start()}", m.group(1)))
    for m in B64_RUN.finditer(text):
        packed = "".join(m.group(0).split())
        try:
            raw = base64.b64decode(packed, validate=True)
        except Exception:                                        # noqa: BLE001
            continue
        blob = _decompress(raw) or raw
        try:
            inner.append((f"{label} payload @{m.start()}", blob.decode("utf-8")))
        except UnicodeDecodeError:
            continue                    # a font or an image, not text
    for lab, txt in inner:
        _expand(lab, txt, out, depth + 1)


def site_texts(site: Path) -> list[tuple[str, str]]:
    """Every readable byte the site ships, with payloads inflated, sorted."""
    out: list[tuple[str, str]] = []
    for f in sorted(site.rglob("*")):
        if not f.is_file():
            continue
        raw = f.read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = _decompress(raw) or raw
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # latin-1 never fails and leaves ASCII intact, so text embedded in
            # a binary is still visible rather than skipped.
            text = raw.decode("latin-1")
        _expand(str(f.relative_to(site)), text, out)
    return out


def _shingles(tokens: list[bytes], breaks: np.ndarray | None = None):
    """Rolling hashes of every RUN_WORDS-token window, and where each starts.

    Windows containing a break (the marker between two sentences) are dropped,
    so a run is never manufactured by two unrelated pieces of text meeting.
    """
    n = len(tokens)
    if n < RUN_WORDS:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int64)
    h = np.fromiter(map(zlib.crc32, tokens), dtype=np.uint64, count=n)
    acc = np.zeros(n - RUN_WORDS + 1, dtype=np.uint64)
    for j in range(RUN_WORDS):
        acc = acc * HASH_P + h[j:n - RUN_WORDS + 1 + j]
    if breaks is None:
        return acc, np.arange(len(acc), dtype=np.int64)
    cum = np.concatenate(([0], np.cumsum(breaks.astype(np.int64))))
    keep = (cum[RUN_WORDS:] - cum[:n - RUN_WORDS + 1]) == 0
    return acc[keep], np.nonzero(keep)[0]


def _is_prose_run(words: list[str]) -> bool:
    """rescore.py's filter: a run of spaced-out single letters is not prose."""
    ones = sum(1 for w in words if len(w) == 1)
    if ones / len(words) > MAX_SINGLE_FRAC:
        return False
    return sum(1 for w in words if len(w) >= 4) >= MIN_LONG_TOKENS


def published_titles(db: Path) -> str:
    """The titles the site is entitled to publish, as one searchable blob.

    Joined with newlines, and every run searched for is a single line of words,
    so a run can never be matched across two titles.
    """
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT title FROM doc WHERE status='ok'").fetchall()
    finally:
        con.close()
    return "\n".join(sorted({" ".join((t or "").split()) for (t,) in rows}))


def scan_for_thesis_text(site: Path, corpus: Path, db: Path, docs: int) -> dict:
    """Look for real thesis text in everything the site ships.

    The site's central claim is that it contains no thesis body text. Every
    other assertion here checks a mechanism; this one checks the outcome, by
    taking actual sentences out of the extraction store and looking for their
    verbatim runs in what was published. Bounded and deterministic: the first
    `docs` documents in sorted shard order, sentences of MIN_SENTENCE_CHARS
    characters or more, runs of RUN_WORDS words.
    """
    sys.path.insert(0, str(HERE))
    import harvest_corpus as H                                   # noqa: PLC0415

    texts = site_texts(site)
    index = np.unique(np.concatenate(
        [_shingles(t.encode("utf-8").split())[0] for _, t in texts]
        or [np.empty(0, dtype=np.uint64)]))
    titles = published_titles(db)

    picked: list[Path] = []
    for shard in sorted(corpus.iterdir()):
        if not shard.is_dir():
            continue
        for f in sorted(shard.iterdir()):
            picked.append(f)
            if len(picked) >= docs:
                break
        if len(picked) >= docs:
            break

    checked = short = matched = in_title = 0
    hits: list[tuple[str, str, str]] = []
    for f in sorted(picked):
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                rows = json.load(fh)["sentences"]
        except Exception:                                        # noqa: BLE001
            continue
        sents: list[list[str]] = []
        starts: list[int] = []
        tokens: list[bytes] = []
        breaks: list[bool] = []
        for r in rows:
            text = " ".join(H.unpack_sentence(r)[0].split())
            if len(text) < MIN_SENTENCE_CHARS:
                continue
            words = text.split()
            if len(words) < RUN_WORDS:
                short += 1              # too short to contain a run at all
                continue
            checked += 1
            starts.append(len(tokens))
            sents.append(words)
            tokens.extend(w.encode("utf-8") for w in words)
            tokens.append(b"\x00")     # sentence boundary; never a real token
            breaks.extend([False] * len(words) + [True])
        if not tokens:
            continue
        found, where = _shingles(tokens, np.array(breaks))
        if not len(found) or not len(index):
            continue
        at = np.searchsorted(index, found)
        at[at >= len(index)] = 0
        first = np.array(starts)
        for k in np.nonzero(index[at] == found)[0]:
            pos = int(where[k])
            si = int(np.searchsorted(first, pos, side="right") - 1)
            off = pos - starts[si]
            words = sents[si][off:off + RUN_WORDS]
            run = " ".join(words)
            if not _is_prose_run(words):
                continue                # spaced-out characters, not a passage
            matched += 1
            if run in titles:
                in_title += 1           # a published title: the site's own job
                continue
            where_seen = next((lab for lab, txt in texts if run in txt), "?")
            hits.append((f.name, where_seen, run))
    return {"docs": len(picked), "sentences": checked, "short": short,
            "files": len(texts), "chars": sum(len(t) for _, t in texts),
            "runs": matched, "titles": in_title, "hits": sorted(set(hits))}


def _flat(text: str) -> str:
    """Collapse whitespace: prose is asserted, not the source's line wrapping."""
    return re.sub(r"\s+", " ", text or "")


def _lum(css_rgb: str) -> float:
    """Rough luminance of a computed rgb() colour, for the light/dark assertion."""
    nums = [float(n) for n in re.findall(r"[\d.]+", css_rgb or "")][:3]
    if len(nums) < 3:
        return -1.0
    r, g, b = nums
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main() -> int:
    site = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "out" / "site"
    if not (site / "index.html").exists():
        print(f"check_site: no site at {site} — run `tg site` first", file=sys.stderr)
        return 1

    port, prof = free_port(), tempfile.mkdtemp(prefix="sitecheck-")
    proc = subprocess.Popen(
        [find_chrome(), "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={prof}", "--no-first-run", "--no-default-browser-check",
         "--remote-allow-origins=*", "--disable-gpu", "--window-size=1400,1000",
         "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    res: list[tuple[bool, str]] = []

    def check(cond, what):
        res.append((bool(cond), what))
        print(f"  {'PASS' if cond else 'FAIL'}  {what}")

    tmp = Path(tempfile.mkdtemp(prefix="siteanalytics-"))
    try:
        ws = None
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list",
                                            timeout=1) as r:
                    t = [x for x in json.load(r) if x.get("type") == "page"]
                if t:
                    ws = t[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        c = CDP(ws)
        for d in ("Runtime", "Log", "Page", "Network"):
            c.send(d + ".enable")

        def goto(name: str, settle: float = 2.0):
            c.events.clear()
            c.send("Page.navigate", url=(site / name).as_uri())
            c.drain(settle)
            errs, reqs = [], []
            for e in c.events:
                m = e.get("method")
                if m == "Log.entryAdded" and \
                        e["params"]["entry"].get("level") in ("error", "warning"):
                    errs.append(e["params"]["entry"].get("text"))
                elif m == "Runtime.exceptionThrown":
                    errs.append(str(e["params"]["exceptionDetails"].get("text"))[:120])
                elif m == "Network.loadingFailed":
                    errs.append("loadingFailed " + str(e["params"].get("errorText")))
                elif m == "Network.requestWillBeSent":
                    reqs.append(e["params"]["request"]["url"])
            return errs, reqs

        pages = ["index.html", "search.html", "methods.html", "findings.html"]
        if (site / "graph.html").exists():
            pages.append("graph.html")

        print("every page")
        for name in pages:
            errs, reqs = goto(name, 3.0 if name in ("search.html", "graph.html") else 1.5)
            off = [u for u in reqs if not u.startswith("file:")]
            ext = c.js("""(function(){var b=[];document.querySelectorAll(
                '[src],link[href],iframe,img,object,embed').forEach(function(e){
                b.push(e.tagName+':'+(e.getAttribute('src')||e.getAttribute('href')));});
                return b;})()""")
            check(not errs and not off and not ext,
                  f"{name}: no console errors, no off-file:// request, no external "
                  f"resource (errs={errs[:1]} off={off[:1]} ext={ext[:1]})")
            check(c.js("document.body.scrollWidth <= window.innerWidth + 2"),
                  f"{name}: no horizontal overflow at 1400px")
            c.send("Emulation.setDeviceMetricsOverride", width=390, height=780,
                   deviceScaleFactor=2, mobile=True)
            time.sleep(0.5)
            check(c.js("document.documentElement.scrollWidth <= 392"),
                  f"{name}: no horizontal overflow at 390px (iPhone width)")
            c.send("Emulation.clearDeviceMetricsOverride")
            time.sleep(0.3)

        print("navigation and theme")
        goto("index.html")
        nav = c.js("""(function(){var a=[];document.querySelectorAll('nav.site a')
            .forEach(function(e){a.push(e.getAttribute('href'))});return a;})()""")
        check(set(nav) >= {"search.html", "methods.html", "findings.html"},
              f"nav links every page that exists ({nav})")
        dead = c.js("""(function(){var bad=[];document.querySelectorAll('a[href]')
            .forEach(function(a){var h=a.getAttribute('href');
              if(/^(https?:|#|mailto:)/.test(h)) return;
              bad.push(h);});return bad;})()""")
        missing = sorted({h for h in dead if not (site / h.split("#")[0]).exists()})
        check(not missing, f"no broken internal link on the landing page ({missing})")
        # Headless Chromium reports prefers-color-scheme: dark, so "auto" and
        # "dark" look identical here. Click through the cycle until each explicit
        # mode has been seen and compare those, rather than assuming the starting
        # state.
        seen = {}
        start = c.js("document.documentElement.getAttribute('data-theme')")
        for _ in range(3):
            c.js("document.getElementById('theme').click()")
            time.sleep(0.25)
            seen[c.js("document.documentElement.getAttribute('data-theme')")] = \
                c.js("getComputedStyle(document.body).backgroundColor")
        check(set(seen) == {"dark", "light", None},
              f"theme button cycles dark / light / auto ({sorted(map(str, seen))})")
        check(seen.get("dark") != seen.get("light"),
              f"dark and light are genuinely different palettes "
              f"({seen.get('dark')} vs {seen.get('light')})")
        check(_lum(seen.get("light", "")) > 200 and _lum(seen.get("dark", "")) < 60,
              f"light is light and dark is dark (luminance "
              f"{_lum(seen.get('light', '')):.0f} / {_lum(seen.get('dark', '')):.0f})")
        check(c.js("document.documentElement.getAttribute('data-theme')") == start,
              "three clicks return the page to the state it started in")

        print("landing page")
        goto("index.html")
        pos = c.js("""(function(){var w=document.querySelector('.warn'),
            g=document.querySelector('.grid');
            return {warn:w?w.getBoundingClientRect().top+window.scrollY:-1,
                    grid:g?g.getBoundingClientRect().top+window.scrollY:-1};})()""")
        check(pos["warn"] > 0 and pos["warn"] < pos["grid"],
              f"limitations sit above the statistics, not buried "
              f"(warning at y={pos['warn']:.0f}, stats at y={pos['grid']:.0f})")
        # textContent keeps the source line breaks; the prose is what is being
        # asserted, not how it happens to be wrapped in the file.
        txt = _flat(c.js("document.querySelector('.warn').textContent"))
        for phrase in ("ranked list of things to look at",
                       "not a finding of misconduct",
                       "cannot detect fabricated data"):
            check(phrase in txt, f"landing states {phrase!r} up front")
        check("24,656" in _flat(c.js("document.body.textContent")),
              "landing states the corpus size")
        check(c.js("document.querySelectorAll('.stat').length") >= 6,
              f"summary statistics present ({c.js('document.querySelectorAll(\".stat\").length')})")
        check(c.js("document.querySelectorAll('.steps li').length") >= 5,
              "how-it-was-built pipeline listed")

        print("search")
        goto("search.html", 4.0)
        check(c.js("document.getElementById('ui').style.display") != "none"
              and c.js("document.getElementById('loading').style.display") == "none",
              "index decompressed and the UI came up")
        n0 = c.js("document.querySelectorAll('#hits li').length")
        total = c.js("""(function(){var m=/of ([\\d,]+) matching/
            .exec(document.getElementById('count').textContent);
            return m?parseInt(m[1].replace(/,/g,''),10):0;})()""")
        check(n0 > 0 and total == 24656,
              f"all {total:,} theses searchable, first page rendered ({n0} rows)")
        payload = c.js("""(function(){
            var t=document.getElementById('ix').textContent.trim();
            return t.length;})()""")
        check(payload > 100000, f"index shipped inside the page ({payload:,} base64 chars)")
        c.js("""(function(){var q=document.getElementById('q');
            q.value='thermal imaging'; q.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.6)
        n1 = c.js("document.querySelectorAll('#hits li').length")
        t1 = c.js("document.getElementById('count').textContent")
        check(0 < n1 <= n0 and "matching" in t1,
              f"free-text query narrows the list ({t1.strip()})")
        check(c.js("document.querySelectorAll('#hits mark').length") > 0,
              "matched words are highlighted in the title")
        href = c.js("""document.querySelector('#hits a.t').getAttribute('href')""")
        check(href.startswith("https://etheses.whiterose.ac.uk/id/eprint/"),
              f"each hit links to its repository record ({href})")
        c.js("""(function(){var q=document.getElementById('q');q.value='';
            q.dispatchEvent(new Event('input'));
            var d=document.getElementById('fd');d.value='Mathematics';
            d.dispatchEvent(new Event('change'));})()""")
        time.sleep(0.6)
        tm = c.js("document.getElementById('count').textContent")
        nm = c.js("""(function(){var m=/of ([\\d,]+) matching/.exec(
            document.getElementById('count').textContent);
            return m?parseInt(m[1].replace(/,/g,''),10):0;})()""")
        subs = c.js("document.querySelectorAll('#fs option').length")
        check(nm == 831, f"discipline filter is exact (Mathematics -> {nm} theses)")
        check(1 < subs < 99, f"sub-field list follows the discipline ({subs - 1} options)")
        c.js("""(function(){var y=document.getElementById('y0');y.value='2020';
            y.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.5)
        ny = c.js("""(function(){var m=/of ([\\d,]+) matching/.exec(
            document.getElementById('count').textContent);
            return m?parseInt(m[1].replace(/,/g,''),10):0;})()""")
        yrs = c.js("""(function(){var bad=0;document.querySelectorAll('#hits .m')
            .forEach(function(e){var m=/\\b(19|20)\\d\\d\\b/.exec(e.textContent);
              if(m && parseInt(m[0],10)<2020) bad++;});return bad;})()""")
        check(0 < ny < nm and yrs == 0,
              f"year filter narrows and every shown row obeys it "
              f"({nm} -> {ny}, {yrs} violations)")
        c.js("document.getElementById('clr').click()")
        time.sleep(0.6)
        check(c.js("""(function(){var m=/of ([\\d,]+) matching/.exec(
            document.getElementById('count').textContent);
            return m?parseInt(m[1].replace(/,/g,''),10):0;})()""") == 24656,
              "clear restores the full corpus")
        c.js("""(function(){var q=document.getElementById('q');q.value='zzqqxnope';
            q.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.5)
        check(c.js("document.querySelectorAll('#hits .empty').length") == 1,
              "a query with no hits shows the empty state, not a blank page")
        c.js("""(function(){var q=document.getElementById('q');q.value='';
            q.dispatchEvent(new Event('input'));})()""")
        time.sleep(0.6)
        b0 = c.js("document.querySelectorAll('#hits li').length")
        c.js("document.getElementById('more').click()")
        time.sleep(0.3)
        check(c.js("document.querySelectorAll('#hits li').length") > b0,
              f"'show more' pages further into the results ({b0} -> "
              f"{c.js('document.querySelectorAll(\"#hits li\").length')})")

        print("copyright guarantee")
        probe = c.js("""(async function(){
            var b64=document.getElementById('ix').textContent.trim();
            var bin=Uint8Array.from(atob(b64),function(ch){return ch.charCodeAt(0);});
            var st=new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
            var D=JSON.parse(await new Response(st).text());
            var keys=Object.keys(D).sort(), longest=0, count=0;
            ['d','s','t'].forEach(function(k){ D[k].forEach(function(v){
                count++; if(v.length>longest) longest=v.length; }); });
            return {keys:keys, longest:longest, strings:count, rows:D.id.length};
        })()""")
        check(set(probe["keys"]) == PAYLOAD_KEYS,
              f"search payload carries only the declared fields ({probe['keys']})")
        check(probe["longest"] <= TITLE_CAP,
              f"no string in the payload exceeds the {TITLE_CAP}-char title cap "
              f"(longest of {probe['strings']:,} is {probe['longest']})")
        check(probe["rows"] == 24656, f"payload holds every thesis ({probe['rows']:,})")

        print("methods")
        goto("methods.html")
        body = _flat(c.js("document.body.textContent")).replace("\u2212", "-")
        for phrase in ("n + w - 1 = 8 + 13 - 1 =", "winnowing".capitalize(),
                       "Schleimer", "303,946,840", "1,363", "222,881",
                       "ranked list of things to look at",
                       "not evidence of intent"):
            check(phrase.lower() in body.lower(), f"methods states {phrase!r}")
        check("i n f o r m a t i o n" in body,
              "methods names the character-spacing artefact it filters")
        check(c.js("document.querySelectorAll('table').length") >= 5,
              "methods carries the filter, verdict and composition tables")
        check("20" in body and "guarantee" in body.lower(),
              "methods states the 20-word detection guarantee")

        # The graceful-degradation claim is only worth anything if both halves
        # run, so both are built here rather than asserted about whichever state
        # out/analytics/ happens to be in when the check runs.
        print("findings, with out/analytics absent")
        (tmp / "empty").mkdir()
        rc = subprocess.run(
            [sys.executable, str(HERE / "tg_site.py"), "--out", str(tmp / "bare"),
             "--analytics", str(tmp / "empty"), "--no-apps"],
            capture_output=True, text=True)
        check(rc.returncode == 0,
              f"build with an empty out/analytics/ succeeds (rc={rc.returncode})")
        c.events.clear()
        c.send("Page.navigate", url=(tmp / "bare" / "findings.html").as_uri())
        c.drain(1.5)
        ftxt = _flat(c.js("document.body.textContent"))
        check("Not built yet" in ftxt, "findings says the analysis has not been run")
        check("tg canon" in ftxt, "findings says how to produce the figures")
        check(c.js("document.querySelectorAll('main table').length") == 1
              and c.js("document.querySelectorAll('.tw table.kv').length") == 0,
              "findings invents no figures when there are none")

        print("findings, with synthetic analytics")
        (tmp / "analytics").mkdir()
        for name, obj in sorted(SYNTHETIC.items()):
            (tmp / "analytics" / name).write_text(
                json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
        (tmp / "analytics" / "citation_age_by_field.csv").write_text(
            "discipline,median_age\nMathematics,14.2\nComputing,6.1\n",
            encoding="utf-8")
        rc = subprocess.run(
            [sys.executable, str(HERE / "tg_site.py"), "--out", str(tmp / "site"),
             "--analytics", str(tmp / "analytics"), "--no-apps"],
            capture_output=True, text=True)
        check(rc.returncode == 0, f"rebuild against a synthetic out/analytics/ "
                                  f"succeeds (rc={rc.returncode})")
        c.events.clear()
        c.send("Page.navigate", url=(tmp / "site" / "findings.html").as_uri())
        c.drain(1.5)
        atxt = _flat(c.js("document.body.textContent"))
        check("Braun & Clarke (2006)" in atxt and "1,234" in atxt,
              "findings renders the canon table it was given")
        check("Citation age" in atxt and "9.4" in atxt,
              "findings renders citation-age JSON")
        check("citation_age_by_field.csv" in atxt and "14.2" in atxt,
              "findings renders a CSV as a table too")
        check("Field flow" in atxt and "Psychology" in atxt,
              "findings renders field flow")
        check("canon.json" in atxt and "sha256" in atxt,
              "findings prints provenance for every figure it shows")
        check("Not built yet" not in atxt, "the 'not built' notice is gone")
        # Two 900-character values, one under a prose key and one not: the cap
        # must let the caveat through in full and cut the other off.
        (tmp / "analytics" / "canon_note.json").write_text(
            json.dumps({"caveat": "C" * 900, "label": "L" * 900}, sort_keys=True),
            encoding="utf-8")
        rc = subprocess.run(
            [sys.executable, str(HERE / "tg_site.py"), "--out", str(tmp / "site2"),
             "--analytics", str(tmp / "analytics"), "--no-apps"],
            capture_output=True, text=True)
        c.events.clear()
        c.send("Page.navigate", url=(tmp / "site2" / "findings.html").as_uri())
        c.drain(1.5)
        gtxt = c.js("document.body.textContent")
        check("L" * 900 not in gtxt and "L" * 200 in gtxt,
              "a 900-char value under an ordinary key is cut to 200, not published")
        # The cap has no exemptions. It used to let 1,600 characters through
        # under a list of prose-sounding key names, which meant a passage
        # written under any of them was published unread.
        check("C" * 900 not in gtxt and "C" * 200 in gtxt,
              "a 900-char value under a prose key (caveat) is cut to 200 too — "
              "the cap keys on nothing")
        check("truncated" in _flat(gtxt),
              "the page says on its face that a value was truncated")
        check("value(s) truncated" in rc.stderr,
              f"the build reports the truncation on stderr "
              f"({[l for l in rc.stderr.splitlines() if 'truncat' in l][:1]})")

        print("findings, against the real out/analytics if it exists")
        goto("findings.html")
        rtxt = _flat(c.js("document.body.textContent"))
        if "Not built yet" in rtxt:
            check(True, "out/analytics/ is absent in this build; findings says so")
        else:
            nt = c.js("document.querySelectorAll('main table').length")
            check(nt >= 3, f"real analytics rendered ({nt} tables)")
            check("sha256" in rtxt, "real analytics carry provenance")
            check("caveat" in rtxt.lower(),
                  "the analysis's own caveats are reproduced")

        print("no thesis text reached the published pages")
        corpus, db = HERE / "corpus" / "text", HERE / "corpus" / "corpus.db"
        if corpus.is_dir() and db.exists():
            r = scan_for_thesis_text(site, corpus, db, 300)
            print(f"  scanned {r['files']} text stream(s), {r['chars']:,} "
                  f"characters — every file in {site.name}/, payloads inflated")
            check(not r["hits"],
                  f"no {RUN_WORDS}-word run of the {r['sentences']:,} real "
                  f"sentences from {r['docs']} theses appears anywhere in what "
                  f"the site ships ({r['runs']:,} matched, {r['titles']:,} of them "
                  f"inside a published title; {r['hits'][:2]})")
            print(f"        {r['short']:,} sentence(s) of fewer than {RUN_WORDS} "
                  f"words cannot form a run and are outside this bound")
        else:
            print(f"  SKIP  {corpus} or {db.name} not present — cannot scan for "
                  f"thesis text")

        print("bridges is linked, not copied")
        check(not (site / "bridges.html").exists(),
              "the bridge reader (which quotes thesis sentences) is NOT in the "
              "site unless --include-bridges asks for it")
        goto("index.html")
        check("separate decision" in _flat(c.js("document.body.textContent")),
              "the landing page explains why bridges is absent")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    total = sum(p.stat().st_size for p in sorted(site.rglob("*")) if p.is_file())
    bad = [w for ok, w in res if not ok]
    print("-" * 70)
    for p in sorted(site.iterdir()):
        if p.is_file():
            print(f"  {p.name:<16} {p.stat().st_size / 1024:>9,.0f} KB")
    print(f"  {'TOTAL':<16} {total / 1024:>9,.0f} KB  ({total / 1e6:.2f} MB deployable)")
    print(f"SITE CHECK {'FAILED' if bad else 'PASSED'} — {len(res) - len(bad)}/{len(res)}")
    for w in bad:
        print("  -", w)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
