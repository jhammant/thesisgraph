#!/usr/bin/env python3
"""
tg_similar.py — "what else in this corpus is about what I am about?"

The rest of the toolkit answers questions about the corpus in aggregate: what it
cites, what methods it uses, how its subjects divide. This module answers the
one question an individual researcher actually asks — *show me the neighbours of
this thesis* — and it exists mainly for the case a subject browse cannot serve:
`--cross-discipline`, which hides everything inside the query's own department
and shows only the work next door. A student browsing their own faculty listing
will never meet the sociologist who solved their sampling problem; a vector
space does not know where the departmental boundary is, so it will.

Two indexes are involved, and the difference between them is the point:

  * corpus/title_emb.npy, which despite its name is NOT titles. subfields.py
    built it from `title + ". " + abstract` truncated to 900 characters, and
    hierarchy.py and subfields.py cluster on it. It is available with no build
    step, so it is the fallback (`--legacy-index`).
  * The index this module builds, which encodes the title and the abstract
    SEPARATELY: the abstract in overlapping ~140-word windows (all-MiniLM-L6-v2
    truncates at 256 word pieces, and 97.3% of these abstracts run past 900
    characters — they average 2,202, counting only the 20,940 documents that
    have one — so a 900-character prefix cuts into nearly all of them),
    mean-pooled and re-normalised. Storing the two vectors apart rather than one
    blended vector makes `--title-weight` a query-time knob instead of a
    rebuild, which is what made it possible to measure the blend rather than
    guess it (`--evaluate`).

On that measurement, honestly, and in both directions. `--evaluate` scores each
variant by how much cited literature its neighbours share, which citations.db
knows independently of any embedding — here over 2,000 sampled theses, ten
neighbours each (`--evaluate-n 2000 --evaluate-top 10`):

    variant                                mean shared refs   >=1 ref  >=3 refs  no abstract
    legacy title+abs[:900] cal 0.50                    1.29     22.6%     11.2%        14.2%
    title only cal 0.50                                1.11     19.2%      9.1%        15.1%
    abstract only cal 0.50                             1.31     22.4%     11.2%        13.5%
    title 0.15 + abstract 0.85 cal 0.50                1.35     22.8%     11.5%        12.8%
    title 0.25 + abstract 0.75 cal 0.50                1.34     22.7%     11.6%        12.5%
    title 0.35 + abstract 0.65 cal 0.50                1.35     22.7%     11.6%        12.2%
    title 0.50 + abstract 0.50 cal 0.50                1.31     22.3%     11.1%        12.2%
    title 0.25 + abstract 0.75 raw cosine              1.38     23.4%     11.8%         6.0%
    (unrelated baseline)                               0.03      2.3%      0.2%        14.4%

Titles alone ARE weak — 17% less shared literature than the blend, and in
practice they latch onto a shared proper noun rather than a shared subject. But
chunking the whole abstract beat truncating it at 900 characters by 0.05 shared
references (1.34 against 1.29) and by 0.1 of a point on the >=1 rate: small, and
the two columns do not agree on how small, so the honest reading is that the two
indexes are close. MiniLM's 256-word-piece window already covers most of a
900-character prefix, and abstracts front-load their topic. The claim for this
index is therefore not that it retrieves better than what subfields.py already
built, but that it separates the two signals, which is what makes
--title-weight, title-only queries and that table possible at all.

The default weight of 0.25 is not a peak and this docstring should not have
called it one. Over 2,000 queries, 0.15, 0.25 and 0.35 sit within 0.01 shared
references and 0.1 of a point of each other on every column: they are
indistinguishable, and an earlier 0.1%-wide "peak" was a smaller gap than the
one this docstring elsewhere calls close. What the sweep does establish is the
shape — the ends (title only, title 0.50) are worse than the middle — so read
0.25 as measured and not beaten, not as optimal.

The one thing this index does badly, stated plainly, because a ranked list
cannot show a reader who is missing from it. 3,716 of the 24,656 status='ok'
documents (15.07%) have no abstract at all. compose() gives those their title
vector, since a zero row would drop them out of every result — but a bare title
vector is not on the same scale as the blended vectors everything else has.
Measured over 200 queries: their cosines against the corpus average 0.020 where
an abstract-bearing document's average 0.036, and their spread is 0.108 against
0.123. Both differences push the same way in the tail that a top-k is, and the
effect is not subtle. On a raw cosine they take 6.9% of top-10 slots (6.0% when
the queries are drawn from the evaluation pool, the table above), against a
15.07% base rate, while a title-only representation, in which every document is
on the same scale, gives them 15.9% — their base rate. The legacy index is
milder, at 11.1%. The cause is the representation, not the corpus, and on this
one axis the index built here is worse than the one it improves on.

Imputing the missing abstract does not fix it: shifting a title vector into
abstract space by the corpus-mean offset between the two moves the share from
6.9% to 7.5%, and averaging the abstract vectors of the ten title-nearest
documents that do have one over-corrects to 49%, because the average of ten
vectors sits near the middle of the space and is close to everything. Whitening
the whole space reaches 9.8% and pays for it in shared references. What works is
to stop comparing raw cosines across representations, and score each document
against its own similarity distribution instead: adj = (cos - mu_d) / sd_d, both
terms computed exactly over the corpus. That is an ordinary scale correction, it
is a property of a vector, and it never asks which documents have an abstract.

At full strength it overshoots — 18.7% of slots — and the citation yardstick
says as much: the abstract-less neighbours it lets in share fewer references
(mean 2.47) than the abstract-bearing ones beside them (2.71), counting only the
neighbours with 25 or more parsed references, which is all the yardstick can
see. The default is therefore half strength (--calibration 0.5), where over the
same 2,000 queries the abstract-less neighbours shown still share MORE cited
literature than the abstract-bearing ones beside them (3.26 against 2.76) while
taking 12.5% of the slots instead of 6.0%. It is not free: the aggregate falls
even so, which means the neighbours pushed past rank 10 were better still. The
paired cost over those 2,000 queries is -0.038 +- 0.014 shared references, about
2.8%, with the >=1 rate falling from 23.4% to 22.7%. That is
the trade this module makes by default — a sixth of the corpus roughly doubles
its visibility, for 2.8% of a proxy measure — and it is reversible with
--no-calibration. A gap to the base rate remains at 12.5% against 15.07%, so the
header of every run prints it, --evaluate reports the share for every variant,
and both numbers can be reproduced from the CLI rather than believed.

Two cautions about the yardstick itself. The evaluation pool is the 7,591
documents with at least 25 parsed references, and reference parsing here is
author-date only (README), so the pool is not a slice of the corpus: Education
4.5% -> 9.6%, Psychology 4.8% -> 9.6%, Business and economics 4.9% -> 7.1% and
Social sciences 8.8% -> 11.8% are over-weighted, while Physical sciences
12.4% -> 4.0%, Engineering 6.4% -> 3.0%, History, philosophy and religion
7.2% -> 3.6%, Mathematics 3.4% -> 1.9% and Computing 4.4% -> 2.9% are
under-weighted. It tilts towards the social sciences and NOT towards the
humanities: footnote-referencing subjects fail a 25-entry author-date test as
badly as the physical sciences do. What the pool is not is blind to the
documents the previous paragraphs are about — 14.5% of it has no abstract
against 15.07% of the corpus — so --evaluate can see them, and now counts them.

Everything printed is bibliographic: id, title, creator, year, discipline, link,
a cosine and its calibrated counterpart. No thesis text and no abstract text is
ever emitted — whether a document has an abstract is read as a length, never as
text — so the output of this command, table or --json, is derived data and safe
to publish. What --build writes is ids.json, meta.json and two float32 matrices
under out/similar, 73 MiB: embeddings and identifiers, which the README classes
as shareable derived data, and not one sentence of any thesis. They are build
output rather than a publication artefact, and out/ is gitignored; nothing in
this module writes anywhere else.

Determinism: ranking ties break on id, both scores are rounded before sorting,
the calibration terms are exact corpus statistics rather than a sample, and the
built index carries no timestamp — no wall clock and no RNG anywhere, including
in the evaluation sample, which is a fixed stride over sorted ids.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "similar"
HELP = "nearest-neighbour theses, by id, free text or title"

DB = HERE / "corpus" / "corpus.db"
CITE_DB = HERE / "corpus" / "citations.db"
LEGACY_EMB = HERE / "corpus" / "title_emb.npy"
LEGACY_IDS = HERE / "corpus" / "title_emb_ids.json"
# corpus/ is read-only for this toolkit, so the built index lives under out/.
DEFAULT_INDEX = HERE / "out" / "similar"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 140          # ~190 word pieces, inside MiniLM's 256 limit
CHUNK_OVERLAP = 30         # words repeated between windows, so no idea is cut in half
MAX_CHUNKS = 6             # abstracts are capped at 4,000 chars upstream; this is slack
TITLE_WEIGHT = 0.25        # chosen by --evaluate, not by taste
CALIBRATION = 0.5          # per-document score calibration; also measured, see below
SCORE_DP = 6               # cosines rounded before sorting, so ties break identically
HOME_PROBE = 25            # neighbours consulted to infer a text query's home discipline

# Share of top-10 slots taken by the 15.07% of documents that have no abstract,
# over query samples of 200 to 2,000 (the docstring has the runs), uncalibrated
# and at the default strength. Quoted in --help and in the header of every run,
# because a ranked list cannot show a reader who is missing from it.
NO_ABS_SHARE = {                            # (legacy index?, calibrated?)
    (False, False): ("6-7%", "far below their 15.1% share of the corpus"),
    (False, True): ("12-14%", "still below their 15.1% share of the corpus"),
    (True, False): ("11-12%", "below their 15.1% share of the corpus"),
    (True, True): ("15%", "about their share of the corpus"),
}


# --------------------------------------------------------------------------- #
# Corpus metadata
# --------------------------------------------------------------------------- #

def _connect(path: Path) -> sqlite3.Connection:
    """Read-only connection. Other agents may be writing these databases."""
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def load_meta(db: Path) -> dict[str, dict]:
    con = _connect(db)
    try:
        rows = con.execute(
            "SELECT id,title,creator,year,discipline,subfield,landing,pdf_url,words,"
            # a length, not the abstract: no thesis text crosses this connection
            "COALESCE(LENGTH(TRIM(abstract)),0) FROM doc WHERE status='ok' "
            "ORDER BY id").fetchall()
    finally:
        con.close()
    # Harvested titles carry the deposit form's line breaks; collapse them so a
    # title is one line in the table and one string in the JSON.
    def flat(v):
        return " ".join((v or "").split())

    return {r[0]: {"id": r[0], "title": flat(r[1]), "creator": flat(r[2]),
                   "year": r[3], "discipline": flat(r[4]), "subfield": flat(r[5]),
                   "url": r[6] or r[7] or "", "words": r[8] or 0,
                   "has_abstract": bool(r[9])} for r in rows}


def resolve_id(meta: dict[str, dict], query: str) -> str:
    """Accept a full oai id, a bare record number, or any unambiguous suffix.

    The repo prefix is never hard-coded: it is recovered from the ids present.
    """
    q = query.strip()
    if q in meta:
        return q
    suffix = [i for i in meta if i.endswith(":" + q)]
    if len(suffix) == 1:
        return suffix[0]
    lower = [i for i in meta if i.lower() == q.lower()]
    if len(lower) == 1:
        return lower[0]
    contains = sorted(i for i in meta if q.lower() in i.lower())
    if len(contains) == 1:
        return contains[0]
    if contains:
        raise ValueError(f"{query!r} matches {len(contains)} ids, e.g. "
                         f"{', '.join(contains[:3])}")
    raise ValueError(f"no thesis with id {query!r} and status='ok'")


def resolve_discipline(meta: dict[str, dict], want: str) -> str:
    known = sorted({m["discipline"] for m in meta.values() if m["discipline"]})
    for d in known:
        if d.lower() == want.lower():
            return d
    hits = [d for d in known if want.lower() in d.lower()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"discipline {want!r} is not one of: {'; '.join(known)}")


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP,
           limit: int = MAX_CHUNKS) -> list[str]:
    """Overlapping word windows that each fit the encoder's context."""
    words = (text or "").split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return out[:limit]


def get_model(device: str):
    """The project's offline MiniLM, with the model name taken from run_analysis.

    run_analysis owns the constant, so the name cannot drift from the sentence
    embeddings already cached in cache/; it is imported lazily because tg.py
    imports every plugin on every invocation and run_analysis is 188 KB of
    pdfplumber-importing pipeline.
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    name = EMBED_MODEL
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    try:
        import run_analysis
        name = run_analysis.EMBED_MODEL
    except Exception:                                   # noqa: BLE001 — advisory only
        pass
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, local_files_only=True, device=device), name


def model_dim(model) -> int:
    """sentence-transformers renamed this method; support both spellings."""
    fn = getattr(model, "get_embedding_dimension", None) or \
        model.get_sentence_embedding_dimension
    return int(fn())


def encode(model, texts: list[str], batch: int):
    import numpy as np
    if not texts:
        return np.zeros((0, model_dim(model)), dtype="float32")
    return model.encode(texts, batch_size=batch, convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")


def encode_long(model, text: str, batch: int):
    """One normalised vector for a text of any length: mean of its windows."""
    import numpy as np
    parts = chunks(text)
    if not parts:
        return None
    v = encode(model, parts, batch).mean(axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).astype("float32") if n else None


# --------------------------------------------------------------------------- #
# Index build / load
# --------------------------------------------------------------------------- #

def signature(db: Path) -> str:
    """Fingerprint of the corpus an index was built from, so staleness is visible.

    Ids and field lengths only, computed in SQL: enough to notice a re-harvest,
    and it moves no thesis text across the connection to do it.
    """
    import hashlib
    h = hashlib.sha256()
    con = _connect(db)
    try:
        for doc_id, tl, al in con.execute(
                "SELECT id,COALESCE(LENGTH(title),0),COALESCE(LENGTH(abstract),0) "
                "FROM doc WHERE status='ok' ORDER BY id"):
            h.update(f"{doc_id}\t{tl}\t{al}\n".encode())
    finally:
        con.close()
    return h.hexdigest()[:16]


def build_index(db: Path, out: Path, device: str, batch: int, limit: int | None,
                log=lambda s: print(s, file=sys.stderr)) -> dict:
    import numpy as np
    con = _connect(db)
    try:
        rows = con.execute("SELECT id,title,abstract FROM doc WHERE status='ok' "
                           "ORDER BY id").fetchall()
    finally:
        con.close()
    sig = signature(db)
    if limit:
        # A bounded build takes every Nth document rather than the first N, so a
        # subset index still spans the whole corpus (and every discipline).
        stride = max(1, len(rows) // limit)
        rows = rows[::stride][:limit]

    model, name = get_model(device)
    dim = model_dim(model)
    ids = [r[0] for r in rows]
    e_title = np.zeros((len(rows), dim), dtype="float32")
    e_abs = np.zeros((len(rows), dim), dtype="float32")

    log(f"  encoding {len(rows):,} titles on {device} ...")
    for start in range(0, len(rows), 1024):
        block = rows[start:start + 1024]
        # An empty title falls back to the id, so no document gets a zero vector.
        e_title[start:start + len(block)] = encode(
            model, [(r[1] or "").strip() or r[0] for r in block], batch)

    # Abstracts are chunked, so every document contributes a variable number of
    # rows to one flat encode: batching across documents keeps the encoder busy.
    todo = [(i, part) for i, r in enumerate(rows) for part in chunks(r[2] or "")]
    log(f"  encoding {len(todo):,} abstract windows from "
        f"{sum(1 for r in rows if (r[2] or '').strip()):,} abstracts ...")
    acc = np.zeros((len(rows), dim), dtype="float32")
    seen = np.zeros(len(rows), dtype="int32")
    for start in range(0, len(todo), 2048):
        block = todo[start:start + 2048]
        vecs = encode(model, [t for _, t in block], batch)
        for (i, _), v in zip(block, vecs):
            acc[i] += v
            seen[i] += 1
        if start and start % 20480 == 0:
            log(f"    {start:,}/{len(todo):,}")
    hit = seen > 0
    acc[hit] /= np.linalg.norm(acc[hit], axis=1, keepdims=True)
    e_abs[hit] = acc[hit]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "emb_title.npy", e_title)
    np.save(out / "emb_abstract.npy", e_abs)
    (out / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    meta = {"model": name, "dim": int(dim), "docs": len(ids),
            "with_abstract": int(hit.sum()), "abstract_windows": len(todo),
            "chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP,
            "max_chunks": MAX_CHUNKS, "device": device,
            "corpus_signature": sig, "partial": bool(limit)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    return meta


class Index:
    """Title and abstract matrices plus the id order they share."""

    def __init__(self, ids, e_title, e_abs, meta, label):
        import numpy as np
        self.ids = ids
        self.pos = {d: i for i, d in enumerate(ids)}
        self.title = e_title
        self.abstract = e_abs if e_abs is not None else None
        self.meta = meta
        self.label = label
        self._np = np

    def compose(self, w_title: float):
        """The document matrix at a given title weight, re-normalised."""
        np = self._np
        if self.abstract is None or w_title >= 1.0:
            return self.title
        if w_title <= 0.0:
            # Documents with no abstract keep their title vector; a zero row
            # would silently drop them out of every result.
            m = self.abstract.copy()
            blank = ~m.any(axis=1)
            m[blank] = self.title[blank]
            return m
        m = w_title * self.title + (1.0 - w_title) * self.abstract
        blank = ~self.abstract.any(axis=1)
        m[blank] = self.title[blank]
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n


def load_index(path: Path, legacy: bool) -> Index:
    import numpy as np
    if legacy:
        if not LEGACY_EMB.exists():
            raise FileNotFoundError(f"no legacy index at {LEGACY_EMB}")
        ids = json.loads(LEGACY_IDS.read_text(encoding="utf-8"))
        e = np.load(LEGACY_EMB)
        return Index(ids, e, None,
                     {"model": EMBED_MODEL, "docs": len(ids), "partial": False,
                      "note": "title + abstract truncated to 900 chars, one vector"},
                     f"{LEGACY_EMB.relative_to(HERE)} (legacy, single vector)")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    ids = json.loads((path / "ids.json").read_text(encoding="utf-8"))
    return Index(ids, np.load(path / "emb_title.npy"),
                 np.load(path / "emb_abstract.npy"), meta,
                 f"{path.relative_to(HERE) if path.is_relative_to(HERE) else path}"
                 f" (title + chunked abstract)")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def calibration_terms(matrix):
    """Each document's own similarity distribution over the corpus: (mean, sd, mean sd).

    Documents are not all on one scale. A document with no abstract is a bare
    title vector, and a title vector scores lower, and over a narrower range,
    against the blended vectors everything else has: measured over 200 queries,
    mean cosine 0.020 against 0.036 and spread 0.108 against 0.123. Both push the
    same way in the tail that a top-k is, and the result is that 15.07% of the
    corpus takes ~6% of the slots (see the module docstring). Scoring a document
    against its own distribution instead of against a raw cosine removes that,
    and it does so without knowing which documents lack an abstract — it is a
    property of the vector, so nothing is singled out.

    Exact, not sampled: with xbar the corpus mean vector and S its second-moment
    matrix, mu = d.xbar and sd = sqrt(d'Sd - mu^2), which is 384x384 of work
    rather than 24,656^2, and identical on every run.
    """
    import numpy as np
    n = len(matrix)
    if n == 0:
        return None
    xbar = matrix.mean(axis=0)
    second = (matrix.T @ matrix) / n
    mu = matrix @ xbar
    var = np.einsum("ij,jk,ik->i", matrix, second, matrix) - mu * mu
    sd = np.sqrt(np.maximum(var, 1e-12))
    return mu, sd, float(sd.mean())


def calibrated(scores, terms, strength: float):
    """Move a raw cosine towards a per-document z-score, `strength` of the way.

    At 1.0 this is the full z-score, (cos - mu_d) / sd_d. Below that the spread is
    shrunk geometrically towards the corpus mean spread, which keeps the number on
    a standard-deviation scale and readable, and changes no ranking — the shrinkage
    factor is one positive constant. The document mean is removed at every strength
    above 0; 0 itself switches the calibration off and is the same thing as
    --no-calibration, so the knob is deliberately not continuous there.
    """
    import numpy as np
    if terms is None or strength <= 0.0:
        return scores
    mu, sd, sd_mean = terms
    scale = np.power(sd, strength) * (sd_mean ** (1.0 - strength))
    return (scores - mu) / scale


def rank(index: Index, matrix, qvec, meta: dict[str, dict], k: int,
         exclude: set[str], discipline: str | None, not_discipline: str | None,
         year_from: int | None, year_to: int | None, min_sim: float,
         terms=None, strength: float = 0.0):
    """Deterministic top-k of (ranking score, cosine, id).

    Ranking is on the calibrated score; the cosine is carried through untouched
    because it is the number the user is shown and --min-sim filters on. Both are
    rounded before the sort, so ties break identically on every run.
    """
    scores = matrix @ qvec
    adj = calibrated(scores, terms, strength)
    out = []
    for i, doc_id in enumerate(index.ids):
        if doc_id in exclude:
            continue
        m = meta.get(doc_id)
        if m is None:
            continue
        if discipline and m["discipline"] != discipline:
            continue
        if not_discipline and m["discipline"] == not_discipline:
            continue
        y = m["year"]
        if year_from is not None and (y is None or y < year_from):
            continue
        if year_to is not None and (y is None or y > year_to):
            continue
        s = round(float(scores[i]), SCORE_DP)
        if s < min_sim:
            continue
        out.append((round(float(adj[i]), SCORE_DP), s, doc_id))
    out.sort(key=lambda t: (-t[0], t[2]))
    return out[:k] if k > 0 else []


def infer_home(index: Index, matrix, qvec, meta: dict[str, dict],
               terms=None, strength: float = 0.0) -> str | None:
    """A free-text query has no department; borrow one from its nearest neighbours."""
    from collections import Counter
    top = rank(index, matrix, qvec, meta, HOME_PROBE, set(), None, None,
               None, None, -1.0, terms, strength)
    counts = Counter(meta[d]["discipline"] for _, _, d in top if meta[d]["discipline"])
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# --------------------------------------------------------------------------- #
# Evaluation — does any of this actually retrieve related work?
# --------------------------------------------------------------------------- #

def cited_works(db: Path) -> dict[str, frozenset]:
    """src -> set of cited works, keyed by (surname, year, title).

    Interned to ints through a dict rather than hashed with the builtin hash(),
    which is salted per process and would make the numbers unreproducible.
    """
    con = _connect(db)
    keys: dict[tuple, int] = {}
    acc: dict[str, set] = {}
    try:
        for src, s, y, t in con.execute(
                "SELECT src,surname,year,title FROM cite "
                "WHERE title<>'' AND length(title)>12"):
            k = keys.setdefault((s, y, t), len(keys))
            acc.setdefault(src, set()).add(k)
    finally:
        con.close()
    return {k: frozenset(v) for k, v in acc.items()}


def evaluate(index: Index, legacy: Index | None, meta: dict[str, dict],
             n: int, weights: list[float], top: int, cites: dict[str, frozenset],
             no_abstract: set[str], strength: float):
    """Score each variant by how much cited literature its neighbours share.

    Shared references are an independent notion of relatedness: citations.db was
    parsed from reference lists and knows nothing about any embedding. A variant
    that retrieves neighbours citing the same works is retrieving related work,
    not lexically similar noise.

    Each variant also reports what share of the neighbours it returned has no
    abstract, against the corpus base rate of 15.07%, because that is the one
    thing this index does badly and it should be visible in the same table as
    everything else rather than only in a docstring.
    """
    import numpy as np
    pool = sorted(d for d in index.ids
                  if len(cites.get(d, ())) >= 25 and d in meta)
    if not pool:
        return []
    stride = max(1, len(pool) // n)
    sample = pool[::stride][:n]

    def tag(w: float) -> str:
        return f"cal {w:.2f}" if w > 0 else "raw cosine"

    variants: list[tuple[str, Index, object, float]] = []
    if legacy is not None:
        variants.append((f"legacy title+abs[:900] {tag(strength)}", legacy,
                         legacy.title, strength))
    if index.abstract is None:
        # A single-vector index has nothing to blend; every weight is the same run.
        variants.append((f"legacy title+abs[:900] {tag(strength)}", index,
                         index.title, strength))
    else:
        variants.append((f"title only {tag(strength)}", index, index.compose(1.0),
                         strength))
        variants.append((f"abstract only {tag(strength)}", index, index.compose(0.0),
                         strength))
        for w in weights:
            variants.append((f"title {w:.2f} + abstract {1 - w:.2f} {tag(strength)}",
                             index, index.compose(w), strength))
        # One contrast row at the default weight under the opposite setting, so
        # what the calibration costs and buys is on the same table as everything
        # else rather than in a claim the reader has to take on trust.
        w = TITLE_WEIGHT if TITLE_WEIGHT in weights else weights[0]
        other = 0.0 if strength > 0 else CALIBRATION
        variants.append((f"title {w:.2f} + abstract {1 - w:.2f} {tag(other)}",
                         index, index.compose(w), other))

    def score(pairs):
        """(mean shared refs, share with >=1, share with >=3, mean Jaccard).

        The mean alone is dominated by a handful of heavily-overlapping pairs,
        so the hit rates are reported beside it as the robust statistic.
        """
        shared = [n for n, _ in pairs]
        return (float(np.mean(shared)),
                float(np.mean([n >= 1 for n, _ in pairs])),
                float(np.mean([n >= 3 for n, _ in pairs])),
                float(np.mean([j for _, j in pairs])))

    results = []
    for label, idx, matrix, w_cal in variants:
        terms = calibration_terms(matrix) if w_cal > 0 else None
        pairs, same_disc, cover, no_abs = [], 0, 0, 0
        for q in sample:
            if q not in idx.pos:
                continue
            qv = matrix[idx.pos[q]]
            hits = rank(idx, matrix, qv, meta, top, {q}, None, None, None, None,
                        -1.0, terms, w_cal)
            cq = cites.get(q, frozenset())
            for _, _, d in hits:
                cd = cites.get(d, frozenset())
                inter = len(cq & cd)
                pairs.append((inter, inter / (len(cq | cd) or 1)))
                same_disc += int(meta[d]["discipline"] == meta[q]["discipline"])
                no_abs += int(d in no_abstract)
                cover += 1
        results.append((label, *score(pairs), same_disc / max(cover, 1),
                        no_abs / max(cover, 1), len(sample)))

    # Baseline: neighbours chosen by position in the sorted id list, i.e. by
    # nothing at all. Deterministic, and no RNG in an artefact.
    pairs, no_abs = [], 0
    for qi, q in enumerate(sample):
        cq = cites.get(q, frozenset())
        for j in range(1, top + 1):
            d = pool[(qi * 7919 + j * 1327) % len(pool)]
            if d == q:
                continue
            cd = cites.get(d, frozenset())
            inter = len(cq & cd)
            pairs.append((inter, inter / (len(cq | cd) or 1)))
            no_abs += int(d in no_abstract)
    # The baseline draws from the evaluation pool rather than from the corpus, so
    # its no-abstract share is the pool's (14.5%), not the corpus's (15.1%).
    results.append(("(unrelated baseline)", *score(pairs), 0.0,
                    no_abs / max(len(pairs), 1), len(sample)))
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _esc(text: str) -> str:
    """argparse %-formats help text, so a literal percent sign has to be doubled."""
    return text.replace("%", "%%")


def add_args(p: argparse.ArgumentParser) -> None:
    q = p.add_argument_group("query (exactly one)")
    q.add_argument("--id", help="oai id, or a bare record number, of a thesis in the corpus")
    q.add_argument("--text", help="free text — an abstract or a description ('-' reads stdin)")
    q.add_argument("--title", help="a title, encoded as a title rather than as a body of text")

    f = p.add_argument_group("filters")
    f.add_argument("-k", "--k", type=int, default=10, help="neighbours to show (default: %(default)s)")
    f.add_argument("--discipline",
                   help="restrict neighbours to this discipline (labels come from "
                        "discipline.py, which the README reports as ~38%% "
                        "low-confidence)")
    f.add_argument("--cross-discipline", action="store_true",
                   help="show only neighbours OUTSIDE the query's own discipline "
                        "(as good, and as bad, as that same label)")
    f.add_argument("--year-from", type=int, help="earliest year of award")
    f.add_argument("--year-to", type=int, help="latest year of award")
    f.add_argument("--min-sim", type=float, default=-1.0,
                   help="drop neighbours below this cosine (the cosine shown, not "
                        "the calibrated score they are ranked on)")

    i = p.add_argument_group("index")
    i.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                   help="directory holding the built index (default: %(default)s)")
    i.add_argument("--legacy-index", action="store_true",
                   help=f"use {LEGACY_EMB.name} instead (no build step; weaker)")
    i.add_argument("--title-weight", type=float, default=TITLE_WEIGHT,
                   help="weight on the title vector, 0..1 (default: %(default)s)")
    i.add_argument("--calibration", type=float, default=CALIBRATION,
                   help="strength, 0..1, of the per-document score calibration "
                        "(default: %(default)s). The 3,716 documents with no "
                        "abstract — 15.07%% of the corpus — are represented by "
                        f"their title alone, and a raw cosine costs them most of "
                        f"their visibility: {_esc(NO_ABS_SHARE[False, False][0])} of "
                        f"top-10 slots against a 15.07%% base rate. Scoring each "
                        f"document against its own similarity distribution lifts "
                        f"them to {_esc(NO_ABS_SHARE[False, True][0])}, for a measured "
                        "2.8%% fall in mean shared cited works (-0.038 +- 0.014 "
                        "over 2,000 queries); a gap to the base rate remains. The "
                        "module docstring has every number")
    i.add_argument("--no-calibration", action="store_true",
                   help=f"rank on the raw cosine instead; abstract-less documents "
                        f"then take about {_esc(NO_ABS_SHARE[False, False][0])} of "
                        f"top-10 slots against a 15.07%% base rate")
    i.add_argument("--build", action="store_true", help="build or refresh the index, then exit")
    i.add_argument("--build-limit", type=int, help="encode only this many documents (a bounded test build)")
    i.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"),
                   help="encoder device; cpu is the reproducible one (default: %(default)s)")
    i.add_argument("--batch", type=int, default=128, help="encoder batch size (default: %(default)s)")

    e = p.add_argument_group("evaluation")
    e.add_argument("--evaluate", action="store_true",
                   help="score the index variants against shared cited literature, "
                        "and report what share of their neighbours has no abstract")
    e.add_argument("--evaluate-n", type=int, default=200, help="query documents to sample (default: %(default)s)")
    e.add_argument("--evaluate-top", type=int, default=5, help="neighbours per query to score (default: %(default)s)")
    e.add_argument("--evaluate-weights", default="0.15,0.25,0.35,0.50",
                   help="title weights to compare (default: %(default)s)")

    p.add_argument("--db", type=Path, default=DB, help="corpus database (default: %(default)s)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")


def _fail(msg: str) -> int:
    print(f"similar: {msg}", file=sys.stderr)
    return 1


def _cut(text: str, width: int) -> str:
    """Truncate to a column width, marking that something was cut."""
    text = text or ""
    return text if len(text) <= width else text[:width - 1].rstrip() + "\u2026"


def run(args: argparse.Namespace) -> int:
    db = Path(args.db)
    if not db.exists():
        return _fail(f"no corpus database at {db}")

    if args.build:
        meta = build_index(db, Path(args.index), args.device, args.batch,
                           args.build_limit)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if args.k <= 0:
        return _fail(f"-k must be a positive integer, not {args.k}")
    if not 0.0 <= args.title_weight <= 1.0:
        return _fail(f"--title-weight must be between 0 and 1, not {args.title_weight}")
    if not 0.0 <= args.calibration <= 1.0:
        return _fail(f"--calibration must be between 0 and 1, not {args.calibration}")
    if args.evaluate and (args.evaluate_n <= 0 or args.evaluate_top <= 0):
        return _fail("--evaluate-n and --evaluate-top must be positive integers")

    meta = load_meta(db)
    no_abstract = {d for d, m in meta.items() if not m["has_abstract"]}
    index_dir = Path(args.index)
    legacy = args.legacy_index
    if not legacy and not (index_dir / "meta.json").exists():
        print(f"similar: no built index at {index_dir}; falling back to "
              f"{LEGACY_EMB.name}. Build the stronger one with "
              f"`tg similar --build`.", file=sys.stderr)
        legacy = True
    try:
        index = load_index(index_dir, legacy)
    except (OSError, ValueError, KeyError) as exc:      # missing, truncated or corrupt
        return _fail(f"could not read the index: {exc}")
    if index.meta.get("partial"):
        print(f"similar: index is partial — {index.meta['docs']:,} of the corpus "
              f"only (built with --build-limit)", file=sys.stderr)
    built_from = index.meta.get("corpus_signature")
    if built_from and built_from != signature(db):
        print(f"similar: index was built from a different corpus "
              f"({built_from}); rebuild with `tg similar --build`", file=sys.stderr)

    if args.evaluate:
        if not CITE_DB.exists():
            return _fail(f"no citation database at {CITE_DB}")
        print("loading citations ...", file=sys.stderr)
        cites = cited_works(CITE_DB)
        other = None
        if not legacy and LEGACY_EMB.exists():
            other = load_index(index_dir, True)
        try:
            weights = sorted({float(x) for x in args.evaluate_weights.split(",") if x.strip()})
        except ValueError:
            return _fail("--evaluate-weights takes a comma-separated list of numbers")
        strength = 0.0 if args.no_calibration else args.calibration
        rows = evaluate(index, other, meta, args.evaluate_n, weights,
                        args.evaluate_top, cites, no_abstract, strength)
        base = len(no_abstract) / max(len(meta), 1)
        if args.json:
            print(json.dumps([{"variant": r[0], "mean_shared_refs": round(r[1], 3),
                               "share_with_1_plus": round(r[2], 4),
                               "share_with_3_plus": round(r[3], 4),
                               "mean_jaccard": round(r[4], 5),
                               "same_discipline_rate": round(r[5], 4),
                               "no_abstract_share": round(r[6], 4),
                               "queries": r[7]} for r in rows]
                             + [{"variant": "(corpus base rate)",
                                 "no_abstract_share": round(base, 4)}],
                             indent=2, sort_keys=True))
            return 0
        print(f"shared cited works among the top {args.evaluate_top} neighbours "
              f"of {rows[0][7]} sampled theses")
        print(f"no-abstract = share of those neighbours with no abstract; "
              f"{len(no_abstract):,} of {len(meta):,} documents ({base:.2%}) have none")
        print(f"{'variant':<40} {'mean':>7} {'>=1 ref':>8} {'>=3 refs':>9} "
              f"{'jaccard':>8} {'same disc':>10} {'no-abstract':>12}")
        for label, mean, h1, h3, jac, sd, na, _ in rows:
            print(f"{label:<40} {mean:>7.2f} {h1:>7.1%} {h3:>8.1%} "
                  f"{jac:>8.4f} {sd:>9.1%} {na:>11.1%}")
        return 0

    # `is not None`, not truthiness: --text "" is a query that was given and is
    # empty, which is a different mistake from not giving one at all.
    given = [(flag, v) for flag, v in
             (("--id", args.id), ("--text", args.text), ("--title", args.title))
             if v is not None]
    if len(given) != 1:
        return _fail("give exactly one of --id, --text or --title")
    flag, value = given[0]
    if not value.strip():
        return _fail(f"{flag} was given an empty value")

    w = args.title_weight
    matrix = index.compose(w)
    strength = 0.0 if args.no_calibration else args.calibration
    terms = calibration_terms(matrix) if strength > 0 else None

    exclude: set[str] = set()
    home: str | None = None
    qmeta: dict | None = None
    if flag == "--id":
        try:
            doc_id = resolve_id(meta, value)
        except ValueError as exc:
            return _fail(str(exc))
        if doc_id not in index.pos:
            return _fail(f"{doc_id} is not in this index "
                         f"({'partial index' if index.meta.get('partial') else 'unexpected'})")
        qvec = matrix[index.pos[doc_id]]
        exclude.add(doc_id)
        qmeta = meta[doc_id]
        home = qmeta["discipline"] or None
        qlabel = f"{doc_id}"
    else:
        text = sys.stdin.read() if value == "-" else value
        text = " ".join(text.split())
        if not text:
            return _fail("the query text is empty")
        try:
            model, _ = get_model(args.device)
        except Exception as exc:                        # noqa: BLE001 — one line, not a stack
            return _fail(f"could not load the encoder {EMBED_MODEL} on "
                         f"{args.device}: {exc}")
        if flag == "--title":
            qvec = encode(model, [text], args.batch)[0]
        else:
            qvec = encode_long(model, text, args.batch)
            if qvec is None:
                return _fail("the query text produced no embedding")
        qlabel = f"{flag} {_cut(text, 60)}"

    want_disc = None
    if args.discipline:
        try:
            want_disc = resolve_discipline(meta, args.discipline)
        except ValueError as exc:
            return _fail(str(exc))

    not_disc = None
    if args.cross_discipline:
        if home is None:
            home = infer_home(index, matrix, qvec, meta, terms, strength)
            if home is None:
                return _fail("--cross-discipline needs a home discipline and none "
                             "could be inferred")
        not_disc = home
        if want_disc and want_disc == not_disc:
            return _fail(f"--discipline {want_disc!r} is the query's own discipline, "
                         f"so --cross-discipline leaves nothing to show")

    hits = rank(index, matrix, qvec, meta, args.k, exclude, want_disc, not_disc,
                args.year_from, args.year_to, args.min_sim, terms, strength)

    n_no_abs = sum(1 for _, _, d in hits if d in no_abstract)
    payload = {
        "query": {"kind": flag.lstrip("-"),
                  "value": qmeta["id"] if qmeta else qlabel,
                  "title": qmeta["title"] if qmeta else None,
                  "creator": qmeta["creator"] if qmeta else None,
                  "year": qmeta["year"] if qmeta else None,
                  "discipline": qmeta["discipline"] if qmeta else None},
        "index": {"path": index.label, "docs": index.meta["docs"],
                  "model": index.meta.get("model"),
                  "title_weight": w if index.abstract is not None else None,
                  "calibration": strength,
                  "partial": bool(index.meta.get("partial"))},
        "no_abstract": {"in_corpus": len(no_abstract),
                        "corpus_share": round(len(no_abstract) / max(len(meta), 1), 4),
                        "in_results": n_no_abs,
                        "note": "documents with no abstract are represented by "
                                "their title alone and are under-represented in "
                                "any ranking here; see --help"},
        "filters": {"k": args.k, "discipline": want_disc,
                    "outside_discipline": not_disc,
                    "year_from": args.year_from, "year_to": args.year_to,
                    "min_sim": args.min_sim if args.min_sim > -1.0 else None},
        "results": [{"rank": n + 1, "id": d, "similarity": s,
                     "ranking_score": adj, "has_abstract": meta[d]["has_abstract"],
                     "title": meta[d]["title"], "creator": meta[d]["creator"],
                     "year": meta[d]["year"], "discipline": meta[d]["discipline"],
                     "subfield": meta[d]["subfield"], "url": meta[d]["url"]}
                    for n, (adj, s, d) in enumerate(hits)],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if qmeta:
        print(f"query   {qmeta['id']}")
        print(f"        {qmeta['title']}")
        print(f"        {qmeta['creator']}, {qmeta['year']} — {qmeta['discipline']}")
    else:
        print(f"query   {qlabel}")
        if not_disc:
            print(f"        home discipline inferred as {not_disc}")
    weight_note = f"title weight {w:.2f}" if index.abstract is not None \
        else "one blended vector, no title weight"
    print(f"index   {index.label}, {index.meta['docs']:,} docs, {weight_note}")
    # Disclosed on every run, not only in --help: a reader cannot see from a
    # ranked list that a sixth of the corpus is competing at a disadvantage.
    base = len(no_abstract) / max(len(meta), 1)
    if strength > 0:
        print(f"scoring sim = cosine, ranked on adj = cosine calibrated per "
              f"document (strength {strength:.2f})")
    else:
        print("scoring sim = cosine, ranked on the raw cosine (--no-calibration)")
    share, verdict = NO_ABS_SHARE[legacy, strength > 0]
    print(textwrap.fill(
        f"{len(no_abstract):,} of {len(meta):,} documents ({base:.1%}) have no "
        f"abstract and are matched on title alone: they take about {share} of "
        f"top-10 slots on this index, {verdict} (see --help)",
        width=86, initial_indent=" " * 8, subsequent_indent=" " * 8))
    shown = [f"k={args.k}"]
    if want_disc:
        shown.append(f"in {want_disc}")
    if not_disc:
        shown.append(f"outside {not_disc}")
    if args.year_from or args.year_to:
        shown.append(f"years {args.year_from or ''}-{args.year_to or ''}")
    print(f"filter  {', '.join(shown)}")
    print()
    if not hits:
        print("  no neighbours matched the filters")
        return 0
    for n, (adj, s, d) in enumerate(hits, 1):
        m = meta[d]
        mark = "" if m["has_abstract"] else "  [no abstract]"
        if strength > 0:
            print(f"{n:>3}  {s:6.3f} {adj:>6.2f}  {m['year'] or '????'}  "
                  f"{_cut(m['discipline'], 22):<22} {_cut(m['title'], 64)}")
            print(f"{'':>21}{_cut(m['creator'], 40):<40} {d}{mark}")
        else:
            print(f"{n:>3}  {s:6.3f}  {m['year'] or '????'}  "
                  f"{_cut(m['discipline'], 22):<22} {_cut(m['title'], 71)}")
            print(f"{'':>14}{_cut(m['creator'], 40):<40} {d}{mark}")
    if n_no_abs:
        print(f"\n{n_no_abs} of the {len(hits)} shown "
              f"{'has' if n_no_abs == 1 else 'have'} no abstract in the corpus and "
              f"{'was' if n_no_abs == 1 else 'were'} matched on title alone")
    return 0


if __name__ == "__main__":
    # tg.py catches these for the plugin path; run standalone, nothing does, and
    # `tg_similar.py ... | head` or a ^C should not end in a stack trace.
    ap = argparse.ArgumentParser(description=HELP)
    add_args(ap)
    try:
        code = run(ap.parse_args())
        sys.stdout.flush()          # so a closed pipe is caught here, not at exit
    except KeyboardInterrupt:
        code = 130
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 141
    raise SystemExit(code)
