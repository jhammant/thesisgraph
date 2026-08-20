#!/usr/bin/env python3
"""
discourse.py — Study 4: writing and discourse features across the corpus,
plus the cross-field method-diffusion analysis.

    discourse   per-thesis writing features -> CSV + trends by period/discipline
    diffusion   when each research method crossed into each discipline

Features are computed on RETAINED text only (references, attributed block
quotes, boilerplate and captions already excluded by the extraction pipeline),
so they describe the author's own prose rather than their bibliography.
"""
from __future__ import annotations
import argparse, csv, gzip, json, multiprocessing as mp, os, re, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE / "corpus" / "studies"

HEDGE = re.compile(r"\b(may|might|could|perhaps|possibly|probably|seem(?:s|ed)?|"
                   r"appear(?:s|ed)?|suggest(?:s|ed)?|indicate(?:s|d)?|tend(?:s|ed)? to|"
                   r"relatively|somewhat|arguably|apparently|presumably|"
                   r"largely|generally|typically|often|likely|unlikely|"
                   r"assum(?:e|es|ed)|imply|implies|implied)\b", re.I)
BOOST = re.compile(r"\b(clearly|obviously|certainly|definitely|undoubtedly|"
                   r"demonstrat(?:e|es|ed)|prove(?:s|d|n)?|establish(?:es|ed)|"
                   r"must|always|never|conclusively|indisputabl\w+|evidently)\b", re.I)
FIRST_SG = re.compile(r"\b(I|my|mine|myself)\b")
FIRST_PL = re.compile(r"\b(we|our|ours|ourselves)\b", re.I)
IMPERSONAL = re.compile(r"\b(it (?:is|was) (?:argued|found|shown|noted|observed|"
                        r"suggested|considered|assumed|believed|thought)|"
                        r"this (?:thesis|study|chapter|research) (?:argues|shows|"
                        r"demonstrates|examines|explores|presents))\b", re.I)
PASSIVE = re.compile(r"\b(?:is|are|was|were|been|being|be)\s+(?:\w+ly\s+)?"
                     r"(\w+(?:ed|en))\b", re.I)
NOMINAL = re.compile(r"\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence|ism)\b", re.I)
CITATION = re.compile(r"\([^)]{0,60}\b(?:19|20)\d{2}[a-z]?\b[^)]{0,30}\)|"
                      r"\b[A-Z][A-Za-z'\-]+\s*\(\s*(?:19|20)\d{2}")
QUESTION = re.compile(r"\?")
VOWELS = re.compile(r"[aeiouy]+")

STOP = frozenset("""the of and to in a is that for it as was with be by on not he
i this are or his from at which but have an they you had were their one all we
her she there would when so been if no more will there its who what which them
than then these into some can only other new also do does did such our may""".split())


def syllables(w: str) -> int:
    return max(1, len(VOWELS.findall(w)))


def features(doc_id: str) -> dict | None:
    try:
        with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
    except Exception:
        return None
    n_sent = n_words = n_syll = n_long = n_stop = 0
    hedge = boost = fsg = fpl = imp = passv = nom = cit = ques = 0
    per_bucket: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        text, pg, pr, bucket, heading, excl = H.unpack_sentence(r)
        if excl:
            continue
        toks = R.match_tokens(text)
        if not toks:
            continue
        n_sent += 1
        n_words += len(toks)
        for t in toks:
            n_syll += syllables(t)
            if len(t) >= 7:
                n_long += 1
            if t in STOP:
                n_stop += 1
        hedge += len(HEDGE.findall(text))
        boost += len(BOOST.findall(text))
        fsg += len(FIRST_SG.findall(text))
        fpl += len(FIRST_PL.findall(text))
        imp += len(IMPERSONAL.findall(text))
        passv += len(PASSIVE.findall(text))
        nom += len(NOMINAL.findall(text))
        cit += len(CITATION.findall(text))
        ques += len(QUESTION.findall(text))
        b = per_bucket[bucket]
        b[0] += 1
        b[1] += len(toks)
    if n_words < 2000:
        return None
    per1k = lambda x: 1000.0 * x / n_words
    return {
        "id": doc_id, "sentences": n_sent, "words": n_words,
        "mean_sentence_len": round(n_words / max(n_sent, 1), 2),
        "long_word_pct": round(100.0 * n_long / n_words, 2),
        "lexical_density": round(100.0 * (1 - n_stop / n_words), 2),
        # Flesch reading ease; lower = harder. Academic prose sits ~20-40.
        "flesch": round(206.835 - 1.015 * (n_words / max(n_sent, 1))
                        - 84.6 * (n_syll / n_words), 1),
        "hedges_per1k": round(per1k(hedge), 2),
        "boosters_per1k": round(per1k(boost), 2),
        "first_sg_per1k": round(per1k(fsg), 2),
        "first_pl_per1k": round(per1k(fpl), 2),
        "impersonal_per1k": round(per1k(imp), 2),
        "passive_per1k": round(per1k(passv), 2),
        "nominal_per1k": round(per1k(nom), 2),
        "citations_per1k": round(per1k(cit), 2),
        "questions_per1k": round(per1k(ques), 2),
        "hedge_boost_ratio": round(hedge / max(boost, 1), 2),
    }


def run_discourse(limit, workers):
    con = H.db()
    rows = con.execute("SELECT id,year,discipline,publisher FROM doc "
                       "WHERE status='ok' ORDER BY id").fetchall()
    if limit:
        rows = rows[:limit]
    meta = {r[0]: r for r in rows}
    ids = [r[0] for r in rows]
    print(f"  {len(ids):,} theses", file=sys.stderr)
    t0, out = time.monotonic(), []
    with mp.Pool(workers) as pool:
        for k, f in enumerate(pool.imap(features, ids, chunksize=16), 1):
            if f:
                out.append(f)
            if k % 4000 == 0:
                print(f"    {k:,}/{len(ids):,}  {(time.monotonic()-t0)/60:.1f} min",
                      file=sys.stderr)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted(out[0].keys())
    with (STUDY_DIR / "discourse_features.csv").open("w", newline="",
                                                     encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "year", "discipline", "publisher"] +
                   [k for k in keys if k != "id"])
        for f in sorted(out, key=lambda x: x["id"]):
            m = meta[f["id"]]
            w.writerow([f["id"], m[1], m[2], m[3]] +
                       [f[k] for k in keys if k != "id"])

    TRACK = ["mean_sentence_len", "flesch", "lexical_density", "first_sg_per1k",
             "first_pl_per1k", "passive_per1k", "hedges_per1k", "boosters_per1k",
             "nominal_per1k", "citations_per1k"]
    print()
    print("=" * 108)
    print(f"STUDY 4 — WRITING AND DISCOURSE   ({len(out):,} theses)")
    print("=" * 108)
    by = defaultdict(list)
    for f in out:
        y = meta[f["id"]][1]
        if y and 1960 <= y <= 2030:
            by[(y // 5) * 5].append(f)
    print(f"\n  BY PERIOD (mean per thesis)")
    print("    " + f"{'period':<11}{'n':>6}" + "".join(f"{t[:13]:>14}" for t in TRACK))
    print("    " + "-" * (17 + 14 * len(TRACK)))
    for per in sorted(by):
        fs = by[per]
        if len(fs) < 20:
            continue
        line = f"    {per}-{per+4:<6}{len(fs):>6}"
        for t in TRACK:
            line += f"{sum(x[t] for x in fs)/len(fs):>14.1f}"
        print(line)
    bd = defaultdict(list)
    for f in out:
        d = meta[f["id"]][2]
        if d:
            bd[d].append(f)
    print(f"\n  BY DISCIPLINE (mean per thesis)")
    print("    " + f"{'discipline':<30}{'n':>6}" +
          "".join(f"{t[:13]:>14}" for t in TRACK))
    print("    " + "-" * (36 + 14 * len(TRACK)))
    for d in sorted(bd, key=lambda x: -len(bd[x])):
        fs = bd[d]
        line = f"    {d:<30}{len(fs):>6}"
        for t in TRACK:
            line += f"{sum(x[t] for x in fs)/len(fs):>14.1f}"
        print(line)
    print(f"\n  per-thesis features -> {STUDY_DIR/'discourse_features.csv'}")
    con.close()


def run_diffusion():
    """When did each method cross into each discipline? Cross-field part 1."""
    con = H.db()
    meta = {r[0]: r for r in con.execute(
        "SELECT id,year,discipline FROM doc WHERE status='ok'")}
    path = STUDY_DIR / "method_features.csv"
    if not path.exists():
        print("run `corpus_studies.py method` first"); return
    METHODS = ["thematic analysis", "grounded theory", "IPA", "discourse analysis",
               "mixed methods", "ethnography", "case study", "systematic review",
               "meta-analysis", "action research", "phenomenology",
               "content analysis", "narrative inquiry"]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    tot = defaultdict(int)
    hit = defaultdict(int)
    first = {}
    for r in rows:
        m = meta.get(r["id"])
        if not m or not m[1] or not m[2] or not (1960 <= int(m[1]) <= 2030):
            continue
        y, d = int(m[1]), m[2]
        tot[d] += 1
        for meth in METHODS:
            k = f"design::{meth}"
            if r.get(k) and int(r[k]) > 0:
                hit[(d, meth)] += 1
                cur = first.get((d, meth))
                if cur is None or y < cur:
                    first[(d, meth)] = y
    discs = sorted(tot, key=lambda d: -tot[d])
    print()
    print("=" * 112)
    print("CROSS-FIELD: METHOD DIFFUSION — prevalence by discipline "
          "(first year observed in brackets)")
    print("=" * 112)
    print(f"  {'discipline':<28}{'n':>6}" +
          "".join(f"{m[:11]:>13}" for m in METHODS[:7]))
    print("  " + "-" * (34 + 13 * 7))
    for d in discs:
        if tot[d] < 100:
            continue
        line = f"  {d:<28}{tot[d]:>6}"
        for m in METHODS[:7]:
            pc = 100.0 * hit[(d, m)] / tot[d]
            line += f"{pc:>8.1f}% " if pc >= 0.05 else f"{'—':>9} "
        print(line)
    print()
    print("  EARLIEST APPEARANCE per discipline (thematic analysis / mixed methods)")
    for m in ("thematic analysis", "mixed methods", "grounded theory"):
        yrs = sorted(((first[(d, m)], d) for d in discs if (d, m) in first))
        if yrs:
            print(f"    {m:<20}" + "  ".join(f"{d[:11]}:{y}" for y, d in yrs[:8]))
    con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discourse")
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 6))
    sub.add_parser("diffusion")
    a = ap.parse_args(argv)
    if a.cmd == "discourse":
        run_discourse(a.limit, a.workers)
    else:
        run_diffusion()
    return 0


if __name__ == "__main__":
    sys.exit(main())
