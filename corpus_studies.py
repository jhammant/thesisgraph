#!/usr/bin/env python3
"""
corpus_studies.py — O(N) studies over the thesis corpus built by harvest_corpus.py.

Unlike the overlap screen (O(N^2), happy with a few hundred documents), every
study here is linear in the corpus and wants as many theses as we can get.

    method     Study 3 — method and reporting practice
    validate   Run the method extractor over the four theses already fetched by
               run_analysis.py, whose contents are known, so the extractor can
               be checked by eye before it is trusted on thousands.

Paths anchor to this file's location, not the shell's working directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE / "corpus" / "studies"


# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #
# Each entry is (label, pattern, scope). Scope decides where a hit counts:
#   "method"  — methodology chapters only. Used for research designs, because
#               "case study" and "ethnography" are discussed constantly in
#               literature reviews without being what the author actually did.
#   "any"     — anywhere in the retained text. Used for things that are stated
#               once, wherever the author happened to put them (ethics, software).

FEATURES: dict[str, list[tuple[str, str, str]]] = {
    "design": [
        ("thematic analysis", r"\bthematic analys[ie]s\b", "method"),
        ("grounded theory", r"\bgrounded theory\b", "method"),
        ("IPA", r"\binterpretat(?:ive|ional) phenomenological analysis\b|\bIPA\b", "method"),
        ("phenomenology", r"\bphenomenolog(?:y|ical)\b", "method"),
        ("discourse analysis", r"\b(?:critical )?discourse analys[ie]s\b", "method"),
        ("content analysis", r"\bcontent analys[ie]s\b", "method"),
        ("narrative inquiry", r"\bnarrative (?:inquiry|enquiry|analysis)\b", "method"),
        ("case study", r"\bcase stud(?:y|ies)\b", "method"),
        ("ethnography", r"\b(?:auto)?ethnograph(?:y|ic)\b", "method"),
        ("action research", r"\baction research\b", "method"),
        ("framework analysis", r"\bframework analys[ie]s\b", "method"),
        ("conversation analysis", r"\bconversation analys[ie]s\b", "method"),
        ("mixed methods", r"\bmixed[- ]methods?\b", "method"),
        ("RCT", r"\brandomi[sz]ed controlled trial\b|\bRCT\b", "method"),
        ("quasi-experimental", r"\bquasi[- ]experimental\b", "method"),
        ("experiment", r"\bexperimental design\b", "method"),
        ("survey", r"\bsurvey\b", "method"),
        ("systematic review", r"\bsystematic review\b", "method"),
        ("meta-analysis", r"\bmeta[- ]analys[ie]s\b", "method"),
        ("Delphi", r"\bDelphi (?:method|technique|study)\b", "method"),
        ("Q-methodology", r"\bQ[- ]methodolog(?:y|ical)\b|\bQ[- ]sort\b", "method"),
    ],
    "collection": [
        ("semi-structured interview", r"\bsemi[- ]structured interview", "method"),
        ("structured interview", r"\bstructured interview", "method"),
        ("unstructured interview", r"\bunstructured interview", "method"),
        ("focus group", r"\bfocus group", "method"),
        ("questionnaire", r"\bquestionnaire", "method"),
        ("observation", r"\b(?:classroom |participant |non[- ]participant )?observation",
         "method"),
        ("document analysis", r"\bdocument(?:ary)? analys[ie]s\b", "method"),
        ("diary", r"\b(?:diary|diaries|journal(?:ling|ing))\b", "method"),
        ("think-aloud", r"\bthink[- ]aloud\b", "method"),
        ("video", r"\bvideo[- ](?:record|data|analysis)", "method"),
    ],
    "sampling": [
        ("purposive", r"\bpurposive sampl", "method"),
        ("convenience", r"\bconvenience sampl", "method"),
        ("snowball", r"\bsnowball sampl", "method"),
        ("random sampling", r"\brandom(?:ly)? (?:sampl|select)", "method"),
        ("theoretical sampling", r"\btheoretical sampl", "method"),
        ("stratified", r"\bstratified sampl", "method"),
    ],
    "rigour": [
        ("ethical approval", r"\bethic(?:s|al) (?:approval|clearance|committee)\b|"
                             r"\bapproved by the .{0,40}ethics\b", "any"),
        ("informed consent", r"\binformed consent\b", "any"),
        ("anonymity", r"\banonymi[sz](?:ed|ation)\b|\bpseudonym", "any"),
        ("triangulation", r"\btriangulat(?:ion|ed)\b", "any"),
        ("member checking", r"\bmember[- ]check", "any"),
        ("saturation", r"\b(?:data |theoretical )?saturation\b", "any"),
        ("inter-rater reliability", r"\binter[- ]?(?:rater|coder) reliabilit", "any"),
        ("Cohen's kappa", r"\b(?:cohen'?s )?kappa\b", "any"),
        ("pilot study", r"\bpilot (?:study|stud|test|interview)", "any"),
        ("positionality", r"\bpositionalit(?:y|ies)\b", "any"),
        ("reflexivity", r"\breflexiv(?:e|ity)\b", "any"),
        ("trustworthiness", r"\btrustworthiness\b", "any"),
        ("credibility framework", r"\b(?:credibility|transferability|dependability|"
                                  r"confirmability)\b", "any"),
        ("limitations section", r"\blimitations? of (?:the|this) (?:study|research)\b",
         "any"),
    ],
    "software": [
        ("NVivo", r"\bNVivo\b", "any"),
        ("SPSS", r"\bSPSS\b", "any"),
        ("R", r"\bR\s*(?:statistical|software|version|studio)\b|\bRStudio\b|"
              r"\bin R\b(?!\w)", "any"),
        ("Stata", r"\bStata\b", "any"),
        ("MAXQDA", r"\bMAXQDA\b", "any"),
        ("ATLAS.ti", r"\bATLAS\.?ti\b", "any"),
        ("Excel", r"\b(?:Microsoft )?Excel\b", "any"),
        ("Mplus", r"\bMplus\b", "any"),
        ("AMOS", r"\bAMOS\b", "any"),
        ("Dedoose", r"\bDedoose\b", "any"),
        ("Python", r"\bPython\b", "any"),
        ("MATLAB", r"\bMATLAB\b", "any"),
    ],
    "open_science": [
        ("pre-registered", r"\bpre[- ]?registrat|pre[- ]?registered\b", "any"),
        ("data availability", r"\bdata (?:are|is) available\b|\bopenly available\b|"
                              r"\bdata availability\b", "any"),
        ("replication", r"\breplicat(?:ion|e) (?:study|of the)", "any"),
    ],
}

# A design is "declared" (rather than merely discussed) when it sits next to
# language claiming it as the author's own choice.
DECLARED_RE = re.compile(
    r"\b(?:this (?:study|research|thesis|chapter|investigation)|the present study|"
    r"the current study|I|we|the researcher)\b[^.]{0,80}\b"
    r"(?:use[sd]?|using|adopt(?:s|ed|ing)?|employ(?:s|ed|ing)?|appl(?:y|ies|ied)|"
    r"draw[s]? on|drew on|utilis(?:e|ed|ing)|chose|chosen|select(?:ed)?|"
    r"follow(?:s|ed)?|take[sn]? a|took a|is based on|was conducted)\b"
    r"|\b(?:was|were|is|are) (?:used|adopted|employed|chosen|selected|applied|"
    r"conducted|undertaken)\b")

SAMPLE_RE = re.compile(
    r"\bn\s*=\s*(\d{1,5})\b"
    r"|\b(\d{1,5})\s+(?:participants?|respondents?|students?|pupils?|teachers?|"
    r"children|interviewees?|informants?|subjects?|volunteers?|cases?)\b"
    r"|\bsample of\s+(\d{1,5})\b", re.IGNORECASE)

METHOD_BUCKETS = {"methodology"}

# Chapter bucketing is derived from heading detection, and heading detection
# fails on some documents (ALHU2025 in the validation set yields 6 methodology
# sentences out of 2,854 — its headings were not recovered). Two defences:
#   1. a sentence also counts as method-scope if its OWN heading looks
#      methodological, which does not depend on the chapter-level state;
#   2. if a document still has almost no method text, fall back to the whole
#      document and FLAG it, so degraded rows can be excluded from the design
#      statistics rather than silently reported as zeros.
METHOD_HEADING_RE = re.compile(
    r"\b(methodolog|method(?:s)?\b|research design|research approach|"
    r"research strategy|data collection|data analysis|procedure|participants|"
    r"sampling|instrument|fieldwork|ethic)", re.IGNORECASE)

MIN_METHOD_SENTENCES = 25

_COMPILED = {cat: [(lab, re.compile(pat, re.IGNORECASE), scope)
                   for lab, pat, scope in items]
             for cat, items in FEATURES.items()}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_features(sentences: list[tuple]) -> dict:
    """sentences: list of (text, page, printed, bucket, heading, excl)."""
    any_txt, method_sents = [], []
    for text, page, printed, bucket, heading, excl in sentences:
        if excl:                                  # refs / quotes / boilerplate out
            continue
        any_txt.append(text)
        if bucket in METHOD_BUCKETS or (heading and METHOD_HEADING_RE.search(heading)):
            method_sents.append(text)
    scope = "chapter"
    if len(method_sents) < MIN_METHOD_SENTENCES:
        # Heading detection failed on this document. Use the whole text and say so.
        scope = "fallback_wholedoc"
        method_sents = list(any_txt)
    blob_any = "\n".join(any_txt)
    blob_method = "\n".join(method_sents)

    out: dict = {"method_sentences": len(method_sents),
                 "retained_sentences": len(any_txt),
                 "method_scope": scope}
    for cat, items in _COMPILED.items():
        for label, rx, scope in items:
            blob = blob_method if scope == "method" else blob_any
            n = len(rx.findall(blob)) if blob else 0
            out[f"{cat}::{label}"] = n
            if scope == "method":
                declared = 0
                for s in method_sents:
                    if rx.search(s) and DECLARED_RE.search(s):
                        declared = 1
                        break
                out[f"{cat}::{label}::declared"] = declared

    sizes = []
    for s in method_sents:
        for m in SAMPLE_RE.finditer(s):
            v = next((g for g in m.groups() if g), None)
            if v and 1 <= int(v) <= 100000:
                sizes.append(int(v))
    out["sample_sizes_found"] = len(sizes)
    out["sample_size_max"] = max(sizes) if sizes else ""
    out["sample_size_median"] = sorted(sizes)[len(sizes) // 2] if sizes else ""
    return out


def _load_corpus_doc(doc_id: str) -> list[tuple]:
    with gzip.open(H.store_path(doc_id), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [H.unpack_sentence(r) for r in payload["sentences"]]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def validate() -> int:
    """Run the extractor over the four theses whose contents we already know."""
    info = R.ensure_pdfs(R.REAL_DOCS, True, R.PDF_DIR)
    print("Validation — the four theses already fetched by run_analysis.py")
    print("=" * 78)
    rows = []
    for spec in R.REAL_DOCS:
        pages = R.load_or_extract(spec, Path(info[spec.key]["path"]),
                                  info[spec.key]["sha256"])
        doc = R.build_document(spec, pages)
        sents = [(s.text, s.pdf_page_start, s.printed_page or "", s.bucket,
                  s.heading or "", ";".join(s.exclusions)) for s in doc.sentences]
        f = extract_features(sents)
        rows.append((spec.short, f))
        print(f"\n{spec.short}  ({spec.label[:58]})")
        print(f"  methodology sentences: {f['method_sentences']:,} "
              f"of {f['retained_sentences']:,} retained   "
              f"[scope: {f['method_scope']}]")
        for cat in ("design", "collection", "sampling"):
            hits = sorted(
                ((lab.split('::')[1], n,
                  f.get(f"{lab}::declared", 0))
                 for lab, n in f.items()
                 if lab.startswith(cat + "::") and not lab.endswith("::declared") and n),
                key=lambda t: -t[1])[:6]
            if hits:
                print(f"  {cat:<11}" + ", ".join(
                    f"{h[0]}×{h[1]}{'*' if h[2] else ''}" for h in hits))
        for cat in ("rigour", "software", "open_science"):
            hits = sorted(((lab.split('::')[1], n) for lab, n in f.items()
                           if lab.startswith(cat + "::") and n), key=lambda t: -t[1])[:8]
            if hits:
                print(f"  {cat:<11}" + ", ".join(f"{h[0]}×{h[1]}" for h in hits))
        print(f"  sample sizes: {f['sample_sizes_found']} found, "
              f"median {f['sample_size_median']}, max {f['sample_size_max']}")
    print("\n  (* = 'declared', i.e. found alongside language claiming it as the")
    print("   author's own choice rather than merely discussing it)")
    print("\nCheck these by eye against what you know of the theses before")
    print("trusting the extractor on thousands of documents.")
    return 0


def method_study(limit: int | None) -> int:
    con = H.db()
    rows = con.execute(
        "SELECT id,title,creator,year,publisher,is_education FROM doc "
        "WHERE status='ok' ORDER BY id").fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("Corpus is empty — run `harvest_corpus.py fetch` first.")
        print("Meanwhile, `corpus_studies.py validate` exercises the extractor")
        print("on the four theses already downloaded by run_analysis.py.")
        return 1
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = STUDY_DIR / "method_features.csv"
    feats, meta = [], []
    for i, (doc_id, title, creator, year, publisher, is_edu) in enumerate(rows, 1):
        try:
            f = extract_features(_load_corpus_doc(doc_id))
        except Exception as e:
            print(f"  skip {doc_id}: {e}", file=sys.stderr)
            continue
        feats.append(f)
        meta.append({"id": doc_id, "title": title, "creator": creator,
                     "year": year, "publisher": publisher, "education": is_edu})
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)

    keys = sorted({k for f in feats for k in f})
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "title", "creator", "year", "publisher", "education"] + keys)
        for m, f in sorted(zip(meta, feats), key=lambda t: t[0]["id"]):
            w.writerow([m["id"], m["title"], m["creator"], m["year"], m["publisher"],
                        m["education"]] + [f.get(k, "") for k in keys])

    n = len(feats)
    print()
    print("=" * 78)
    print(f"STUDY 3 — METHOD AND REPORTING PRACTICE   ({n:,} theses)")
    print("=" * 78)
    for cat in ("design", "collection", "sampling", "rigour", "software",
                "open_science"):
        labels = [lab for lab, _, _ in FEATURES[cat]]
        print(f"\n  {cat.upper()}")
        print(f"    {'feature':<28}{'theses':>8}{'prevalence':>12}{'declared':>10}")
        print("    " + "-" * 58)
        stats = []
        for lab in labels:
            k = f"{cat}::{lab}"
            used = sum(1 for f in feats if f.get(k, 0) > 0)
            dec = sum(1 for f in feats if f.get(k + "::declared", 0) > 0)
            stats.append((used, lab, dec))
        for used, lab, dec in sorted(stats, reverse=True):
            if not used:
                continue
            print(f"    {lab:<28}{used:>8}{100*used/n:>11.1f}%"
                  + (f"{100*dec/n:>9.1f}%" if dec else f"{'—':>10}"))

    years = defaultdict(list)
    for m, f in zip(meta, feats):
        if m["year"]:
            years[(m["year"] // 5) * 5].append(f)
    if len(years) > 1:
        track = [("design", "thematic analysis"), ("design", "grounded theory"),
                 ("design", "mixed methods"), ("rigour", "positionality"),
                 ("rigour", "ethical approval"), ("software", "NVivo")]
        print("\n  TRENDS (share of theses, by 5-year period)")
        hdr = "    " + f"{'period':<10}{'n':>6}"
        for _, lab in track:
            hdr += f"{lab[:13]:>15}"
        print(hdr)
        print("    " + "-" * (16 + 15 * len(track)))
        for per in sorted(years):
            fs = years[per]
            line = f"    {per}-{per+4:<5}{len(fs):>6}"
            for cat, lab in track:
                k = f"{cat}::{lab}"
                c = sum(1 for f in fs if f.get(k, 0) > 0)
                line += f"{100*c/len(fs):>14.0f}%"
            print(line)
    print(f"\n  per-thesis features written to {out_csv}")
    con.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("method", help="study 3 — method and reporting practice")
    m.add_argument("--limit", type=int, default=None)
    sub.add_parser("validate", help="run the extractor on the four known theses")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        return validate()
    return method_study(a.limit)


if __name__ == "__main__":
    sys.exit(main())
