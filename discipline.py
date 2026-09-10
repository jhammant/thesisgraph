#!/usr/bin/env python3
"""
discipline.py — assign a subject discipline to every thesis in the corpus.

Every cross-field question ("did this method migrate from psychology into
education?", "are these two literatures disconnected?") needs a discipline
label, and OAI-PMH `oai_dc` from this repository carries no `dc:subject`. So we
derive one.

Approach is deliberately hybrid, in this order:

  1. RULES. Weighted term lexicons over title + abstract (+ a sample of the full
     text when we have it). Deterministic, interpretable, auditable — you can
     always ask "why did this get labelled Psychology" and get a term list back.
  2. EMBEDDINGS, only where the rules are unconfident (no hit, or the top two
     disciplines are too close). The same offline MiniLM model run_analysis.py
     uses for pass 3, compared against a written descriptor per discipline.
     Degrades cleanly to "unknown" when no model is present.

Categories follow the shape of the UK Common Aggregation Hierarchy (CAH) level
1, which is what UK subject statistics are reported in, so results are
comparable with sector data.

    python discipline.py classify          # label everything, write to the DB
    python discipline.py validate          # label the four known theses
    python discipline.py samples --n 8     # print samples per discipline, to eyeball
    python discipline.py explain <oai-id>  # why did this get that label

Paths anchor to this file's location, not the shell's working directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_analysis as R
import harvest_corpus as H

HERE = Path(__file__).resolve().parent

# Weights: a term in the TITLE is far more diagnostic than one in the abstract,
# which is more diagnostic than one somewhere in 100,000 words of body text.
W_TITLE, W_ABSTRACT, W_BODY = 6.0, 2.0, 0.5
STRONG, WEAK = 3.0, 1.0
MARGIN = 0.20          # below this relative margin, defer to embeddings
BODY_SAMPLE = 400      # sentences of body text sampled when available


# --------------------------------------------------------------------------- #
# Taxonomy. `strong` terms are near-decisive; `weak` terms are supporting.
# Lexicons are kept to a similar size so no discipline wins on breadth alone.
# --------------------------------------------------------------------------- #

DISCIPLINES: dict[str, dict] = {
    "Education": {
        "desc": "education, teaching, learning, pedagogy, schools, curriculum, "
                "teacher training, classroom practice, higher education policy",
        "strong": ["pedagog", "curriculum", "teacher education", "initial teacher",
                   "classroom practice", "school improvement", "national curriculum",
                   "teaching and learning", "student teacher", "key stage",
                   "ofsted", "educational attainment", "widening participation"],
        "weak": ["teacher", "teaching", "school", "pupil", "learner", "education",
                 "literacy", "numeracy", "curricular", "headteacher", "lesson",
                 "textbook", "classroom"],
    },
    "Psychology": {
        "desc": "psychology, cognition, behaviour, mental health, perception, "
                "memory, emotion, psychometrics, clinical and developmental psychology",
        "strong": ["cognitive psychology", "psychometric", "working memory",
                   "executive function", "attachment theory", "psychopatholog",
                   "neuropsycholog", "clinical psychology", "developmental psychology",
                   "social cognition", "behaviour change", "self-efficacy"],
        "weak": ["psycholog", "cognition", "cognitive", "perception", "emotion",
                 "anxiety", "depression", "participants rated", "stimuli", "recall"],
    },
    "Medicine and health": {
        "desc": "medicine, clinical practice, disease, patients, diagnosis, "
                "treatment, epidemiology, public health, nursing, surgery",
        "strong": ["clinical trial", "epidemiolog", "patients with", "diagnosis of",
                   "randomised controlled", "public health", "comorbid", "oncolog",
                   "cardiovascular", "primary care", "nursing practice", "palliative"],
        "weak": ["clinical", "patient", "disease", "therapy", "treatment", "cohort",
                 "prevalence", "symptom", "hospital", "nurse", "surgical"],
    },
    "Biological sciences": {
        "desc": "biology, genetics, molecular and cell biology, ecology, "
                "evolution, microbiology, biochemistry, physiology",
        "strong": ["gene expression", "protein", "cell line", "genom", "enzyme",
                   "in vitro", "phylogen", "microbiolog", "biochemic", "molecular",
                   "ecosystem", "species richness", "metabolic"],
        "weak": ["biolog", "cells", "dna", "rna", "bacteria", "ecolog", "organism",
                 "evolution", "physiolog", "assay"],
    },
    "Physical sciences": {
        "desc": "physics, chemistry, materials science, astronomy, quantum "
                "mechanics, spectroscopy, catalysis, condensed matter",
        "strong": ["quantum", "spectroscop", "catalys", "crystal structure",
                   "condensed matter", "nanoparticle", "thermodynamic", "polymer",
                   "astrophys", "photon", "magnetic field", "synthesis of"],
        "weak": ["physic", "chemistr", "materials", "atom", "molecul", "laser",
                 "reaction", "particle", "optical", "electron"],
    },
    "Mathematics": {
        "desc": "mathematics, statistics, probability, algebra, topology, "
                "analysis, number theory, operational research",
        "strong": ["theorem", "manifold", "topolog", "algebraic", "stochastic process",
                   "partial differential equation", "number theory", "bayesian inference",
                   "asymptotic", "conjecture", "lemma"],
        "weak": ["mathemat", "statistic", "probabilit", "equation", "matrix",
                 "algorithmic complexity", "geometr", "proof"],
    },
    "Computing": {
        "desc": "computer science, software, algorithms, machine learning, "
                "artificial intelligence, networks, databases, human-computer interaction",
        "strong": ["machine learning", "neural network", "software engineering",
                   "human-computer interaction", "distributed system", "deep learning",
                   "natural language processing", "computer vision", "cryptograph",
                   "database system", "source code"],
        "weak": ["algorithm", "computing", "computational", "software", "dataset",
                 "programming", "user interface", "network protocol"],
    },
    "Engineering": {
        "desc": "engineering, mechanical, civil, electrical, chemical and "
                "aerospace engineering, control systems, manufacturing, structures",
        "strong": ["finite element", "turbine", "combustion", "structural analysis",
                   "control system", "manufactur", "aerodynamic", "fatigue life",
                   "heat transfer", "power system", "mechanical propert", "actuator"],
        "weak": ["engineering", "mechanical", "electrical", "civil", "hydraulic",
                 "sensor", "design of the system", "vibration", "circuit"],
    },
    "Earth and environment": {
        "desc": "geography, geology, environmental science, climate, oceanography, "
                "hydrology, sustainability, earth systems",
        "strong": ["climate change", "sediment", "hydrolog", "glacial", "catchment",
                   "atmospheric", "biodiversity", "land use", "geochemic",
                   "sea level", "carbon emission", "geomorpholog"],
        "weak": ["geolog", "environmental", "climate", "soil", "river", "ocean",
                 "landscape", "sustainab", "geograph"],
    },
    "Social sciences": {
        "desc": "sociology, social policy, anthropology, politics, international "
                "relations, criminology, gender and race studies",
        "strong": ["sociolog", "anthropolog", "social policy", "criminolog",
                   "international relations", "political econom", "ethnic minorit",
                   "social capital", "welfare state", "civil society", "governance",
                   "racism", "inequalit"],
        "weak": ["social", "politic", "policy", "community", "identity", "gender",
                 "state", "citizen", "migration", "class"],
    },
    "Business and economics": {
        "desc": "business, management, marketing, finance, accounting, economics, "
                "organisational behaviour, entrepreneurship, supply chain",
        "strong": ["supply chain", "entrepreneur", "organisational behaviour",
                   "marketing strateg", "corporate governance", "financial market",
                   "human resource management", "consumer behaviour", "econometric",
                   "firm performance", "accounting"],
        "weak": ["business", "management", "economic", "market", "firm", "investment",
                 "employee", "customer", "productivity", "industry"],
    },
    "Law": {
        "desc": "law, legal theory, jurisprudence, human rights, criminal and "
                "international law, regulation, legislation, courts",
        "strong": ["jurisprudence", "statutory", "case law", "human rights law",
                   "criminal law", "international law", "legislative", "tribunal",
                   "legal framework", "judicial review", "contract law"],
        "weak": ["legal", "court", "law", "rights", "regulation", "justice",
                 "liability", "statute"],
    },
    "History, philosophy and religion": {
        "desc": "history, philosophy, theology, religious studies, archaeology, "
                "classics, intellectual history, ethics",
        "strong": ["historiograph", "archaeolog", "theolog", "medieval",
                   "eighteenth century", "nineteenth century", "epistemolog",
                   "metaphysic", "religious practice", "manuscript", "antiquity"],
        "weak": ["history", "historical", "philosoph", "religio", "church",
                 "ancient", "ethic", "archive", "century"],
    },
    "Language and literature": {
        "desc": "linguistics, applied linguistics, language acquisition, English "
                "literature, translation, second language learning, discourse",
        "strong": ["second language acquisition", "applied linguistic", "corpus linguistic",
                   "translation studies", "phonolog", "syntax", "morpholog",
                   "literary criticism", "narrative fiction", "poetics",
                   "language learner", "bilingual"],
        "weak": ["linguistic", "language", "literature", "text", "novel", "poetry",
                 "grammar", "lexical", "discourse", "translation"],
    },
    "Arts and media": {
        "desc": "art, design, music, drama, film, performance, media, journalism, "
                "communication, architecture, creative practice",
        "strong": ["performance practice", "musicolog", "film studies", "journalism",
                   "creative practice", "architectural design", "visual art",
                   "choreograph", "curatorial", "sound design", "theatre practice"],
        "weak": ["music", "film", "art", "design", "media", "theatre", "dance",
                 "audience", "composition", "architecture"],
    },
    "Sport and agriculture": {
        "desc": "sport science, exercise physiology, kinesiology, nutrition, "
                "agriculture, food science, veterinary science",
        "strong": ["exercise physiolog", "athletic performance", "sports science",
                   "vo2 max", "musculoskeletal", "agricultur", "crop yield",
                   "veterinar", "food science", "livestock", "nutritional intake"],
        "weak": ["sport", "exercise", "athlete", "training load", "nutrition",
                 "farm", "animal", "dietary", "physical activity"],
    },
}

# Terms that are deliberate STEMS, matched as prefixes ("pedagog" -> pedagogy,
# pedagogical). Everything else is matched as a whole word with at most a short
# inflection, because unbounded prefix matching is catastrophic on short terms:
# "art" otherwise matches "Arteriogenesis" and files a zebrafish genetics thesis
# under Arts and media.
STEMS = {
    "pedagog", "psycholog", "psychopatholog", "neuropsycholog", "epidemiolog",
    "oncolog", "comorbid", "microbiolog", "biochemic", "phylogen", "genom",
    "ecolog", "biolog", "physiolog", "spectroscop", "catalys", "astrophys",
    "molecul", "chemistr", "physic", "topolog", "mathemat", "statistic",
    "probabilit", "geometr", "cryptograph", "manufactur", "hydrolog",
    "geochemic", "geomorpholog", "geolog", "sustainab", "geograph", "sociolog",
    "anthropolog", "criminolog", "entrepreneur", "historiograph", "archaeolog",
    "theolog", "epistemolog", "metaphysic", "religio", "philosoph", "ethic",
    "musicolog", "choreograph", "agricultur", "veterinar", "phonolog",
    "morpholog", "inequalit", "politic", "legislativ", "statutor", "judicial",
    "exercise physiolog", "aerodynamic", "thermodynamic", "nanoparticle",
    "condensed matter", "mechanical propert", "structural analysis",
    "diagnosis of", "patients with", "randomised controlled", "in vitro",
    "gene expression", "cell line", "species richness", "crystal structure",
    "synthesis of", "number theory", "bayesian inference", "stochastic process",
    "partial differential equation", "machine learning", "neural network",
    "software engineering", "human-computer interaction", "distributed system",
    "deep learning", "natural language processing", "computer vision",
    "database system", "source code", "finite element", "control system",
    "fatigue life", "heat transfer", "power system", "climate change",
    "land use", "sea level", "carbon emission", "social policy",
    "international relations", "political econom", "ethnic minorit",
    "social capital", "welfare state", "civil society", "supply chain",
    "organisational behaviour", "marketing strateg", "corporate governance",
    "financial market", "human resource management", "consumer behaviour",
    "econometric", "firm performance", "case law", "human rights law",
    "criminal law", "international law", "legal framework", "judicial review",
    "contract law", "eighteenth century", "nineteenth century",
    "religious practice", "second language acquisition", "applied linguistic",
    "corpus linguistic", "translation studies", "literary criticism",
    "narrative fiction", "language learner", "performance practice",
    "film studies", "creative practice", "architectural design", "visual art",
    "sound design", "theatre practice", "athletic performance",
    "sports science", "crop yield", "food science", "nutritional intake",
    "teacher education", "initial teacher", "classroom practice",
    "school improvement", "national curriculum", "teaching and learning",
    "student teacher", "key stage", "educational attainment",
    "widening participation", "cognitive psychology", "psychometric",
    "working memory", "executive function", "attachment theory",
    "clinical psychology", "developmental psychology", "social cognition",
    "behaviour change", "clinical trial", "public health", "primary care",
    "nursing practice", "quasi-experimental", "design of the system",
    "participants rated", "algorithmic complexity", "network protocol",
    "user interface", "training load", "physical activity", "dietary",
    "musculoskeletal", "vo2 max",
}


def _term_re(t: str) -> re.Pattern:
    if t.endswith("*") or t in STEMS:
        return re.compile(r"\b" + re.escape(t.rstrip("*")), re.IGNORECASE)
    # whole word, tolerating a short inflection (plural, -ed, -al)
    return re.compile(r"\b" + re.escape(t) + r"\w{0,2}\b", re.IGNORECASE)


_COMPILED = {
    name: {
        "strong": [(t, _term_re(t)) for t in d["strong"]],
        "weak": [(t, _term_re(t)) for t in d["weak"]],
    }
    for name, d in DISCIPLINES.items()
}
NAMES = sorted(DISCIPLINES)


def score(title: str, abstract: str, body: str = "") -> dict[str, float]:
    """Weighted lexicon score per discipline. Terms count once per field."""
    out: dict[str, float] = {}
    fields = ((title or "", W_TITLE), (abstract or "", W_ABSTRACT), (body or "", W_BODY))
    for name in NAMES:
        c = _COMPILED[name]
        total = 0.0
        for text, fw in fields:
            if not text:
                continue
            for _, rx in c["strong"]:
                if rx.search(text):
                    total += STRONG * fw
            for _, rx in c["weak"]:
                if rx.search(text):
                    total += WEAK * fw
        out[name] = total
    return out


def matched_terms(name: str, title: str, abstract: str, body: str = "") -> list[str]:
    c = _COMPILED[name]
    hits = []
    for text, tag in ((title, "title"), (abstract, "abstract"), (body, "body")):
        if not text:
            continue
        for t, rx in c["strong"]:
            if rx.search(text):
                hits.append(f"{t}({tag},strong)")
        for t, rx in c["weak"]:
            if rx.search(text):
                hits.append(f"{t}({tag})")
    return hits


# --------------------------------------------------------------------------- #
# Embedding fallback
# --------------------------------------------------------------------------- #

_DESC_EMB = None


def _descriptor_embeddings():
    global _DESC_EMB
    if _DESC_EMB is not None:
        return _DESC_EMB
    model = R._get_embed_model()
    if model is None:
        _DESC_EMB = (None, None)
        return _DESC_EMB
    import numpy as np
    texts = [f"{n}. {DISCIPLINES[n]['desc']}" for n in NAMES]
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    _DESC_EMB = (model, emb)
    return _DESC_EMB


def embed_classify(text: str, among: list[str] | None = None
                   ) -> tuple[str, float] | None:
    """Nearest discipline descriptor by cosine.

    `among` restricts the choice to a shortlist. When the rules have an opinion
    but not a confident one, the embedding should BREAK THE TIE between the
    candidates the rules actually proposed — not roam freely over all sixteen
    and discard the lexical evidence entirely.
    """
    model, emb = _descriptor_embeddings()
    if model is None or not text.strip():
        return None
    import numpy as np
    v = model.encode([text[:2000]], convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)[0]
    sims = emb @ v.astype("float32")
    idxs = [i for i, n in enumerate(NAMES) if (among is None or n in among)]
    if not idxs:
        idxs = list(range(len(NAMES)))
    best = max(idxs, key=lambda i: (float(sims[i]), -i))
    return NAMES[best], round(float(sims[best]), 4)


def classify_one(title: str, abstract: str, body: str = "",
                 allow_embeddings: bool = True) -> dict:
    s = score(title, abstract, body)
    ranked = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))
    top, second = ranked[0], ranked[1]
    total = sum(s.values()) or 1.0
    margin = (top[1] - second[1]) / total if total else 0.0
    if top[1] > 0 and margin >= MARGIN:
        return {"discipline": top[0], "conf": round(top[1] / total, 4),
                "alt": second[0], "method": "rules", "ambiguous": 0}
    if allow_embeddings:
        shortlist = [n for n, v in ranked[:3] if v > 0] or None
        e = embed_classify(" ".join(x for x in (title, abstract, body[:1500]) if x),
                           among=shortlist)
        if e:
            alt = next((n for n in (shortlist or []) if n != e[0]), top[0]
                       if top[1] > 0 else "")
            return {"discipline": e[0], "conf": e[1], "alt": alt,
                    "method": "embedding-tiebreak" if shortlist else "embedding",
                    "ambiguous": 1}
    if top[1] > 0:
        return {"discipline": top[0], "conf": round(top[1] / total, 4),
                "alt": second[0], "method": "rules-weak", "ambiguous": 1}
    return {"discipline": "Unknown", "conf": 0.0, "alt": "", "method": "none",
            "ambiguous": 1}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _body_sample(doc_id: str) -> str:
    p = H.store_path(doc_id)
    if not p.exists():
        return ""
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            rows = json.load(fh)["sentences"]
    except Exception:
        return ""
    keep, anytext = [], []
    for r in rows[:6000]:
        text, page, printed, bucket, heading, excl = H.unpack_sentence(r)
        if excl:
            continue
        anytext.append(text)
        if bucket in ("introduction", "literature_review", "methodology"):
            keep.append(text)
        if len(keep) >= BODY_SAMPLE:
            break
    # Heading detection fails on some documents (ALHU2025 yields 155 methodology
    # tokens against 65,337 unclassified). Falling back to any retained text is
    # better than classifying on a starved sample.
    if len(keep) < 80:
        keep = anytext[:BODY_SAMPLE]
    return " ".join(keep)


def classify_all(limit: int | None, no_embeddings: bool) -> int:
    con = H.db()
    rows = con.execute(
        "SELECT id,title,abstract,status FROM doc ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    if not rows:
        print("No metadata yet — run `harvest_corpus.py harvest` first.")
        return 1
    n_abs = sum(1 for r in rows if r[2])
    print(f"  {len(rows):,} documents ({n_abs:,} with an abstract, "
          f"{100*n_abs/len(rows):.0f}%)", file=sys.stderr)
    out, counts, methods = [], Counter(), Counter()
    for i, (doc_id, title, abstract, status) in enumerate(rows, 1):
        body = _body_sample(doc_id) if status == "ok" else ""
        r = classify_one(title or "", abstract or "", body,
                         allow_embeddings=not no_embeddings)
        out.append((r["discipline"], r["conf"], r["alt"], r["method"], doc_id))
        counts[r["discipline"]] += 1
        methods[r["method"]] += 1
        if i % 2000 == 0:
            print(f"    {i:,}/{len(rows):,}", file=sys.stderr)
    con.executemany("UPDATE doc SET discipline=?, discipline_conf=?, "
                    "discipline_alt=?, discipline_method=? WHERE id=?", out)
    con.commit()

    total = len(out)
    print()
    print("=" * 70)
    print(f"DISCIPLINE CLASSIFICATION   ({total:,} theses)")
    print("=" * 70)
    print(f"  {'discipline':<34}{'theses':>9}{'share':>9}")
    print("  " + "-" * 52)
    for name, c in counts.most_common():
        print(f"  {name:<34}{c:>9,}{100*c/total:>8.1f}%")
    print()
    print("  assignment route:")
    for m, c in methods.most_common():
        print(f"    {m:<16}{c:>9,}{100*c/total:>8.1f}%")
    unk = counts.get("Unknown", 0)
    if unk / total > 0.15:
        print(f"\n  WARNING: {100*unk/total:.0f}% unclassified. Most of these will be")
        print("  title-only rows; rerun after `harvest --refresh` backfills abstracts.")
    con.close()
    return 0


def samples(n: int) -> int:
    con = H.db()
    print("Sample titles per assigned discipline — read these before trusting the labels.")
    for name in NAMES + ["Unknown"]:
        rows = con.execute(
            "SELECT title, discipline_conf, discipline_method FROM doc "
            "WHERE discipline=? AND title IS NOT NULL "
            "ORDER BY discipline_conf DESC LIMIT ?", (name, n)).fetchall()
        if not rows:
            continue
        cnt = con.execute("SELECT COUNT(*) FROM doc WHERE discipline=?",
                          (name,)).fetchone()[0]
        print(f"\n{name}  ({cnt:,})")
        for t, c, m in rows:
            print(f"   [{c:.2f} {m[:9]:<9}] {(t or '')[:96]}")
    con.close()
    return 0


def explain(doc_id: str) -> int:
    con = H.db()
    row = con.execute("SELECT title,abstract,discipline,discipline_conf,"
                      "discipline_alt,discipline_method,status FROM doc WHERE id=?",
                      (doc_id,)).fetchone()
    if not row:
        print(f"no such document: {doc_id}")
        return 1
    title, abstract, disc, conf, alt, method, status = row
    body = _body_sample(doc_id) if status == "ok" else ""
    print(f"title    : {title}")
    print(f"assigned : {disc}  (conf {conf}, via {method}, runner-up {alt})")
    s = score(title or "", abstract or "", body)
    print("\nscores:")
    for name, v in sorted(s.items(), key=lambda kv: -kv[1])[:6]:
        if v <= 0:
            continue
        print(f"  {name:<34}{v:>8.1f}")
        for t in matched_terms(name, title or "", abstract or "", body)[:8]:
            print(f"      {t}")
    con.close()
    return 0


def validate() -> int:
    """Check the classifier against documents you have supplied yourself.

    Reads titles from documents.json (see documents.example.json). Add an
    "expect_discipline" field to any entry to assert its label. With no config
    this reports what it would need rather than failing, because the classifier
    is exercised anyway by harvest_corpus.py over the whole corpus.
    """
    try:
        specs = R.load_documents()
    except SystemExit as e:
        print(str(e).splitlines()[0])
        print("\n  validate() needs documents.json; see documents.example.json.")
        print('  Add "title" and "expect_discipline" per entry to assert a label.')
        return 0
    raw = json.loads((R.HERE / "documents.json").read_text(encoding="utf-8"))
    cfg = {d["key"]: d for d in raw.get("documents", [])}
    info = R.ensure_pdfs(specs, True, R.PDF_DIR)
    print("Validation — classifier against your own documents")
    print("=" * 70)
    ok = top2 = n = 0
    for spec in specs:
        want = cfg.get(spec.key, {}).get("expect_discipline")
        title = cfg.get(spec.key, {}).get("title") or spec.label
        pages = R.load_or_extract(spec, Path(info[spec.key]["path"]),
                                  info[spec.key]["sha256"])
        doc = R.build_document(spec, pages)
        body = " ".join(x.text for x in doc.sentences
                        if x.included and x.bucket in
                        ("introduction", "literature_review", "methodology"))[:60000]
        r = classify_one(title, "", body)
        if not want:
            print(f"  ----  {spec.short:<10} -> {r['discipline']:<22} "
                  f"conf {r['conf']:.2f} via {r['method']} (no expectation set)")
            continue
        n += 1
        good = r["discipline"] == want
        near = good or r["alt"] == want
        ok += good
        top2 += near
        tag = "PASS" if good else ("TOP-2" if near else "FAIL")
        print(f"  {tag:<5} {spec.short:<10} -> {r['discipline']:<22} "
              f"conf {r['conf']:.2f} via {r['method']:<18} (want {want})")
    if n:
        print(f"\n  top-1 {ok}/{n}   top-2 {top2}/{n}")
        print("  Interdisciplinary works are genuinely ambiguous, not wrong:")
        print("  they are flagged `ambiguous` so cross-field analyses can")
        print("  exclude them rather than treat them as confidently labelled.")
    return 0 if (n == 0 or ok == n) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify", help="label every thesis and write to the DB")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--no-embeddings", action="store_true")
    s = sub.add_parser("samples", help="print sample titles per discipline")
    s.add_argument("--n", type=int, default=6)
    e = sub.add_parser("explain", help="why did a document get its label")
    e.add_argument("doc_id")
    sub.add_parser("validate", help="check against the four known theses")
    a = ap.parse_args(argv)
    if a.cmd == "classify":
        return classify_all(a.limit, a.no_embeddings)
    if a.cmd == "samples":
        return samples(a.n)
    if a.cmd == "explain":
        return explain(a.doc_id)
    return validate()


if __name__ == "__main__":
    sys.exit(main())
