# thesisgraph

Reproducible tools for measuring **textual overlap between documents** and for
**mapping a corpus of doctoral theses** — what it cites, what methods it uses,
how it writes, and how its subjects connect.

Built from openly deposited UK doctoral theses. No thesis text is redistributed
by this repository; see [Copyright](#copyright).

---

## Two things live here

**1. A pairwise overlap analyser** (`run_analysis.py`) — measures verbatim and
near-verbatim overlap between two documents against a control baseline, and
emits an auditable report, a CSV of every match, and an interactive
side-by-side viewer.

**2. A corpus toolkit** — harvests a repository over OAI-PMH, extracts and
analyses tens of thousands of theses, screens the whole corpus for textual
reuse, and builds an interactive graph of the result.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python run_analysis.py --selftest     # synthetic fixtures, asserted
.venv/bin/python run_analysis.py                # download + analyse a pair
```

## The pairwise analyser

```bash
run_analysis.py                     # full run
run_analysis.py --selftest          # planted-passage fixtures; 34 unit + 22 integration checks
run_analysis.py --determinism-check # produce artefacts twice, diff byte-for-byte
check_viewer.py / check_app.py      # render the viewers headless, assert behaviour
```

Extraction uses `pdfplumber` at word level with x/y geometry (needed for
block-quote indentation and heading detection), with `pypdf` as a per-page and
whole-document fallback. Paragraphs are rebuilt across line breaks; a hyphenated
line-end split is rejoined only when the continuation begins lowercase.
Sentence splitting is abbreviation-aware. Reference lists, attributed indented
block quotes, boilerplate and captions are excluded and counted separately — an
indented passage with *no* nearby attribution is deliberately retained.

Three matching passes: verbatim runs of 8+ words (maximally extended,
contained runs dropped), `rapidfuzz` `token_set_ratio` ≥ 85, and optional
sentence-embedding cosine ≥ 0.90 which degrades cleanly to "not run".

**Determinism is verified, not asserted:** no wall-clock, no RNG, no reliance on
set or dict ordering; every collection sorted before writing; artefacts produced
twice and diffed.

## The corpus toolkit

| Script | Does |
|---|---|
| `harvest_corpus.py` | OAI-PMH metadata harvest, polite fetch, extract, per-pair screen |
| `screen_corpus.py` | Winnowed-fingerprint index; whole-corpus screening |
| `rescore.py` | Re-score pairs after removing extraction artefacts |
| `discipline.py` | Assign a subject discipline (rules + embedding tie-break) |
| `subfields.py` / `hierarchy.py` | Derive an expandable subject tree from the data |
| `corpus_studies.py` | Method and reporting practice |
| `discourse.py` | Writing/discourse features; method diffusion across fields |
| `citations.py` | Parse reference lists into a citation graph |
| `build_graph.py` / `make_graph_app.py` | Build the interactive graph explorer |

### Screening at corpus scale

All-pairs is impossible — 24,656 theses is 303 million pairs. `screen_corpus.py`
uses winnowing (Schleimer, Wilkerson & Aiken 2003): hash every 8-gram, keep the
minimum in each window of 13. That retains ~1/13 of fingerprints while
**guaranteeing** that any shared passage of `n + w - 1 = 20` or more words shares
a fingerprint. Verified empirically: 300/300 detection at 20 words, 210/300 at
14 (below the bound, as theory predicts).

In practice: **303,946,840 pairs → 222,881 candidates, a 1,363× reduction.**

## What this measures — and what it does not

It measures **textual reuse**. It cannot detect fabricated data, ghostwriting or
a purchased thesis; those leave no textual trace.

Output is **a ranked list of things to look at**, never a finding about a person.
Shared text has many innocent causes, and in this corpus most of it is exactly
that. Measured on a 400-thesis pilot:

- **20%** of same-discipline pairs share an 8-word run; **5.9%** share a 20+ word run
- **95.5%** of all shared runs also appear in an unrelated third thesis
- Only **1.34%** of non-zero pairs survive filtering

The top of an unfiltered ranking is dominated by **PDF character-spacing
artefacts** (`i n f o r m a t i o n s y s t e m s`), duplicate deposits, software
licences, ethics templates, published instruments, and repository deposit forms
bound into theses. Every one of those is a filter, and the code implements them.

**Textual overlap is not evidence of intent and is not a finding of misconduct.**
Only an institution can make that determination.

## Copyright

Theses are in copyright. The UK text-and-data-mining exception (s.29A CDPA)
permits computational analysis of lawfully accessed works; it does **not** permit
redistributing them.

`.gitignore` therefore excludes `pdfs/`, `corpus/`, `cache/` and any generated
artefact that embeds thesis text (`out/viewer.html`, `out/app.html`). Derived
data — statistics, embeddings, citation edges, graphs — is shareable; the text
is not.

Crawling respects `robots.txt` and the published `Crawl-delay`, identifies
itself, and never retains PDFs: each is downloaded, extracted and deleted, so a
25k-thesis corpus costs ~10 GB of text rather than ~72 GB of PDFs.

## Known limitations

- **Reference parsing is author-date only** — ~46% of parenthesised-year entries
  and no Vancouver style, biasing the citation graph toward social sciences and
  humanities. GROBID or AnyStyle is the proper fix.
- **Chapter/heading detection fails on some documents**; affected rows are
  flagged (`method_scope`, unclassified share) rather than silently reported.
- **The discourse time series has a digitisation confound** at ~2010 where
  retro-digitised theses give way to born-digital ones.
- **Discipline labels are ~38% low-confidence** and flagged as such.

## Licence

MIT — see [LICENSE](LICENSE).
