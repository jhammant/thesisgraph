# thesisgraph — state as of 2026-09-11

## What this is
A navigable map of what UK doctoral research actually reads: 24,656 White Rose
theses (Sheffield/Leeds/York), 1.7bn words, now 3.9M parsed citations. Live at
thesisgraph.hammantlabs.com, source at github.com/jhammant/thesisgraph.

## Where it stands

**Done**
- Graph page (index.html) is now the front door: labels radiate outward and
  never overprint, every node labelled, word-boundary trimming, single ring at
  0.88R so the map fills the canvas. Deployed and verified live over HTTPS.
- The bridges cross-link is emitted by the generator, not hand-patched after
  every rebuild.
- Demo video re-shot against the graph (19s, 3.0MB) plus three 1600x1000
  thumbnails, all produced by `capture_demo.py` on every run.
- **Citation reparse COMPLETE**: 24,656/24,656 theses, **3,918,144 references
  (2.58x the old 1,517,377)**, 445,279 carrying a DOI, 0 errors. Used stored
  text, so nothing was re-crawled (~50 min vs ~68 hours of polite fetching).
  New `refseg.py` (8 asserted fixtures) + `regrobid.py`, output in
  `corpus/citations_grobid.db`. The old `citations.db` is untouched.
- **Bridge-type analysis COMPLETE**: 12,534 cross-field works classified on the
  local model (free, ~50 min, gemma-3-4b) via `bridge_types.py`.

**Blocked**
- OpenAlex reconciliation (`reconcile.py`) is written and works, but the
  anonymous pool returns 429 above ~2 req/s. Needs a `--mailto` decision.

**Unpushed**: 6 commits on main, tree clean.

## The finding — and why the LinkedIn post must not go out as drafted
Tracked post #18 asserts "I expected fields to be joined by ideas. They're
joined by methods." **The full corpus contradicts this**: THEORY 45.9% vs
METHOD 35.3% across all 12,522 labelled cross-field works.

The real result is a clean gradient by how far a work travels:

| reach | METHOD | THEORY |
|---|---|---|
| 3-4 fields | 30.6% | 47.3% |
| 5-8 fields | 42.4% | 44.5% |
| 9-16 fields | 59.3% | 33.8% |

Neighbouring fields trade theory; works that cross the whole university are
methods. Most method-driven pairs: Computing x Maths 70.5%, Engineering x Maths
70.4%. Least: History x Social sciences 24.0%, Arts x History 24.7%.

## Session infrastructure built this session (lives in ~/.claude, NOT in this repo)
- `/closecode` now **never** commits transcripts to a public repo — it gitignores
  `.sessions/` instead. Previously it asked; now it does not, because a public
  repo is forever and redaction catches secret shapes, not unreleased drafts or
  third-party names.
- `~/.claude/skills/closecode/scripts/backup-sessions` mirrors `~/.claude/projects`
  into **private** `github.com/jhammant/claude-transcript-archive`, scrubbed and
  gzipped (750MB raw -> 313MB repo). 881 transcripts, 140 secrets redacted.
- launchd agent `com.jhammant.claude-transcript-backup` runs it every 2h and at
  login; logs to `~/Library/Logs/claude-transcript-backup.log`.
- Restore is tested, not assumed: a real transcript was deleted and recovered
  byte-identical. `--restore DIR` clones, `--reinstall` expands back into
  `~/.claude/projects`. Existing transcripts are never overwritten without
  `--force`. Slugs start with `-`, so use `--project=-Users-...`.
- **`~/.claude` is not version controlled**, so all of the above exists only on
  this machine. Worth folding into a dotfiles repo.

## Next steps
1. Rewrite post #18 around the gradient, not "fields are joined by methods".
   Also update Braun & Clarke: 683 -> **857** citing theses.
2. Decide on `--mailto` for OpenAlex, then `reconcile.py --min-theses 2`
   (238,219 works, ~40 min with the polite pool, ~9 hours without).
3. Rebuild graph.json/hierarchy.json off `citations_grobid.db` so the live map
   uses 3.9M references instead of 1.5M, then redeploy.
4. Re-run `bridge_types.py` after reconciliation; merge editions of the same
   work (Denzin 2005/2011/2018 are legitimately distinct, but may want merging).
5. A GROBID container (`grobid/grobid:0.9.1-crf`, native arm64) is still running
   and has been up 14h. Needed only for `regrobid.py`; `docker rm -f grobid` to
   reclaim the memory. Note the Docker *daemon* went unresponsive under the full
   reparse load — the container itself was fine throughout.

## Open questions (not mine to decide)
- Send jhammant@gmail.com to OpenAlex for the polite pool? Not done unasked.
- Publish post #18 at all, and when? 0 of 16 tracked posts are measured, so the
  model predicts 10,699 for everything and cannot calibrate.
- Expand to EThOS scale (650k records, ~420k fetchable, 3-5 days crawl, ~250GB)?
  The CC0 dataset now sits behind a British Library login.

## Ideas worth keeping
- **Segmentation, not parsing, was the bottleneck.** GROBID parses a reference
  well but needs one entry per input; the corpus had run-on text. Recovering
  boundaries was worth 2.58x, and needed no network at all.
- **Prove structurally, not statistically.** Rejecting reference sections by
  digit density killed 9 of 25 valid ACS lists, because ACS *is* digit-dense.
  "Does it split into citation-shaped entries?" is the sound test.
- **Separate 'parser failed' from 'not a reference list'.** Archival lists,
  tables and footnote prose are not recoverable text and must not be summed
  into a recall figure.
- **Route by constraint.** Deterministic matching against an authority file
  beats an LLM; the LLM earns its place only on the ambiguous middle and on
  high-volume labelling. Local pool is memory-bound and free, so bulk
  classification belongs there.
- **Benchmarks can mislead**: `plan`/`bench` assumed decode-bound work and were
  out by 3x on a prefill-dominated batch (259 prompt -> 4 completion tokens).
  Measure on the real workload.

---
_Updated by `forkcode close` on 2026-09-11T13:35+01:00_

---
_Updated by `forkcode close` on 2026-09-11T13:45+01:00_
