#!/bin/bash
# run_corpus_study.sh — drive the whole corpus study to completion.
#
# Stages run STRICTLY IN SEQUENCE. Two stages must never hit the repository at
# once: the host publishes Crawl-delay: 10 and each stage rate-limits itself
# independently, so running them concurrently would double the request rate.
#
# Every stage is resumable. Killing this script and rerunning it is safe.
#
# Paths anchor to this script's own location, not the shell's cwd.
set -u
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
N_DOCS="${1:-400}"

say() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

say "waiting for any running metadata harvest to finish"
while pgrep -f "harvest_corpus.py harvest" > /dev/null; do sleep 20; done

say "stage 1/6 — metadata harvest (resumes if incomplete)"
$PY harvest_corpus.py harvest || echo "harvest returned $? (resumable)"

say "stage 2/6 — fetch + extract ${N_DOCS} doctoral education theses"
echo "  PDFs are deleted immediately after extraction; only text is kept."
$PY harvest_corpus.py fetch --limit "$N_DOCS" --doctoral --education \
  || echo "fetch returned $? (resumable)"

say "stage 3/6 — all-pairs verbatim screen"
$PY harvest_corpus.py screen || echo "screen returned $? (resumable)"

say "stage 4/6 — false-positive suppression"
$PY harvest_corpus.py verify || echo "verify returned $?"

say "stage 5/6 — focal pair against the corpus (third-document test)"
$PY harvest_corpus.py focal-check 2>&1 | tee corpus_focal_check.txt \
  || echo "focal-check returned $?"

say "stage 6/6 — report"
$PY harvest_corpus.py report | tee corpus_report.txt

say "done"
$PY harvest_corpus.py status
