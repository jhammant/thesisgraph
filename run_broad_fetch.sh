#!/bin/bash
# run_broad_fetch.sh — build the LARGE corpus for the O(N) studies.
#
# The overlap screen is O(N^2) and is happy with a few hundred documents. The
# citation-graph, method-reporting and discourse studies are all O(N) and want
# every document we can get. Their binding constraint is not compute, it is the
# published Crawl-delay of 10s: ~10s per thesis, so ~80 hours for all 28,780.
#
# This runs in batches, indefinitely, and is fully resumable. Stop it whenever.
# It waits for the overlap study to finish first so that only one process is
# ever talking to the repository.
set -u
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
BATCH="${1:-500}"

say() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
pending() {
  $PY - <<'EOF'
import sqlite3
c = sqlite3.connect("corpus/corpus.db")
print(c.execute("SELECT COUNT(*) FROM doc WHERE status='pending' "
                "AND pdf_url<>'' AND is_doctoral=1").fetchone()[0])
EOF
}

say "waiting for the overlap study to finish (one process on the host at a time)"
while pgrep -f "run_corpus_study.sh" > /dev/null; do sleep 30; done
while pgrep -f "harvest_corpus.py (harvest|fetch|screen)" > /dev/null; do sleep 30; done

# Metadata-only, 288 requests (~48 min). Abstracts massively improve discipline
# classification: on titles alone 43% of theses fall through to the embedding
# path because the rule lexicons have nothing to match.
say "backfilling abstracts (metadata refresh, ~48 min)"
$PY harvest_corpus.py harvest --refresh || echo "refresh returned $? (resumable)"

say "re-classifying disciplines now abstracts are present"
$PY discipline.py classify || echo "classify returned $?"

say "broad fetch starting — all doctoral theses, no discipline filter"
say "$(pending) documents pending"

while true; do
  before=$(pending)
  if [ "$before" -eq 0 ]; then say "nothing left to fetch"; break; fi
  say "batch of ${BATCH}; ${before} pending (~$(( before * 10 / 3600 )) h remaining)"
  $PY harvest_corpus.py fetch --limit "$BATCH" --doctoral || true
  after=$(pending)
  if [ "$after" -ge "$before" ]; then
    say "no progress in the last batch (${before} -> ${after}); stopping"
    break
  fi
  df -h . | tail -1
done

say "broad fetch finished"
$PY harvest_corpus.py status
