#!/bin/bash
# Supervisor for the exact-verification stage.
# The machine runs the user's own applications (82 GB RSS observed) and macOS
# jetsam has already killed this stage once under memory pressure. Every stage
# is resumable — pairs already written are skipped — so the correct response is
# to restart rather than to fight for memory or hold a large pool open.
set -u
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
for attempt in $(seq 1 40); do
  before=$($PY -c "import sqlite3;print(sqlite3.connect('corpus/corpus.db').execute('SELECT COUNT(*) FROM pair').fetchone()[0])")
  printf '\n=== [%s] attempt %s — %s pairs so far\n' "$(date +%H:%M:%S)" "$attempt" "$before"
  $PY screen_corpus.py exact2 --workers 5 >> corpus_exact2b.log 2>&1
  rc=$?
  after=$($PY -c "import sqlite3;print(sqlite3.connect('corpus/corpus.db').execute('SELECT COUNT(*) FROM pair').fetchone()[0])")
  echo "  exit=$rc  pairs now $after"
  if grep -q "pairs with at least one shared run written" corpus_exact2b.log && [ "$rc" -eq 0 ]; then
    echo "  COMPLETE"; break
  fi
  if [ "$after" -le "$before" ] && [ "$attempt" -gt 2 ]; then
    echo "  no progress in this attempt; stopping"; break
  fi
  sleep 10
done
$PY -c "
import sqlite3;c=sqlite3.connect('corpus/corpus.db')
print('FINAL pairs:', f\"{c.execute('SELECT COUNT(*) FROM pair').fetchone()[0]:,}\")"
