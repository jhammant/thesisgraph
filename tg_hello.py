#!/usr/bin/env python3
"""
tg_hello.py — the smallest capability that does something real, kept as living
documentation of the tg.py plugin contract and as something for discovery to
find in a tree where every other module may still be half-written.

It answers the question worth asking before any longer run: is the corpus
actually there, and how big is it? Row counts only — no thesis text is read and
none is printed, so this output is safe to paste anywhere.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

NAME = "hello"
HELP = "row counts from corpus/corpus.db — check the corpus is present"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, default=HERE / "corpus" / "corpus.db",
                   help="corpus database to count (default: %(default)s)")


def run(args: argparse.Namespace) -> int:
    db = Path(args.db)
    if not db.exists():
        print(f"hello: no database at {db}", file=sys.stderr)
        return 1

    # Read-only: corpus/ is data the toolkit reads and never writes.
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"))
        rows = [(t, con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0])
                for t in tables]
        if "doc" in tables:
            rows.append(("doc where status='ok'", con.execute(
                "SELECT COUNT(*) FROM doc WHERE status='ok'").fetchone()[0]))
    finally:
        con.close()

    width = max((len(label) for label, _ in rows), default=0)
    print(db)
    for label, n in rows:
        print(f"  {label:<{width}}  {n:>10,}")
    return 0


if __name__ == "__main__":
    # tg.py turns an exception into one clear line for the plugin path; run
    # standalone, nothing does, and a malformed --db should not end in a trace.
    ap = argparse.ArgumentParser(description=HELP)
    add_args(ap)
    try:
        raise SystemExit(run(ap.parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (sqlite3.Error, OSError, ValueError) as exc:
        print("%s: %s: %s" % (NAME, type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
