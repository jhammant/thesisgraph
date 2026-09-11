#!/usr/bin/env python3
"""bridge_types.py — what KIND of work connects two fields?

The map says which works two disciplines share. It does not say what those works
are, and that is the more interesting question: fields might borrow each other's
theories, their instruments, or merely their statistics textbooks. Those imply
very different things about how research actually travels.

Each cross-field work is labelled METHOD / THEORY / INSTRUMENT / DATASET /
REVIEW / FINDING / REFERENCE by a local model (see local-llm batch), and this
script aggregates those labels against the citing disciplines.

The load-bearing comparison is near pairs against distant pairs. If distant
fields are joined by methods while neighbouring fields share findings and
theory, then methodology is the currency of interdisciplinarity — and that is a
claim the corpus can support or refute, rather than an impression from reading
four titles.

    bridge_types.py --labels /tmp/cls_all.jsonl
"""
from __future__ import annotations
import argparse, collections, json, sqlite3, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CITE_DB = HERE / "corpus" / "citations_grobid.db"
CORPUS_DB = HERE / "corpus" / "corpus.db"
LABELS = {"METHOD", "THEORY", "INSTRUMENT", "DATASET", "REVIEW", "FINDING",
          "REFERENCE"}


def load_labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("ok") or not r.get("response"):
            continue
        tok = r["response"].strip().upper().replace("*", "").split()
        if tok and tok[0] in LABELS:
            out[r.get("id")] = tok[0]
    return out


def work_fields() -> dict[str, set[str]]:
    """work key -> set of disciplines whose theses cite it."""
    c = sqlite3.connect(f"file:{CITE_DB}?mode=ro", uri=True, timeout=60)
    c.execute(f"ATTACH DATABASE 'file:{CORPUS_DB}?mode=ro' AS m")
    out = collections.defaultdict(set)
    for k, disc in c.execute("""
            SELECT CASE WHEN ct.doi<>'' THEN ct.doi
                        ELSE ct.surname||'|'||ct.year||'|'||LOWER(SUBSTR(ct.title,1,80)) END,
                   d.discipline
            FROM cite ct JOIN m.doc d ON d.id=ct.src
            WHERE ct.title<>'' AND length(ct.title)>14 AND d.discipline<>''"""):
        out[k].add(disc)
    return out


def report(labels: dict[str, str], wf: dict[str, set[str]], top: int):
    have = {k: v for k, v in labels.items() if k in wf}
    print(f"labelled cross-field works: {len(have):,}\n")

    overall = collections.Counter(have.values())
    tot = sum(overall.values()) or 1
    print("What kind of work travels between fields?")
    for lab, n in overall.most_common():
        print(f"  {lab:<11}{n:>7,}  {100*n/tot:>5.1f}%  {'#'*int(44*n/tot)}")

    # Reach: a work cited in many fields versus one cited in three.
    print("\nBy reach (number of disciplines citing the work):")
    buckets = [(3, 4, "3-4 fields"), (5, 8, "5-8 fields"), (9, 16, "9-16 fields")]
    print(f"  {'reach':<12}{'works':>8}   " +
          "".join(f"{l[:6]:>8}" for l in ("METHOD", "THEORY", "FINDING", "REVIEW")))
    for lo, hi, name in buckets:
        sub = [have[k] for k in have if lo <= len(wf[k]) <= hi]
        if not sub:
            continue
        c = collections.Counter(sub)
        n = len(sub)
        cells = "".join(f"{100*c[l]/n:>7.1f}%" for l in
                        ("METHOD", "THEORY", "FINDING", "REVIEW"))
        print(f"  {name:<12}{n:>8,}   {cells}")

    # The load-bearing comparison: near pairs vs distant pairs.
    pair_lab = collections.defaultdict(collections.Counter)
    for k, lab in have.items():
        fs = sorted(wf[k])
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                pair_lab[(fs[i], fs[j])][lab] += 1

    scored = []
    for pair, c in pair_lab.items():
        n = sum(c.values())
        if n >= 40:
            scored.append((c["METHOD"] / n, n, pair, c))
    scored.sort()

    print(f"\nField pairs where shared literature is LEAST method-driven "
          f"(top {top}):")
    for frac, n, pair, c in scored[:top]:
        print(f"  {frac*100:>5.1f}% method  n={n:<6} {pair[0]} x {pair[1]}")
        print(f"          {', '.join(f'{l} {v}' for l, v in c.most_common(3))}")
    print(f"\nField pairs MOST method-driven (top {top}):")
    for frac, n, pair, c in scored[-top:][::-1]:
        print(f"  {frac*100:>5.1f}% method  n={n:<6} {pair[0]} x {pair[1]}")
        print(f"          {', '.join(f'{l} {v}' for l, v in c.most_common(3))}")

    print("\nMost-cited work of each kind:")
    c = sqlite3.connect(f"file:{CITE_DB}?mode=ro", uri=True, timeout=60)
    titles = {}
    for k, t, n in c.execute("""
            SELECT CASE WHEN doi<>'' THEN doi
                        ELSE surname||'|'||year||'|'||LOWER(SUBSTR(title,1,80)) END,
                   MIN(title), COUNT(DISTINCT src)
            FROM cite WHERE title<>'' GROUP BY 1"""):
        titles[k] = (t, n)
    for lab in ("METHOD", "THEORY", "INSTRUMENT", "DATASET", "REVIEW",
                "FINDING", "REFERENCE"):
        best = sorted(((titles.get(k, ("", 0))[1], k) for k in have if have[k] == lab),
                      reverse=True)[:2]
        for n, k in best:
            print(f"  {lab:<11}{n:>5} theses, {len(wf[k]):>2} fields  "
                  f"{titles.get(k, ('', 0))[0][:56]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="/tmp/cls_all.jsonl")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    p = Path(a.labels)
    if not p.exists():
        sys.exit(f"no {p} — run the local-llm batch first")
    labels = load_labels(p)
    if not labels:
        sys.exit("no usable labels in that file")
    report(labels, work_fields(), a.top)


if __name__ == "__main__":
    sys.exit(main())
