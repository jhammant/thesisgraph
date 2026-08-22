#!/usr/bin/env python3
"""
tg.py — one entry point for the thesisgraph toolkit, built as a plugin registry
rather than as one big CLI file.

The toolkit is a dozen independent capabilities over the same corpus, and they
have very different appetites: one wants sqlite and nothing else, the next wants
torch and a 1.5 GB fingerprint index. Three properties follow from that, and all
three argue for a registry:

  * A new capability is a new file. Nothing central has to be edited to add one,
    so capabilities can be written in parallel without colliding on a shared
    dispatcher — which is exactly how this toolkit is being built.
  * A capability that is half-written, missing a dependency or plain broken
    degrades to one warning on stderr and everything else still runs. In a
    monolith the same mistake is an import error at the top of the file and it
    takes every subcommand down with it.
  * Every module stays runnable on its own (`python tg_foo.py --help`), so the
    scripts remain usable, testable and reviewable whether or not this file
    exists. This is a convenience layer, not a dependency.

The contract a capability module must satisfy, in full:

    NAME: str                                  # subcommand name, e.g. "similar"
    HELP: str                                  # one line, shown in `tg --help`
    def add_args(p: argparse.ArgumentParser) -> None
    def run(args: argparse.Namespace) -> int   # 0 on success, non-zero on failure

Modules are imported eagerly, in sorted filename order, so `tg --help` can show
a real HELP line for each and so a broken one is reported the moment anybody
runs anything rather than at the moment somebody needs it. The cost of that is
paid on every invocation, so keep module-level import cheap: pull heavy
dependencies (torch, sentence-transformers, embedding matrices) inside run().

Discovery is sorted, registration is sorted and the warnings follow discovery
order, so two runs over the same tree emit byte-identical output. Determinism is
a project value here and that includes the CLI.
"""

from __future__ import annotations

import argparse
import difflib
import importlib
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
VERSION = "1.0"
PREFIX = "tg_"


class Plugin(NamedTuple):
    name: str          # module.NAME — the subcommand
    help: str          # module.HELP — the one-liner
    stem: str          # module file stem, for error messages
    module: ModuleType


def discover(root: Path = HERE) -> list[Path]:
    """Every tg_*.py beside this file, sorted — discovery must be reproducible."""
    return sorted(p for p in root.glob(f"{PREFIX}*.py") if p.is_file())


def _contract_problem(mod: ModuleType) -> str | None:
    """Describe how a module breaks the plugin contract, or None if it keeps it."""
    name = getattr(mod, "NAME", None)
    if not isinstance(name, str) or not name.strip():
        return "no usable NAME (expected a non-empty str)"
    if name != name.strip() or " " in name or name.startswith("-"):
        return f"NAME {name!r} is not usable as a subcommand"
    if not isinstance(getattr(mod, "HELP", None), str) or not mod.HELP.strip():
        return "no usable HELP (expected a non-empty str)"
    for fn in ("add_args", "run"):
        if not callable(getattr(mod, fn, None)):
            return f"no {fn}() function"
    return None


def load(paths: list[Path]) -> tuple[list[Plugin], list[tuple[str, str]]]:
    """Import each path, returning the plugins that loaded and why the rest did not.

    Nothing raised by a plugin escapes this function. A module that explodes on
    import is a warning, not the end of the CLI.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    candidates: list[Plugin] = []
    problems: list[tuple[str, str]] = []

    for path in paths:
        stem = path.stem
        try:
            mod = importlib.import_module(stem)
        except (Exception, SystemExit) as exc:      # noqa: BLE001 — that is the point
            problems.append((stem, f"{type(exc).__name__}: {exc}"))
            continue

        bad = _contract_problem(mod)
        if bad:
            problems.append((stem, bad))
            continue

        # Exercise add_args on a throwaway parser: a module whose arguments do
        # not build is broken now, not when somebody first types its name.
        try:
            mod.add_args(argparse.ArgumentParser(add_help=False))
        except (Exception, SystemExit) as exc:      # noqa: BLE001
            problems.append((stem, f"add_args() failed: {type(exc).__name__}: {exc}"))
            continue

        candidates.append(Plugin(mod.NAME, mod.HELP.strip(), stem, mod))

    # Two modules can claim one subcommand. Resolve it on merit rather than on
    # alphabetical accident: tg_<NAME>.py is the file named for the command, so
    # it wins; otherwise the first in sorted order does. Either way the answer
    # is the same on every run.
    plugins: list[Plugin] = []
    for name in sorted({c.name for c in candidates}):
        rivals = sorted((c for c in candidates if c.name == name),
                        key=lambda c: (c.stem != f"{PREFIX}{name}", c.stem))
        plugins.append(rivals[0])
        for loser in rivals[1:]:
            problems.append((loser.stem, f"subcommand {name!r} is provided by "
                                         f"{rivals[0].stem}.py"))

    problems.sort()
    return plugins, problems


def build_parser(plugins: list[Plugin]) -> argparse.ArgumentParser:
    epilog = ("run `tg <command> --help` for a command's own options"
              if plugins else
              f"no {PREFIX}*.py capability modules were found next to {Path(__file__).name}")
    ap = argparse.ArgumentParser(
        prog="tg",
        description="thesisgraph — corpus and overlap tools over a thesis corpus.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"tg {VERSION} (thesisgraph)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress warnings about modules that failed to load")
    ap.add_argument("--traceback", action="store_true",
                    help="show the full traceback if a command raises")

    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    for pl in plugins:
        sp = sub.add_parser(pl.name, help=pl.help, description=pl.help)
        pl.module.add_args(sp)
        sp.set_defaults(_plugin=pl)
    return ap


def _exit_code(pl: Plugin, rc: object) -> int:
    """A plugin's return value, coerced to a process exit code."""
    if rc is None:                      # returning nothing means "fine"
        return 0
    if isinstance(rc, bool) or not isinstance(rc, int):
        print(f"tg: warning: {pl.stem}.py run() returned {rc!r}, not an int",
              file=sys.stderr)
        return 1
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Read early, by hand: load warnings are emitted before parse_args, because
    # --help exits inside it and a broken module should still be reported there.
    quiet = "-q" in argv or "--quiet" in argv

    plugins, problems = load(discover())
    if not quiet:
        for stem, why in problems:
            print(f"tg: warning: ignoring {stem}.py — {why}", file=sys.stderr)

    ap = build_parser(plugins)

    # argparse's "invalid choice" is unhelpful when the command exists but is
    # one of the modules that failed to load, or is simply a typo.
    if argv and not argv[0].startswith("-"):
        known = {p.name for p in plugins}
        if argv[0] not in known:
            print(f"tg: unknown command {argv[0]!r}", file=sys.stderr)
            near = difflib.get_close_matches(argv[0], sorted(known), n=3, cutoff=0.6)
            if near:
                print(f"tg: did you mean: {', '.join(near)}", file=sys.stderr)
            if any(stem == f"{PREFIX}{argv[0]}" for stem, _ in problems):
                print(f"tg: {PREFIX}{argv[0]}.py is present but failed to load "
                      f"(see the warning above)", file=sys.stderr)
            print("tg: run `tg --help` for the commands that are available",
                  file=sys.stderr)
            return 2

    args = ap.parse_args(argv)
    pl: Plugin | None = getattr(args, "_plugin", None)
    if pl is None:
        ap.print_help(sys.stderr)
        return 2

    try:
        rc = pl.module.run(args)
    except KeyboardInterrupt:
        print(f"tg: {pl.name} interrupted", file=sys.stderr)
        return 130
    except SystemExit as exc:               # a module that exits instead of returning
        if isinstance(exc.code, int):
            return exc.code
        if exc.code is not None:
            print(exc.code, file=sys.stderr)
            return 1
        return 0
    except Exception as exc:                # noqa: BLE001 — one bad command, not a crash
        if args.traceback:
            traceback.print_exc()
        print(f"tg: {pl.name} failed — {type(exc).__name__}: {exc}", file=sys.stderr)
        if not args.traceback:
            print("tg: re-run with --traceback for the full traceback", file=sys.stderr)
        return 1
    return _exit_code(pl, rc)


if __name__ == "__main__":
    sys.exit(main())
