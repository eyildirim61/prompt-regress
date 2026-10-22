"""Command line: ``prompt-regress run`` and ``prompt-regress diff``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .compare import compare_runs
from .report import render_comparison, render_run, write
from .runner import load_run, run_suite, save_run
from .suite import SuiteError, load_suite

ICONS = {"pass": "✓", "fail": "✗", "flaky": "~", "error": "!"}


def _print_comparison(cmp, quiet: bool) -> None:
    c = cmp["counts"]
    print(f"Compared with baseline: {c['regressed']} regressed, {c['improved']} improved, "
          f"{c['changed']} output changed, {c['same']} unchanged, {c['added']} added, {c['removed']} removed")
    for ch in cmp["setup_changes"]:
        if ch["what"].startswith("prompt."):
            print(f"  setup: {ch['what']} changed")
        else:
            print(f"  setup: {ch['what']}: {ch['base']} -> {ch['new']}")
    if not quiet:
        for r in cmp["rows"]:
            if r["change"] in ("regressed", "improved"):
                extra = f"  now failing: {'; '.join(r['newly_failing'])}" if r["newly_failing"] else ""
                print(f"  {r['change']:<9} {r['id']}: {r['base_status']} -> {r['new_status']}{extra}")


def cmd_run(args) -> int:
    try:
        suite = load_suite(args.suite, args.set)
        progress = None if args.quiet else (lambda cid, st: print(f"  {ICONS[st]} {st:<5} {cid}"))
        if not args.quiet:
            print(f"Running {suite.name}: {len(suite.cases)} cases, provider {suite.provider.get('type')}"
                  f"{' / ' + str(suite.provider.get('model')) if suite.provider.get('model') else ''}")
        run = run_suite(suite, concurrency=args.concurrency, progress=progress)
    except SuiteError as exc:
        print(f"prompt-regress: {exc}", file=sys.stderr)
        return 2

    json_path = save_run(run, args.out)
    html_path = write(render_run(run), json_path.with_suffix(".html"))
    s = run["summary"]
    print(f"{s['pass']}/{s['cases']} pass, {s['fail']} fail, {s['flaky']} flaky, {s['error']} error"
          f" · {s['input_tokens'] + s['output_tokens']} tokens · avg {s['avg_latency_ms']} ms")
    print(f"Run:    {json_path}")
    print(f"Report: {html_path}")

    exit_code = 0
    if args.baseline:
        try:
            cmp = compare_runs(load_run(args.baseline), run)
        except (OSError, ValueError) as exc:
            print(f"prompt-regress: cannot read baseline: {exc}", file=sys.stderr)
            return 2
        cmp_path = write(render_comparison(cmp), json_path.with_name(json_path.stem + "-vs-baseline.html"))
        _print_comparison(cmp, args.quiet)
        print(f"Diff:   {cmp_path}")
        if args.fail_on_regression and cmp["counts"]["regressed"]:
            exit_code = 1
    if args.fail_on_failure and (s["fail"] or s["error"] or s["flaky"]):
        exit_code = 1
    return exit_code


def cmd_diff(args) -> int:
    try:
        base, new = load_run(args.base), load_run(args.new)
    except (OSError, ValueError) as exc:
        print(f"prompt-regress: {exc}", file=sys.stderr)
        return 2
    cmp = compare_runs(base, new)
    out = args.out or str(Path(args.new).with_name(Path(args.new).stem + "-vs-" + Path(args.base).stem + ".html"))
    path = write(render_comparison(cmp), out)
    _print_comparison(cmp, args.quiet)
    print(f"Diff:   {path}")
    return 1 if args.fail_on_regression and cmp["counts"]["regressed"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="prompt-regress", description="Regression tests for LLM prompts.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a suite and save the results")
    r.add_argument("suite", help="suite file (.yaml or .json)")
    r.add_argument("-o", "--out", default="runs", help="directory for run files (default: %(default)s)")
    r.add_argument("--baseline", help="earlier run JSON to compare against")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a suite setting, e.g. --set provider.model=other-model (repeatable)")
    r.add_argument("-j", "--concurrency", type=int, default=4, help="parallel requests (default: %(default)s)")
    r.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any case regressed vs baseline")
    r.add_argument("--fail-on-failure", action="store_true", help="exit 1 if any case is not passing")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("diff", help="compare two saved runs")
    d.add_argument("base", help="baseline run JSON")
    d.add_argument("new", help="new run JSON")
    d.add_argument("-o", "--out", help="HTML output path")
    d.add_argument("--fail-on-regression", action="store_true")
    d.add_argument("-q", "--quiet", action="store_true")
    d.set_defaults(func=cmd_diff)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
