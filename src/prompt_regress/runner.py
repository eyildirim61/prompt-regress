"""Run a suite against a provider and produce a run record (plain dict, saved as JSON)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .checks import run_checks, validate_checks
from .providers import Provider, make_provider
from .suite import Case, Suite

STATUSES = ("pass", "flaky", "fail", "error")


def case_status(attempts: List[Dict[str, Any]]) -> str:
    """pass: every attempt passed. fail: none did. flaky: some did. error: no attempt got an answer."""
    if all(a["error"] for a in attempts):
        return "error"
    passed = [a["passed"] for a in attempts]
    if all(passed):
        return "pass"
    if any(passed):
        return "flaky"
    return "fail"


def _attempt(provider: Provider, suite: Suite, case: Case, index: int) -> Dict[str, Any]:
    prompt = suite.render(case)
    response = provider.complete(prompt["system"], prompt["user"], case.id, index)
    if response.error:
        return {
            "output": "",
            "error": response.error,
            "latency_ms": response.latency_ms,
            "usage": response.usage,
            "checks": [],
            "passed": False,
        }
    results = run_checks(response.text, case.checks)
    return {
        "output": response.text,
        "error": None,
        "latency_ms": response.latency_ms,
        "usage": response.usage,
        "checks": results,
        "passed": all(r["passed"] for r in results),
    }


def run_suite(
    suite: Suite,
    provider: Optional[Provider] = None,
    concurrency: int = 4,
    progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    for case in suite.cases:
        validate_checks(case.checks)
    provider = provider or make_provider(suite.provider)

    jobs: List[Tuple[int, int]] = [(ci, ai) for ci, case in enumerate(suite.cases) for ai in range(case.repeat)]
    started = datetime.now(timezone.utc)

    def work(job: Tuple[int, int]) -> Tuple[int, int, Dict[str, Any]]:
        ci, ai = job
        return ci, ai, _attempt(provider, suite, suite.cases[ci], ai)

    attempts: Dict[int, Dict[int, Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for ci, ai, result in pool.map(work, jobs):
            attempts.setdefault(ci, {})[ai] = result

    cases = []
    for ci, case in enumerate(suite.cases):
        ordered = [attempts[ci][ai] for ai in range(case.repeat)]
        status = case_status(ordered)
        if progress:
            progress(case.id, status)
        cases.append(
            {
                "id": case.id,
                "vars": case.vars,
                "status": status,
                "pass_rate": round(sum(a["passed"] for a in ordered) / len(ordered), 3),
                "attempts": ordered,
            }
        )

    counts = {s: sum(1 for c in cases if c["status"] == s) for s in STATUSES}
    all_attempts = [a for c in cases for a in c["attempts"]]
    return {
        "tool": "prompt-regress",
        "version": __version__,
        "suite": suite.name,
        "description": suite.description,
        "started_at": started.isoformat(timespec="seconds"),
        "provider": provider.describe(),
        "prompt": {"system": suite.system, "user": suite.user, "hash": suite.prompt_hash},
        "summary": {
            "cases": len(cases),
            **counts,
            "attempts": len(all_attempts),
            "input_tokens": sum(a["usage"].get("input_tokens", 0) for a in all_attempts),
            "output_tokens": sum(a["usage"].get("output_tokens", 0) for a in all_attempts),
            "avg_latency_ms": int(sum(a["latency_ms"] for a in all_attempts) / max(1, len(all_attempts))),
        },
        "cases": cases,
    }


def save_run(run: Dict[str, Any], out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = run["started_at"].replace(":", "").replace("-", "").replace("+0000", "Z")[:15]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in run["suite"])
    path = out_dir / f"{safe}-{stamp}.json"
    n = 2
    while path.exists():
        path = out_dir / f"{safe}-{stamp}-{n}.json"
        n += 1
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_run(path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("tool") != "prompt-regress" or "cases" not in data:
        raise ValueError(f"{path} is not a prompt-regress run file")
    return data
