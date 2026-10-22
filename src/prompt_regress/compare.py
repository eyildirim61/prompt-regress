"""Compare two runs: what broke, what got fixed, what changed."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List

# Higher is better. A move to a lower rank is a regression, to a higher rank an improvement.
RANK = {"error": 0, "fail": 1, "flaky": 2, "pass": 3}


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 3)


def _first_output(case: Dict[str, Any]) -> str:
    for attempt in case["attempts"]:
        if not attempt["error"]:
            return attempt["output"]
    return ""


def _failed_checks(case: Dict[str, Any]) -> List[str]:
    failed = []
    for attempt in case["attempts"]:
        for check in attempt["checks"]:
            if not check["passed"] and check["check"] not in failed:
                failed.append(check["check"])
    return failed


def compare_runs(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    base_cases = {c["id"]: c for c in base["cases"]}
    new_cases = {c["id"]: c for c in new["cases"]}

    rows = []
    for case_id in list(new_cases) + [cid for cid in base_cases if cid not in new_cases]:
        b, n = base_cases.get(case_id), new_cases.get(case_id)
        if b is None:
            rows.append({"id": case_id, "change": "added", "base_status": None, "new_status": n["status"],
                         "new_output": _first_output(n), "base_output": "", "similarity": None,
                         "newly_failing": _failed_checks(n), "now_passing": []})
            continue
        if n is None:
            rows.append({"id": case_id, "change": "removed", "base_status": b["status"], "new_status": None,
                         "base_output": _first_output(b), "new_output": "", "similarity": None,
                         "newly_failing": [], "now_passing": []})
            continue

        b_out, n_out = _first_output(b), _first_output(n)
        sim = similarity(b_out, n_out)
        b_rank, n_rank = RANK[b["status"]], RANK[n["status"]]
        if n_rank < b_rank:
            change = "regressed"
        elif n_rank > b_rank:
            change = "improved"
        elif sim < 1.0:
            change = "changed"
        else:
            change = "same"
        b_failed, n_failed = _failed_checks(b), _failed_checks(n)
        rows.append({
            "id": case_id,
            "change": change,
            "base_status": b["status"],
            "new_status": n["status"],
            "base_output": b_out,
            "new_output": n_out,
            "similarity": sim,
            "newly_failing": [c for c in n_failed if c not in b_failed],
            "now_passing": [c for c in b_failed if c not in n_failed],
        })

    order = {"regressed": 0, "improved": 1, "changed": 2, "added": 3, "removed": 4, "same": 5}
    rows.sort(key=lambda r: order[r["change"]])
    counts = {k: sum(1 for r in rows if r["change"] == k) for k in order}

    setup_changes = []
    for key in sorted(set(base["provider"]) | set(new["provider"])):
        if base["provider"].get(key) != new["provider"].get(key):
            setup_changes.append({"what": f"provider.{key}", "base": base["provider"].get(key),
                                  "new": new["provider"].get(key)})
    if base["prompt"]["hash"] != new["prompt"]["hash"]:
        for part in ("system", "user"):
            if base["prompt"][part] != new["prompt"][part]:
                setup_changes.append({"what": f"prompt.{part}", "base": base["prompt"][part],
                                      "new": new["prompt"][part]})

    return {
        "suite": new["suite"],
        "base": {"started_at": base["started_at"], "summary": base["summary"], "prompt_hash": base["prompt"]["hash"],
                 "provider": base["provider"]},
        "new": {"started_at": new["started_at"], "summary": new["summary"], "prompt_hash": new["prompt"]["hash"],
                "provider": new["provider"]},
        "setup_changes": setup_changes,
        "counts": counts,
        "rows": rows,
    }


def word_diff(a: str, b: str) -> List[tuple]:
    """Token-level diff as (op, text) pairs with op in {'=', '-', '+'}; whitespace is kept as tokens."""
    import re

    ta, tb = re.findall(r"\s+|\w+|[^\w\s]", a), re.findall(r"\s+|\w+|[^\w\s]", b)
    out: List[tuple] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(("=", "".join(ta[i1:i2])))
        else:
            if i2 > i1:
                out.append(("-", "".join(ta[i1:i2])))
            if j2 > j1:
                out.append(("+", "".join(tb[j1:j2])))
    return out
