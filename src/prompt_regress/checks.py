"""Assertions on model output. Each check returns (passed, detail)."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .suite import SuiteError

Result = Tuple[bool, str]

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL | re.IGNORECASE)
_MISSING = object()


def parse_json(text: str, allow_fenced: bool = True) -> Tuple[Any, Optional[str]]:
    """Parse model output as JSON. Returns (value, error). Markdown ```json fences are accepted by default."""
    candidate = text.strip()
    if allow_fenced:
        match = _FENCE.match(candidate)
        if match:
            candidate = match.group(1)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc.msg} at line {exc.lineno} col {exc.colno})"


def get_path(data: Any, path: str) -> Any:
    """Read ``a.b.0.c`` from nested dicts/lists. Returns _MISSING when absent."""
    current = data
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return _MISSING
    return current


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _norm(text: str, spec: Dict[str, Any]) -> str:
    return text if spec.get("case_sensitive") else text.casefold()


def _need(spec: Dict[str, Any], key: str) -> Any:
    if key not in spec:
        raise SuiteError(f"check '{spec['type']}' needs '{key}'")
    return spec[key]


def check_contains(text, spec):
    values = _need(spec, "value")
    values = values if isinstance(values, list) else [values]
    missing = [v for v in values if _norm(str(v), spec) not in _norm(text, spec)]
    return (not missing, "contains all" if not missing else f"missing {', '.join(repr(m) for m in missing)}")


def check_not_contains(text, spec):
    values = _need(spec, "value")
    values = values if isinstance(values, list) else [values]
    found = [v for v in values if _norm(str(v), spec) in _norm(text, spec)]
    return (not found, "none present" if not found else f"found {', '.join(repr(f) for f in found)}")


def check_regex(text, spec):
    flags = 0 if spec.get("case_sensitive", True) else re.IGNORECASE
    pattern = _need(spec, "pattern")
    ok = re.search(pattern, text, flags | re.MULTILINE) is not None
    return ok, f"/{pattern}/ {'matched' if ok else 'not found'}"


def check_not_regex(text, spec):
    ok, _ = check_regex(text, spec)
    return (not ok, f"/{spec['pattern']}/ {'absent' if not ok else 'present'}")


def check_equals(text, spec):
    expected = str(_need(spec, "value"))
    actual = text.strip() if spec.get("strip", True) else text
    ok = _norm(actual, spec) == _norm(expected, spec)
    return ok, "equal" if ok else f"expected {_short(expected)!s}, got {_short(actual)!s}"


def check_one_of(text, spec):
    options = [str(v) for v in _need(spec, "value")]
    actual = text.strip()
    ok = any(_norm(actual, spec) == _norm(o, spec) for o in options)
    return ok, f"'{_short(actual)}' {'is' if ok else 'is not'} one of {options}"


def check_starts_with(text, spec):
    prefix = str(_need(spec, "value"))
    ok = _norm(text.lstrip(), spec).startswith(_norm(prefix, spec))
    return ok, f"{'starts' if ok else 'does not start'} with {prefix!r}"


def check_max_chars(text, spec):
    limit = int(_need(spec, "value"))
    return len(text) <= limit, f"{len(text)} chars (max {limit})"


def check_min_chars(text, spec):
    limit = int(_need(spec, "value"))
    return len(text) >= limit, f"{len(text)} chars (min {limit})"


def check_valid_json(text, spec):
    value, error = parse_json(text, spec.get("allow_fenced", True))
    if error:
        return False, error
    required = spec.get("required_keys") or []
    if required:
        if not isinstance(value, dict):
            return False, "JSON is not an object"
        missing = [k for k in required if k not in value]
        if missing:
            return False, f"missing keys: {', '.join(missing)}"
    return True, "valid JSON"


def check_json_field(text, spec):
    path = str(_need(spec, "path"))
    value, error = parse_json(text, spec.get("allow_fenced", True))
    if error:
        return False, error
    actual = get_path(value, path)
    if actual is _MISSING:
        return False, f"{path}: missing"
    if "equals" in spec:
        expected = spec["equals"]
        if isinstance(expected, str) and isinstance(actual, str) and not spec.get("case_sensitive"):
            ok = actual.casefold() == expected.casefold()
        else:
            ok = actual == expected
        return ok, f"{path} = {_short(actual)}" + ("" if ok else f" (expected {_short(expected)})")
    if "one_of" in spec:
        ok = actual in spec["one_of"]
        return ok, f"{path} = {_short(actual)}" + ("" if ok else f" (expected one of {spec['one_of']})")
    if "matches" in spec:
        ok = isinstance(actual, str) and re.search(spec["matches"], actual) is not None
        return ok, f"{path} = {_short(actual)}" + ("" if ok else f" (should match /{spec['matches']}/)")
    return True, f"{path} present"


CHECKS: Dict[str, Callable[[str, Dict[str, Any]], Result]] = {
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "not_regex": check_not_regex,
    "equals": check_equals,
    "one_of": check_one_of,
    "starts_with": check_starts_with,
    "max_chars": check_max_chars,
    "min_chars": check_min_chars,
    "valid_json": check_valid_json,
    "json_field": check_json_field,
}


def describe(spec: Dict[str, Any]) -> str:
    """Short human label for a check, used in reports."""
    if spec.get("label"):
        return str(spec["label"])
    kind = spec["type"]
    if kind == "json_field":
        for key in ("equals", "one_of", "matches"):
            if key in spec:
                return f"json_field {spec.get('path')} {key} {_short(spec[key], 40)}"
        return f"json_field {spec.get('path')} exists"
    for key in ("value", "pattern"):
        if key in spec:
            return f"{kind} {_short(spec[key], 40)}"
    return kind


def run_checks(text: str, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for spec in specs:
        kind = spec["type"]
        if kind not in CHECKS:
            raise SuiteError(f"Unknown check type '{kind}'. Available: {', '.join(sorted(CHECKS))}")
        passed, detail = CHECKS[kind](text, spec)
        results.append({"check": describe(spec), "type": kind, "passed": bool(passed), "detail": detail})
    return results


_REQUIRED = {
    "contains": "value", "not_contains": "value", "equals": "value", "one_of": "value",
    "starts_with": "value", "max_chars": "value", "min_chars": "value",
    "regex": "pattern", "not_regex": "pattern", "json_field": "path",
}


def validate_checks(specs: List[Dict[str, Any]]) -> None:
    """Fail before any API call if a suite uses an unknown check type, omits a field or has a bad regex."""
    for spec in specs:
        kind = spec["type"]
        if kind not in CHECKS:
            raise SuiteError(f"Unknown check type '{kind}'. Available: {', '.join(sorted(CHECKS))}")
        if kind in _REQUIRED:
            _need(spec, _REQUIRED[kind])
        for key in ("pattern", "matches"):
            if key in spec:
                try:
                    re.compile(spec[key])
                except re.error as exc:
                    raise SuiteError(f"check '{kind}': invalid regex {spec[key]!r} ({exc})") from exc
