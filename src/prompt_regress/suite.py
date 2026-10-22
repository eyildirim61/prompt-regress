"""Load and validate test suites (YAML or JSON)."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_VAR = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class SuiteError(Exception):
    """Raised when a suite file is missing, malformed or inconsistent."""


@dataclass
class Case:
    id: str
    vars: Dict[str, Any] = field(default_factory=dict)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    repeat: int = 1


@dataclass
class Suite:
    name: str
    provider: Dict[str, Any]
    system: str
    user: str
    cases: List[Case]
    description: str = ""
    path: Optional[str] = None

    @property
    def prompt_hash(self) -> str:
        """Short fingerprint of the prompt templates, so a run records which prompt version it used."""
        digest = hashlib.sha256(f"{self.system}\x00{self.user}".encode("utf-8")).hexdigest()
        return digest[:12]

    def render(self, case: Case) -> Dict[str, str]:
        return {"system": render(self.system, case.vars, case.id), "user": render(self.user, case.vars, case.id)}


def render(template: str, variables: Dict[str, Any], case_id: str = "?") -> str:
    """Replace ``{{name}}`` placeholders. A missing variable is an error, not an empty string."""

    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in variables:
            raise SuiteError(f"case '{case_id}': template uses {{{{{key}}}}} but vars has no '{key}'")
        value = variables[key]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return _VAR.sub(repl, template or "")


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SuiteError(f"Suite file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path.name}: invalid JSON ({exc})") from exc
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SuiteError("YAML suites need PyYAML: pip install pyyaml (or use a .json suite)") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(f"{path.name}: invalid YAML ({exc})") from exc
    if not isinstance(data, dict):
        raise SuiteError(f"{path.name}: the top level must be a mapping")
    return data


def set_dotted(data: Dict[str, Any], dotted: str, raw_value: str) -> None:
    """Apply a ``--set provider.model=xyz`` override. Values are parsed as JSON when possible."""
    try:
        value: Any = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    keys = dotted.split(".")
    target = data
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def load_suite(path, overrides: Optional[List[str]] = None) -> Suite:
    path = Path(path)
    data = copy.deepcopy(_read(path))
    for item in overrides or []:
        if "=" not in item:
            raise SuiteError(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        set_dotted(data, key.strip(), value.strip())

    provider = data.get("provider")
    if not isinstance(provider, dict) or not provider.get("type"):
        raise SuiteError(f"{path.name}: 'provider' with a 'type' is required")

    # Relative file paths in the provider (e.g. mock responses) are relative to the suite file.
    if isinstance(provider.get("responses"), str):
        provider["responses"] = str((path.parent / provider["responses"]).resolve())

    prompt = data.get("prompt") or {}
    if not isinstance(prompt, dict) or not prompt.get("user"):
        raise SuiteError(f"{path.name}: 'prompt.user' is required")

    defaults = data.get("defaults") or {}
    default_checks = defaults.get("checks") or []
    default_repeat = int(defaults.get("repeat", 1))

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError(f"{path.name}: 'cases' must be a non-empty list")

    cases: List[Case] = []
    seen = set()
    for i, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise SuiteError(f"{path.name}: case #{i} must be a mapping")
        case_id = str(raw.get("id") or f"case-{i}")
        if case_id in seen:
            raise SuiteError(f"{path.name}: duplicate case id '{case_id}'")
        seen.add(case_id)
        checks = list(default_checks) + list(raw.get("checks") or [])
        for check in checks:
            if not isinstance(check, dict) or "type" not in check:
                raise SuiteError(f"{path.name}: case '{case_id}' has a check without a 'type'")
        repeat = int(raw.get("repeat", default_repeat))
        if repeat < 1:
            raise SuiteError(f"{path.name}: case '{case_id}' repeat must be >= 1")
        cases.append(Case(id=case_id, vars=dict(raw.get("vars") or {}), checks=checks, repeat=repeat))

    suite = Suite(
        name=str(data.get("name") or path.stem),
        description=str(data.get("description") or ""),
        provider=provider,
        system=str(prompt.get("system") or ""),
        user=str(prompt["user"]),
        cases=cases,
        path=str(path),
    )
    for case in suite.cases:  # fail fast on missing template variables
        suite.render(case)
    return suite
