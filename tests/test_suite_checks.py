import json

import pytest

from prompt_regress.checks import parse_json, run_checks, validate_checks
from prompt_regress.suite import SuiteError, load_suite, render, set_dotted


def write(tmp_path, text, name="suite.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


BASIC = """
name: demo
provider: {type: mock, responses: {a: "hello"}}
prompt:
  system: "Be brief."
  user: "Say {{word}}"
defaults:
  checks:
    - {type: min_chars, value: 1}
cases:
  - id: a
    vars: {word: hi}
    checks:
      - {type: contains, value: hello}
"""


# --- suite loading --------------------------------------------------------------------

def test_load_suite_merges_default_checks(tmp_path):
    suite = load_suite(write(tmp_path, BASIC))
    assert suite.name == "demo"
    assert [c["type"] for c in suite.cases[0].checks] == ["min_chars", "contains"]
    assert suite.render(suite.cases[0]) == {"system": "Be brief.", "user": "Say hi"}


def test_json_suite(tmp_path):
    data = {"provider": {"type": "mock"}, "prompt": {"user": "x"}, "cases": [{"id": "a"}]}
    suite = load_suite(write(tmp_path, json.dumps(data), "s.json"))
    assert suite.name == "s"


def test_missing_template_variable_fails_early(tmp_path):
    with pytest.raises(SuiteError, match="no 'word'"):
        load_suite(write(tmp_path, BASIC.replace("vars: {word: hi}", "vars: {}")))


def test_provider_is_required(tmp_path):
    text = BASIC.replace('provider: {type: mock, responses: {a: "hello"}}', "provider: {}")
    with pytest.raises(SuiteError, match="provider"):
        load_suite(write(tmp_path, text))


def test_cases_must_not_be_empty(tmp_path):
    text = BASIC.split("cases:")[0] + "cases: []\n"
    with pytest.raises(SuiteError, match="cases"):
        load_suite(write(tmp_path, text))


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(SuiteError, match="not found"):
        load_suite(tmp_path / "nope.yaml")
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(write(tmp_path, "a: [1, 2"))


def test_duplicate_case_ids(tmp_path):
    text = BASIC + "  - id: a\n    vars: {word: x}\n"
    with pytest.raises(SuiteError, match="duplicate"):
        load_suite(write(tmp_path, text))


def test_set_overrides_and_relative_paths(tmp_path):
    (tmp_path / "r.json").write_text('{"a": "x"}')
    suite = load_suite(write(tmp_path, BASIC), ["provider.model=small", "provider.responses=r.json", "provider.temperature=0.5"])
    assert suite.provider["model"] == "small"
    assert suite.provider["temperature"] == 0.5
    assert suite.provider["responses"] == str((tmp_path / "r.json").resolve())


def test_prompt_hash_changes_with_prompt(tmp_path):
    a = load_suite(write(tmp_path, BASIC))
    b = load_suite(write(tmp_path, BASIC), ["prompt.system=Be detailed."])
    assert a.prompt_hash != b.prompt_hash and len(a.prompt_hash) == 12


def test_render_non_string_values():
    assert render("x={{v}}", {"v": {"a": 1}}) == 'x={"a": 1}'


def test_set_dotted_creates_nested():
    d = {}
    set_dotted(d, "a.b.c", "3")
    assert d == {"a": {"b": {"c": 3}}}


# --- checks ---------------------------------------------------------------------------

def results(text, *specs):
    return [(r["passed"], r["detail"]) for r in run_checks(text, list(specs))]


def passed(text, spec):
    return run_checks(text, [spec])[0]["passed"]


def test_text_checks():
    assert passed("Hello World", {"type": "contains", "value": "hello"})
    assert not passed("Hello World", {"type": "contains", "value": "hello", "case_sensitive": True})
    assert passed("abc", {"type": "contains", "value": ["a", "c"]})
    assert not passed("abc", {"type": "not_contains", "value": ["x", "B"]})
    assert passed("  yes ", {"type": "equals", "value": "YES"})
    assert passed("Maybe", {"type": "one_of", "value": ["yes", "maybe"]})
    assert passed("Dear customer, hi", {"type": "starts_with", "value": "dear"})
    assert passed("12345", {"type": "max_chars", "value": 5})
    assert not passed("123456", {"type": "max_chars", "value": 5})
    assert passed("abc", {"type": "min_chars", "value": 3})
    assert passed("order 4821", {"type": "regex", "pattern": r"\d{4}"})
    assert passed("no digits", {"type": "not_regex", "pattern": r"\d"})


def test_contains_reports_what_is_missing():
    [(ok, detail)] = results("abc", {"type": "contains", "value": ["a", "z"]})
    assert not ok and "'z'" in detail


def test_parse_json_with_and_without_fences():
    assert parse_json('{"a": 1}') == ({"a": 1}, None)
    assert parse_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```', allow_fenced=False)[1] is not None
    assert parse_json("Sure! {\"a\": 1}")[1].startswith("not valid JSON")


def test_valid_json_and_required_keys():
    assert passed('{"a": 1, "b": 2}', {"type": "valid_json", "required_keys": ["a", "b"]})
    [(ok, detail)] = results('{"a": 1}', {"type": "valid_json", "required_keys": ["a", "b"]})
    assert not ok and "b" in detail
    assert not passed("[1, 2]", {"type": "valid_json", "required_keys": ["a"]})


def test_json_field():
    out = '{"category": "Refund", "items": [{"sku": "A-1"}], "escalate": true}'
    assert passed(out, {"type": "json_field", "path": "category", "equals": "refund"})
    assert not passed(out, {"type": "json_field", "path": "category", "equals": "refund", "case_sensitive": True})
    assert passed(out, {"type": "json_field", "path": "escalate", "equals": True})
    assert passed(out, {"type": "json_field", "path": "items.0.sku", "matches": r"^A-\d$"})
    assert passed(out, {"type": "json_field", "path": "category", "one_of": ["Refund", "Other"]})
    assert passed(out, {"type": "json_field", "path": "items"})
    [(ok, detail)] = results(out, {"type": "json_field", "path": "missing.key"})
    assert not ok and "missing" in detail


def test_custom_label():
    r = run_checks("x", [{"type": "contains", "value": "x", "label": "says x"}])
    assert r[0]["check"] == "says x"


def test_validate_checks():
    validate_checks([{"type": "contains", "value": "a"}, {"type": "valid_json"}])
    with pytest.raises(SuiteError, match="Unknown check"):
        validate_checks([{"type": "nope"}])
    with pytest.raises(SuiteError, match="needs 'value'"):
        validate_checks([{"type": "contains"}])
    with pytest.raises(SuiteError, match="needs 'path'"):
        validate_checks([{"type": "json_field"}])
    with pytest.raises(SuiteError, match="invalid regex"):
        validate_checks([{"type": "regex", "pattern": "("}])
