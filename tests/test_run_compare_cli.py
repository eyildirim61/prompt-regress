import io
import json
import urllib.error

import pytest

from prompt_regress import providers
from prompt_regress.cli import main
from prompt_regress.compare import compare_runs, similarity, word_diff
from prompt_regress.providers import AnthropicProvider, MockProvider, OpenAICompatibleProvider, make_provider
from prompt_regress.report import render_comparison, render_run
from prompt_regress.runner import case_status, load_run, run_suite, save_run
from prompt_regress.suite import SuiteError, load_suite

SUITE = """
name: tiny
provider:
  type: mock
  model: m1
  responses: {responses}
prompt:
  user: "Classify: {{{{text}}}}"
cases:
  - id: good
    vars: {{text: a}}
    checks: [{{type: valid_json}}, {{type: json_field, path: label, equals: pos}}]
  - id: shaky
    repeat: 3
    vars: {{text: b}}
    checks: [{{type: contains, value: ok}}]
  - id: missing
    vars: {{text: c}}
    checks: [{{type: contains, value: x}}]
"""


def make_suite(tmp_path, responses, name="suite.yaml"):
    (tmp_path / "r.json").write_text(json.dumps(responses), encoding="utf-8")
    path = tmp_path / name
    path.write_text(SUITE.format(responses="r.json"), encoding="utf-8")
    return path


RESP_A = {"good": '{"label": "pos"}', "shaky": "ok"}
RESP_B = {"good": '{"label": "neg"}', "shaky": ["ok", "no", "ok"], "missing": "x"}


# --- runner ---------------------------------------------------------------------------

def test_case_status():
    a = lambda passed, error=None: {"passed": passed, "error": error}
    assert case_status([a(True), a(True)]) == "pass"
    assert case_status([a(False)]) == "fail"
    assert case_status([a(True), a(False)]) == "flaky"
    assert case_status([a(False, "boom")]) == "error"


def test_run_suite_statuses_and_summary(tmp_path):
    run = run_suite(load_suite(make_suite(tmp_path, RESP_B)))
    status = {c["id"]: c["status"] for c in run["cases"]}
    assert status == {"good": "fail", "shaky": "flaky", "missing": "pass"}
    shaky = next(c for c in run["cases"] if c["id"] == "shaky")
    assert shaky["pass_rate"] == pytest.approx(0.667, abs=0.001) and len(shaky["attempts"]) == 3
    assert run["summary"]["cases"] == 3 and run["summary"]["flaky"] == 1
    assert run["provider"] == {"type": "mock", "model": "m1", "responses": "r.json"}


def test_provider_error_becomes_error_status(tmp_path):
    run = run_suite(load_suite(make_suite(tmp_path, RESP_A)))
    missing = next(c for c in run["cases"] if c["id"] == "missing")
    assert missing["status"] == "error"
    assert "no response" in missing["attempts"][0]["error"]


def test_attempts_keep_order_with_concurrency(tmp_path):
    run = run_suite(load_suite(make_suite(tmp_path, RESP_B)), concurrency=8)
    shaky = next(c for c in run["cases"] if c["id"] == "shaky")
    assert [a["output"] for a in shaky["attempts"]] == ["ok", "no", "ok"]


def test_save_and_load_run(tmp_path):
    run = run_suite(load_suite(make_suite(tmp_path, RESP_A)))
    p1, p2 = save_run(run, tmp_path / "runs"), save_run(run, tmp_path / "runs")
    assert p1 != p2  # same timestamp does not overwrite
    assert load_run(p1)["suite"] == "tiny"
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError):
        load_run(tmp_path / "other.json")


# --- compare --------------------------------------------------------------------------

def test_compare_runs(tmp_path):
    base = run_suite(load_suite(make_suite(tmp_path, RESP_A)))
    new = run_suite(load_suite(make_suite(tmp_path, RESP_B), ["provider.model=m2"]))
    cmp = compare_runs(base, new)
    change = {r["id"]: r["change"] for r in cmp["rows"]}
    assert change == {"good": "regressed", "shaky": "regressed", "missing": "improved"}
    good = next(r for r in cmp["rows"] if r["id"] == "good")
    assert good["newly_failing"] == ["json_field label equals pos"]
    assert {"what": "provider.model", "base": "m1", "new": "m2"} in cmp["setup_changes"]
    assert cmp["rows"][0]["change"] == "regressed"  # regressions sort first


def test_compare_same_changed_added_removed(tmp_path):
    base = run_suite(load_suite(make_suite(tmp_path, RESP_A)))
    new = json.loads(json.dumps(base))
    new["cases"][1]["attempts"][0]["output"] = "ok!"  # still passes, different text
    new["cases"].pop(2)
    extra = json.loads(json.dumps(base["cases"][0]))
    extra["id"] = "brand-new"
    new["cases"].append(extra)
    change = {r["id"]: r["change"] for r in compare_runs(base, new)["rows"]}
    assert change == {"good": "same", "shaky": "changed", "missing": "removed", "brand-new": "added"}


def test_prompt_change_is_reported(tmp_path):
    base = run_suite(load_suite(make_suite(tmp_path, RESP_A)))
    new = run_suite(load_suite(make_suite(tmp_path, RESP_A), ["prompt.system=Be careful."]))
    assert [c["what"] for c in compare_runs(base, new)["setup_changes"]] == ["prompt.system"]


def test_similarity_and_word_diff():
    assert similarity("abc", "abc") == 1.0
    assert similarity("abc", "xyz") < 0.5
    assert word_diff("the cat sat", "the dog sat") == [("=", "the "), ("-", "cat"), ("+", "dog"), ("=", " sat")]


# --- reports --------------------------------------------------------------------------

def test_reports_escape_output(tmp_path):
    responses = {"good": "<script>alert(1)</script>", "shaky": "ok"}
    run = run_suite(load_suite(make_suite(tmp_path, responses)))
    page = render_run(run)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    page = render_comparison(compare_runs(run, run))
    assert "<script>alert(1)</script>" not in page


# --- providers ------------------------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def capture(monkeypatch, body, calls):
    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if isinstance(body, Exception):
            raise body
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)


def test_anthropic_request_and_parse(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []
    capture(monkeypatch, {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 5, "output_tokens": 2}}, calls)
    p = AnthropicProvider({"type": "anthropic", "model": "claude-x", "temperature": 0, "max_tokens": 50})
    r = p.complete("sys", "user msg", "c", 0)
    assert r.text == "hi" and r.usage == {"input_tokens": 5, "output_tokens": 2} and r.error is None
    req = calls[0]
    assert req.full_url == "https://api.anthropic.com/v1/messages"
    assert req.get_header("X-api-key") == "test-key"
    body = json.loads(req.data)
    assert body == {"model": "claude-x", "max_tokens": 50, "messages": [{"role": "user", "content": "user msg"}],
                    "system": "sys", "temperature": 0}
    assert "api_key" not in json.dumps(p.describe()) and "test-key" not in json.dumps(p.describe())


def test_anthropic_needs_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SuiteError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider({"type": "anthropic", "model": "m"})


def test_openai_compatible_local_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    capture(monkeypatch, {"choices": [{"message": {"content": "yo"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}, calls)
    p = OpenAICompatibleProvider({"type": "openai", "model": "llama3.1", "base_url": "http://localhost:11434/v1"})
    r = p.complete("", "hello", "c", 0)
    assert r.text == "yo" and r.usage == {"input_tokens": 3, "output_tokens": 1}
    assert calls[0].full_url == "http://localhost:11434/v1/chat/completions"
    assert json.loads(calls[0].data)["messages"] == [{"role": "user", "content": "hello"}]
    assert calls[0].get_header("Authorization") is None


def test_http_error_is_returned_not_raised(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    err = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"error": "bad model"}'))
    capture(monkeypatch, err, [])
    r = OpenAICompatibleProvider({"type": "openai", "model": "x", "retries": 0}).complete("", "q", "c", 0)
    assert r.error.startswith("HTTP 400") and "bad model" in r.error


def test_retry_then_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    state = {"n": 0}

    def flaky(request, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError("u", 429, "slow down", {}, io.BytesIO(b""))
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "done"}}]}).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", flaky)
    r = OpenAICompatibleProvider({"type": "openai", "model": "x"}).complete("", "q", "c", 0)
    assert r.text == "done" and state["n"] == 2


def test_mock_variants(tmp_path):
    assert MockProvider({"echo": True}).complete("", "same text", "x", 0).text == "same text"
    with pytest.raises(SuiteError):
        MockProvider({"responses": str(tmp_path / "missing.json")})
    with pytest.raises(SuiteError, match="Unknown provider"):
        make_provider({"type": "nope"})


# --- CLI ------------------------------------------------------------------------------

def test_cli_run_baseline_and_diff(tmp_path, capsys):
    suite = make_suite(tmp_path, RESP_A)
    out = tmp_path / "runs"
    assert main(["run", str(suite), "-o", str(out), "-q"]) == 0
    base = sorted(out.glob("*.json"))[0]
    (tmp_path / "r2.json").write_text(json.dumps(RESP_B))
    code = main(["run", str(suite), "-o", str(out), "--set", "provider.responses=r2.json",
                 "--baseline", str(base), "--fail-on-regression"])
    assert code == 1
    text = capsys.readouterr().out
    assert "2 regressed" in text and "1 improved" in text
    assert list(out.glob("*-vs-baseline.html"))

    new = sorted(p for p in out.glob("*.json") if p != base)[0]
    assert main(["diff", str(base), str(new), "-q"]) == 0
    assert main(["diff", str(base), str(new), "-q", "--fail-on-regression"]) == 1


def test_cli_errors(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.yaml")]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2


def test_cli_fail_on_failure(tmp_path):
    suite = make_suite(tmp_path, RESP_A)  # 'missing' case errors
    assert main(["run", str(suite), "-o", str(tmp_path / "o"), "-q", "--fail-on-failure"]) == 1


def test_example_suite_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "examples" / "support-triage" / "suite.yaml"
    suite = load_suite(example)
    run = run_suite(suite)
    assert run["summary"]["cases"] == 8 and run["summary"]["pass"] == 7
