"""Model providers. Only the standard library is used for HTTP, so there is no SDK dependency."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .suite import SuiteError


@dataclass
class Response:
    text: str = ""
    latency_ms: int = 0
    error: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


class Provider:
    """Base class. ``complete`` must never raise for API problems; it returns Response.error instead."""

    name = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    def model(self) -> str:
        return str(self.config.get("model", ""))

    def describe(self) -> Dict[str, Any]:
        """What gets recorded in a run, so two runs can be compared. Never includes secrets."""
        keep = ("type", "model", "temperature", "max_tokens", "base_url", "top_p")
        return {k: self.config[k] for k in keep if k in self.config}

    def complete(self, system: str, user: str, case_id: str, attempt: int) -> Response:  # pragma: no cover
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic provider for demos and tests.

    ``responses`` maps case id to a string, or to a list of strings used in turn for repeated attempts.
    It can be a mapping in the suite or a path to a JSON file. ``echo: true`` returns the user prompt.
    """

    name = "mock"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        responses = config.get("responses", {})
        if isinstance(responses, str):
            path = Path(responses)
            if not path.exists():
                raise SuiteError(f"mock responses file not found: {path}")
            responses = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(responses, dict):
            raise SuiteError("mock 'responses' must be a mapping of case id to text")
        self.responses: Dict[str, Any] = responses

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        source = self.config.get("responses")
        if isinstance(source, str):
            info["responses"] = Path(source).name
        return info

    def complete(self, system, user, case_id, attempt):
        if self.config.get("echo"):
            return Response(text=user)
        if case_id not in self.responses:
            return Response(error=f"mock has no response for case '{case_id}'")
        value = self.responses[case_id]
        if isinstance(value, list):
            value = value[attempt % len(value)]
        return Response(text=str(value))


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class HTTPProvider(Provider):
    retry_statuses = (429, 500, 502, 503, 504, 529)

    def _call(self, url, payload, headers) -> Response:
        retries = int(self.config.get("retries", 2))
        timeout = float(self.config.get("timeout", 60))
        delay = 1.0
        start = time.monotonic()
        for attempt in range(retries + 1):
            try:
                data = _post_json(url, payload, headers, timeout)
                response = self._parse(data)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                return response
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in self.retry_statuses and attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return Response(error=f"HTTP {exc.code}: {body}", latency_ms=int((time.monotonic() - start) * 1000))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return Response(error=f"connection error: {exc}", latency_ms=int((time.monotonic() - start) * 1000))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                return Response(error=f"unexpected response format: {exc}")
        return Response(error="no response")  # pragma: no cover

    def _parse(self, data: Dict[str, Any]) -> Response:  # pragma: no cover
        raise NotImplementedError

    def _key(self, env_default: str, required: bool = True) -> str:
        env_name = self.config.get("api_key_env", env_default)
        key = os.environ.get(env_name, "")
        if required and not key:
            raise SuiteError(f"Set the {env_name} environment variable to use the '{self.name}' provider.")
        return key


class AnthropicProvider(HTTPProvider):
    name = "anthropic"

    def __init__(self, config):
        super().__init__(config)
        if not self.model:
            raise SuiteError("anthropic provider needs a 'model'")
        self.api_key = self._key("ANTHROPIC_API_KEY")
        self.base_url = str(config.get("base_url", "https://api.anthropic.com")).rstrip("/")

    def complete(self, system, user, case_id, attempt):
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(self.config.get("max_tokens", 1024)),
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            payload["system"] = system
        for key in ("temperature", "top_p"):
            if key in self.config:
                payload[key] = self.config[key]
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        return self._call(f"{self.base_url}/v1/messages", payload, headers)

    def _parse(self, data):
        text = "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        usage = data.get("usage") or {}
        return Response(
            text=text,
            usage={"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)},
        )


class OpenAICompatibleProvider(HTTPProvider):
    """OpenAI Chat Completions API, and anything compatible with it (e.g. Ollama, vLLM, LM Studio)."""

    name = "openai"

    def __init__(self, config):
        super().__init__(config)
        if not self.model:
            raise SuiteError("openai provider needs a 'model'")
        self.base_url = str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        local = self.base_url.startswith(("http://localhost", "http://127.0.0.1"))
        self.api_key = self._key("OPENAI_API_KEY", required=not local)

    def complete(self, system, user, case_id, attempt):
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload: Dict[str, Any] = {"model": self.model, "messages": messages}
        for key in ("temperature", "top_p", "max_tokens"):
            if key in self.config:
                payload[key] = self.config[key]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self._call(f"{self.base_url}/chat/completions", payload, headers)

    def _parse(self, data):
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        return Response(
            text=text,
            usage={"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
        )


PROVIDERS = {"mock": MockProvider, "anthropic": AnthropicProvider, "openai": OpenAICompatibleProvider}


def make_provider(config: Dict[str, Any]) -> Provider:
    kind = str(config.get("type", "")).lower()
    if kind not in PROVIDERS:
        raise SuiteError(f"Unknown provider type '{kind}'. Use one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[kind](config)
