"""The gateway must speak OpenAI's shape, including when it fails.

A client written against the OpenAI API breaks on an unexpected error shape, so
errors are checked here as carefully as successes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meshweave.gateway.app import CoordinatorClient, create_app, flatten_messages


class FakeCoordinator(CoordinatorClient):
    """Stands in for the coordinator, so no deployment is needed to test this."""

    def __init__(self, deployments, prompt_reply, result=None) -> None:
        super().__init__("http://unused")
        self._deployments = deployments
        self._prompt_reply = prompt_reply
        self._result = result or {
            "prompt": "hi",
            "text": "hello",
            "prompt_tokens": 3,
            "output_tokens": 2,
            "ms_per_token": 12.5,
            "ttft_ms": 40.0,
            "stage_roundtrip_ms": 500.0,
            "activation_bytes_per_token": 1792.0,
            "error": None,
        }
        self.submitted = []

    async def deployments(self):
        return self._deployments

    def _last_prompt_id(self):
        return self._prompt_reply.get("prompt_id") if isinstance(self._prompt_reply, dict) else "p-1"

    async def get_deployment(self, deployment_id: str):
        # The gateway polls until a result carries the id it was given, which is
        # what proves the request was answered rather than assumed.
        return {
            "deployment_id": deployment_id,
            "results": [dict(self._result, prompt_id=self._last_prompt_id())],
            "stages": [],
        }

    async def submit_prompt(self, deployment_id: str, request: dict):
        self.submitted.append((deployment_id, request))
        if isinstance(self._prompt_reply, Exception):
            raise self._prompt_reply
        return self._prompt_reply


DEPLOYMENTS = [
    {
        "deployment_id": "dep-1",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "status": "RUNNING",
        "stages": [
            {"node_id": "alpha", "layers": "[0,12)", "phase": "running"},
            {"node_id": "beta", "layers": "[12,24)", "phase": "running"},
        ],
    }
]


def _client(deployments=None, prompt_reply=None, api_key="", result=None):
    fake = FakeCoordinator(
        deployments if deployments is not None else DEPLOYMENTS,
        prompt_reply if prompt_reply is not None else {"prompt_id": "p-1"},
        result,
    )
    return TestClient(create_app(fake, api_key=api_key)), fake


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #


def test_messages_are_flattened_into_one_prompt() -> None:
    text = flatten_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert "system: be terse" in text
    assert text.endswith("assistant:")


# --------------------------------------------------------------------------- #
# responses
# --------------------------------------------------------------------------- #


def test_a_completion_has_the_openai_shape() -> None:
    client, fake = _client()
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["usage"]["completion_tokens"] == 2
    assert body["meshweave"]["stages"][1]["node_id"] == "beta"
    # The prompt really went to the cluster that owns the model.
    assert fake.submitted[0][0] == "dep-1"


def test_the_text_completion_endpoint_works_too() -> None:
    client, _ = _client()
    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "prompt": "once", "max_tokens": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "hello"


def test_models_are_listed_from_live_deployments() -> None:
    client, _ = _client()
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["Qwen/Qwen2.5-0.5B-Instruct"]
    assert body["data"][0]["owned_by"] == "float16-vdc"


def test_a_model_with_no_deployment_is_a_503_not_an_empty_answer() -> None:
    client, _ = _client()
    response = client.post(
        "/v1/chat/completions",
        json={"model": "some/other-model", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 503
    assert "no deployment" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


def test_bad_request_shapes_return_openai_errors() -> None:
    client, _ = _client()

    no_messages = client.post("/v1/chat/completions", json={"model": "m"})
    assert no_messages.status_code == 400
    assert no_messages.json()["error"]["type"] == "invalid_request_error"

    bad_json = client.post(
        "/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"}
    )
    assert bad_json.status_code == 400


def test_one_request_at_a_time_is_enforced() -> None:
    """Rank 0 serialises generation, so a second request cannot queue forever."""
    client, _ = _client()
    holder = client.app.state.gateway  # type: ignore[attr-defined]
    # Simulate a request already in flight.
    holder["inflight"]["dep-1"] = "chatcmpl-other"

    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    holder["inflight"].clear()


def test_auth_is_enforced_when_a_key_is_configured() -> None:
    client, _ = _client(api_key="secret")
    denied = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "x"}]},
    )
    assert denied.status_code == 401

    allowed = client.get("/v1/models", headers={"authorization": "Bearer secret"})
    assert allowed.status_code == 200