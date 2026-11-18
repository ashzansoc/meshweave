"""A separate ASGI service: the OpenAI-compatible door in front of a cluster.

It deliberately does not embed the model or talk to a node. It finds the
deployment running on the cluster, submits a prompt through the coordinator's
queue, and waits for the result. That keeps the node's control surface closed:
nothing dials in, and the only process that can reach a stage's unix socket is
the agent on that host.

Two limitations are structural, not accidental:

  * one request per deployment at a time, because rank 0 serialises generation
    on its KV cache; a second request gets 429 rather than a queue that never
    drains
  * ``messages`` is flattened to a single prompt. The model has no chat template
    applied, so a multi-turn conversation is concatenated text, not a role-aware
    prompt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("meshweave.gateway")

COORDINATOR_ENV = "MESHWEAVE_GATEWAY_COORDINATOR"
API_KEY_ENV = "MESHWEAVE_GATEWAY_KEY"
DEFAULT_TIMEOUT_S = 600.0


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Turn an OpenAI message list into one prompt.

    Without a chat template this is the honest approximation: role markers in
    plain text. It is labelled in the docs rather than presented as templating.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):  # multimodal content blocks
            content = " ".join(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        if content is None:
            continue
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


class CoordinatorClient:
    """The only outbound calls the gateway makes."""

    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.request(
                method, f"{self.base_url}{path}", json=body
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail")
                except (json.JSONDecodeError, AttributeError):
                    detail = response.text[:300]
                raise RuntimeError(f"coordinator {response.status_code}: {detail}")
            return response.json() if response.content else {}

    async def deployments(self) -> list[dict]:
        payload = await self._request("GET", "/v1/deployments")
        return payload.get("deployments", [])

    async def get_deployment(self, deployment_id: str) -> dict:
        return await self._request("GET", f"/v1/deployments/{deployment_id}")

    async def submit_prompt(self, deployment_id: str, request: dict) -> dict:
        return await self._request("POST", f"/v1/deployments/{deployment_id}/prompt", request)


def create_app(
    coordinator: CoordinatorClient | None = None,
    api_key: str | None = None,
) -> FastAPI:
    coordinator = coordinator or CoordinatorClient(
        os.environ.get(COORDINATOR_ENV, "http://127.0.0.1:8700")
    )
    api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
    state = {
        "coordinator": coordinator,
        "api_key": api_key,
        "inflight": {},
        "log": [],
    }

    app = FastAPI(title="Float16 VDC gateway", version="0.1.0")
    # Exposed so an operator (or a test) can see what is in flight rather than
    # having to infer it from a 429.
    app.state.gateway = state

    # ------------------------------------------------------------------ auth
    def _authorize(authorization: str | None) -> None:
        if not state["api_key"]:
            return
        supplied = ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization.split(" ", 1)[1].strip()
        if supplied != state["api_key"]:
            raise HTTPException(
                status_code=401,
                detail="invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def _error(status_code: int, message: str, code: str = "invalid_request_error"):
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": "invalid_request_error", "code": code}},
        )

    # ---------------------------------------------------------------- routes
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "api_key_required": bool(state["api_key"])}

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict:
        _authorize(authorization)
        try:
            deployments = await coordinator.deployments()
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        models = []
        seen = set()
        for deployment in deployments:
            name = deployment.get("model")
            if not name or name in seen:
                continue
            seen.add(name)
            models.append(
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "float16-vdc",
                    "status": deployment.get("status"),
                }
            )
        return {"object": "list", "data": models}

    async def _resolve_deployment(requested: str | None) -> dict:
        deployments = await coordinator.deployments()
        if requested:
            matches = [
                d
                for d in deployments
                if d.get("model") == requested and d.get("status") in ("RUNNING", "LOADING")
            ]
        else:
            matches = [
                d for d in deployments if d.get("status") in ("RUNNING", "LOADING")
            ]
        if not matches:
            raise HTTPException(
                status_code=503,
                detail="no deployment is serving this model; deploy it on a cluster first",
            )
        return matches[0]

    async def _wait_for_result(
        deployment_id: str, prompt_id: str, timeout_s: float
    ) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            deployment = await coordinator.get_deployment(deployment_id)
            for result in deployment.get("results", []):
                if result.get("prompt_id") == prompt_id:
                    return result
            await asyncio.sleep(0.5)
        raise HTTPException(status_code=504, detail="generation timed out")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        _authorize(authorization)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "body must be valid JSON")

        model = body.get("model")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _error(400, "'messages' must be a non-empty array")
        prompt = flatten_messages(messages)

        deployment = await _resolve_deployment(model)
        deployment_id = deployment["deployment_id"]

        if deployment_id in state["inflight"]:
            return _error(
                429,
                f"{model} is already generating a request; this POC serves one at a time",
                code="rate_limit_exceeded",
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        state["inflight"][deployment_id] = request_id
        state["log"].append({"request_id": request_id, "model": model, "at": time.time()})
        state["log"] = state["log"][-200:]

        try:
            queued = await coordinator.submit_prompt(
                deployment_id,
                {
                    "prompt": prompt,
                    "max_new_tokens": int(body.get("max_tokens") or 256),
                    "temperature": float(body.get("temperature") or 0.0),
                    "top_p": float(body.get("top_p") or 1.0),
                },
            )
            prompt_id = queued.get("prompt_id")
            if not prompt_id:
                raise HTTPException(status_code=502, detail="coordinator did not accept the prompt")

            result = await _wait_for_result(
                deployment_id, prompt_id, float(body.get("timeout_s") or DEFAULT_TIMEOUT_S)
            )
        finally:
            state["inflight"].pop(deployment_id, None)

        if result.get("error"):
            return _error(502, str(result["error"]), code="model_error")

        prompt_tokens = int(result.get("prompt_tokens") or 0)
        output_tokens = int(result.get("output_tokens") or 0)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": deployment.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.get("text") or ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            },
            "meshweave": {
                "deployment_id": deployment_id,
                "stages": [
                    {
                        "node_id": stage.get("node_id"),
                        "layers": stage.get("layers"),
                        "phase": stage.get("phase"),
                    }
                    for stage in deployment.get("stages", [])
                ],
                "stage_roundtrip_ms": result.get("stage_roundtrip_ms"),
                "activation_bytes_per_token": result.get("activation_bytes_per_token"),
            },
        }

    @app.post("/v1/completions")
    async def completions(request: Request, authorization: str | None = Header(default=None)):
        _authorize(authorization)
        body = await request.json()
        prompt = str(body.get("prompt") or "")
        if not prompt:
            return _error(400, "'prompt' is required")

        deployment = await _resolve_deployment(body.get("model"))
        deployment_id = deployment["deployment_id"]
        if deployment_id in state["inflight"]:
            return _error(429, "model is already generating a request")

        request_id = f"cmpl-{uuid.uuid4().hex}"
        state["inflight"][deployment_id] = request_id
        try:
            queued = await coordinator.submit_prompt(
                deployment_id,
                {
                    "prompt": prompt,
                    "max_new_tokens": int(body.get("max_tokens") or 256),
                    "temperature": float(body.get("temperature") or 0.0),
                    "top_p": float(body.get("top_p") or 1.0),
                },
            )
            result = await _wait_for_result(
                deployment_id, queued["prompt_id"], float(body.get("timeout_s") or DEFAULT_TIMEOUT_S)
            )
        finally:
            state["inflight"].pop(deployment_id, None)

        if result.get("error"):
            return _error(502, str(result["error"]), code="model_error")

        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": deployment.get("model"),
            "choices": [
                {"index": 0, "text": result.get("text") or "", "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": int(result.get("prompt_tokens") or 0),
                "completion_tokens": int(result.get("output_tokens") or 0),
                "total_tokens": int(result.get("prompt_tokens") or 0)
                + int(result.get("output_tokens") or 0),
            },
        }

    return app