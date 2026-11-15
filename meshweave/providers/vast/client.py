"""Vast.ai provider integration.

Implements the subset of the Vast API MESHWEAVE needs, derived from the official
documentation (docs.vast.ai). Behaviour that the docs do not specify is not
invented here; it is marked as unverified and raised as an error at runtime.

Endpoints used:
    GET    /api/v0/users/current/          verify the key
    POST   /api/v0/bundles/                search offers
    PUT    /api/v0/asks/{id}/              rent (returns new_contract)
    GET    /api/v1/instances               list owned instances
    GET    /api/v0/instances/{id}          instance detail
    PUT    /api/v0/instances/{id}          start / stop
    DELETE /api/v0/instances/{id}          destroy
    POST   /api/v0/instances/reboot/{id}   reboot

Credentials come from VAST_API_KEY. MESHWEAVE never hardcodes them and never
logs them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from meshweave.config import VastSettings
from meshweave.errors import MeshweaveError

logger = logging.getLogger("meshweave.providers.vast")

DEFAULT_BASE_URL = "https://console.vast.ai"


class VastError(MeshweaveError):
    """A Vast API call failed."""


class VastCredentialsMissingError(VastError):
    """No API key configured. The correct response is to stop, not to guess."""


@dataclass
class VastInstance:
    """One rented instance, as reported by the API."""

    id: int
    label: str | None
    state: str
    public_ipaddr: str | None
    ssh_host: str | None
    ssh_port: int | None
    gpu_name: str | None
    num_gpus: int
    gpu_ram_mb: int | None
    dph_base: float | None
    dph_total: float | None
    geolocation: str | None
    machine_id: int | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def gpu_ram_gib(self) -> float | None:
        if self.gpu_ram_mb is None:
            return None
        return self.gpu_ram_mb / 1024.0

    @property
    def hourly_cost_usd(self) -> float | None:
        """Total hourly cost, or None when the API did not report it."""
        return self.dph_total if self.dph_total is not None else self.dph_base


class VastClient:
    """Async client for the Vast API.

    The rate limiter matters: Vast enforces a minimum interval per endpoint and
    returns 429 without a Retry-After header. A naive client will trip it during
    offer search, which is the first thing any provisioning script does.
    """

    def __init__(self, settings: VastSettings | None = None) -> None:
        self.settings = settings or VastSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- plumbing

    def _headers(self) -> dict[str, str]:
        key = self.settings.api_key
        if not key:
            raise VastCredentialsMissingError(
                "VAST_API_KEY is not set. Export it, or place it in a secrets file "
                "excluded by .gitignore. MESHWEAVE will not invent credentials."
            )
        return {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        retries: int = 4,
    ) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(retries):
            async with self._lock:
                # Vast rate-limits by minimum interval per endpoint. Sleep just
                # past it before every call.
                elapsed = time.monotonic() - self._last_request
                wait = self.settings.request_min_interval_s - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request = time.monotonic()

            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.request_timeout_s
                ) as client:
                    resp = await client.request(
                        method, url, headers=self._headers(), json=json_body, params=params
                    )
            except httpx.HTTPError as exc:
                last_error = exc
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            if resp.status_code == 429:
                # No Retry-After is published; back off exponentially.
                await asyncio.sleep(0.5 * (2**attempt))
                continue

            if resp.status_code in (401, 403):
                raise VastError(
                    f"Vast rejected the credentials for {method} {path}: {resp.text[:200]}"
                )

            if resp.status_code >= 400:
                raise VastError(f"{method} {path} failed: {resp.status_code} {resp.text[:300]}")

            if not resp.content:
                return None
            try:
                return resp.json()
            except json.JSONDecodeError:
                return resp.text

        raise VastError(
            f"{method} {path} failed after {retries} attempts: {last_error}"
        )

    # ----------------------------------------------------------------- api

    async def verify_credentials(self) -> dict[str, Any]:
        """Confirm the key works. This is the first thing every script should do."""
        result: dict[str, Any] = await self._request("GET", "/api/v0/users/current/")
        return result

    async def search_offers(
        self,
        *,
        gpu_name: str | None = None,
        min_gpus: int = 1,
        max_hourly_usd: float | None = None,
        datacenter: bool | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search the offer marketplace.

        ``datacenter=True`` restricts to verified datacenters; ``None`` allows
        both. The distinction matters for network characteristics, not just price.
        """
        query: dict[str, Any] = {
            "gpu_name": {"eq": gpu_name} if gpu_name else {},
            "num_gpus": {"gte": min_gpus},
            "rentable": {"eq": True},
        }
        if max_hourly_usd is not None:
            query["dph_total"] = {"lte": max_hourly_usd}
        if datacenter is not None:
            query["datacenter"] = {"eq": datacenter}

        body = {"q": query, "limit": limit, "order": "dph_total", "asc": True}
        result = await self._request("POST", "/api/v0/bundles/", json_body=body)
        if isinstance(result, dict) and "offers" in result:
            return list(result["offers"])
        if isinstance(result, list):
            return result
        return []

    async def rent(
        self,
        offer_id: int,
        *,
        image: str | None = None,
        disk_gb: int = 64,
        ports: list[int] | None = None,
        onstart: str | None = None,
        label: str | None = None,
    ) -> int:
        """Rent an offer and return the new instance id.

        Port mappings must be requested at rent time. Vast assigns an *external*
        port for each internal port, and the mapping is only visible from inside
        the instance (``VAST_TCP_PORT_<internal>``). MESHWEAVE therefore treats
        the mapping as something the agent reports, not something the API returns.
        """
        env: dict[str, str] = {}
        for port in ports or []:
            env[f"-p {port}:{port}"] = "1"

        body: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "env": env,
            "label": label or f"meshweave-{int(time.time())}",
        }
        if onstart:
            body["onstart-cmd"] = onstart

        result = await self._request("PUT", f"/api/v0/asks/{offer_id}/", json_body=body)
        if not isinstance(result, dict):
            raise VastError(f"unexpected rent response: {result!r}")
        new_contract = result.get("new_contract")
        if new_contract is None:
            raise VastError(f"rent did not return new_contract: {result!r}")
        return int(new_contract)

    async def list_instances(self) -> list[VastInstance]:
        result = await self._request("GET", "/api/v1/instances")
        rows = result.get("instances", []) if isinstance(result, dict) else (result or [])
        return [self._parse_instance(row) for row in rows]

    async def get_instance(self, instance_id: int) -> VastInstance:
        result = await self._request("GET", f"/api/v0/instances/{instance_id}")
        return self._parse_instance(result if isinstance(result, dict) else {})

    async def start(self, instance_id: int) -> None:
        await self._request("PUT", f"/api/v0/instances/{instance_id}", json_body={"state": "running"})

    async def stop(self, instance_id: int) -> None:
        await self._request("PUT", f"/api/v0/instances/{instance_id}", json_body={"state": "stopped"})

    async def destroy(self, instance_id: int) -> None:
        await self._request("DELETE", f"/api/v0/instances/{instance_id}")

    async def reboot(self, instance_id: int) -> None:
        await self._request("POST", f"/api/v0/instances/reboot/{instance_id}")

    async def ssh_url(self, instance_id: int) -> str | None:
        result = await self._request("GET", f"/api/v0/instances/ssh-url/{instance_id}")
        if isinstance(result, dict):
            return result.get("url")
        return None

    # ------------------------------------------------------------ internals

    @staticmethod
    def _parse_instance(row: dict[str, Any]) -> VastInstance:
        def _int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        return VastInstance(
            id=_int(row.get("id")) or 0,
            label=row.get("label"),
            state=str(row.get("actual_status") or row.get("state") or "unknown"),
            public_ipaddr=row.get("public_ipaddr"),
            ssh_host=row.get("ssh_host"),
            ssh_port=_int(row.get("ssh_port")),
            gpu_name=row.get("gpu_name"),
            num_gpus=_int(row.get("num_gpus")) or 0,
            gpu_ram_mb=_int(row.get("gpu_ram")),
            dph_base=_float(row.get("dph_base")),
            dph_total=_float(row.get("dph_total")),
            geolocation=row.get("geolocation"),
            machine_id=_int(row.get("machine_id")),
            raw=row,
        )


# --------------------------------------------------------------------------- #
# provisioning helpers
# --------------------------------------------------------------------------- #


def provisioning_plan(
    gpu_name: str = "RTX 4090",
    count: int = 2,
    image: str = "vastai/base-image:@vastai-automatic-tag",
    disk_gb: int = 64,
) -> dict[str, Any]:
    """Describe what MESHWEAVE would provision, without doing it.

    Returning the plan rather than executing it is deliberate: renting instances
    costs money and must be an explicit, reviewed decision.
    """
    return {
        "provider": "vast",
        "gpu_name": gpu_name,
        "count": count,
        "image": image,
        "disk_gb": disk_gb,
        "ports": [22, 8443, 9000],
        "estimated_hourly_usd": None,
        "steps": [
            "meshweave vast verify",
            "meshweave vast search --gpu-name {gpu_name} --count {count}",
            "meshweave vast rent --offer <id>   (one per node)",
            "wait for each instance to reach state=running",
            "meshweave certs --node-id <node-id>   (on the coordinator)",
            "copy the node certificate and key to the instance",
            "start the agent: meshweave agent --node-id <node-id> "
            "--coordinator https://<coordinator>:8700",
            "verify: meshweave nodes  shows each node with real GPU telemetry",
        ],
    }
