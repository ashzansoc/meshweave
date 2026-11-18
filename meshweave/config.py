"""Central configuration.

Nothing sensitive is read from the source tree. Credentials come from the
environment or a secrets file that .gitignore excludes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class CoordinatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MESHWEAVE_COORD_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8700
    tls_cert: Path | None = None
    tls_key: Path | None = None
    # Agents without a certificate may only register if this is true. Off in prod.
    allow_insecure_registration: bool = False
    database_url: str = "sqlite:///./reports/meshweave.sqlite3"
    heartbeat_timeout_s: float = 15.0
    heartbeat_grace_s: float = 30.0

    # Open enrollment. When set, an agent presenting this shared token may
    # register itself without a coordinator-issued credential. This is what makes
    # "run one command on a new host" possible.
    #
    # It is a *bootstrap* secret, not a per-node identity: possession of it lets a
    # host join the cluster and report telemetry. It does not grant workload
    # execution rights -- those are still scoped per workload.
    #
    # Empty means enrollment is closed and only pre-authorised nodes may register.
    enrollment_token: str | None = None


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MESHWEAVE_AGENT_", env_file=".env", extra="ignore")

    coordinator_url: str = "https://127.0.0.1:8700"
    node_id: str | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None
    tls_ca: Path | None = None
    poll_interval_s: float = 5.0
    heartbeat_interval_s: float = 5.0
    # Set to true to run the agent against synthetic GPUs. Never for experiments.
    simulation: bool = False
    workload_root: Path = REPO_ROOT / "reports" / "workloads"


class VastSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VAST_", env_file=".env", extra="ignore")

    api_key: str | None = None
    base_url: str = "https://console.vast.ai"
    request_min_interval_s: float = 0.25
    request_timeout_s: float = 30.0

    @field_validator("api_key", mode="before")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "VAST_API_KEY is not set. Export it or place it in a secrets file "
                "excluded by .gitignore. MESHWEAVE will not invent credentials."
            )
        return self.api_key


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"
    coordinator: CoordinatorSettings = Field(default_factory=CoordinatorSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    vast: VastSettings = Field(default_factory=VastSettings)


def load_settings() -> Settings:
    return Settings()
