"""Every documented POST body must actually be a body.

FastAPI resolves a handler's annotation against module globals. With
``from __future__ import annotations`` a model declared *inside* ``create_app``
becomes an unresolvable string, and FastAPI silently treats it as a required
query parameter named ``request``. The endpoint then fails every call with a 422
that mentions a field the caller never sent.

That happened twice while building deployments and had to be fixed twice. This
test fails for the whole class of bug rather than for one route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meshweave.config import CoordinatorSettings
from meshweave.coordinator import app as coordinator_app

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    async def fake_config(repo: str) -> str:
        return json.dumps({})

    monkeypatch.setattr(coordinator_app, "fetch_model_config", fake_config)
    settings = CoordinatorSettings(port=8797, deployments_file=tmp_path / "deployments.json")
    return TestClient(coordinator_app.create_app(settings))


def test_no_body_model_is_read_as_a_query_parameter(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    offenders: list[str] = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if method not in {"post", "put", "patch"}:
                continue
            for parameter in operation.get("parameters", []):
                if parameter.get("in") == "query" and parameter.get("name") == "request":
                    offenders.append(f"{method.upper()} {path}")
    assert not offenders, (
        "these routes treat their body model as a query parameter, which means "
        f"the model is declared inside create_app: {offenders}"
    )


def test_no_request_model_is_declared_inside_the_app_factory() -> None:
    """A source-level check for the same mistake, so it is caught without a client."""
    import ast

    tree = ast.parse((REPO / "meshweave" / "coordinator" / "app.py").read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_app":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "BaseModel" for base in child.bases
            ):
                offenders.append(child.name)
    assert not offenders, f"BaseModel subclasses declared inside create_app: {offenders}"
