"""EXP-NET-011: Shared public IP / potential co-location investigation.

HOST-A and HOST-C share the public IP 1.208.108.242. This experiment determines
whether that means they share a physical machine.

It does not decide from the IP. It compares hardware identifiers that actually
distinguish physical machines, then issues a graded verdict:

    same-host confirmed  -- strong identifiers agree and RTT is local
    same-host suspected  -- strong identifiers agree but latency is unmeasured
    different-host       -- strong identifiers disagree
    undetermined         -- not enough comparable evidence

The experiment is deliberately conservative. Getting this wrong would be worse
than admitting we cannot tell: it would send a model's layers to a "local" pair
that is really across the internet.

Usage:
    uv run python -m experiments.network.exp_net_011
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshweave.network.colocation import (  # noqa: E402
    HardwareFingerprint,
    assess,
    compare_fingerprints,
)
from meshweave.remote import HOSTS, RAW_DIR, Host, RemoteError, run  # noqa: E402

FINGERPRINT_FIELDS: list[tuple[str, str]] = [
    ("system_vendor", "cat /sys/class/dmi/id/sys_vendor 2>/dev/null || echo ''"),
    ("system_product_name", "cat /sys/class/dmi/id/product_name 2>/dev/null || echo ''"),
    ("board_vendor", "cat /sys/class/dmi/id/board_vendor 2>/dev/null || echo ''"),
    ("board_name", "cat /sys/class/dmi/id/board_name 2>/dev/null || echo ''"),
    ("board_serial", "cat /sys/class/dmi/id/board_serial 2>/dev/null || echo ''"),
    ("system_uuid", "cat /sys/class/dmi/id/product_uuid 2>/dev/null || echo ''"),
    ("machine_id", "cat /etc/machine-id 2>/dev/null || echo ''"),
    ("boot_id", "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo ''"),
    ("kernel_version", "uname -r"),
    ("cpu_model", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//'"),
    ("hostname", "hostname"),
]


def _clean(value: str) -> str | None:
    text = value.strip()
    return text or None


async def fingerprint(host: Host) -> dict:
    """Collect one host's fingerprint."""
    record: dict = {
        "node_id": host.node_id,
        "label": host.label,
        "endpoint": {"host": host.host, "port": host.port},
    }

    values: dict[str, str | None] = {}
    for field_name, command in FINGERPRINT_FIELDS:
        try:
            result = await run(host, command, timeout_s=25.0, check=False)
            values[field_name] = _clean(result.stdout)
        except RemoteError as exc:
            record.setdefault("errors", []).append(f"{field_name}: {exc}")
            values[field_name] = None

    record["fingerprint"] = values

    # GPU UUIDs from nvidia-smi, for completeness.
    try:
        result = await run(
            host,
            "nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null",
            timeout_s=25.0,
            check=False,
        )
        record["gpu_uuids"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
    except RemoteError:
        record["gpu_uuids"] = []

    return record


def to_fingerprint(record: dict) -> HardwareFingerprint:
    fp = record["fingerprint"]
    return HardwareFingerprint(
        system_vendor=fp.get("system_vendor"),
        system_product_name=fp.get("system_product_name"),
        system_uuid=fp.get("system_uuid"),
        board_vendor=fp.get("board_vendor"),
        board_name=fp.get("board_name"),
        board_serial=fp.get("board_serial"),
        cpu_model=fp.get("cpu_model"),
        kernel_version=fp.get("kernel_version"),
        gpu_uuids=record.get("gpu_uuids", []),
        machine_id=fp.get("machine_id"),
        boot_id=fp.get("boot_id"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-NET-011 co-location investigation")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = asyncio.run(_collect())
    by_id = {r["node_id"]: r for r in records}

    print(f"\n{'=' * 72}")
    print("EXP-NET-011: shared public IP / co-location")
    print(f"{'=' * 72}")

    # Show the raw identifiers side by side. This is the evidence a reader needs
    # to judge the verdict, so it is printed before the verdict.
    fields = [
        "hostname",
        "system_vendor",
        "board_name",
        "board_serial",
        "kernel_version",
        "cpu_model",
        "boot_id",
        "machine_id",
    ]
    print("\nraw identifiers")
    print(f"  {'field':<18}" + "".join(f"{r['label']:<28}" for r in records))
    for field_name in fields:
        row = f"  {field_name:<18}"
        for record in records:
            value = record["fingerprint"].get(field_name)
            text = (value or "(missing)")[:26]
            row += f"{text:<28}"
        print(row)

    pairs: list[dict] = []
    ids = [r["node_id"] for r in records]
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            a_fp = to_fingerprint(by_id[a_id])
            b_fp = to_fingerprint(by_id[b_id])
            evidence = compare_fingerprints(a_fp, b_fp)
            verdict, reasons = assess(evidence)

            shared_ip = by_id[a_id]["endpoint"]["host"] == by_id[b_id]["endpoint"]["host"]
            pairs.append(
                {
                    "a": a_id,
                    "b": b_id,
                    "shared_public_ip": shared_ip,
                    "verdict": verdict,
                    "reasons": reasons,
                    "evidence": [
                        {
                            "name": e.name,
                            "supports_same_host": e.supports_same_host,
                            "detail": e.detail,
                            "weight": e.weight,
                        }
                        for e in evidence
                    ],
                }
            )

    print("\nverdicts")
    for pair in pairs:
        a_label = by_id[pair["a"]]["label"]
        b_label = by_id[pair["b"]]["label"]
        shared = "shares IP" if pair["shared_public_ip"] else "distinct IPs"
        print(f"\n  {a_label} <-> {b_label}  ({shared})")
        print(f"    verdict: {pair['verdict']}")
        for reason in pair["reasons"]:
            print(f"      - {reason}")

    # The headline finding for the shared-IP pair, stated plainly.
    shared_pairs = [p for p in pairs if p["shared_public_ip"]]
    if shared_pairs:
        pair = shared_pairs[0]
        print(f"\n{'=' * 72}")
        print("Shared-IP pair finding")
        print(f"{'=' * 72}")
        print(f"  {by_id[pair['a']]['label']} and {by_id[pair['b']]['label']} share "
              f"public IP {by_id[pair['a']]['endpoint']['host']}")
        print(f"  verdict: {pair['verdict']}")
        if pair["verdict"] == "different-host":
            print("  A shared public IP does NOT imply a shared physical machine.")
            print("  These are separate hosts reached through the same NAT/port map.")
            print("  They must not be treated as a fast local GPU pair for placement.")

    out = args.out or (RAW_DIR / "colocation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hosts": records,
        "pairs": pairs,
        "note": (
            "A shared public IP is recorded but is not evidence of co-location. "
            "machine_id is excluded as evidence because containers inherit it "
            "from the image. boot_id and board identifiers are the discriminators."
        ),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")

    return 0


async def _collect() -> list[dict]:
    return list(await asyncio.gather(*[fingerprint(h) for h in HOSTS]))


if __name__ == "__main__":
    raise SystemExit(main())
