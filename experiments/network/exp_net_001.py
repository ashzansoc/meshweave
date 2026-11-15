"""EXP-NET-001: Actual host discovery.

Connects to the three Vast hosts, verifies SSH authentication, and collects a
hardware inventory from each. Every value comes from a command actually run on the
host; nothing is inferred from the Vast listing.

The distinction this experiment enforces: the provider *claims* a GPU, the host
*reports* a GPU. When they disagree, the report says so.

Usage:
    uv run python -m experiments.network.exp_net_001 --hosts all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from meshweave.remote import (  # noqa: E402
    CLAIMED_SPECS,
    HOSTS,
    RAW_DIR,
    Host,
    RemoteError,
    check_auth,
)

# Each entry: (key, command, human description)
INVENTORY_COMMANDS: list[tuple[str, str]] = [
    ("gpu_nvidia_smi", "nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || nvidia-smi"),
    ("gpu_uuid_short", "nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null || true"),
    ("cuda_nvcc", "nvcc --version 2>/dev/null || echo 'nvcc not found'"),
    ("cuda_driver_api", "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true"),
    ("os_uname", "uname -a"),
    ("os_release", "cat /etc/os-release 2>/dev/null | head -6 || true"),
    ("kernel", "uname -r"),
    ("python", "python3 --version 2>&1 || echo 'python3 missing'"),
    ("cpu_model", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//' || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown"),
    ("cpu_cores", "nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown"),
    ("ram", "free -h 2>/dev/null | head -3 || vm_stat 2>/dev/null | head -4 || true"),
    ("disk", "df -h / 2>/dev/null | tail -2 || true"),
    ("interfaces", "ip -o addr show 2>/dev/null || ifconfig -a 2>/dev/null || true"),
    ("routes", "ip route 2>/dev/null || netstat -rn 2>/dev/null | head -10 || true"),
    ("link_stats", "ip -s link 2>/dev/null || true"),
    ("listening", "ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null || true"),
    ("pci_gpus", "lspci 2>/dev/null | grep -iE 'vga|nvidia|3d' || echo 'lspci unavailable'"),
    # Co-location inputs. Missing values are fine; they are recorded as null.
    ("dmi_product", "dmidecode -s system-product-name 2>/dev/null || cat /sys/class/dmi/id/product_name 2>/dev/null || echo ''"),
    ("dmi_uuid", "dmidecode -s system-uuid 2>/dev/null || cat /sys/class/dmi/id/product_uuid 2>/dev/null || echo ''"),
    ("dmi_board_serial", "dmidecode -s baseboard-serial-number 2>/dev/null || cat /sys/class/dmi/id/board_serial 2>/dev/null || echo ''"),
    ("machine_id", "cat /etc/machine-id 2>/dev/null || echo ''"),
    ("boot_id", "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo ''"),
    ("virt", "systemd-detect-virt 2>/dev/null || echo unknown"),
    ("tun_device", "test -e /dev/net/tun && echo present || echo absent"),
    ("vast_env", "env | grep -E '^VAST_(TCP|UDP)_PORT_|^PUBLIC_IPADDR' | sort || echo 'no VAST_* env'"),
]


def _clean(value: str) -> str | None:
    """Normalise command output: empty or placeholder becomes None."""
    text = value.strip()
    if not text or text in {"", "unknown", "n/a"}:
        return None
    return text


def _parse_nvidia_smi(output: str) -> list[dict]:
    """Parse ``nvidia-smi --format=csv,noheader`` rows into structured facts.

    A failure to parse yields an empty list; the raw output is kept separately so
    nothing is silently lost.
    """
    gpus: list[dict] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "uuid": parts[1] if len(parts) > 1 else None,
                "driver_version": parts[2] if len(parts) > 2 else None,
                "memory_total": parts[3] if len(parts) > 3 else None,
                "memory_free": parts[4] if len(parts) > 4 else None,
                "temperature_c": parts[5] if len(parts) > 5 else None,
                "power_draw_w": parts[6] if len(parts) > 6 else None,
            }
        )
    return gpus


async def inspect(host: Host) -> dict:
    """Run the inventory for one host.

    Returns a dict with the collected facts, the raw output per command, and any
    command that failed. A failure is recorded, never dropped.
    """
    record: dict = {
        "node_id": host.node_id,
        "label": host.label,
        "endpoint": {"host": host.host, "port": host.port, "user": host.user},
        "claimed": CLAIMED_SPECS.get(host.node_id, {}),
        "collected_at": None,
        "commands": {},
        "failed_commands": [],
        "verified_gpus": [],
        "verification": {},
    }

    ok, detail = await check_auth(host)
    record["auth_ok"] = ok
    record["auth_detail"] = detail
    if not ok:
        record["error"] = "authentication failed"
        return record

    import datetime

    record["collected_at"] = datetime.datetime.now(datetime.UTC).isoformat()

    for key, command in INVENTORY_COMMANDS:
        try:
            result = await _run_quiet(host, command)
        except RemoteError as exc:
            record["failed_commands"].append({"key": key, "command": command, "error": str(exc)})
            continue

        record["commands"][key] = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(result.duration_s, 3),
        }

    # Structured verification from nvidia-smi, the authoritative source.
    smi_raw = record["commands"].get("gpu_nvidia_smi", {}).get("stdout", "")
    record["verified_gpus"] = _parse_nvidia_smi(smi_raw)

    claimed = CLAIMED_SPECS.get(host.node_id, {})
    claimed_name = str(claimed.get("gpu_name", "")).lower()
    claimed_vram = claimed.get("vram_gib")

    mismatches: list[str] = []
    for gpu in record["verified_gpus"]:
        reported = str(gpu.get("name", "")).lower()
        if claimed_name and reported and claimed_name not in reported and reported not in claimed_name:
            mismatches.append(
                f"gpu name: listing says {claimed_name!r}, nvidia-smi reports {reported!r}"
            )
    if not record["verified_gpus"]:
        mismatches.append("nvidia-smi reported no GPU")

    record["verification"] = {
        "claimed_gpu": claimed.get("gpu_name"),
        "claimed_vram_gib": claimed_vram,
        "reported_gpus": [g.get("name") for g in record["verified_gpus"]],
        "mismatches": mismatches,
        "matches_claim": not mismatches,
    }

    return record


async def _run_quiet(host: Host, command: str):
    """Run without raising on non-zero exit, so one bad command does not abort."""
    from meshweave.remote import run

    return await run(host, command, timeout_s=45.0, check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-NET-001 host discovery")
    parser.add_argument("--hosts", default="all", help="all, or comma-separated node ids")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.hosts == "all":
        targets = list(HOSTS)
    else:
        ids = [h.strip() for h in args.hosts.split(",")]
        targets = [h for h in HOSTS if h.node_id in ids]

    results = asyncio.run(_collect(targets))

    out = args.out or (RAW_DIR / "host_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'=' * 70}")
    print("EXP-NET-001: host discovery")
    print(f"{'=' * 70}")

    authenticated = [r for r in results if r.get("auth_ok")]
    blocked = [r for r in results if not r.get("auth_ok")]

    for record in results:
        print(f"\n{record['label']} ({record['node_id']})")
        print(f"  endpoint: {record['endpoint']['host']}:{record['endpoint']['port']}")
        if not record.get("auth_ok"):
            print(f"  AUTH FAILED: {record.get('auth_detail')}")
            continue

        for gpu in record["verified_gpus"]:
            print(f"  GPU: {gpu['name']}  uuid={gpu['uuid']}")
            print(f"       driver={gpu['driver_version']}  vram_total={gpu['memory_total']}")
        verification = record["verification"]
        print(f"  listing claims: {verification['claimed_gpu']}")
        if verification["mismatches"]:
            for m in verification["mismatches"]:
                print(f"  MISMATCH: {m}")
        else:
            print("  verified: reported GPU matches the listing")

        commands = record["commands"]
        print(f"  os: {_clean(commands.get('os_uname', {}).get('stdout', ''))}")
        print(f"  cpu: {_clean(commands.get('cpu_model', {}).get('stdout', ''))}")
        print(f"  tun: {_clean(commands.get('tun_device', {}).get('stdout', ''))}")
        if record["failed_commands"]:
            print(f"  failed commands: {len(record['failed_commands'])}")

    print(f"\nwrote {out}")
    print(f"authenticated: {len(authenticated)}/{len(results)}")
    if blocked:
        print("\nBLOCKED hosts:")
        for record in blocked:
            print(f"  {record['node_id']}: {record.get('auth_detail')}")
        print("\nSee docs/PHASE_2A_BLOCKED.md")

    return 0 if not blocked else 1


async def _collect(targets: list[Host]) -> list[dict]:
    return list(await asyncio.gather(*[inspect(h) for h in targets]))


if __name__ == "__main__":
    raise SystemExit(main())
