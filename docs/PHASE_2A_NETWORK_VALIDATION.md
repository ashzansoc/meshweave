# Phase 2A — Network Validation

Status: **partially complete.** Hardware discovery and co-location analysis are
done against real hosts. Direct TCP connectivity was measured on 2 of 6 paths
before an SSH outage interrupted the run.

Every result below is tagged by how it was obtained.

---

## 1. Host inventory (measured)

Verified from `nvidia-smi` and `/sys/class/dmi/`, not from the provider listing.

| | HOST-A | HOST-B | HOST-C |
|---|---|---|---|
| Node id | `meshweave-host-a` | `meshweave-host-b` | `meshweave-host-c` |
| Endpoint | `1.208.108.242:51400` | `83.63.1.187:24014` | `1.208.108.242:46079` |
| Container | `70d0b1aea811` | `2d6e3ce8541f` | `5c00204b15f7` |
| GPU | GTX 1660 SUPER | **Quadro** RTX 4000 | RTX 2070 SUPER |
| VRAM | 6 GiB | 8 GiB | 8 GiB |
| GPU UUID | `GPU-b921eed5…` | `GPU-ce05c91f…` | `GPU-8490e6e2…` |
| Driver | 580.159.03 | 595.99.02 | 580.178.04 |
| CPU | Xeon E5-2630 v4 | Xeon W-2123 | Xeon E5-1650 v3 |
| Kernel | 5.15.0-186 | 7.0.0-42 | 6.8.0-147 |
| Motherboard | MACHINIST X99-D8 MAX | HP 81C5 | Supermicro X10SRA |
| Board serial | MB-202509082025 | PFPPQ0MCYEM4JM | ZM158S037512 |

Aggregate: **22 GiB VRAM across 3 GPUs on 3 physical machines.**

## 2. GPU verification (measured)

The provider listing for HOST-B says "NVIDIA RTX 4000". `nvidia-smi` reports
**"Quadro RTX 4000"**. These are different products with different memory
bandwidth and driver branches. The listing is imprecise; the host is correct.

All other GPUs matched their listing.

## 3. Co-location: HOST-A and HOST-C (measured)

They share public IP `1.208.108.242`, which raises the question of whether they
are one physical machine. **They are not.**

| Evidence | Verdict |
|---|---|
| Motherboard | MACHINIST X99-D8 MAX vs Supermicro X10SRA — **different** |
| Board serial | MB-202509082025 vs ZM158S037512 — **different** |
| Kernel | 5.15.0-186 vs 6.8.0-147 — **different boot sessions** |
| CPU | E5-2630 v4 vs E5-1650 v3 — **different** |
| `/etc/machine-id` | **identical** — but this is not evidence |

**Verdict: `different-host`.** A shared public IP means these two containers are
reached through the same NAT and port map. They are separate machines.

The identical `machine-id` is the trap: containers inherit it from the image, so
it is shared by every instance created from the same Vast image regardless of
which physical host it lands on. An earlier version of the classifier treated a
machine-id match as strong evidence of co-location and would have produced the
wrong answer. That was corrected, and a test now locks it in
(`test_shared_ip_is_not_evidence_by_itself`).

### Why this matters for placement

A and C are **not** a fast local pair. They are two machines whose traffic
crosses a NAT and the public internet, even though they share an IP.

## 4. SSH connectivity (measured)

All three hosts authenticated with `id_rsa_vastai` after the key was installed.

```
HOST-A  AUTH_OK  70d0b1aea811
HOST-B  AUTH_OK  2d6e3ce8541f
HOST-C  AUTH_OK  5c00204b15f7
```

**Currently down.** See section 9.

## 5. Direct TCP connectivity (measured, 2 of 6 paths)

Measured host-to-host, no SSH tunnels, using Vast's existing port mappings.
Listeners ran as foreground processes inside held SSH sessions, because
backgrounded processes (`nohup`, and even `tmux`) were reaped on two of three
hosts.

| Path | Result | Connect | Throughput | RTT |
|---|---|---|---|---|
| C → A | **works** | 2.3 ms | 79.4 Mbps | 6.60 ms |
| B → A | **works** | 307 ms | 1.5 Mbps | 342 ms |
| A → B | not measured | — | — | — |
| A → C | not measured | — | — | — |
| B → C | not measured | — | — | — |
| C → B | not measured | — | — | — |

The two working paths differ by a factor of 50 in latency, which is itself a
finding: the link quality is not uniform even between the same pairs of machines
in the same run.

The other four paths returned `ConnectionResetError` and `ConnectionRefusedError`
in the run that completed. Those failures are attributable to listeners that had
been reaped mid-run, not to the network, and the measurement must be repeated
before any conclusion is drawn about them.

### Constraint discovered

Vast maps only the ports requested at rent time. There is no API to open a new
one, and the external port is visible only from inside the container
(`VAST_TCP_PORT_<internal>`). The first attempt used internal port 9100, which is
not mapped on any host, so no path could work. Port 10100 is a Vast default
mapping present and free on all three and is what the successful runs used.

## 6. Private network: not deployed

`/dev/net/tun` is **absent on all three hosts** (measured). That removes the two
overlay options that would otherwise be considered:

- WireGuard needs `/dev/net/tun`.
- SSH TUN (`ssh -w`) needs TUN support plus `PermitTunnel` on the server.

Per the approved plan, no overlay software is installed until direct TCP is
characterised. Direct TCP now has partial evidence of working, which argues for
resolving it rather than building an overlay.

Whether UDP is reachable at all is **not yet measured**, so no conclusion is drawn
about WireGuard's viability on these hosts.

## 7. Not yet measured

- Four of six direct TCP paths (blocked by the SSH outage).
- RTT percentiles and jitter.
- Bandwidth across payload sizes.
- Persistent connection overhead, including TLS.
- Reconnection behaviour.
- UDP reachability.
- Actual GPU-to-GPU throughput between hosts.

## 8. Recommended topology (provisional, pending measurements)

On current evidence:

- **HOST-A** has the smallest GPU (6 GiB) but the lowest-latency inbound path
  (C → A at 6.6 ms RTT). It is a candidate for a pipeline *stage*, not a replica.
- **HOST-C** is the best-balanced node: 8 GiB, fast to A.
- **HOST-B** is the outlier: 342 ms RTT to A on the one measurement taken. If that
  holds, B should not sit in the middle of a pipeline.

This is provisional and must not drive placement until all six paths are measured
in both directions.

## 9. Incident: SSH outage

During P2 an automated cleanup routine killed `sshd` on all three hosts.

The cause was a defect I introduced: a helper intended to stop a stale listener on
one port parsed every `pid=` from `ss -tlnp` and killed them whenever the target
port appeared anywhere in the output. It terminated every listening process,
including the SSH daemon.

All three hosts now refuse SSH. The instances themselves are unaffected; only
sshd is down.

**Recovery required:** restart sshd from the Vast console, or stop and start each
instance. The data disk survives a stop/start.

The defective code has been removed, not patched. A measurement tool must not kill
processes on a host to obtain a port; the server now reuses an existing listener
and never terminates one it did not start.

This was the second process-management error in this phase. Earlier,
`pkill -f mw_echo` matched the SSH command line itself and killed its own session.
Both are the same mistake: acting on a name pattern instead of a specific PID.