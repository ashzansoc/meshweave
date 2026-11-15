import { Card, Stat, Measured, EmptyState, Badge, ErrorBanner } from '../lib/format'
import { usePolling } from '../lib/api'

interface NetworkPath {
  source: string
  target: string
  endpoint: string | null
  rtt_p50_ms: number | null
  rtt_p95_ms: number | null
  rtt_p99_ms: number | null
  jitter_ms: number | null
  packet_loss_pct: number | null
    sustained_mbps: number | null
  measured: boolean
}

interface NetworkData {
  measured: boolean
  paths: NetworkPath[]
  topology: { nodes: string[] }
  endpoints: Record<string, [string, number]>
  "co-location": {
    available: boolean
    pairs?: Array<{ a: string; b: string; shared_public_ip: boolean; verdict: string }>
    network_adjacency?: Record<string, Record<string, number>>
    host_count?: number
  }
}

/** Colour a path by its measured latency class. */
function latencyTone(p: NetworkPath): { label: string; cls: string } {
  if (!p.measured || p.rtt_p50_ms === null) return { label: 'unmeasured', cls: 'text-neutral-500' }
  if (p.rtt_p50_ms <= 5) return { label: 'low latency', cls: 'text-emerald-400' }
  if (p.rtt_p50_ms <= 50) return { label: 'moderate', cls: 'text-amber-400' }
  return { label: 'high latency', cls: 'text-red-400' }
}

export function Network() {
  const net = usePolling<NetworkData>('/v1/network', 10000)

  if (net.error) {
    return (
      <div className="space-y-5">
        <h2 className="text-lg font-medium text-neutral-100">Network</h2>
        <ErrorBanner message={net.error} />
      </div>
    )
  }

  const data = net.data

  if (!data || !data.measured) {
    return (
      <div className="space-y-5">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Network</h2>
          <p className="mt-0.5 text-sm text-neutral-500">
            Phase 2A measurements between the three Vast hosts.
          </p>
        </div>
        <Card>
          <EmptyState title="No network measurements yet">
            <p className="mb-3">
              Run the benchmark against the real hosts to populate this page.
            </p>
            <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`uv run python -m experiments.network.exp_net_002   # connectivity, 6 paths
uv run python -m experiments.network.exp_net_005   # RTT, jitter, bandwidth`}
            </pre>
          </EmptyState>
        </Card>
      </div>
    )
  }

  const paths = data.paths
  const measured = paths.filter((p) => p.measured)
  const fast = measured.filter((p) => p.rtt_p50_ms !== null && p.rtt_p50_ms <= 5)
  const slow = measured.filter((p) => p.rtt_p50_ms !== null && p.rtt_p50_ms > 50)
  const bestTput = measured.reduce(
    (acc, p) => (p.sustained_mbps !== null && p.sustained_mbps > acc ? p.sustained_mbps : acc),
    0,
  )

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">Network</h2>
        <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
          Measured between the three Vast hosts. Every value is from an instrumented
          transfer; nothing here is derived from advertised hardware specs.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6 rounded-lg border border-neutral-800 bg-neutral-900/40 p-5 sm:grid-cols-4">
        <Stat label="Nodes">{data.topology.nodes.length}</Stat>
        <Stat label="Paths measured">
          {measured.length}
          <span className="text-neutral-500">/{paths.length}</span>
        </Stat>
        <Stat label="Low-latency paths">
          <span className={fast.length ? 'text-emerald-400' : ''}>{fast.length}</span>
        </Stat>
        <Stat label="Best sustained">
          <Measured value={bestTput || null} unit="Mbps" digits={0} />
        </Stat>
      </div>

      {/* Path table */}
      <Card
        title="Directional paths"
        subtitle="Each direction measured separately; uplinks are frequently asymmetric"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-neutral-400">
              <tr>
                <th className="px-3 py-2 font-medium">Path</th>
                <th className="px-3 py-2 font-medium">Class</th>
                <th className="px-3 py-2 font-medium">RTT p50</th>
                <th className="px-3 py-2 font-medium">p99</th>
                <th className="px-3 py-2 font-medium">Jitter</th>
                <th className="px-3 py-2 font-medium">Loss</th>
                <th className="px-3 py-2 font-medium">Sustained</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800">
              {paths.map((p) => {
                const tone = latencyTone(p)
                return (
                  <tr key={`${p.source}-${p.target}`} className="text-neutral-300">
                    <td className="px-3 py-2 font-mono text-[11px]">
                      {p.source} → {p.target}
                    </td>
                    <td className={`px-3 py-2 ${tone.cls}`}>{tone.label}</td>
                    <td className="px-3 py-2">
                      <Measured value={p.rtt_p50_ms} unit="ms" digits={2} />
                    </td>
                    <td className="px-3 py-2">
                      <Measured value={p.rtt_p99_ms} unit="ms" digits={2} />
                    </td>
                    <td className="px-3 py-2">
                      <Measured value={p.jitter_ms} unit="ms" digits={2} />
                    </td>
                    <td className="px-3 py-2">
                      <Measured value={p.packet_loss_pct} unit="%" digits={1} />
                    </td>
                    <td className="px-3 py-2">
                      <Measured value={p.sustained_mbps} unit="Mbps" digits={0} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {slow.length > 0 && (
          <p className="mt-3 rounded border border-amber-900/60 bg-amber-950/25 px-3 py-2 text-xs text-amber-300">
            {slow.length} path{slow.length === 1 ? '' : 's'} exceed 50 ms RTT. At that
            latency a pipeline stage boundary adds more than 50 ms to every generated
            token, so these links are suitable for independent replicas only — not
            for coupled execution.
          </p>
        )}
      </Card>

      {/* Co-location */}
      {data['co-location']?.available && (
        <Card
          title="HOST-A / HOST-C co-location"
          subtitle="They share a public IP. Measured network adjacency tells a different story than the hardware identifiers."
        >
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              {(data['co-location'].pairs ?? []).map((p) => (
                <span
                  key={`${p.a}-${p.b}`}
                  className="flex items-center gap-2 rounded border border-neutral-800 bg-neutral-950/50 px-2.5 py-1 text-xs"
                >
                  <span className="font-mono text-[11px] text-neutral-400">
                    {p.a} ↔ {p.b}
                  </span>
                  {p.shared_public_ip && <Badge tone="warn">shared IP</Badge>}
                  <Badge
                    tone={
                      p.verdict === 'different-host'
                        ? 'info'
                        : p.verdict === 'same-host confirmed'
                          ? 'good'
                          : 'neutral'
                    }
                  >
                    {p.verdict}
                  </Badge>
                </span>
              ))}
            </div>
            {data['co-location'].network_adjacency?.a_to_c && (
              <p className="text-xs text-neutral-400">
                Despite different virtual hardware, A↔C measures{' '}
                {data['co-location'].network_adjacency.a_to_c.rtt_p50_ms} ms RTT at{' '}
                {data['co-location'].network_adjacency.a_to_c.sustained_mbps} Mbps
                sustained — datacenter-class. They are separate VMs on one physical
                host, and form MESHWEAVE&apos;s only fast pair.
              </p>
            )}
          </div>
        </Card>
      )}

      <p className="text-[11px] text-neutral-600">
        Source: experiments/network/raw/. One-way latency is not reported: the clocks
        are unsynchronised, so RTT/2 would be an estimate assuming path symmetry.
      </p>
    </div>
  )
}