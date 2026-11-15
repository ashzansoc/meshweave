import { Card, Stat, Measured, EmptyState, ErrorBanner, Badge } from '../lib/format'
import { usePolling } from '../lib/api'
import { TopologyGraph } from '../components/TopologyGraph'
import type { NodeListResponse, TopologyResponse } from '../api'

export function Topology() {
  const topology = usePolling<TopologyResponse>('/v1/topology')
  const nodes = usePolling<NodeListResponse>('/v1/nodes', 10000)

  const matrix = topology.data?.matrix
  const environments = topology.data?.environments ?? {}
  const registered = nodes.data?.nodes ?? []

  const measuredLinks = matrix?.links.filter((l) => l.rtt_ms_p50 !== null) ?? []
  const failedLinks = matrix?.links.filter((l) => l.error) ?? []

  const rtts = measuredLinks
    .map((l) => l.rtt_ms_p50)
    .filter((v): v is number => v !== null)
  const meanRtt = rtts.length ? rtts.reduce((a, b) => a + b, 0) / rtts.length : null
  const maxRtt = rtts.length ? Math.max(...rtts) : null

  const bandwidths = matrix?.links
    .map((l) => l.sustained_mbps ?? l.tcp_mbps_single_stream)
    .filter((v): v is number => v !== null && v !== undefined) ?? []
  const minBandwidth = bandwidths.length ? Math.min(...bandwidths) : null

  const nodeIds = matrix?.nodes ?? registered.map((n) => n.node_id)

  const envCounts = Object.values(environments).reduce<Record<string, number>>(
    (acc, env) => {
      acc[env] = (acc[env] ?? 0) + 1
      return acc
    },
    {},
  )

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Network topology</h2>
          <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
            Drawn from measured links only. A link's class comes from its measured
            round-trip time, never from what a host advertises about its hardware.
          </p>
        </div>
        <button
          onClick={() => topology.refresh()}
          className="rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          refresh
        </button>
      </div>

      {topology.error && topology.error.includes('404') && (
        <Card>
          <EmptyState title="No topology has been measured">
            <p className="mb-3">
              MESHWEAVE measures every node pair before any model is loaded, because
              "the network was slow" is only a finding if it was measured.
            </p>
            <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`# 2 nodes -> 2 directed probes; 8 nodes -> 56 directed probes
uv run meshweave bench network \\
  --nodes host1:9000,host2:9000 \\
  --ids node-a,node-b \\
  --out reports/topology.json

# then submit the matrix to the coordinator
curl -X POST http://127.0.0.1:8700/v1/topology \\
  -H 'content-type: application/json' \\
  -d @reports/topology.json`}
            </pre>
          </EmptyState>
        </Card>
      )}

      {topology.error && !topology.error.includes('404') && (
        <ErrorBanner message={topology.error} />
      )}

      {matrix && (
        <>
          <div className="grid grid-cols-2 gap-6 rounded-lg border border-neutral-800 bg-neutral-900/40 p-5 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Nodes">{matrix.nodes.length}</Stat>
            <Stat label="Directed links">{matrix.links.length}</Stat>
            <Stat label="Measured">
              {measuredLinks.length}
              <span className="text-neutral-500">/{matrix.links.length}</span>
            </Stat>
            <Stat label="Mean RTT">
              <Measured value={meanRtt} unit="ms" digits={1} />
            </Stat>
            <Stat label="Worst RTT">
              <Measured value={maxRtt} unit="ms" digits={1} />
            </Stat>
            <Stat
              label="Slowest link"
              hint="Minimum sustained or single-stream bandwidth across measured links."
            >
              <Measured value={minBandwidth} unit="Mbps" digits={0} />
            </Stat>
          </div>

          {Object.keys(envCounts).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {Object.entries(envCounts).map(([env, count]) => (
                <Badge
                  key={env}
                  tone={
                    env === 'A_DATACENTER'
                      ? 'good'
                      : env === 'B_REGIONAL'
                        ? 'warn'
                        : 'bad'
                  }
                >
                  {env}: {count} node{count === 1 ? '' : 's'}
                </Badge>
              ))}
            </div>
          )}

          <Card
            title="Measured links"
            subtitle="Hover a row to highlight that link in the graph"
          >
            <TopologyGraph
              nodes={nodeIds}
              links={matrix.links}
              environments={environments}
            />
          </Card>

          {failedLinks.length > 0 && (
            <Card
              title="Links that could not be measured"
              subtitle="Recorded rather than dropped, because a failed probe is a finding"
            >
              <ul className="space-y-2 text-xs">
                {failedLinks.map((l) => (
                  <li
                    key={`${l.source}-${l.target}`}
                    className="rounded border border-amber-900/60 bg-amber-950/20 px-3 py-2"
                  >
                    <span className="font-mono text-[11px] text-amber-200">
                      {l.source} → {l.target}
                    </span>
                    <p className="mt-1 text-amber-300/80">{l.error}</p>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          <p className="text-[11px] text-neutral-600">
            Matrix generated {new Date(matrix.generated_at).toLocaleString()}. Each
            unordered pair is probed in both directions, so 4 nodes produce 12
            directed measurements and 8 nodes produce 56.
          </p>
        </>
      )}
    </div>
  )
}