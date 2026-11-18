import { Badge, Card, EmptyState, ErrorBanner, Measured, Stat, timeAgo } from '../lib/format'
import { usePolling } from '../lib/api'

interface ReadinessNode {
  node_id: string
  status: string
  gpus: number
  gpu_name: string | null
  vram_gib: number
  can_host: boolean
  agent_version: string | null
}

interface Readiness {
  nodes: ReadinessNode[]
  ready: number
  total: number
  note: string
}

/**
 * What each node can actually do.
 *
 * Two separate questions, deliberately not merged:
 *   - is the node reachable and does it have a GPU?  (can_host)
 *   - is an inference stack installed?              (not reported here)
 *
 * A node that has not reported a stack is shown as unknown rather than assumed
 * ready. MESHWEAVE does not guess capability.
 */
export function Readiness() {
  const { data, error, loading, lastFetched, refresh } = usePolling<Readiness>(
    '/v1/readiness',
    5000,
  )

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Readiness</h2>
          <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
            Whether each node can host a model, and what it has reported. A node
            with no GPU telemetry is not assumed capable.
          </p>
        </div>
        <button
          onClick={() => refresh()}
          className="rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          refresh
        </button>
      </div>

      {error && <ErrorBanner message={error} />}

      {!data && !error && (
        <Card>
          <p className="text-sm text-neutral-500">{loading ? 'connecting…' : 'no data'}</p>
        </Card>
      )}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-6 rounded-lg border border-neutral-800 bg-neutral-900/40 p-5 sm:grid-cols-4">
            <Stat label="Nodes">
              {data.total}
            </Stat>
            <Stat label="Can host">
              <span className={data.ready ? 'text-emerald-400' : ''}>{data.ready}</span>
            </Stat>
            <Stat label="Reported a GPU">
              {data.nodes.filter((n) => n.gpus > 0).length}
            </Stat>
            <Stat label="Updated">
              <span className="text-sm">
                {lastFetched ? timeAgo(new Date(lastFetched).toISOString()) : '—'}
              </span>
            </Stat>
          </div>

          {data.nodes.length === 0 ? (
            <Card>
              <EmptyState title="No nodes registered">
                <p className="mb-3">
                  Run the installer on a GPU host to register it.
                </p>
                <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`curl -fsSL https://raw.githubusercontent.com/ashzansoc/meshweave/main/deploy/install.sh \\
  | sudo bash -s -- --coordinator <url> --token <token> \\
      --node-id <name> --enroll --provision`}
                </pre>
              </EmptyState>
            </Card>
          ) : (
            <Card
              title="Nodes"
              subtitle="Provisioning installs the inference stack locally on each host; the coordinator cannot run commands on a node."
            >
              <ul className="divide-y divide-neutral-800">
                {data.nodes.map((n) => (
                  <li key={n.node_id} className="flex flex-wrap items-center justify-between gap-3 py-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span
                          className={`inline-block h-2 w-2 rounded-full ${
                            n.status === 'HEALTHY'
                              ? 'bg-emerald-500'
                              : n.status === 'UNREACHABLE'
                                ? 'bg-red-500'
                                : 'bg-neutral-500'
                          }`}
                        />
                        <span className="font-mono text-[12px] text-neutral-200">
                          {n.node_id}
                        </span>
                        {n.can_host ? (
                          <Badge tone="good">can host</Badge>
                        ) : (
                          <Badge tone="warn">
                            {n.gpus === 0 ? 'no GPU reported' : 'unavailable'}
                          </Badge>
                        )}
                      </div>
                      <div className="mt-1 text-xs text-neutral-500">
                        {n.gpu_name ?? 'no GPU'} ·{' '}
                        <Measured value={n.vram_gib || null} unit="GiB" />
                      </div>
                    </div>
                    <span className="text-[11px] text-neutral-600">
                      agent {n.agent_version ?? 'unknown'}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {data.nodes.some((n) => n.gpus === 0) && (
            <p className="rounded border border-amber-900/60 bg-amber-950/25 px-3 py-2 text-xs text-amber-300">
              {data.nodes
                .filter((n) => n.gpus === 0)
                .map((n) => n.node_id)
                .join(', ')}{' '}
              reported no GPU. That usually means the agent is running an older
              build — restart it so it picks up the NVIDIA bindings.
            </p>
          )}
        </>
      )}
    </div>
  )
}
