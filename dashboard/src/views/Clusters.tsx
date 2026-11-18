import { useState } from 'react'
import { Badge, Card, Measured } from '../lib/format'
import { apiPost, usePolling } from '../lib/api'

interface Cluster {
  cluster_id: string
  name: string
  members: string[]
  member_state: Record<string, string>
  missing: string[]
  health: string
  capacity: {
    gpus: number
    vram_total_gib: number
    vram_free_gib: number
    cpu_cores: number
    kind: string
    note: string
  }
  warnings: string[]
  created_at: string
}

interface Node {
  node_id: string
  gpu_name: string | null
  vram_gib: number
  can_host: boolean
}

const HEALTH_TONE: Record<string, 'good' | 'warn' | 'bad'> = {
  HEALTHY: 'good',
  DEGRADED: 'warn',
  OFFLINE: 'bad',
}

/**
 * Virtual clusters: several machines named as one, with the aggregate stated as
 * distributed rather than unified. The capacity note is part of the data model,
 * not a disclaimer bolted on here.
 */
export function Clusters() {
  const clusters = usePolling<{ clusters: Cluster[] }>('/v1/clusters', 5000)
  const nodes = usePolling<{ nodes: Node[] }>('/v1/readiness', 5000)
  const [name, setName] = useState('F16-VDC')
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function create() {
    setBusy(true)
    setError(null)
    try {
      await apiPost('/v1/clusters', { name, node_ids: selected })
      setSelected([])
      await clusters.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function remove(clusterId: string) {
    setBusy(true)
    try {
      await fetch(`/api/v1/clusters/${clusterId}`, { method: 'DELETE' })
      await clusters.refresh()
    } finally {
      setBusy(false)
    }
  }

  const available = (nodes.data?.nodes ?? []).filter((n) => n.can_host)

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">Virtual clusters</h2>
        <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
          Group physical nodes under one identifier. The combined capacity is the
          sum of separate GPUs on separate machines, not a single pool of
          addressable memory.
        </p>
      </div>

      {error && (
        <p className="rounded border border-red-900/60 bg-red-950/25 px-3 py-2 text-xs text-red-300">
          {error}
        </p>
      )}

      <Card title="Form a cluster">
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-56 rounded border border-neutral-700 bg-neutral-900 px-2.5 py-1.5 font-mono text-sm text-neutral-200 outline-none focus:border-neutral-500"
          />
          <span className="text-[11px] text-neutral-500">
            ids are assigned as F16-VDC-001, 002, …
          </span>
        </div>

        <div className="mt-3 space-y-1.5">
          {available.length === 0 && (
            <p className="text-sm text-neutral-500">no GPU nodes available</p>
          )}
          {available.map((node) => {
            const checked = selected.includes(node.node_id)
            return (
              <label
                key={node.node_id}
                className={`flex cursor-pointer items-center gap-3 rounded border px-3 py-1.5 text-sm ${
                  checked
                    ? 'border-neutral-600 bg-neutral-800/60'
                    : 'border-neutral-800 hover:bg-neutral-900'
                }`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() =>
                    setSelected((prev) =>
                      prev.includes(node.node_id)
                        ? prev.filter((n) => n !== node.node_id)
                        : [...prev, node.node_id],
                    )
                  }
                  className="h-3.5 w-3.5 accent-neutral-300"
                />
                <span className="font-mono text-[12px] text-neutral-200">{node.node_id}</span>
                <span className="text-[11px] text-neutral-500">
                  {node.gpu_name ?? 'no GPU'} · <Measured value={node.vram_gib} unit="GiB" />
                </span>
              </label>
            )
          })}
        </div>

        <button
          onClick={() => void create()}
          disabled={busy || selected.length === 0 || name.trim() === ''}
          className="mt-3 rounded border border-neutral-600 bg-neutral-800 px-3 py-1.5 text-xs text-neutral-100 hover:bg-neutral-700 disabled:opacity-40"
        >
          {busy ? 'working…' : `create cluster from ${selected.length} node(s)`}
        </button>
      </Card>

      {(clusters.data?.clusters.length ?? 0) === 0 ? (
        <Card>
          <p className="text-sm text-neutral-500">
            No clusters. A single GPU node can already host a model on its own;
            a cluster is for when you want several machines to share the work.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {clusters.data?.clusters.map((cluster) => (
            <Card
              key={cluster.cluster_id}
              title={`${cluster.cluster_id} · ${cluster.name}`}
              subtitle={`${cluster.members.length} member(s) · created ${new Date(
                cluster.created_at,
              ).toLocaleString()}`}
            >
              <div className="flex flex-wrap items-center gap-3">
                <Badge tone={HEALTH_TONE[cluster.health] ?? 'neutral'}>{cluster.health.toLowerCase()}</Badge>
                <span className="text-[11px] text-neutral-500">
                  {cluster.capacity.gpus} GPU(s) ·{' '}
                  <Measured value={cluster.capacity.vram_total_gib} unit="GiB total" /> ·{' '}
                  <Measured value={cluster.capacity.vram_free_gib} unit="GiB free" /> ·{' '}
                  {cluster.capacity.cpu_cores} cores
                </span>
                <button
                  onClick={() => void remove(cluster.cluster_id)}
                  className="ml-auto rounded border border-neutral-700 px-2 py-1 text-[11px] text-neutral-400 hover:bg-neutral-800"
                >
                  remove
                </button>
              </div>

              <ul className="mt-2 divide-y divide-neutral-800">
                {cluster.members.map((member) => (
                  <li key={member} className="flex items-center justify-between py-1.5">
                    <span className="font-mono text-[12px] text-neutral-300">{member}</span>
                    <Badge
                      tone={
                        cluster.member_state[member] === 'AVAILABLE'
                          ? 'good'
                          : cluster.member_state[member] === 'BUSY'
                            ? 'warn'
                            : cluster.member_state[member] === 'OFFLINE'
                              ? 'bad'
                              : 'neutral'
                      }
                    >
                      {(cluster.member_state[member] ?? 'unknown').toLowerCase()}
                    </Badge>
                  </li>
                ))}
                {cluster.missing.map((member) => (
                  <li key={member} className="flex items-center justify-between py-1.5">
                    <span className="font-mono text-[12px] text-neutral-500">{member}</span>
                    <Badge tone="bad">unregistered</Badge>
                  </li>
                ))}
              </ul>

              <p className="mt-2 rounded bg-neutral-950/60 px-2.5 py-2 text-[11px] leading-relaxed text-neutral-500">
                {cluster.capacity.note}
              </p>

              {cluster.warnings.length > 0 && (
                <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[11px] text-amber-400">
                  {cluster.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}