import { useMemo, useState } from 'react'
import { useCluster } from './useCluster'
import type { LinkMeasurement, NodeRecord } from './api'

const GIB = 1024 ** 3

/** Render a value that may not have been measured. Never substitutes zero. */
function Measured({
  value,
  unit,
  digits = 1,
}: {
  value: number | null | undefined
  unit: string
  digits?: number
}) {
  if (value === null || value === undefined) {
    return <span className="text-neutral-400">not measured</span>
  }
  return (
    <span>
      {value.toFixed(digits)}
      <span className="ml-1 text-neutral-500">{unit}</span>
    </span>
  )
}

function StatusDot({ status }: { status: string }) {
  const colour =
    status === 'HEALTHY'
      ? 'bg-emerald-500'
      : status === 'DEGRADED'
        ? 'bg-amber-500'
        : status === 'UNREACHABLE'
          ? 'bg-red-500'
          : 'bg-neutral-500'
  return <span className={`inline-block h-2 w-2 rounded-full ${colour}`} />
}

function Card({
  title,
  children,
  hint,
}: {
  title: string
  children: React.ReactNode
  hint?: string
}) {
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/50 p-5">
      <h2 className="text-sm font-medium tracking-wide text-neutral-300 uppercase">
        {title}
      </h2>
      {hint && <p className="mt-1 text-xs text-neutral-500">{hint}</p>}
      <div className="mt-4">{children}</div>
    </section>
  )
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="mt-1 text-lg text-neutral-100">{children}</div>
    </div>
  )
}

function TopologyMatrix({ links, nodes }: { links: LinkMeasurement[]; nodes: string[] }) {
  if (links.length === 0) {
    return (
      <p className="text-sm text-neutral-500">
        No links measured. Run{' '}
        <code className="text-neutral-300">meshweave bench network</code> and submit the
        result to the coordinator.
      </p>
    )
  }

  const worst = links.reduce(
    (acc, l) => (l.rtt_ms_p50 !== null && l.rtt_ms_p50 > acc ? l.rtt_ms_p50 : acc),
    0,
  )

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Nodes">
          {nodes.length}
        </Stat>
        <Stat label="Directed links">
          {links.length}
        </Stat>
        <Stat label="Worst RTT">
          <Measured value={worst} unit="ms" />
        </Stat>
        <Stat label="Measured">
          {links.filter((l) => l.rtt_ms_p50 !== null).length}/{links.length}
        </Stat>
      </div>
      <div className="max-h-64 overflow-auto rounded border border-neutral-800">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-neutral-900 text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">Link</th>
              <th className="px-3 py-2 font-medium">RTT p50</th>
              <th className="px-3 py-2 font-medium">Jitter</th>
              <th className="px-3 py-2 font-medium">Loss</th>
              <th className="px-3 py-2 font-medium">Sustained</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {links.map((l) => (
              <tr key={`${l.source}-${l.target}`} className="text-neutral-300">
                <td className="px-3 py-2 font-mono text-[11px]">
                  {l.source} → {l.target}
                </td>
                <td className="px-3 py-2">
                  <Measured value={l.rtt_ms_p50} unit="ms" digits={2} />
                </td>
                <td className="px-3 py-2">
                  <Measured value={l.jitter_ms} unit="ms" digits={2} />
                </td>
                <td className="px-3 py-2">
                  <Measured value={l.packet_loss_pct} unit="%" />
                </td>
                <td className="px-3 py-2">
                  <Measured value={l.sustained_mbps} unit="Mbps" digits={0} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function App() {
  const { nodes, metrics, topology, plan, error, loading, requestPlan } = useCluster()
  const [model, setModel] = useState('Qwen2.5-7B-Instruct')
  const [quant, setQuant] = useState('bf16')

  const nodeList: NodeRecord[] = useMemo(() => nodes?.nodes ?? [], [nodes])
  const totalVram = nodeList.reduce((sum, n) => {
    if (!n.inventory) return sum
    return sum + n.inventory.gpus.reduce((s, g) => s + g.vram_total_bytes, 0)
  }, 0)

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-200">
      <header className="border-b border-neutral-800 px-6 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h1 className="text-xl font-medium tracking-tight">MESHWEAVE</h1>
            <p className="text-xs text-neutral-500">
              Many Machines. One Intelligence.
            </p>
          </div>
          <div className="text-xs text-neutral-500">
            {loading ? 'connecting…' : `polling every 3s`}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">
        {error && (
          <div className="rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
            <strong className="font-medium">Coordinator unreachable.</strong>{' '}
            {error}
            <div className="mt-1 text-xs text-red-400/80">
              Start it with <code>uv run meshweave coordinator --port 8700</code>.
            </div>
          </div>
        )}

        <Card
          title="Cluster overview"
          hint="Every figure below comes from a coordinator that received it from an agent. Nothing is estimated here."
        >
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Connected nodes">
              {nodes ? nodes.summary.healthy : '—'}
              <span className="text-neutral-500">/{nodes ? nodes.summary.total : '—'}</span>
            </Stat>
            <Stat label="Total GPUs">
              {nodes ? nodes.summary.total_gpus : '—'}
            </Stat>
            <Stat label="Aggregate VRAM">
              <Measured value={totalVram / GIB} unit="GiB" />
            </Stat>
            <Stat label="Free VRAM">
              <Measured value={nodes ? nodes.summary.free_vram_gib : null} unit="GiB" />
            </Stat>
            <Stat label="Cluster cost">
              {nodes?.summary.hourly_cost_usd != null ? (
                <>
                  ${nodes.summary.hourly_cost_usd.toFixed(2)}
                  <span className="ml-1 text-neutral-500">/hr</span>
                </>
              ) : (
                <span className="text-neutral-400">unknown</span>
              )}
            </Stat>
            <Stat label="Topology">
              {metrics?.measured_topology ? (
                <span className="text-emerald-400">measured</span>
              ) : (
                <span className="text-neutral-400">not measured</span>
              )}
            </Stat>
          </div>

          {nodes && nodes.summary.hourly_cost_usd === null && (
            <p className="mt-4 text-xs text-neutral-500">
              Hourly cost is unknown because at least one node has no price recorded.
              MESHWEAVE reports this rather than guessing.
            </p>
          )}
        </Card>

        <Card title="Nodes">
          {nodeList.length === 0 ? (
            <p className="text-sm text-neutral-500">
              No nodes registered. Start an agent on each GPU host:
              <br />
              <code className="text-neutral-300">
                uv run meshweave agent --node-id node-a --coordinator
                https://&lt;coordinator&gt;:8700
              </code>
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-neutral-400">
                  <tr>
                    <th className="px-2 py-2 font-medium">Node</th>
                    <th className="px-2 py-2 font-medium">Status</th>
                    <th className="px-2 py-2 font-medium">GPU</th>
                    <th className="px-2 py-2 font-medium">VRAM free</th>
                    <th className="px-2 py-2 font-medium">Temp</th>
                    <th className="px-2 py-2 font-medium">Reliability</th>
                    <th className="px-2 py-2 font-medium">$/hr</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {nodeList.map((n) => {
                    const gpu = n.inventory?.gpus[0]
                    return (
                      <tr key={n.node_id} className="text-neutral-300">
                        <td className="px-2 py-2 font-mono text-[11px]">{n.node_id}</td>
                        <td className="px-2 py-2">
                          <span className="inline-flex items-center gap-2">
                            <StatusDot status={n.status} />
                            {n.status}
                          </span>
                        </td>
                        <td className="px-2 py-2">
                          {gpu ? gpu.name : <span className="text-neutral-500">—</span>}
                        </td>
                        <td className="px-2 py-2">
                          <Measured value={gpu ? gpu.vram_free_bytes / GIB : null} unit="GiB" />
                        </td>
                        <td className="px-2 py-2">
                          <Measured value={gpu?.temperature_c ?? null} unit="°C" />
                        </td>
                        <td className="px-2 py-2">
                          {n.reliability_score != null ? (
                            n.reliability_score.toFixed(3)
                          ) : (
                            <span className="text-neutral-500">
                              unknown ({n.observations} obs)
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-2">
                          {n.hourly_cost_usd != null ? (
                            `$${n.hourly_cost_usd.toFixed(2)}`
                          ) : (
                            <span className="text-neutral-500">unknown</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card
          title="Network topology"
          hint="Measured before any model is loaded. A link is classified from its measured RTT, never from what the host advertises."
        >
          <TopologyMatrix
            links={topology?.matrix.links ?? []}
            nodes={topology?.matrix.nodes ?? []}
          />
        </Card>

        <Card
          title="Execution plan"
          hint="The planner's current answer, including when that answer is to use a single GPU."
        >
          <div className="flex flex-wrap items-end gap-3">
            <label className="text-xs">
              <span className="block text-neutral-500">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="mt-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm"
              >
                <option>Qwen2.5-1.5B-Instruct</option>
                <option>Qwen2.5-7B-Instruct</option>
                <option>Qwen2.5-14B-Instruct</option>
                <option>Qwen2.5-32B-Instruct</option>
              </select>
            </label>
            <label className="text-xs">
              <span className="block text-neutral-500">Precision</span>
              <select
                value={quant}
                onChange={(e) => setQuant(e.target.value)}
                className="mt-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-sm"
              >
                <option>bf16</option>
                <option>int8</option>
                <option>int4</option>
              </select>
            </label>
            <button
              onClick={() => void requestPlan(model, quant)}
              className="rounded bg-neutral-100 px-3 py-1.5 text-sm font-medium text-neutral-900 hover:bg-white"
            >
              Ask the planner
            </button>
          </div>

          {plan && (
            <div className="mt-4 space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Strategy">{plan.strategy}</Stat>
                <Stat label="Environment">{plan.environment}</Stat>
                <Stat label="Est. throughput">
                  <Measured value={plan.estimated_throughput_tps} unit="tok/s" />
                </Stat>
                <Stat label="Est. cost">
                  {plan.estimated_cost_per_m_output_tokens != null ? (
                    <>
                      ${plan.estimated_cost_per_m_output_tokens.toFixed(2)}
                      <span className="ml-1 text-neutral-500">/M tok</span>
                    </>
                  ) : (
                    <span className="text-neutral-400">unknown</span>
                  )}
                </Stat>
              </div>

              {plan.rationale.length > 0 && (
                <ul className="list-inside list-disc space-y-1 text-neutral-400">
                  {plan.rationale.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              )}

              {plan.disconnect_impact && (
                <p className="rounded border border-amber-900/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
                  <strong className="font-medium">If a node disconnects:</strong>{' '}
                  {plan.disconnect_impact}
                </p>
              )}

              {plan.warnings.length > 0 && (
                <ul className="space-y-1 text-xs text-amber-400">
                  {plan.warnings.map((w) => (
                    <li key={w}>⚠ {w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </Card>

        <Card
          title="Recorded performance"
          hint="Derived only from completed workloads. A value appears here only after an execution recorded it."
        >
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Workloads">
              {metrics ? metrics.workloads_total : '—'}
            </Stat>
            <Stat label="Succeeded">
              {metrics ? metrics.workloads_successful : '—'}
            </Stat>
            <Stat label="Failed">
              {metrics ? metrics.workloads_failed : '—'}
            </Stat>
            <Stat label="Mean throughput">
              <Measured value={metrics?.mean_tps ?? null} unit="tok/s" />
            </Stat>
            <Stat label="Mean TTFT">
              <Measured value={metrics?.mean_ttft_ms ?? null} unit="ms" />
            </Stat>
            <Stat label="Mean cost">
              {metrics?.mean_cost_per_m_output_tokens_usd != null ? (
                <>
                  ${metrics.mean_cost_per_m_output_tokens_usd.toFixed(2)}
                  <span className="ml-1 text-neutral-500">/M tok</span>
                </>
              ) : (
                <span className="text-neutral-400">not measured</span>
              )}
            </Stat>
          </div>
          {metrics && metrics.workloads_total === 0 && (
            <p className="mt-4 text-xs text-neutral-500">
              No workload has been executed yet. Nothing is shown rather than a
              placeholder — a chart of zeros would be a fabrication.
            </p>
          )}
        </Card>
      </main>
    </div>
  )
}