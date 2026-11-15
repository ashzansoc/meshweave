import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  EmptyState,
  ErrorBanner,
  Badge,
  Code,
} from '../lib/format'
import { usePolling, useSamples } from '../lib/api'
import { Sparkline } from '../components/Sparkline'
import type { Metrics, NodeListResponse } from '../api'

function SectionHeading({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle?: string
  children?: ReactNode
}) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">{title}</h2>
        {subtitle && <p className="mt-0.5 text-sm text-neutral-500">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

export function Overview() {
  const nodes = usePolling<NodeListResponse>('/v1/nodes')
  const metrics = usePolling<Metrics>('/v1/metrics')

  const freeVram = nodes.data?.summary.free_vram_gib
  const tps = metrics.data?.mean_tps
  const cost = metrics.data?.mean_cost_per_m_output_tokens_usd

  const freeVramSamples = useSamples(freeVram)
  const tpsSamples = useSamples(tps)
  const costSamples = useSamples(cost)

  const summary = nodes.data?.summary
  const nodeList = nodes.data?.nodes ?? []
  const healthyNodes = nodeList.filter((n) => n.status === 'HEALTHY')
  const unhealthy = nodeList.filter((n) => n.status !== 'HEALTHY')

  const down = Boolean(nodes.error && metrics.error)

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Cluster overview"
        subtitle="Every figure is reported by an agent to the coordinator. Nothing here is estimated."
      />

      {down && <ErrorBanner message={nodes.error ?? 'unreachable'} />}

      {!down && !nodes.data && !nodes.loading && (
        <ErrorBanner message={nodes.error ?? 'no response'} />
      )}

      {/* Primary stats */}
      <Card>
        <div className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Connected nodes">
            {summary ? (
              <>
                {summary.healthy}
                <span className="text-neutral-500">/{summary.total}</span>
              </>
            ) : (
              '—'
            )}
          </Stat>
          <Stat label="Total GPUs">
            {summary ? summary.total_gpus : '—'}
          </Stat>
          <Stat
            label="Aggregate VRAM"
            hint="Sum of every GPU's total memory. This is arithmetic, not evidence of aggregation."
          >
            <Measured value={summary ? summary.total_vram_gib : null} unit="GiB" />
          </Stat>
          <Stat label="Free VRAM">
            <Measured value={summary ? summary.free_vram_gib : null} unit="GiB" />
          </Stat>
          <Stat
            label="Cluster cost"
            hint="Sum of the hourly price recorded for each node. Unknown if any node has no price."
          >
            <Money value={summary?.hourly_cost_usd} per="/hr" />
          </Stat>
          <Stat label="Topology">
            {metrics.data ? (
              metrics.data.measured_topology ? (
                <span className="text-emerald-400">measured</span>
              ) : (
                <span className="text-neutral-400">not measured</span>
              )
            ) : (
              '—'
            )}
          </Stat>
        </div>

        {summary && summary.hourly_cost_usd === null && (
          <p className="mt-5 border-t border-neutral-800 pt-4 text-xs text-neutral-500">
            Hourly cost is unknown because at least one node has no price recorded.
            MESHWEAVE reports the gap rather than guessing a number.
          </p>
        )}
      </Card>

      {/* Live trends */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card
          title="Free VRAM"
          subtitle={`${freeVramSamples.length} samples this session`}
        >
          {freeVram !== null && freeVram !== undefined ? (
            <div className="flex items-end justify-between gap-3">
              <div className="text-2xl tabular-nums">
                {freeVram.toFixed(1)}
                <span className="ml-1 text-sm text-neutral-500">GiB</span>
              </div>
              <Sparkline
                samples={freeVramSamples}
                stroke="#10b981"
                label="Free VRAM observed by this browser"
              />
            </div>
          ) : (
            <EmptyState title="No VRAM reported" />
          )}
        </Card>

        <Card
          title="Mean throughput"
          subtitle="Derived from completed workloads"
        >
          {tps !== null && tps !== undefined ? (
            <div className="flex items-end justify-between gap-3">
              <div className="text-2xl tabular-nums">
                {tps.toFixed(1)}
                <span className="ml-1 text-sm text-neutral-500">tok/s</span>
              </div>
              <Sparkline
                samples={tpsSamples}
                stroke="#38bdf8"
                label="Mean throughput observed by this browser"
              />
            </div>
          ) : (
            <EmptyState title="No workload has completed">
              Throughput appears only after an execution records it.
            </EmptyState>
          )}
        </Card>

        <Card
          title="Mean cost"
          subtitle="Per million output tokens"
        >
          {cost !== null && cost !== undefined ? (
            <div className="flex items-end justify-between gap-3">
              <div className="text-2xl tabular-nums">
                ${cost.toFixed(2)}
                <span className="ml-1 text-sm text-neutral-500">/M</span>
              </div>
              <Sparkline
                samples={costSamples}
                stroke="#a78bfa"
                label="Mean cost observed by this browser"
              />
            </div>
          ) : (
            <EmptyState title="Cost not yet known">
              Requires a completed workload and a recorded node price.
            </EmptyState>
          )}
        </Card>
      </div>

      {/* Node health */}
      <Card
        title="Node health"
        subtitle={
          nodeList.length > 0
            ? `${healthyNodes.length} healthy, ${unhealthy.length} needing attention`
            : undefined
        }
        actions={
          <Link to="/nodes" className="text-xs text-neutral-400 hover:text-neutral-200">
            details →
          </Link>
        }
      >
        {nodeList.length === 0 ? (
          <EmptyState title="No nodes registered">
            Start an agent on each GPU host:
            <div className="mt-2 text-left">
              <Code>uv run meshweave agent --node-id node-a --coordinator https://&lt;host&gt;:8700</Code>
            </div>
          </EmptyState>
        ) : (
          <ul className="divide-y divide-neutral-800">
            {nodeList.map((n) => {
              const gpu = n.inventory?.gpus[0]
              return (
                <li
                  key={n.node_id}
                  className="flex flex-wrap items-center justify-between gap-2 py-2 text-sm"
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${
                        n.status === 'HEALTHY'
                          ? 'bg-emerald-500'
                          : n.status === 'DEGRADED'
                            ? 'bg-amber-500'
                            : n.status === 'UNREACHABLE'
                              ? 'bg-red-500'
                              : 'bg-neutral-500'
                      }`}
                    />
                    <span className="font-mono text-[12px] text-neutral-300">
                      {n.node_id}
                    </span>
                    {n.status !== 'HEALTHY' && (
                      <Badge tone={n.status === 'UNREACHABLE' ? 'bad' : 'warn'}>
                        {n.status}
                      </Badge>
                    )}
                  </span>
                  <span className="flex flex-wrap items-center gap-4 text-xs text-neutral-400">
                    <span>{gpu ? gpu.name : 'no GPU reported'}</span>
                    <span>
                      <Measured value={gpu ? gpu.vram_free_bytes / 1024 ** 3 : null} unit="GiB free" />
                    </span>
                    <span>
                      {n.reliability_score !== null ? (
                        <>reliability {n.reliability_score.toFixed(2)}</>
                      ) : (
                        <span className="text-neutral-600 italic">
                          reliability unknown
                        </span>
                      )}
                    </span>
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </Card>

      <p className="text-[11px] text-neutral-600">
        Trend lines show only what this browser observed while the page was open.
        They are not a stored time series and reset on reload.
      </p>
    </div>
  )
}