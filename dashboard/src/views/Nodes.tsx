import { useState } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  Badge,
  StatusDot,
  EmptyState,
  ErrorBanner,
  Code,
  timeAgo,
} from '../lib/format'
import { usePolling } from '../lib/api'
import type { NodeDetailResponse, NodeListResponse, GpuDevice } from '../api'

function GpuRow({ gpu, nodeId }: { gpu: GpuDevice; nodeId: string }) {
  const usedPct =
    gpu.vram_total_bytes > 0
      ? (gpu.vram_used_bytes / gpu.vram_total_bytes) * 100
      : 0

  return (
    <div className="rounded border border-neutral-800 bg-neutral-950/50 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm text-neutral-200">
          {gpu.name}
          {gpu.architecture && (
            <span className="ml-2 text-xs text-neutral-500">{gpu.architecture}</span>
          )}
        </span>
        <span className="font-mono text-[10px] text-neutral-600">{gpu.uuid}</span>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-neutral-800">
        <div
          className="h-full rounded-full bg-sky-500/70"
          style={{ width: `${Math.min(100, usedPct).toFixed(1)}%` }}
          role="img"
          aria-label={`${usedPct.toFixed(0)} percent of VRAM used on ${nodeId} GPU ${gpu.index}`}
        />
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-4">
        {[
          ['VRAM total', `${(gpu.vram_total_bytes / 1024 ** 3).toFixed(1)} GiB`],
          ['VRAM free', `${(gpu.vram_free_bytes / 1024 ** 3).toFixed(1)} GiB`],
          ['Temperature', gpu.temperature_c !== null ? `${gpu.temperature_c.toFixed(0)} °C` : null],
          ['Power', gpu.power_draw_w !== null ? `${gpu.power_draw_w.toFixed(0)} W` : null],
          ['GPU util', gpu.utilization_gpu_pct !== null ? `${gpu.utilization_gpu_pct.toFixed(0)} %` : null],
          [
            'Memory BW',
            gpu.memory_bandwidth_gbps !== null ? `${gpu.memory_bandwidth_gbps.toFixed(0)} GB/s` : null,
          ],
          ['Compute cap', gpu.compute_capability ? gpu.compute_capability.join('.') : null],
          ['Driver', gpu.driver_version],
        ].map(([label, value]) => (
          <div key={String(label)}>
            <dt className="text-neutral-500">{label}</dt>
            <dd className="text-neutral-300">
              {value ?? <span className="text-neutral-600 italic">not reported</span>}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function NodeDetail({ nodeId }: { nodeId: string }) {
  const detail = usePolling<NodeDetailResponse>(`/v1/nodes/${nodeId}`, 5000)
  const node = detail.data?.node

  return (
    <Card
      title={nodeId}
      subtitle={node?.registration?.hostname ?? undefined}
      actions={
        <button
          onClick={() => detail.refresh()}
          className="text-xs text-neutral-400 hover:text-neutral-200"
        >
          refresh
        </button>
      }
    >
      {detail.error && <p className="text-xs text-red-400">{detail.error}</p>}

      {node && (
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex items-center gap-2 text-sm">
              <StatusDot status={node.status} />
              {node.status}
            </span>
            {detail.data?.environment && (
              <Badge tone="info">link class {detail.data.environment}</Badge>
            )}
            <Badge>{node.registration.provider}</Badge>
            {detail.data?.degrade_reason && (
              <Badge tone="warn">{detail.data.degrade_reason}</Badge>
            )}
          </div>

          <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
            <Stat label="Reliability">
              {node.reliability_score !== null ? (
                node.reliability_score.toFixed(3)
              ) : (
                <span className="text-sm text-neutral-500 italic">
                  unknown ({node.observations} obs)
                </span>
              )}
            </Stat>
            <Stat label="Missed heartbeats">{node.missed_heartbeats}</Stat>
            <Stat label="Last heartbeat">
              <span className="text-sm">{timeAgo(node.last_heartbeat_at)}</span>
            </Stat>
            <Stat label="Cost">
              <Money value={node.hourly_cost_usd} per="/hr" />
            </Stat>
          </div>

          {node.reliability_score === null && (
            <p className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2 text-[11px] text-neutral-500">
              A node is not reported as reliable until it has been observed for at
              least ten heartbeats. A node that has never been seen is unknown, not
              100% reliable.
            </p>
          )}

          <div className="space-y-2">
            <h3 className="text-xs tracking-wide text-neutral-400 uppercase">
              GPUs ({node.inventory?.gpus.length ?? 0})
            </h3>
            {node.inventory?.gpus.map((gpu) => (
              <GpuRow key={gpu.uuid} gpu={gpu} nodeId={nodeId} />
            ))}
          </div>

          {node.inventory && (
            <div className="grid gap-4 text-xs sm:grid-cols-2">
              <div>
                <h3 className="mb-2 tracking-wide text-neutral-400 uppercase">
                  Host
                </h3>
                <dl className="space-y-1">
                  {[
                    ['CPU', node.inventory.host.cpu_model],
                    ['Cores', node.inventory.host.cpu_cores?.toString() ?? null],
                    [
                      'RAM',
                      node.inventory.host.ram_total_bytes
                        ? `${(node.inventory.host.ram_total_bytes / 1024 ** 3).toFixed(0)} GiB`
                        : null,
                    ],
                    ['OS', node.inventory.host.os_name],
                    ['Kernel', node.inventory.host.kernel_version],
                    ['Agent', node.inventory.agent_version],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="flex justify-between gap-3">
                      <dt className="text-neutral-500">{k}</dt>
                      <dd className="text-right text-neutral-300">
                        {v ?? <span className="text-neutral-600 italic">—</span>}
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
              <div>
                <h3 className="mb-2 tracking-wide text-neutral-400 uppercase">
                  Network
                </h3>
                <dl className="space-y-1">
                  {[
                    ['Private IP', node.inventory.network.private_ip],
                    ['External IP', node.inventory.network.external_ip],
                    ['Region', node.inventory.network.geolocation],
                    [
                      'NAT',
                      node.inventory.network.nat === null
                        ? null
                        : node.inventory.network.nat
                          ? 'yes'
                          : 'no',
                    ],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="flex justify-between gap-3">
                      <dt className="text-neutral-500">{k}</dt>
                      <dd className="text-right text-neutral-300">
                        {v ?? <span className="text-neutral-600 italic">—</span>}
                      </dd>
                    </div>
                  ))}
                </dl>
                {Object.keys(node.inventory.network.mapped_ports).length > 0 && (
                  <p className="mt-2 text-[11px] text-neutral-500">
                    Mapped ports:{' '}
                    {Object.entries(node.inventory.network.mapped_ports)
                      .map(([internal, external]) => `${internal}→${external}`)
                      .join(', ')}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export function Nodes() {
  const nodes = usePolling<NodeListResponse>('/v1/nodes')
  const [selected, setSelected] = useState<string | null>(null)

  const list = nodes.data?.nodes ?? []
  const active = selected ?? list[0]?.node_id ?? null

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Nodes</h2>
          <p className="mt-0.5 text-sm text-neutral-500">
            Live inventory reported by each agent. Values an agent does not report
            are shown as missing, never as zero.
          </p>
        </div>
        <button
          onClick={() => nodes.refresh()}
          className="rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          refresh
        </button>
      </div>

      {nodes.error && <ErrorBanner message={nodes.error} />}

      {list.length === 0 && !nodes.error && (
        <Card>
          <EmptyState title="No nodes registered">
            Start an agent on each GPU host:
            <div className="mt-2 text-left">
              <Code>uv run meshweave agent --node-id node-a --coordinator https://&lt;host&gt;:8700</Code>
            </div>
          </EmptyState>
        </Card>
      )}

      {list.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
          <div className="space-y-2">
            {list.map((n) => {
              const gpus = n.inventory?.gpus.length ?? 0
              const isActive = active === n.node_id
              return (
                <button
                  key={n.node_id}
                  onClick={() => setSelected(n.node_id)}
                  className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    isActive
                      ? 'border-neutral-600 bg-neutral-800/60'
                      : 'border-neutral-800 bg-neutral-900/40 hover:border-neutral-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2">
                      <StatusDot status={n.status} />
                      <span className="font-mono text-[12px] text-neutral-200">
                        {n.node_id}
                      </span>
                    </span>
                    <span className="text-[11px] text-neutral-500">{gpus} GPU</span>
                  </div>
                  <div className="mt-1 flex items-center justify-between gap-2 text-[11px] text-neutral-500">
                    <span>{n.inventory?.gpus[0]?.name ?? 'no GPU reported'}</span>
                    <span>
                      <Measured
                        value={
                          n.inventory
                            ? n.inventory.gpus.reduce((s, g) => s + g.vram_free_bytes, 0) /
                              1024 ** 3
                            : null
                        }
                        unit="GiB free"
                      />
                    </span>
                  </div>
                </button>
              )
            })}
          </div>

          <div>{active && <NodeDetail nodeId={active} />}</div>
        </div>
      )}
    </div>
  )
}