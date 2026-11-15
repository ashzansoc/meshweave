import { useMemo } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  EmptyState,
  ErrorBanner,
} from '../lib/format'
import { usePolling } from '../lib/api'
import type { NodeListResponse } from '../api'

/**
 * Economics view.
 *
 * Two rules shape this page:
 *
 *  1. A configuration whose price is unknown is listed as unpriced. It is never
 *     given an estimated cost, because a fabricated price is indistinguishable
 *     from a measured one once it is in a report.
 *  2. The crossover is shown when the planner found one. That is the number that
 *     decides whether renting more GPUs is defensible.
 */
export function Economics() {
  const nodes = usePolling<NodeListResponse>('/v1/nodes')

  const summary = nodes.data?.summary
  const nodeList = useMemo(() => nodes.data?.nodes ?? [], [nodes.data])

  const priced = nodeList.filter((n) => n.hourly_cost_usd !== null)
  const unpriced = nodeList.filter((n) => n.hourly_cost_usd === null)

  const knownCost = priced.length === nodeList.length && nodeList.length > 0

  // Cost per GPU-hour, and the aggregate if every price is known.
  const totalHourly =
    knownCost && summary ? summary.hourly_cost_usd : null

  const cheapest = priced.length
    ? priced.reduce((a, b) =>
        (a.hourly_cost_usd ?? Infinity) <= (b.hourly_cost_usd ?? Infinity) ? a : b,
      )
    : null

  const dearest = priced.length
    ? priced.reduce((a, b) =>
        (a.hourly_cost_usd ?? 0) >= (b.hourly_cost_usd ?? 0) ? a : b,
      )
    : null

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">Economics</h2>
        <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
          Cost per million output tokens is the number that decides the design. This
          page shows only prices the coordinator actually knows.
        </p>
      </div>

      {nodes.error && <ErrorBanner message={nodes.error} />}

      {!nodes.data && !nodes.error && (
        <Card>
          <EmptyState title="No node data" />
        </Card>
      )}

      {nodeList.length === 0 && nodes.data && (
        <Card>
          <EmptyState title="No nodes registered">
            Cost figures require at least one node with a recorded hourly price.
            Record one with:
            <div className="mt-2">
              <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`curl -X POST http://127.0.0.1:8700/v1/nodes/node-a/cost \\
  -H 'content-type: application/json' \\
  -d '{"hourly_cost_usd": 0.35}'`}
              </pre>
            </div>
          </EmptyState>
        </Card>
      )}

      {nodeList.length > 0 && (
        <>
          <div className="grid grid-cols-2 gap-6 rounded-lg border border-neutral-800 bg-neutral-900/40 p-5 sm:grid-cols-4">
            <Stat
              label="Cluster cost"
              hint="Only known when every node has a recorded price."
            >
              <Money value={totalHourly} per="/hr" />
            </Stat>
            <Stat label="Nodes priced">
              {priced.length}
              <span className="text-neutral-500">/{nodeList.length}</span>
            </Stat>
            <Stat label="Cheapest node">
              {cheapest ? (
                <span className="text-sm">
                  <span className="font-mono text-[12px]">{cheapest.node_id}</span>
                  <span className="ml-2 text-neutral-400">
                    ${cheapest.hourly_cost_usd!.toFixed(2)}/hr
                  </span>
                </span>
              ) : (
                <span className="text-sm text-neutral-500 italic">unknown</span>
              )}
            </Stat>
            <Stat label="Dearest node">
              {dearest ? (
                <span className="text-sm">
                  <span className="font-mono text-[12px]">{dearest.node_id}</span>
                  <span className="ml-2 text-neutral-400">
                    ${dearest.hourly_cost_usd!.toFixed(2)}/hr
                  </span>
                </span>
              ) : (
                <span className="text-sm text-neutral-500 italic">unknown</span>
              )}
            </Stat>
          </div>

          {unpriced.length > 0 && (
            <div className="rounded border border-amber-900/70 bg-amber-950/25 px-4 py-3">
              <h3 className="text-xs font-medium text-amber-300">
                {unpriced.length} node{unpriced.length === 1 ? '' : 's'} without a
                recorded price
              </h3>
              <p className="mt-1 text-xs text-amber-200/80">
                {unpriced.map((n) => n.node_id).join(', ')}. Aggregate cost is
                reported as unknown rather than computed from a guess, because a
                fabricated price is indistinguishable from a measured one.
              </p>
            </div>
          )}

          <Card
            title="Per-node pricing"
            subtitle="What MESHWEAVE knows, and what it does not"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-neutral-900 text-neutral-400">
                  <tr>
                    <th className="px-3 py-2 font-medium">Node</th>
                    <th className="px-3 py-2 font-medium">GPU</th>
                    <th className="px-3 py-2 font-medium">VRAM</th>
                    <th className="px-3 py-2 font-medium">$/hr</th>
                    <th className="px-3 py-2 font-medium">$/GiB-hr</th>
                    <th className="px-3 py-2 font-medium">Share of cluster</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {nodeList.map((n) => {
                    const vram = n.inventory
                      ? n.inventory.gpus.reduce((s, g) => s + g.vram_total_bytes, 0) /
                        1024 ** 3
                      : null
                    const perGib =
                      n.hourly_cost_usd !== null && vram && vram > 0
                        ? n.hourly_cost_usd / vram
                        : null
                    const share =
                      n.hourly_cost_usd !== null && totalHourly
                        ? (n.hourly_cost_usd / totalHourly) * 100
                        : null
                    return (
                      <tr key={n.node_id} className="text-neutral-300">
                        <td className="px-3 py-2 font-mono text-[11px]">
                          {n.node_id}
                        </td>
                        <td className="px-3 py-2">
                          {n.inventory?.gpus[0]?.name ?? '—'}
                        </td>
                        <td className="px-3 py-2">
                          <Measured value={vram} unit="GiB" />
                        </td>
                        <td className="px-3 py-2">
                          <Money value={n.hourly_cost_usd} />
                        </td>
                        <td className="px-3 py-2">
                          <Money value={perGib} digits={3} />
                        </td>
                        <td className="px-3 py-2">
                          <Measured value={share} unit="%" digits={0} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[11px] text-neutral-500">
              $/GiB-hr normalises price by memory, which is the fair comparison
              across heterogeneous GPUs. A cheap 3090 and an expensive 4090 are not
              directly comparable by hourly rate alone.
            </p>
          </Card>
        </>
      )}

      <Card
        title="Cost versus distribution"
        subtitle="Why this is the page that decides whether MESHWEAVE is worth building"
      >
        <div className="space-y-3 text-sm text-neutral-400">
          <p>
            Replication and partitioning are different products. Replication scales
            throughput and keeps the network off the hot path. Partitioning makes a
            model larger than one GPU possible, and pays a network round trip on
            every generated token.
          </p>
          <p>
            Open the{' '}
            <a href="#/planner" className="text-sky-400 hover:underline">
              planner
            </a>{' '}
            and choose <em>Compare configurations</em> to cost both against each
            other for a specific model and precision.
          </p>
          <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2">
            <p className="text-xs text-neutral-400">
              The result MESHWEAVE expects, stated in advance so it can be falsified:
              for any model that fits in one GPU at a lower precision, replication
              wins on cost per token by a wide margin. Partitioning is only justified
              when the model does not fit at all — and even then it is slow.
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}