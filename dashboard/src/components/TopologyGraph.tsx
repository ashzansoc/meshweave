import { useMemo, useState } from 'react'
import type { LinkMeasurement } from '../api'
import { Measured, EmptyState } from '../lib/format'

/**
 * Topology graph drawn from measured links only.
 *
 * Node placement is a deterministic circle, not a force simulation: a force
 * layout would re-arrange on every poll and make the picture impossible to
 * read. Circle placement keeps a node in the same place between refreshes.
 *
 * Edge colour encodes the *measured* environment class. An unmeasured link is
 * drawn as a dashed grey line rather than being omitted, because "we did not
 * measure this pair" is itself information.
 */

const ENV_COLOUR: Record<string, string> = {
  A_DATACENTER: '#10b981',
  B_REGIONAL: '#f59e0b',
  C_INTERNET: '#ef4444',
}

const ENV_LABEL: Record<string, string> = {
  A_DATACENTER: 'datacenter',
  B_REGIONAL: 'regional',
  C_INTERNET: 'internet',
}

function classify(rtt: number | null): string {
  if (rtt === null) return 'C_INTERNET'
  if (rtt <= 2) return 'A_DATACENTER'
  if (rtt <= 20) return 'B_REGIONAL'
  return 'C_INTERNET'
}

export function TopologyGraph({
  nodes,
  links,
  environments,
  width = 720,
  height = 420,
}: {
  nodes: string[]
  links: LinkMeasurement[]
  environments: Record<string, string>
  width?: number
  height?: number
}) {
  const [hovered, setHovered] = useState<string | null>(null)

  const positions = useMemo(() => {
    const cx = width / 2
    const cy = height / 2
    const radius = Math.min(width, height) / 2 - 70
    const map = new Map<string, { x: number; y: number }>()
    nodes.forEach((id, i) => {
      const angle = (i / Math.max(1, nodes.length)) * Math.PI * 2 - Math.PI / 2
      map.set(id, {
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
      })
    })
    return map
  }, [nodes, width, height])

  // One undirected edge per measured pair, so the drawing does not double up
  // when both directions were probed.
  const edges = useMemo(() => {
    const seen = new Set<string>()
    const out: Array<{ a: string; b: string; link: LinkMeasurement }> = []
    for (const link of links) {
      const key = [link.source, link.target].sort().join('|')
      if (seen.has(key)) continue
      seen.add(key)
      out.push({ a: link.source, b: link.target, link })
    }
    return out
  }, [links])

  if (nodes.length === 0) {
    return (
      <EmptyState title="No nodes to draw">
        Register at least one agent before the topology can be shown.
      </EmptyState>
    )
  }

  if (links.length === 0) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-neutral-400">
          {nodes.length} node{nodes.length === 1 ? '' : 's'} registered, but no link
          has been measured. MESHWEAVE draws only measured links, so there is no
          graph to show yet.
        </p>
        <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-[11px] text-neutral-400">
{`# measure every pair, then submit the matrix
uv run meshweave bench network \\
  --nodes host1:9000,host2:9000,host3:9000 \\
  --ids node-a,node-b,node-c \\
  --out reports/topology.json`}
        </pre>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4 text-xs">
        {Object.entries(ENV_LABEL).map(([key, label]) => (
          <span key={key} className="inline-flex items-center gap-1.5 text-neutral-400">
            <span
              className="inline-block h-0.5 w-6 rounded"
              style={{ background: ENV_COLOUR[key] }}
            />
            {label}
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5 text-neutral-400">
          <span className="inline-block h-0.5 w-6 rounded border-t border-dashed border-neutral-600" />
          unmeasured
        </span>
      </div>

      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Network topology: ${nodes.length} nodes, ${edges.length} measured links`}
      >
        {edges.map(({ a, b, link }) => {
          const pa = positions.get(a)
          const pb = positions.get(b)
          if (!pa || !pb) return null
          const env = link.rtt_ms_p50 !== null ? classify(link.rtt_ms_p50) : 'UNMEASURED'
          const colour =
            env === 'UNMEASURED' ? '#525252' : ENV_COLOUR[env] ?? '#525252'
          const isHovered = hovered === `${a}|${b}` || hovered === `${b}|${a}`
          return (
            <line
              key={`${a}-${b}`}
              x1={pa.x}
              y1={pa.y}
              x2={pb.x}
              y2={pb.y}
              stroke={colour}
              strokeWidth={isHovered ? 3 : 1.5}
              strokeOpacity={hovered && !isHovered ? 0.25 : 0.8}
              strokeDasharray={env === 'UNMEASURED' ? '4 4' : undefined}
            />
          )
        })}

        {nodes.map((id) => {
          const p = positions.get(id)
          if (!p) return null
          const env = environments[id] ?? 'C_INTERNET'
          const colour = ENV_COLOUR[env] ?? '#737373'
          return (
            <g key={id}>
              <circle cx={p.x} cy={p.y} r={9} fill="#0a0a0a" stroke={colour} strokeWidth={2} />
              <text
                x={p.x}
                y={p.y + 26}
                textAnchor="middle"
                className="fill-neutral-400"
                style={{ fontSize: 10, fontFamily: 'ui-monospace, monospace' }}
              >
                {id}
              </text>
            </g>
          )
        })}
      </svg>

      <div className="max-h-72 overflow-auto rounded border border-neutral-800">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-neutral-900 text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">Link</th>
              <th className="px-3 py-2 font-medium">Class</th>
              <th className="px-3 py-2 font-medium">RTT p50</th>
              <th className="px-3 py-2 font-medium">Jitter</th>
              <th className="px-3 py-2 font-medium">Loss</th>
              <th className="px-3 py-2 font-medium">Sustained</th>
              <th className="px-3 py-2 font-medium">UDP</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {edges.map(({ a, b, link }) => {
              const env = link.rtt_ms_p50 !== null ? classify(link.rtt_ms_p50) : null
              const key = `${a}|${b}`
              return (
                <tr
                  key={key}
                  className="text-neutral-300 hover:bg-neutral-800/40"
                  onMouseEnter={() => setHovered(key)}
                  onMouseLeave={() => setHovered(null)}
                >
                  <td className="px-3 py-2 font-mono text-[11px]">
                    {a} ↔ {b}
                  </td>
                  <td className="px-3 py-2">
                    {env ? (
                      <span
                        className="inline-flex items-center gap-1.5"
                        style={{ color: ENV_COLOUR[env] }}
                      >
                        <span
                          className="inline-block h-1.5 w-1.5 rounded-full"
                          style={{ background: ENV_COLOUR[env] }}
                        />
                        {ENV_LABEL[env]}
                      </span>
                    ) : (
                      <span className="text-neutral-500 italic">unmeasured</span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <Measured value={link.rtt_ms_p50} unit="ms" digits={2} />
                  </td>
                  <td className="px-3 py-2">
                    <Measured value={link.jitter_ms} unit="ms" digits={2} />
                  </td>
                  <td className="px-3 py-2">
                    <Measured value={link.packet_loss_pct} unit="%" />
                  </td>
                  <td className="px-3 py-2">
                    <Measured value={link.sustained_mbps} unit="Mbps" digits={0} />
                  </td>
                  <td className="px-3 py-2">
                    {link.udp_supported === null || link.udp_supported === undefined ? (
                      <span className="text-neutral-500 italic">—</span>
                    ) : link.udp_supported ? (
                      <span className="text-emerald-400">yes</span>
                    ) : (
                      <span className="text-neutral-400">no</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-neutral-500">
        {edges.length} measured link{edges.length === 1 ? '' : 's'} across {nodes.length}{' '}
        nodes. Each pair is probed in both directions because residential uplinks
        are frequently asymmetric.
      </p>
    </div>
  )
}