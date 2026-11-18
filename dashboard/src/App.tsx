import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { Overview } from './views/Overview'
import { Nodes } from './views/Nodes'
import { Topology } from './views/Topology'
import { Planner } from './views/Planner'
import { Workloads } from './views/Workloads'
import { Experiments } from './views/Experiments'
import { Economics } from './views/Economics'
import { Network } from './views/Network'
import { Clusters } from './views/Clusters'
import { Deploy } from './views/Deploy'
import { Readiness } from './views/Readiness'
import { usePolling } from './lib/api'
import { timeAgo } from './lib/format'

interface Health {
  status: string
  version: string
  nodes: number
  healthy_nodes: number
  topology_measured: boolean
}

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/nodes', label: 'Nodes' },
  { to: '/readiness', label: 'Readiness' },
  { to: '/deploy', label: 'Deploy' },
  { to: '/clusters', label: 'Clusters' },
  { to: '/topology', label: 'Topology' },
  { to: '/network', label: 'Network' },
  { to: '/planner', label: 'Planner' },
  { to: '/workloads', label: 'Workloads' },
  { to: '/experiments', label: 'Experiments' },
  { to: '/economics', label: 'Economics' },
]

function NavItem({
  to,
  label,
  end,
}: {
  to: string
  label: string
  end?: boolean
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `block rounded px-3 py-1.5 text-sm transition-colors ${
          isActive
            ? 'bg-neutral-800 text-neutral-100'
            : 'text-neutral-400 hover:bg-neutral-900 hover:text-neutral-200'
        }`
      }
    >
      {label}
    </NavLink>
  )
}

export default function App() {
  const health = usePolling<Health>('/healthz', 5000)

  const h = health.data
  const statusColour =
    health.error
      ? 'bg-red-500'
      : h?.status === 'ok'
        ? 'bg-emerald-500'
        : 'bg-amber-500'

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-200">
      <div className="mx-auto flex max-w-[1500px]">
        {/* Sidebar */}
        <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-neutral-800 px-3 py-5 md:flex">
          <div className="px-2">
            <h1 className="text-base font-medium tracking-tight text-neutral-100">
              MESHWEAVE
            </h1>
            <p className="mt-0.5 text-[11px] text-neutral-500">
              Many Machines. One Intelligence.
            </p>
          </div>

          <nav className="mt-6 space-y-0.5">
            {NAV.map((item) => (
              <NavItem key={item.to} {...item} />
            ))}
          </nav>

          <div className="mt-auto space-y-2 px-2 text-[11px] text-neutral-500">
            <div className="flex items-center gap-2">
              <span
                className={`inline-block h-2 w-2 rounded-full ${statusColour}`}
                aria-hidden
              />
              <span>
                {health.error
                  ? 'coordinator down'
                  : h
                    ? `coordinator v${h.version}`
                    : 'connecting…'}
              </span>
            </div>
            {h && !health.error && (
              <>
                <div>
                  {h.healthy_nodes}/{h.nodes} nodes healthy
                </div>
                <div className={h.topology_measured ? 'text-emerald-500' : ''}>
                  topology {h.topology_measured ? 'measured' : 'not measured'}
                </div>
              </>
            )}
            {health.lastFetched && (
              <div className="text-neutral-600">
                updated {timeAgo(new Date(health.lastFetched).toISOString())}
              </div>
            )}
          </div>
        </aside>

        {/* Main */}
        <main className="min-w-0 flex-1">
          {/* Mobile nav */}
          <div className="sticky top-0 z-10 flex gap-1 overflow-x-auto border-b border-neutral-800 bg-neutral-950/95 px-3 py-2 backdrop-blur md:hidden">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `shrink-0 rounded px-2.5 py-1 text-xs ${
                    isActive
                      ? 'bg-neutral-800 text-neutral-100'
                      : 'text-neutral-400'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>

          <div className="px-5 py-6 lg:px-8">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/nodes" element={<Nodes />} />
              <Route path="/readiness" element={<Readiness />} />
              <Route path="/deploy" element={<Deploy />} />
              <Route path="/clusters" element={<Clusters />} />
              <Route path="/topology" element={<Topology />} />
              <Route path="/network" element={<Network />} />
              <Route path="/planner" element={<Planner />} />
              <Route path="/workloads" element={<Workloads />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/economics" element={<Economics />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}