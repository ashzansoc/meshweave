import { useState } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  Badge,
  EmptyState,
  ErrorBanner,
} from '../lib/format'
import { usePolling } from '../lib/api'
import type { ExperimentListResponse, ExperimentReport } from '../api'

const STATUS_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral' | 'info'> = {
  completed: 'good',
  blocked: 'warn',
  failed: 'bad',
  partial: 'info',
}

function ReportDetail({ experimentId }: { experimentId: string }) {
  const report = usePolling<ExperimentReport>(`/v1/experiments/${experimentId}`, 0)

  if (report.error) {
    return (
      <Card title={experimentId}>
        <p className="text-sm text-red-400">{report.error}</p>
      </Card>
    )
  }

  const r = report.data
  if (!r) {
    return (
      <Card title={experimentId}>
        <p className="text-sm text-neutral-500">loading…</p>
      </Card>
    )
  }

  return (
    <Card
      title={`${r.experiment_id} · ${r.title}`}
      subtitle={r.hypothesis}
      actions={
        <div className="flex items-center gap-2">
          <Badge tone={STATUS_TONE[r.status] ?? 'neutral'}>{r.status}</Badge>
          {r.provenance.synthetic && <Badge tone="warn">synthetic</Badge>}
        </div>
      }
    >
      <div className="space-y-5">
        {r.status === 'blocked' && (
          <div className="rounded border border-amber-900 bg-amber-950/30 px-3 py-2">
            <h3 className="text-xs font-medium text-amber-300">Blocked</h3>
            <p className="mt-1 text-sm text-amber-200/90">{r.blocked_reason}</p>
            {r.ready_to_run_command && (
              <div className="mt-2">
                <p className="mb-1 text-[11px] text-amber-400/80">
                  Ready to run once unblocked:
                </p>
                <pre className="overflow-x-auto rounded bg-neutral-950/80 p-2 text-[11px] text-amber-200">
{r.ready_to_run_command}
                </pre>
              </div>
            )}
          </div>
        )}

        {r.provenance.synthetic && (
          <p className="rounded border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-300">
            This report used synthetic GPU telemetry. It is structurally valid but is
            not evidence of real distributed execution, and MESHWEAVE will not count
            it as such.
          </p>
        )}

        {r.measurements.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
              Measurements ({r.measurements.length})
            </h3>
            <div className="overflow-x-auto rounded border border-neutral-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-neutral-900 text-neutral-400">
                  <tr>
                    <th className="px-3 py-2 font-medium">Measurement</th>
                    <th className="px-3 py-2 font-medium">Value</th>
                    <th className="px-3 py-2 font-medium">Unit</th>
                    <th className="px-3 py-2 font-medium">Samples</th>
                    <th className="px-3 py-2 font-medium">Note</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800">
                  {r.measurements.map((m, i) => (
                    <tr key={`${m.name}-${i}`} className="text-neutral-300">
                      <td className="px-3 py-2 font-mono text-[11px]">{m.name}</td>
                      <td className="px-3 py-2">
                        <Measured value={m.value} unit="" digits={3} />
                      </td>
                      <td className="px-3 py-2 text-neutral-500">{m.unit}</td>
                      <td className="px-3 py-2 tabular-nums text-neutral-500">
                        {m.samples}
                      </td>
                      <td className="px-3 py-2 text-[11px] text-neutral-500">
                        {m.note ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {r.economics && (
          <div>
            <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
              Economics
            </h3>
            {r.economics.cost_known ? (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Total cost">
                  <Money value={r.economics.total_cost_usd} />
                </Stat>
                <Stat label="Per M input tok">
                  <Money value={r.economics.cost_per_m_input_tokens_usd} />
                </Stat>
                <Stat label="Per M output tok">
                  <Money value={r.economics.cost_per_m_output_tokens_usd} />
                </Stat>
                <Stat label="Per request">
                  <Money value={r.economics.cost_per_request_usd} />
                </Stat>
              </div>
            ) : (
              <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2">
                <p className="text-xs text-neutral-400">
                  Cost is unknown for this experiment.
                </p>
                {r.economics.unknown_components.length > 0 && (
                  <ul className="mt-1 list-inside list-disc text-[11px] text-neutral-500">
                    {r.economics.unknown_components.map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        )}

        {r.findings.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
              Findings
            </h3>
            <ul className="list-inside list-disc space-y-1 text-sm text-neutral-300">
              {r.findings.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </div>
        )}

        {r.failures.failures.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs tracking-wide text-red-400 uppercase">
              Failures ({r.failures.failures.length})
            </h3>
            <ul className="space-y-1.5">
              {r.failures.failures.map((f, i) => (
                <li
                  key={i}
                  className="rounded border border-red-900/60 bg-red-950/20 px-3 py-2 text-xs"
                >
                  <span className="font-medium text-red-300">{f.kind}</span>
                  <p className="mt-0.5 text-red-200/80">{f.detail}</p>
                </li>
              ))}
            </ul>
          </div>
        )}

        {r.failures.failures.length === 0 && r.failures.failures_investigated && (
          <p className="text-xs text-neutral-500">
            No failures were recorded, and the run states that failures were
            investigated.
          </p>
        )}

        {r.evidence.length > 0 && (
          <div className="border-t border-neutral-800 pt-3">
            <h3 className="mb-1 text-[11px] tracking-wide text-neutral-500 uppercase">
              Evidence
            </h3>
            {r.evidence.map((e) => (
              <p key={e} className="font-mono text-[11px] text-neutral-500">
                {e}
              </p>
            ))}
          </div>
        )}

        <div className="border-t border-neutral-800 pt-3 text-[11px] text-neutral-600">
          <p>
            Generated {new Date(r.provenance.generated_at).toLocaleString()} on{' '}
            {r.provenance.host}
          </p>
          <p>
            commit {r.provenance.git_commit?.slice(0, 12) ?? 'unknown'}
            {r.provenance.git_dirty ? ' (dirty)' : ''} · meshweave{' '}
            {r.provenance.meshweave_version}
          </p>
        </div>
      </div>
    </Card>
  )
}

export function Experiments() {
  const experiments = usePolling<ExperimentListResponse>('/v1/experiments', 10000)
  const [selected, setSelected] = useState<string | null>(null)

  const list = experiments.data?.experiments ?? []
  const active = selected ?? list[0]?.experiment_id ?? null

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Experiments</h2>
          <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
            Reports written by the experiment harnesses. A blocked experiment is a
            legitimate result; a fabricated one is rejected by the report validator
            before it can be written.
          </p>
        </div>
        <button
          onClick={() => experiments.refresh()}
          className="rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          refresh
        </button>
      </div>

      {experiments.data && (
        <div className="grid grid-cols-2 gap-6 rounded-lg border border-neutral-800 bg-neutral-900/40 p-5 sm:grid-cols-4">
          <Stat label="Reports">{experiments.data.total}</Stat>
          <Stat label="Completed">{experiments.data.completed}</Stat>
          <Stat label="Blocked">{experiments.data.blocked}</Stat>
          <Stat label="Synthetic">
            {experiments.data.experiments.filter((e) => e.synthetic).length}
          </Stat>
        </div>
      )}

      {experiments.error && !experiments.error.includes('404') && (
        <ErrorBanner message={experiments.error} />
      )}

      {list.length === 0 && !experiments.error && (
        <Card>
          <EmptyState title="No experiment reports yet">
            <p className="mb-3">
              Run an experiment to produce one. Every report records its provenance,
              its failures, and its cost — or states what is missing.
            </p>
            <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`# always dry-run first: checks prerequisites, spends nothing
uv run python -m experiments.exp011_distributed_vram --dry-run

# EXP-011 is the project's first technical milestone:
# a model partitioned across two physical GPUs`}
            </pre>
          </EmptyState>
        </Card>
      )}

      {list.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
          <div className="space-y-2">
            {list.map((e) => (
              <button
                key={e.experiment_id}
                onClick={() => setSelected(e.experiment_id)}
                className={`w-full rounded-lg border px-3 py-2.5 text-left transition-colors ${
                  active === e.experiment_id
                    ? 'border-neutral-600 bg-neutral-800/60'
                    : 'border-neutral-800 bg-neutral-900/40 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] text-neutral-400">
                    {e.experiment_id}
                  </span>
                  <Badge tone={STATUS_TONE[e.status] ?? 'neutral'}>{e.status}</Badge>
                </div>
                <div className="mt-1 text-sm text-neutral-200">
                  {e.title ?? 'untitled'}
                </div>
                <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-neutral-500">
                  <span>{e.measurement_count} measurements</span>
                  {e.failure_count > 0 && (
                    <span className="text-red-400">{e.failure_count} failures</span>
                  )}
                  {e.synthetic && <span className="text-amber-400">synthetic</span>}
                </div>
              </button>
            ))}
          </div>
          <div>{active && <ReportDetail experimentId={active} />}</div>
        </div>
      )}
    </div>
  )
}