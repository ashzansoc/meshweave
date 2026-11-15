import { useState } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  Badge,
  EmptyState,
  ErrorBanner,
  timeAgo,
} from '../lib/format'
import { usePolling } from '../lib/api'
import type { WorkloadDetail, WorkloadRecord } from '../api'

const STATUS_TONE: Record<string, 'neutral' | 'good' | 'warn' | 'bad' | 'info'> = {
  COMPLETED: 'good',
  RUNNING: 'info',
  STREAMING: 'info',
  FAILED: 'bad',
  CANCELLED: 'warn',
  PENDING: 'neutral',
}

function WorkloadRow({
  workload,
  onClick,
}: {
  workload: WorkloadRecord
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="w-full px-3 py-2.5 text-left hover:bg-neutral-800/40"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-neutral-400">
            {workload.workload_id}
          </span>
          <Badge tone={STATUS_TONE[workload.success ? 'COMPLETED' : 'FAILED'] ?? 'neutral'}>
            {workload.success ? 'completed' : 'failed'}
          </Badge>
        </span>
        <span className="text-[11px] text-neutral-500">
          {timeAgo(workload.started_at)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-neutral-500">
        <span className="text-neutral-400">{workload.model}</span>
        <span>{workload.strategy}</span>
        <span>{workload.nodes.length} node(s)</span>
        {workload.output_tokens > 0 && (
          <span>{workload.output_tokens} tok</span>
        )}
        {workload.tokens_per_second !== null && (
          <span>{workload.tokens_per_second.toFixed(1)} tok/s</span>
        )}
        {workload.error && <span className="text-red-400">{workload.error}</span>}
      </div>
    </button>
  )
}

function WorkloadDetailPanel({ workloadId }: { workloadId: string }) {
  const detail = usePolling<WorkloadDetail>(`/v1/workloads/${workloadId}`, 5000)
  const w = detail.data?.workload

  return (
    <Card
      title={workloadId}
      subtitle={w ? `${w.model} · ${w.strategy}` : undefined}
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

      {w && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
            <Stat label="Prompt tokens">{w.prompt_tokens}</Stat>
            <Stat label="Output tokens">{w.output_tokens}</Stat>
            <Stat label="Throughput">
              <Measured value={w.tokens_per_second} unit="tok/s" />
            </Stat>
            <Stat label="TTFT">
              <Measured value={w.ttft_ms} unit="ms" />
            </Stat>
            <Stat label="Inter-token p50">
              <Measured value={w.inter_token_latency_ms_p50} unit="ms" />
            </Stat>
            <Stat label="Cost">
              <Money value={w.cost_usd} />
            </Stat>
            <Stat label="Cost / M out">
              <Money value={w.cost_per_m_output_tokens_usd} />
            </Stat>
            <Stat label="Environment">
              <span className="text-sm">{w.environment}</span>
            </Stat>
          </div>

          {w.nodes.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
                Nodes
              </h3>
              <div className="flex flex-wrap gap-2">
                {w.nodes.map((n) => (
                  <span
                    key={n}
                    className="rounded border border-neutral-800 bg-neutral-950/50 px-2 py-1 font-mono text-[11px] text-neutral-300"
                  >
                    {n}
                  </span>
                ))}
              </div>
            </div>
          )}

          {detail.data && detail.data.events.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
                Event timeline
              </h3>
              <ol className="space-y-1.5 border-l border-neutral-800 pl-4">
                {detail.data.events.map((e, i) => (
                  <li key={i} className="relative text-xs">
                    <span className="absolute top-1.5 -left-[21px] h-2 w-2 rounded-full bg-neutral-700" />
                    <span className="text-neutral-500">
                      {new Date(e.at).toLocaleTimeString()}
                    </span>
                    <span className="ml-2 text-neutral-300">{e.kind}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

export function Workloads() {
  const workloads = usePolling<{ workloads: WorkloadRecord[]; active: number }>(
    '/v1/workloads',
  )
  const [selected, setSelected] = useState<string | null>(null)

  const list = workloads.data?.workloads ?? []
  const active = selected ?? list[0]?.workload_id ?? null

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-medium text-neutral-100">Workloads</h2>
          <p className="mt-0.5 text-sm text-neutral-500">
            Execution records. Each one is what the coordinator observed, including
            failures.
          </p>
        </div>
        <button
          onClick={() => workloads.refresh()}
          className="rounded border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300 hover:bg-neutral-800"
        >
          refresh
        </button>
      </div>

      {workloads.error && <ErrorBanner message={workloads.error} />}

      {list.length === 0 && !workloads.error && (
        <Card>
          <EmptyState title="No workload has been submitted">
            <p className="mb-3">
              Submit one from the planner, or run an experiment:
            </p>
            <pre className="overflow-x-auto rounded border border-neutral-800 bg-neutral-950 p-3 text-left text-[11px] text-neutral-400">
{`# plan and submit through the API
curl -X POST http://127.0.0.1:8700/v1/workloads \\
  -H 'content-type: application/json' \\
  -d '{"model":"Qwen2.5-7B-Instruct","quantization":"bf16"}'`}
            </pre>
          </EmptyState>
        </Card>
      )}

      {list.length > 0 && (
        <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
          <Card
            title={`${list.length} workload${list.length === 1 ? '' : 's'}`}
            subtitle={
              workloads.data
                ? `${workloads.data.active} active`
                : undefined
            }
            className="overflow-hidden"
          >
            <div className="-mx-5 -my-5 divide-y divide-neutral-800">
              {list.map((w) => (
                <WorkloadRow
                  key={w.workload_id}
                  workload={w}
                  onClick={() => setSelected(w.workload_id)}
                />
              ))}
            </div>
          </Card>
          <div>{active && <WorkloadDetailPanel workloadId={active} />}</div>
        </div>
      )}
    </div>
  )
}