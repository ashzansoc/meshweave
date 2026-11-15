import { useState } from 'react'
import {
  Card,
  Stat,
  Measured,
  Money,
  Badge,
  EmptyState,
} from '../lib/format'
import { apiPost } from '../lib/api'
import type { ExecutionPlan, PlanComparison } from '../api'

const MODELS = [
  'Qwen2.5-1.5B-Instruct',
  'Qwen2.5-3B-Instruct',
  'Qwen2.5-7B-Instruct',
  'Qwen2.5-14B-Instruct',
  'Qwen2.5-32B-Instruct',
  'Qwen2.5-72B-Instruct',
]

const STRATEGY_LABEL: Record<string, string> = {
  SINGLE_GPU: 'single GPU',
  REPLICATED: 'replicated',
  PIPELINE: 'pipeline',
  TENSOR: 'tensor parallel',
  EXPERT: 'expert parallel',
}

function PlanResult({ plan }: { plan: ExecutionPlan }) {
  const stages = Object.entries(plan.stage_assignment).sort(
    (a, b) => Number(a[0]) - Number(b[0]),
  )

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={plan.viable ? 'good' : 'bad'}>
          {plan.viable ? 'viable' : 'rejected'}
        </Badge>
        <Badge tone="info">{STRATEGY_LABEL[plan.strategy] ?? plan.strategy}</Badge>
        <Badge>{plan.environment}</Badge>
        <span className="font-mono text-[10px] text-neutral-600">{plan.plan_id}</span>
      </div>

      {!plan.viable && plan.rejected_reason && (
        <p className="rounded border border-red-900 bg-red-950/40 px-3 py-2 text-sm text-red-300">
          {plan.rejected_reason}
        </p>
      )}

      <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
        <Stat label="Est. throughput">
          <Measured value={plan.estimated_throughput_tps} unit="tok/s" />
        </Stat>
        <Stat label="Est. inter-token">
          <Measured value={plan.estimated_itl_ms} unit="ms" />
        </Stat>
        <Stat label="Est. TTFT">
          <Measured value={plan.estimated_ttft_ms} unit="ms" />
        </Stat>
        <Stat label="Est. cost">
          <Money value={plan.estimated_cost_per_m_output_tokens} per="/M tok" />
        </Stat>
      </div>

      {stages.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
            Stage assignment
          </h3>
          <ol className="space-y-1.5">
            {stages.map(([index, nodeId]) => {
              const layers = plan.layer_assignment[nodeId]
              return (
                <li
                  key={index}
                  className="flex flex-wrap items-center gap-3 rounded border border-neutral-800 bg-neutral-950/50 px-3 py-2 text-sm"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-neutral-800 text-[11px] text-neutral-300">
                    {index}
                  </span>
                  <span className="font-mono text-[12px] text-neutral-200">
                    {nodeId}
                  </span>
                  {layers && (
                    <span className="text-xs text-neutral-500">
                      layers {layers[0]}–{layers[1]! - 1}
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>
      )}

      {plan.replica_nodes.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
            Replicas ({plan.replica_nodes.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {plan.replica_nodes.map((n) => (
              <span
                key={n}
                className="rounded border border-neutral-800 bg-neutral-950/50 px-2 py-1 font-mono text-[11px] text-neutral-300"
              >
                {n}
              </span>
            ))}
          </div>
          <p className="mt-2 text-xs text-neutral-500">
            Each replica holds the full model. Requests are distributed across them;
            no activation crosses the network.
          </p>
        </div>
      )}

      {plan.rationale.length > 0 && (
        <div>
          <h3 className="mb-2 text-xs tracking-wide text-neutral-400 uppercase">
            Why this plan
          </h3>
          <ul className="list-inside list-disc space-y-1 text-sm text-neutral-400">
            {plan.rationale.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {plan.disconnect_impact && (
        <div className="rounded border border-amber-900/60 bg-amber-950/25 px-3 py-2">
          <h3 className="text-xs font-medium text-amber-300">
            If a node disconnects
          </h3>
          <p className="mt-1 text-sm text-amber-200/90">{plan.disconnect_impact}</p>
        </div>
      )}

      {plan.warnings.length > 0 && (
        <ul className="space-y-1">
          {plan.warnings.map((w) => (
            <li key={w} className="text-xs text-amber-400">
              ⚠ {w}
            </li>
          ))}
        </ul>
      )}

      {plan.assumptions.length > 0 && (
        <div className="border-t border-neutral-800 pt-3">
          <h3 className="mb-1 text-[11px] tracking-wide text-neutral-500 uppercase">
            Assumptions
          </h3>
          {plan.assumptions.map((a) => (
            <p key={a} className="text-[11px] text-neutral-500">
              {a}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

function ComparisonTable({ comparison }: { comparison: PlanComparison }) {
  const rows = comparison.configurations
  if (rows.length === 0) {
    return <EmptyState title="No configurations could be costed" />
  }

  const best = comparison.cheapest_by_cost_per_m_output_tokens?.label

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto rounded border border-neutral-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-neutral-900 text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">Configuration</th>
              <th className="px-3 py-2 font-medium">Nodes</th>
              <th className="px-3 py-2 font-medium">$/hr</th>
              <th className="px-3 py-2 font-medium">tok/s</th>
              <th className="px-3 py-2 font-medium">$/M output tok</th>
              <th className="px-3 py-2 font-medium">tok/s per $</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {rows.map((r) => (
              <tr
                key={r.label}
                className={
                  r.label === best
                    ? 'bg-emerald-950/20 text-emerald-100'
                    : 'text-neutral-300'
                }
              >
                <td className="px-3 py-2">
                  <span className="flex items-center gap-2">
                    {r.label}
                    {r.label === best && <Badge tone="good">cheapest</Badge>}
                  </span>
                </td>
                <td className="px-3 py-2 tabular-nums">{r.nodes}</td>
                <td className="px-3 py-2">
                  <Money value={r.hourly_cost_usd} />
                </td>
                <td className="px-3 py-2">
                  <Measured value={r.throughput_tps} unit="" />
                </td>
                <td className="px-3 py-2">
                  {r.cost_per_m_output_tokens_usd !== null ? (
                    `$${r.cost_per_m_output_tokens_usd.toFixed(2)}`
                  ) : (
                    <span className="text-neutral-500 italic">unknown</span>
                  )}
                </td>
                <td className="px-3 py-2">
                  <Measured value={r.throughput_per_dollar} digits={1} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {comparison.unpriced.length > 0 && (
        <p className="text-xs text-neutral-500">
          Not costed because a node price is unknown:{' '}
          {comparison.unpriced.join(', ')}. MESHWEAVE does not estimate a missing
          price.
        </p>
      )}

      {comparison.crossover && (
        <div className="rounded border border-neutral-800 bg-neutral-950/60 px-3 py-2">
          <h4 className="text-xs font-medium text-neutral-300">Crossover</h4>
          <p className="mt-1 text-xs text-neutral-400">
            {comparison.crossover.explanation}
          </p>
          {comparison.crossover.cost_before !== null &&
            comparison.crossover.cost_after !== null && (
              <p className="mt-1 text-[11px] text-neutral-500">
                ${comparison.crossover.cost_before.toFixed(2)} → $
                {comparison.crossover.cost_after.toFixed(2)} per million output
                tokens
              </p>
            )}
        </div>
      )}
    </div>
  )
}

export function Planner() {
  const [model, setModel] = useState('Qwen2.5-7B-Instruct')
  const [quantization, setQuantization] = useState('bf16')
  const [plan, setPlan] = useState<ExecutionPlan | null>(null)
  const [comparison, setComparison] = useState<PlanComparison | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async (mode: 'plan' | 'compare') => {
    setBusy(true)
    setError(null)
    try {
      const body = { model, quantization, prompt_tokens: 128, max_new_tokens: 256 }
      if (mode === 'plan') {
        setPlan(await apiPost<ExecutionPlan>('/v1/plan', body))
      } else {
        setComparison(await apiPost<PlanComparison>('/v1/plan/compare', body))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">
          Execution Intelligence
        </h2>
        <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
          The deterministic planner answers nine questions — including when the
          answer is to use a single GPU. It can refuse a configuration, and when
          it does it says why.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs">
            <span className="mb-1 block text-neutral-500">Model</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="rounded border border-neutral-700 bg-neutral-900 px-2.5 py-1.5 text-sm text-neutral-200"
            >
              {MODELS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>

          <label className="text-xs">
            <span className="mb-1 block text-neutral-500">Precision</span>
            <select
              value={quantization}
              onChange={(e) => setQuantization(e.target.value)}
              className="rounded border border-neutral-700 bg-neutral-900 px-2.5 py-1.5 text-sm text-neutral-200"
            >
              <option value="bf16">bf16</option>
              <option value="fp16">fp16</option>
              <option value="int8">int8</option>
              <option value="int4">int4</option>
            </select>
          </label>

          <button
            onClick={() => void run('plan')}
            disabled={busy}
            className="rounded bg-neutral-100 px-3 py-1.5 text-sm font-medium text-neutral-900 hover:bg-white disabled:opacity-50"
          >
            Plan
          </button>
          <button
            onClick={() => void run('compare')}
            disabled={busy}
            className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-200 hover:bg-neutral-800 disabled:opacity-50"
          >
            Compare configurations
          </button>
        </div>

        {error && (
          <p className="mt-4 rounded border border-red-900 bg-red-950/40 px-3 py-2 text-sm text-red-300">
            {error}
          </p>
        )}
      </Card>

      {plan && (
        <Card
          title="Selected plan"
          subtitle={`${plan.model} at ${quantization}`}
        >
          <PlanResult plan={plan} />
        </Card>
      )}

      {comparison && (
        <Card
          title="Configuration comparison"
          subtitle="Replication versus partitioning, costed per million output tokens"
        >
          <ComparisonTable comparison={comparison} />
        </Card>
      )}

      {!plan && !comparison && !error && (
        <Card>
          <EmptyState title="No plan requested yet">
            Choose a model and precision, then ask the planner. If the model fits on
            one GPU, the planner will say so rather than distributing it — forcing
            distribution would be a defect, not a conservative default.
          </EmptyState>
        </Card>
      )}
    </div>
  )
}