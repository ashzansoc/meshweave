import { useState } from 'react'
import { Badge, Card, ErrorBanner, Measured, timeAgo } from '../lib/format'
import { apiPost, usePolling } from '../lib/api'

interface ReadinessNode {
  node_id: string
  status: string
  gpus: number
  gpu_name: string | null
  vram_gib: number
  ram_available_gib: number | null
  disk_available_gib: number | null
  can_host: boolean
  mapped_ports: Record<string, number>
  external_ip: string | null
}

interface StageView {
  node_id: string
  rank: number
  layers: string
  count: number
  resident_gib: number
  checkpoint_gib: number
  address: string | null
  phase: string
  detail: string | null
}

interface DeploymentView {
  deployment_id: string
  model: string
  status: string
  message: string | null
  created_at: string
  strategy: string | null
  dtype: string | null
  layers: number | null
  stages: StageView[]
  results: {
    prompt?: string
    text?: string | null
    output_tokens?: number
    ms_per_token?: number | null
    ttft_ms?: number | null
    error?: string | null
    at?: string
  }[]
  pending_prompts: number
}

const PHASE_TONE: Record<string, 'good' | 'warn' | 'bad' | 'neutral'> = {
  running: 'good',
  loading: 'warn',
  starting: 'warn',
  pending: 'neutral',
  failed: 'bad',
  stopped: 'neutral',
}

/**
 * Put a model on a set of nodes.
 *
 * The plan is computed by the coordinator from the model's own config.json and
 * each node's reported free VRAM and disk. Nothing here decides that a model
 * fits because the totals add up: a refusal names the node and the shortfall.
 */
export function Deploy() {
  const [model, setModel] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [selected, setSelected] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('The capital of France is')
  const [active, setActive] = useState<string | null>(null)

  const readiness = usePolling<{ nodes: ReadinessNode[] }>('/v1/readiness', 5000)
  const deployments = usePolling<{ deployments: DeploymentView[] }>('/v1/deployments', 2500)

  const nodes = readiness.data?.nodes ?? []
  const current = deployments.data?.deployments.find((d) => d.deployment_id === active) ?? null

  function toggle(nodeId: string) {
    setSelected((prev) =>
      prev.includes(nodeId) ? prev.filter((n) => n !== nodeId) : [...prev, nodeId],
    )
  }

  async function launch() {
    setBusy(true)
    setError(null)
    try {
      const created = await apiPost<DeploymentView>('/v1/deployments', {
        model,
        node_ids: selected,
      })
      setActive(created.deployment_id)
      await deployments.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function sendPrompt() {
    if (!current) return
    setBusy(true)
    setError(null)
    try {
      await apiPost(`/v1/deployments/${current.deployment_id}/prompt`, {
        prompt,
        max_new_tokens: 32,
        temperature: 0,
      })
      await deployments.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  async function stopDeployment() {
    if (!current) return
    setBusy(true)
    try {
      await apiPost(`/v1/deployments/${current.deployment_id}/stop`, {})
      await deployments.refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-medium text-neutral-100">Deploy</h2>
        <p className="mt-0.5 max-w-3xl text-sm text-neutral-500">
          Paste a Hugging Face repository, choose the nodes, and the model is split
          by decoder layer across them. Each node holds only its own layers in VRAM;
          every node downloads the full checkpoint to disk.
        </p>
      </div>

      {error && <ErrorBanner message={error} />}

      <Card title="Model and nodes">
        <label className="block text-xs uppercase tracking-wide text-neutral-500">Model</label>
        <input
          value={model}
          onChange={(e) => setModel(e.target.value)}
          spellCheck={false}
          className="mt-1 w-full rounded border border-neutral-700 bg-neutral-900 px-2.5 py-1.5 font-mono text-sm text-neutral-200 outline-none focus:border-neutral-500"
          placeholder="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
        />

        <div className="mt-4 space-y-2">
          {nodes.length === 0 && (
            <p className="text-sm text-neutral-500">no nodes registered</p>
          )}
          {nodes.map((node) => {
            const ports = Object.keys(node.mapped_ports ?? {}).filter((p) => p !== '22')
            const selectedHere = selected.includes(node.node_id)
            return (
              <label
                key={node.node_id}
                className={`flex cursor-pointer flex-wrap items-center justify-between gap-3 rounded border px-3 py-2 ${
                  selectedHere
                    ? 'border-neutral-600 bg-neutral-800/60'
                    : 'border-neutral-800 hover:bg-neutral-900'
                }`}
              >
                <span className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selectedHere}
                    onChange={() => toggle(node.node_id)}
                    className="h-3.5 w-3.5 accent-neutral-300"
                  />
                  <span className="font-mono text-[12px] text-neutral-200">{node.node_id}</span>
                  {node.can_host ? <Badge tone="good">can host</Badge> : <Badge tone="bad">no GPU</Badge>}
                </span>
                <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-neutral-500">
                  <span>{node.gpu_name ?? 'no GPU'}</span>
                  <span>
                    <Measured value={node.vram_gib} unit="GiB VRAM" />
                  </span>
                  <span>
                    disk <Measured value={node.disk_available_gib ?? null} unit="GiB free" />
                  </span>
                  <span className={ports.length ? 'text-neutral-500' : 'text-amber-400'}>
                    {ports.length ? `ports ${ports.slice(0, 3).join(', ')}` : 'no published port'}
                  </span>
                </span>
              </label>
            )
          })}
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={() => void launch()}
            disabled={busy || selected.length === 0}
            className="rounded border border-neutral-600 bg-neutral-800 px-3 py-1.5 text-xs text-neutral-100 hover:bg-neutral-700 disabled:opacity-40"
          >
            {busy ? 'working…' : `deploy to ${selected.length || 0} node(s)`}
          </button>
          <span className="text-[11px] text-neutral-600">
            layers are split evenly across the selected nodes
          </span>
        </div>
      </Card>

      {(deployments.data?.deployments.length ?? 0) > 0 && (
        <Card title="Deployments">
          <ul className="divide-y divide-neutral-800">
            {deployments.data?.deployments.map((dep) => (
              <li key={dep.deployment_id} className="py-2">
                <button
                  onClick={() => setActive(dep.deployment_id)}
                  className="flex w-full flex-wrap items-center justify-between gap-2 text-left"
                >
                  <span className="font-mono text-[12px] text-neutral-200">{dep.model}</span>
                  <span className="flex items-center gap-3 text-[11px] text-neutral-500">
                    <Badge tone={dep.status === 'RUNNING' ? 'good' : dep.status === 'FAILED' ? 'bad' : 'warn'}>
                      {dep.status.toLowerCase()}
                    </Badge>
                    <span>{timeAgo(dep.created_at)}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {current && (
        <Card title={`${current.model}`} subtitle={`${current.deployment_id} · ${current.strategy ?? 'PIPELINE'} · ${current.dtype ?? ''}`}>
          {current.message && (
            <p className="mb-3 rounded border border-amber-900/60 bg-amber-950/25 px-3 py-2 text-xs text-amber-300">
              {current.message}
            </p>
          )}
          <ul className="divide-y divide-neutral-800">
            {current.stages.map((stage) => (
              <li key={stage.node_id} className="flex flex-wrap items-center justify-between gap-3 py-2">
                <span className="flex items-center gap-2">
                  <Badge tone={PHASE_TONE[stage.phase] ?? 'neutral'}>{stage.phase}</Badge>
                  <span className="font-mono text-[12px] text-neutral-200">{stage.node_id}</span>
                  <span className="text-[11px] text-neutral-500">
                    rank {stage.rank} · layers {stage.layers} ({stage.count})
                  </span>
                </span>
                <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-neutral-500">
                  <span>
                    VRAM <Measured value={stage.resident_gib} unit="GiB" />
                  </span>
                  <span>
                    disk <Measured value={stage.checkpoint_gib} unit="GiB" />
                  </span>
                  <span className="font-mono text-neutral-600">{stage.address ?? 'local'}</span>
                </span>
              </li>
            ))}
          </ul>
          {current.stages.some((s) => s.detail) && (
            <pre className="mt-3 max-h-32 overflow-auto rounded bg-neutral-950 p-2 text-[11px] text-neutral-500">
              {current.stages
                .filter((s) => s.detail)
                .map((s) => `${s.node_id}: ${s.detail}`)
                .join('\n')}
            </pre>
          )}

          <div className="mt-4 border-t border-neutral-800 pt-3">
            <label className="block text-xs uppercase tracking-wide text-neutral-500">Prompt</label>
            <div className="mt-1 flex gap-2">
              <input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-2.5 py-1.5 text-sm text-neutral-200 outline-none focus:border-neutral-500"
              />
              <button
                onClick={() => void sendPrompt()}
                disabled={busy || current.status !== 'RUNNING'}
                className="rounded border border-neutral-600 bg-neutral-800 px-3 py-1.5 text-xs text-neutral-100 hover:bg-neutral-700 disabled:opacity-40"
              >
                run
              </button>
              <button
                onClick={() => void stopDeployment()}
                disabled={busy}
                className="rounded border border-neutral-700 px-3 py-1.5 text-xs text-neutral-400 hover:bg-neutral-800 disabled:opacity-40"
              >
                stop
              </button>
            </div>
            {current.status !== 'RUNNING' && (
              <p className="mt-2 text-[11px] text-neutral-600">
                the prompt is accepted once every stage reports running
              </p>
            )}
          </div>

          {current.results.length > 0 && (
            <ul className="mt-3 space-y-2">
              {[...current.results].reverse().map((result, index) => (
                <li key={index} className="rounded border border-neutral-800 bg-neutral-900/50 p-2.5">
                  <div className="font-mono text-[11px] text-neutral-500">{result.prompt}</div>
                  {result.error ? (
                    <div className="mt-1 text-xs text-red-400">{result.error}</div>
                  ) : (
                    <>
                      <div className="mt-1 text-sm text-neutral-200">{result.text}</div>
                      <div className="mt-1 flex flex-wrap gap-x-4 text-[11px] text-neutral-500">
                        <span>{result.output_tokens} tokens</span>
                        <span>{result.ms_per_token ?? '—'} ms/token</span>
                        <span>{result.ttft_ms ?? '—'} ms to first token</span>
                      </div>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}
    </div>
  )
}
