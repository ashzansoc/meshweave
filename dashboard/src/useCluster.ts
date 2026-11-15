import { useCallback, useEffect, useState } from 'react'
import type {
  ExecutionPlan,
  Metrics,
  NodeListResponse,
  TopologyResponse,
} from './api'

const POLL_MS = 3000

/** Fetch wrapper that never invents data: a failed call yields an error, not {}. */
async function get<T>(path: string): Promise<T> {
  const resp = await fetch(`/api${path}`)
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`)
  }
  return (await resp.json()) as T
}

export interface ClusterState {
  nodes: NodeListResponse | null
  metrics: Metrics | null
  topology: TopologyResponse | null
  plan: ExecutionPlan | null
  error: string | null
  loading: boolean
}

export function useCluster() {
  const [state, setState] = useState<ClusterState>({
    nodes: null,
    metrics: null,
    topology: null,
    plan: null,
    error: null,
    loading: true,
  })

  const refresh = useCallback(async () => {
    try {
      const [nodes, metrics] = await Promise.all([
        get<NodeListResponse>('/v1/nodes'),
        get<Metrics>('/v1/metrics'),
      ])
      let topology: TopologyResponse | null = null
      try {
        topology = await get<TopologyResponse>('/v1/topology')
      } catch {
        // No topology measured yet is a valid state, not an error.
        topology = null
      }
      setState((prev) => ({ ...prev, nodes, metrics, topology, error: null, loading: false }))
    } catch (err) {
      setState((prev) => ({
        ...prev,
        error: err instanceof Error ? err.message : String(err),
        loading: false,
      }))
    }
  }, [])

  const requestPlan = useCallback(
    async (model: string, quantization: string) => {
      try {
        const plan = await fetch('/api/v1/plan', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model, quantization, prompt_tokens: 128 }),
        }).then(async (r) => {
          if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`)
          return (await r.json()) as ExecutionPlan
        })
        setState((prev) => ({ ...prev, plan }))
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : String(err),
        }))
      }
    },
    [],
  )

  useEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_MS)
    return () => clearInterval(timer)
  }, [refresh])

  return { ...state, refresh, requestPlan }
}