import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Polling data layer.
 *
 * Two rules shape this file:
 *
 *  1. A failed request yields an error, never an empty object. Rendering `{}`
 *     as if it were data is how a dashboard starts lying.
 *  2. Historical samples are kept only for the current session, and are labelled
 *     as such. They are not a time series from a database; they are what this
 *     browser observed while it was open.
 */

const POLL_MS = 3000

export async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(`/api${path}`)
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`)
  }
  return (await resp.json()) as T
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`${resp.status}: ${text.slice(0, 300)}`)
  }
  return (await resp.json()) as T
}

export interface Sample {
  at: number
  value: number
}

/** Keeps a bounded window of the last N observations of one metric. */
export function useSamples(value: number | null | undefined, limit = 40): Sample[] {
  const [samples, setSamples] = useState<Sample[]>([])
  const last = useRef<number | null>(null)

  useEffect(() => {
    if (value === null || value === undefined) return
    if (last.current !== null && Math.abs(last.current - value) < 1e-9) return
    last.current = value
    setSamples((prev) => [...prev, { at: Date.now(), value }].slice(-limit))
  }, [value, limit])

  return samples
}

export function usePolling<T>(path: string | null, intervalMs = POLL_MS) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [lastFetched, setLastFetched] = useState<number | null>(null)

  const refresh = useCallback(async () => {
    if (!path) return
    try {
      const result = await apiGet<T>(path)
      setData(result)
      setError(null)
      setLastFetched(Date.now())
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [path])

  useEffect(() => {
    void refresh()
    if (intervalMs <= 0) return
    const timer = setInterval(() => void refresh(), intervalMs)
    return () => clearInterval(timer)
  }, [refresh, intervalMs])

  return { data, error, loading, lastFetched, refresh, setData }
}