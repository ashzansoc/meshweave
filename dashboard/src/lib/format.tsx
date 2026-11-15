import type { ReactNode } from 'react'

/** Bytes are always shown in GiB with one decimal — the unit engineers read. */
export const GIB = 1024 ** 3

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—'
  return `${(bytes / GIB).toFixed(1)} GiB`
}

export function formatNumber(
  value: number | null | undefined,
  digits = 1,
): string {
  if (value === null || value === undefined) return '—'
  return value.toFixed(digits)
}

/**
 * A value that may not have been measured.
 *
 * Renders "not measured" rather than 0. This is the single most important
 * primitive in the frontend: every number that could be absent goes through it.
 */
export function Measured({
  value,
  unit,
  digits = 1,
  className = '',
}: {
  value: number | null | undefined
  unit?: string
  digits?: number
  className?: string
}) {
  if (value === null || value === undefined) {
    return (
      <span className={`text-neutral-500 italic ${className}`} title="Not measured">
        not measured
      </span>
    )
  }
  return (
    <span className={className}>
      {value.toFixed(digits)}
      {unit && <span className="ml-1 text-neutral-500">{unit}</span>}
    </span>
  )
}

/** Money is only shown when the coordinator actually knows the price. */
export function Money({
  value,
  per,
  digits = 2,
  className = '',
}: {
  value: number | null | undefined
  per?: string
  digits?: number
  className?: string
}) {
  if (value === null || value === undefined) {
    return (
      <span className={`text-neutral-500 italic ${className}`} title="Cost unknown">
        unknown
      </span>
    )
  }
  return (
    <span className={className}>
      ${value.toFixed(digits)}
      {per && <span className="ml-1 text-neutral-500">{per}</span>}
    </span>
  )
}

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-lg border border-neutral-800 bg-neutral-900/40 ${className}`}
    >
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-2 border-b border-neutral-800 px-5 py-3">
          <div>
            {title && (
              <h2 className="text-sm font-medium tracking-wide text-neutral-200">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-neutral-500">{subtitle}</p>
            )}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

export function Stat({
  label,
  children,
  hint,
}: {
  label: string
  children: ReactNode
  hint?: string
}) {
  return (
    <div title={hint}>
      <div className="text-[11px] tracking-wide text-neutral-500 uppercase">
        {label}
      </div>
      <div className="mt-1 text-lg text-neutral-100 tabular-nums">{children}</div>
    </div>
  )
}

const STATUS_STYLES: Record<string, string> = {
  HEALTHY: 'bg-emerald-500',
  DEGRADED: 'bg-amber-500',
  UNREACHABLE: 'bg-red-500',
  DRAINING: 'bg-sky-500',
  RETIRED: 'bg-neutral-600',
  REGISTERING: 'bg-neutral-500',
}

export function StatusDot({ status }: { status: string }) {
  const colour = STATUS_STYLES[status] ?? 'bg-neutral-500'
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${colour}`}
      aria-label={status}
    />
  )
}

export function Badge({
  children,
  tone = 'neutral',
}: {
  children: ReactNode
  tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'info'
}) {
  const tones = {
    neutral: 'border-neutral-700 bg-neutral-800/60 text-neutral-300',
    good: 'border-emerald-900 bg-emerald-950/50 text-emerald-300',
    warn: 'border-amber-900 bg-amber-950/50 text-amber-300',
    bad: 'border-red-900 bg-red-950/50 text-red-300',
    info: 'border-sky-900 bg-sky-950/50 text-sky-300',
  }
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  )
}

export function EmptyState({
  title,
  children,
}: {
  title: string
  children?: ReactNode
}) {
  return (
    <div className="rounded border border-dashed border-neutral-800 px-4 py-8 text-center">
      <p className="text-sm text-neutral-400">{title}</p>
      {children && (
        <div className="mx-auto mt-2 max-w-xl text-xs text-neutral-500">{children}</div>
      )}
    </div>
  )
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
      <strong className="font-medium">Coordinator unreachable.</strong> {message}
      <div className="mt-1 text-xs text-red-400/80">
        Start it with <code className="text-red-200">uv run meshweave coordinator --port 8700</code>
      </div>
    </div>
  )
}

export function Code({ children }: { children: ReactNode }) {
  return (
    <code className="rounded bg-neutral-800/80 px-1.5 py-0.5 font-mono text-[11px] text-neutral-300">
      {children}
    </code>
  )
}

/** Relative time, computed client-side from a timestamp the server supplied. */
export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return 'unknown'
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}