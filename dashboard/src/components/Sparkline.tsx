import { useId } from 'react'
import type { Sample } from '../lib/api'

/**
 * A sparkline of what this browser actually observed.
 *
 * There is no historical database behind this. Each point is one poll the user's
 * browser made while this page was open. When the page reloads the history
 * resets. That is a real limitation and the chart is labelled with it rather
 * than implying a stored time series.
 */
export function Sparkline({
  samples,
  width = 220,
  height = 40,
  stroke = 'currentColor',
  label,
}: {
  samples: Sample[]
  width?: number
  height?: number
  stroke?: string
  label?: string
}) {
  const gradientId = useId()

  if (samples.length < 2) {
    return (
      <div
        className="flex items-center text-[11px] text-neutral-600 italic"
        style={{ height }}
      >
        collecting samples…
      </div>
    )
  }

  const values = samples.map((s) => s.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const pad = 3

  const points = samples.map((s, i) => {
    const x = pad + (i / (samples.length - 1)) * (width - pad * 2)
    const y = height - pad - ((s.value - min) / span) * (height - pad * 2)
    return [x, y] as const
  })

  const path = points
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(' ')
  const areaPath = `${path} L${points[points.length - 1]![0].toFixed(2)},${height - pad} L${points[0]![0].toFixed(2)},${height - pad} Z`

  const last = values[values.length - 1]!

  return (
    <div className="inline-block" title={label}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={
          label ??
          `Recent values, from ${min.toFixed(2)} to ${max.toFixed(2)}, latest ${last.toFixed(2)}`
        }
      >
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={stroke} stopOpacity="0.28" />
            <stop offset="100%" stopColor={stroke} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill={`url(#${gradientId})`} />
        <path
          d={path}
          fill="none"
          stroke={stroke}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle
          cx={points[points.length - 1]![0]}
          cy={points[points.length - 1]![1]}
          r="2.5"
          fill={stroke}
        />
      </svg>
    </div>
  )
}