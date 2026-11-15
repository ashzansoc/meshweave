import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * Catches render failures and shows them.
 *
 * Without this, any uncaught render error leaves an empty root and the user sees
 * a blank white page with no indication of what went wrong. That is the worst
 * possible failure mode for a diagnostic tool.
 */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error('MESHWEAVE dashboard crashed:', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div className="min-h-screen bg-neutral-950 p-6 text-neutral-200">
        <div className="mx-auto max-w-2xl">
          <h1 className="text-lg font-medium text-red-300">
            The dashboard failed to render
          </h1>
          <p className="mt-2 text-sm text-neutral-400">
            This is a real error, not a blank state. The message is below so it can
            be reported.
          </p>
          <pre className="mt-4 overflow-x-auto rounded border border-red-900 bg-neutral-900 p-4 text-xs text-red-200">
{error.message}
{'\n\n'}
{error.stack}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            className="mt-4 rounded border border-neutral-700 px-3 py-1.5 text-sm hover:bg-neutral-800"
          >
            Try again
          </button>
        </div>
      </div>
    )
  }
}