import { JSDOM } from 'jsdom'

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'http://localhost:5273/',
  pretendToBeVisual: true,
})
const w = dom.window as any
for (const key of ['window','document','HTMLElement','Element','Node','getComputedStyle','CustomEvent','Event','MutationObserver','DocumentFragment','SVGElement']) {
  try { (globalThis as any)[key] = w[key] ?? w } catch {}
}
try { Object.defineProperty(globalThis, 'navigator', { value: w.navigator, configurable: true }) } catch {}
;(globalThis as any).requestAnimationFrame = (cb: any) => setTimeout(cb, 0)
;(globalThis as any).cancelAnimationFrame = clearTimeout
;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true

const errors: string[] = []
const origError = console.error
console.error = (...args: any[]) => { errors.push(args.map(String).join(' ')) }

try {
  const { createRoot } = await import('react-dom/client')
  const React = await import('react')
  const App = (await import('./src/App')).default
  const { MemoryRouter } = await import('react-router-dom')
  const root = createRoot(document.getElementById('root')!)
  root.render(React.createElement(MemoryRouter, { initialEntries: ['/'] }, React.createElement(App)))
  await new Promise((r) => setTimeout(r, 900))
  console.log('HTML length:', document.getElementById('root')!.innerHTML.length)
  console.log('TEXT:', (document.body.textContent || '').replace(/\s+/g,' ').slice(0, 400))
} catch (e) {
  console.log('THREW:', (e as Error).message)
  console.log((e as Error).stack?.split('\n').slice(0,6).join('\n'))
}
console.log('\nconsole.error calls:', errors.length)
for (const e of errors.slice(0, 4)) console.log(' !', e.slice(0, 500))
