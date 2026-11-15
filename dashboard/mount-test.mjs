import { JSDOM } from 'jsdom'
import { readFileSync } from 'node:fs'
import { TextEncoder, TextDecoder } from 'node:util'

const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'http://127.0.0.1:5273/', pretendToBeVisual: true,
})
const w = dom.window
globalThis.window = w; globalThis.document = w.document
globalThis.TextEncoder = TextEncoder; globalThis.TextDecoder = TextDecoder
w.TextEncoder = TextEncoder; w.TextDecoder = TextDecoder
Object.defineProperty(globalThis, 'navigator', { value: w.navigator, configurable: true })
for (const k of ['HTMLElement','Element','Node','Event','CustomEvent','getComputedStyle','MutationObserver','DocumentFragment','SVGElement']) {
  if (w[k] && !globalThis[k]) globalThis[k] = w[k]
}
globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0)
globalThis.cancelAnimationFrame = clearTimeout
w.requestAnimationFrame = globalThis.requestAnimationFrame
w.cancelAnimationFrame = globalThis.cancelAnimationFrame

const errors = []
w.addEventListener('error', e => errors.push('window error: ' + e.message))
const origErr = console.error
console.error = (...a) => { errors.push(a.map(x => x?.stack || String(x)).join(' ').slice(0, 300)) }

w.fetch = (u, o) => fetch(String(u).startsWith('http') ? u : 'http://127.0.0.1:8700' + u, o)
globalThis.fetch = w.fetch

const bundle = readFileSync('./dist/assets/' + process.argv[2], 'utf8')
try { w.eval(bundle) } catch (e) { errors.push('THREW: ' + e.message + '\n' + (e.stack||'').split('\n').slice(0,4).join('\n')) }

await new Promise(r => setTimeout(r, 1800))
const root = w.document.getElementById('root')
console.log('root children:', root.children.length)
console.log('TEXT:', (root.textContent || '').replace(/\s+/g, ' ').slice(0, 300))
console.log('\nERRORS:', errors.length)
for (const e of errors.slice(0, 3)) console.log(' !', e)
