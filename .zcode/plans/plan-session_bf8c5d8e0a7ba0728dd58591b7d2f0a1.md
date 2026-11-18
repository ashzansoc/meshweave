## FLOAT16 × Meshweave — Foundation Slice Plan

### What this is

Integrate Meshweave's distributed GPU capabilities **into the existing Float16 Next.js app** (`/Users/ashutosh/Downloads/shakti`) as native Float16 screens, wired to the real Meshweave coordinator through a server-side proxy. No separate dashboard. Meshweave branding never appears in the UI.

**Scope: Phases 1–4 of the 16-part spec.** Concretely: audit + IA docs, a typed meshweave client library, Next.js proxy API routes with auth, and five real screens (Overview, GPU Inventory, Nodes, Virtual Datacenters, Network Intelligence). Later phases (inference deployments, serving, benchmarks, billing wiring) are left with clean seams but not built.

### Key facts that shape the design (from reading both repos)

- **Float16** is Next.js 16 / React 19 / Tailwind v4 (CSS-first `@theme` in `src/app/globals.css`), App Router, Firebase Auth + HMAC session cookie, granular IAM (`src/lib/iam/`), custom UI primitives in `src/components/ui/`, nav driven entirely by `src/lib/nav.ts`. Design tokens: `--f16-*` brand, Inter + JetBrains Mono, dark default. There is **no component test framework** (Node `--test` only).
- **Meshweave coordinator** is FastAPI on `127.0.0.1:8700` with an already-F16-shaped API: `/v1/nodes`, `/v1/readiness`, `/v1/deployments`, `/v1/clusters` (IDs literally `F16-VDC-001`), `/v1/topology`, `/v1/network/measure`, `/v1/network/intelligence`, `/v1/plan`, `/v1/metrics`, `/v1/experiments`. **Almost nothing is authenticated** — only `DELETE /v1/nodes/{id}` and `/v1/admin/reap` require the admin secret. Therefore the coordinator must never face the browser; the Next.js backend proxies every call and holds secrets server-side.
- **Meshweave's honesty rule matches Float16's**: missing values are `None`/nullable, never 0. I will preserve this end-to-end (nullable TS types → `<Measured>`-style rendering "not measured").

### Phase 1 — Frontend audit doc

Deliver `docs/FLOAT16_FRONTEND_AUDIT.md`: stack, route inventory, component/token inventory (pointing at real files), auth/RBAC summary, billing surface, the trust/fleet nodes model, and an inventory of reusable components per new screen. (I have most of this from the exploration; I'll verify details by reading the files before writing.)

### Phase 2 — Information architecture doc

Deliver `docs/FLOAT16_PRODUCT_INFORMATION_ARCHITECTURE.md`: new route tree, page-by-page data contracts, empty/loading/error states, and an explicit mapping table from each FLOAT16 spec API group (`/api/infrastructure/*`, `/api/vdc/*`, `/api/network/*`…) to Meshweave coordinator endpoints, marking which are real today vs not-yet-implemented (streaming, tensor parallelism, per-request cost for unpriced nodes, persistent time-series — these get "not measured"/"not yet available" states, never fake numbers).

### Phase 3 — Integration layer (`src/lib/meshweave/`)

Modeled on the existing `src/lib/baremetal/` provider pattern:

- `types.ts` — TypeScript mirrors of Meshweave `core.py` models: `NodeRecord`, `GpuDevice`, `HostResources`, `NetworkInfo`, `LinkMeasurement`, `TopologyMatrix`, `ClusterSummary`, `DeploymentSummary`, `PlanComparison`, `ReadinessEntry`, `Metrics`, `ExperimentReport`. **Every numeric field nullable.**
- `client.ts` — `server-only` HTTP client for the coordinator with timeout + typed errors; reads `MESHWEAVE_COORDINATOR_URL` and `MESHWEAVE_COORD_ADMIN_TOKEN` from env.
- `mock.ts` — deterministic mock client implementing the same interface, with fixture data based on the recorded POC snapshot shape (`reports/deployments.json`, RTX 2070S + GTX 1660S style data). Every mock response tagged `source: "mock"`.
- `index.ts` — provider selection: coordinator client when configured and reachable, mock otherwise; a `source` field flows to the UI so a **"Demo data"** badge renders wherever mock data appears. Never silently presented as live.
- `flags.ts` — `FLOAT16_MESHWEAVE_ENABLED` flag, consistent with the baremetal flags pattern.

### Phase 3b — Proxy API routes

Thin authenticated routes under `src/app/api/meshweave/*` (one catch-all-ish set mirroring the paths we need: `nodes`, `nodes/[id]`, `readiness`, `metrics`, `clusters`, `clusters/[id]`, `network/topology`, `network/measurements`, `network/intelligence`, `experiments`). Each route: `requireSession` + `requirePermission("meshweave.*")` + org/project scoping, following `src/app/api/compute/instances/route.ts` as the template. Admin-only coordinator operations (delete node, reap) additionally require the coordinator admin token server-side; the browser never sees it.

### Phase 3c — IAM + nav wiring

- Add a `meshweave` permission group to `src/lib/iam/permissions.ts` (`meshweave.nodes.read/manage`, `meshweave.vdc.read/manage`, `meshweave.network.read`, `meshweave.deploy.*` for later phases) and grant reads to `developer`/`viewer`, writes to `admin`/`owner` in `roles.ts`.
- Add nav entries to `src/lib/nav.ts`: a **"Distributed GPU"** group — Overview (or fold into existing `/`), GPU Inventory, Nodes, Virtual Datacenters, Network Intelligence — with `permission` fields so the sidebar auto-filters. Also add command-palette items.

### Phase 4 — Screens

All screens reuse `src/components/ui/*` (Card, Table, Badge/StatusBadge, Button, Select, Input, Tabs, Dialog, EmptyState, PageHeader, Skeleton) and `src/components/charts/metrics.tsx`; tokens via `text-muted-foreground`, `bg-card`, etc. Each screen: real loading skeletons, actionable empty states ("No nodes connected — here's the installer command"), and error banners.

1. **Overview additions** — a "Distributed Fleet" section on the existing `/` page (not a new dashboard): totals from `/v1/metrics` + node summary + cluster count, each rendered from real data or "not measured".
2. **GPU Inventory** (`/meshweave/gpus`) — flat table of every GPU across all nodes (name, model, VRAM total/free, util, temp, power, node, status), filters (vendor, model, availability), click-through to node detail. Data: `/v1/nodes`.
3. **Nodes** (`/meshweave/nodes` + `[nodeId]`) — list with status/last-heartbeat/reliability; detail page with full hardware inventory (CPU/RAM/disk/OS/network/ports), session telemetry series chart, and the **enrollment command generator** (`curl … --coordinator … --token …` built from actual config; token fetched from a new `POST /api/meshweave/enrollment-tokens` that proxies node registration prep — never hardcoded, never shown twice).
4. **Virtual Datacenters** (`/meshweave/vdcs` + wizard + `[id]`) — list from `/v1/clusters` with aggregate capacity **labelled "aggregate distributed capacity, not a unified memory pool"** (spec requirement). Guided create wizard (General → Select nodes → Network check (runs `/v1/network/measure` and shows real RTT matrix) → Policies → Review → `POST /v1/clusters`). Detail page: members, health, capacity, validations (`/v1/clusters/{id}/validate`), topology view.
5. **Network Intelligence** (`/meshweave/network`) — connectivity matrix (RTT p50/p95 per measured pair; unmeasured pairs shown as "not measured", never omitted or faked), link details (bandwidth, jitter, stability), `/v1/network/intelligence` verdicts per pair (PIPELINE/REPLICATE/TENSOR with rationale), and a "Measure connectivity" action.

### Phase 5 — Verification

- `npx tsc --noEmit` clean; `npm run lint` clean.
- Write tests in the existing style (`tests/*.test.mjs`, Node test runner) for: meshweave client provider-selection logic, proxy route auth (401 without session, 403 without permission), and mock/real data parity (mock responses satisfy the same TS types — checked via a type-level assertion test).
- Manual: run with `FLOAT16_MESHWEAVE_ENABLED=true` + mock provider (default when no coordinator URL set) → all screens render with "Demo data" badges, empty states work with `FLOAT16_MESHWEAVE_ENABLED=false`... actually flag-off hides nav; flag-on without coordinator shows mocks badged as demo.
- Deliver `docs/FLOAT16_INTEGRATION.md`: new routes, components, API contracts, backend dependencies, feature completion status per spec section, known limitations (streaming, TP, concurrency, fault recovery not wired, coordinator auth gaps mitigated only by proxy).

### Out of scope (explicitly deferred, seams noted)

Model catalogue & deploy wizard, deployment dashboard & pipeline visualization, OpenAI endpoint management, performance center/benchmarks UI, billing integration for GPU-hours, KYC gating of new routes. The types and nav permission groups will be defined so Phase 5+ plugs in without rework.

### Risks / notes

- The coordinator API is unauthenticated by design; the proxy layer is the only security boundary — I'll note this prominently in docs and add an env-based allowlist (`MESHWEAVE_COORDINATOR_URL`) so prod deployments can't accidentally point at a public coordinator.
- Meshweave session telemetry is in-memory (240 samples) — charts will be labelled "session window" honestly, matching the existing dashboard's approach.
- I will not modify the Python repo; if the agent/installer needs changes, that's a follow-up.