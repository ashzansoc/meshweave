/**
 * MESHWEAVE coordinator API types.
 *
 * Every numeric field is nullable. That is deliberate and load-bearing: the
 * coordinator reports null when a value has not been measured, and the UI must
 * render "not measured" rather than 0. A zero would be a fabrication.
 */

export type NodeStatus =
  | 'REGISTERING'
  | 'HEALTHY'
  | 'DEGRADED'
  | 'UNREACHABLE'
  | 'DRAINING'
  | 'RETIRED'

export interface GpuDevice {
  index: number
  uuid: string
  name: string
  manufacturer: string
  architecture: string | null
  compute_capability: [number, number] | null
  cuda_version: string | null
  driver_version: string | null
  vram_total_bytes: number
  vram_free_bytes: number
  vram_used_bytes: number
  temperature_c: number | null
  power_draw_w: number | null
  power_limit_w: number | null
  utilization_gpu_pct: number | null
  utilization_memory_pct: number | null
  memory_bandwidth_gbps: number | null
}

export interface NetworkInfo {
  hostname: string | null
  private_ip: string | null
  external_ip: string | null
  geolocation: string | null
  nat: boolean | null
  mapped_ports: Record<string, number>
  overlay_interface: string | null
}

export interface HostResources {
  cpu_model: string | null
  cpu_cores: number | null
  ram_total_bytes: number | null
  ram_available_bytes: number | null
  os_name: string | null
  kernel_version: string | null
}

export interface GpuInventory {
  gpus: GpuDevice[]
  host: HostResources
  network: NetworkInfo
  agent_version: string
  collected_at: string
}

export interface NodeRecord {
  node_id: string
  status: NodeStatus
  hourly_cost_usd: number | null
  reliability_score: number | null
  uptime_fraction: number | null
  observations: number
  missed_heartbeats: number
  last_heartbeat_at: string | null
  registration: {
    node_id: string
    hostname: string
    agent_version: string
    provider: string
    provider_instance_id: string | null
    region: string | null
  }
  inventory: GpuInventory | null
}

export interface NodeSummary {
  total: number
  healthy: number
  total_gpus: number
  total_vram_gib: number
  free_vram_gib: number
  hourly_cost_usd: number | null
}

export interface NodeListResponse {
  nodes: NodeRecord[]
  summary: NodeSummary
}

export interface NodeDetailResponse {
  node: NodeRecord
  degrade_reason: string | null
  environment: string | null
}

export interface Metrics {
  workloads_total: number
  workloads_successful: number
  workloads_failed: number
  tokens_generated: number
  mean_tps: number | null
  mean_ttft_ms: number | null
  mean_cost_per_m_output_tokens_usd: number | null
  cluster_hourly_cost_usd: number | null
  total_vram_gib: number
  free_vram_gib: number
  measured_topology: boolean
}

export interface LinkMeasurement {
  source: string
  target: string
  measured_at: string
  rtt_ms_p50: number | null
  rtt_ms_p95: number | null
  rtt_ms_min: number | null
  one_way_ms_estimate: number | null
  jitter_ms: number | null
  packet_loss_pct: number | null
  tcp_mbps_single_stream: number | null
  tcp_mbps_multi_stream: number | null
  sustained_mbps: number | null
  transfer_time_s: number | null
  connection_success: boolean | null
  connection_stability_pct: number | null
  udp_supported: boolean | null
  link_kind: string
  error: string | null
}

export interface TopologyResponse {
  matrix: {
    generated_at: string
    nodes: string[]
    links: LinkMeasurement[]
  }
  environments: Record<string, string>
}

export interface ExecutionPlan {
  plan_id: string
  model: string
  strategy: string
  environment: string
  stage_assignment: Record<string, string>
  layer_assignment: Record<string, number[]>
  replica_nodes: string[]
  estimated_ttft_ms: number | null
  estimated_itl_ms: number | null
  estimated_throughput_tps: number | null
  estimated_cost_per_m_output_tokens: number | null
  viable: boolean
  rejected_reason: string | null
  assumptions: string[]
  warnings: string[]
  rationale: string[]
  disconnect_impact: string | null
}

export interface PlanComparisonRow {
  label: string
  strategy: string
  nodes: number
  gpus: number
  hourly_cost_usd: number | null
  throughput_tps: number | null
  cost_per_m_output_tokens_usd: number | null
  throughput_per_dollar: number | null
}

export interface PlanComparison {
  configurations: PlanComparisonRow[]
  cheapest_by_cost_per_m_output_tokens: PlanComparisonRow | null
  fastest_by_throughput_per_dollar: PlanComparisonRow | null
  unpriced: string[]
  crossover: {
    beyond_node_count: number | null
    at_node_count: number | null
    cost_before: number | null
    cost_after: number | null
    explanation: string
  } | null
}

export interface WorkloadRecord {
  workload_id: string
  model: string
  strategy: string
  nodes: string[]
  environment: string
  started_at: string
  finished_at: string | null
  prompt_tokens: number
  output_tokens: number
  ttft_ms: number | null
  inter_token_latency_ms_p50: number | null
  tokens_per_second: number | null
  cost_usd: number | null
  cost_per_m_output_tokens_usd: number | null
  success: boolean
  error: string | null
}

export interface WorkloadEvent {
  at: string
  kind: string
  detail: Record<string, unknown>
}

export interface WorkloadDetail {
  workload: WorkloadRecord
  events: WorkloadEvent[]
  assignments: Array<Record<string, unknown>>
}

export interface ExperimentSummary {
  experiment_id: string
  title: string | null
  status: string
  synthetic: boolean
  measurement_count: number
  failure_count: number
  blocked_reason: string | null
  path: string
}

export interface ExperimentListResponse {
  experiments: ExperimentSummary[]
  total: number
  completed: number
  blocked: number
}

export interface Measurement {
  name: string
  value: number | null
  unit: string
  samples: number
  note: string | null
}

export interface ExperimentReport {
  schema_version: string
  experiment_id: string
  title: string
  hypothesis: string
  provenance: {
    git_commit: string | null
    git_dirty: boolean | null
    generated_at: string
    host: string
    python_version: string
    meshweave_version: string
    synthetic: boolean
    command: string
    config: Record<string, unknown>
  }
  status: string
  nodes: Array<Record<string, unknown>>
  strategy: string | null
  measurements: Measurement[]
  economics: {
    cost_known: boolean
    hourly_cost_usd: number | null
    total_cost_usd: number | null
    cost_per_m_input_tokens_usd: number | null
    cost_per_m_output_tokens_usd: number | null
    cost_per_request_usd: number | null
    unknown_components: string[]
  }
  failures: {
    failures: Array<{ kind: string; detail: string; at: string }>
    failures_investigated: boolean
    notes: string | null
  }
  findings: string[]
  evidence: string[]
  blocked_reason: string | null
  ready_to_run_command: string | null
}