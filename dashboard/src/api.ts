/**
 * Types mirroring the MESHWEAVE coordinator API.
 *
 * Every numeric field here is nullable. That is not laziness: the coordinator
 * reports `null` when a value has not been measured, and the dashboard must
 * render "not measured" rather than 0. A zero would be a lie.
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
  architecture: string | null
  compute_capability: [number, number] | null
  vram_total_bytes: number
  vram_free_bytes: number
  vram_used_bytes: number
  temperature_c: number | null
  power_draw_w: number | null
  utilization_gpu_pct: number | null
  memory_bandwidth_gbps: number | null
}

export interface NodeRecord {
  node_id: string
  status: NodeStatus
  hourly_cost_usd: number | null
  reliability_score: number | null
  observations: number
  missed_heartbeats: number
  last_heartbeat_at: string | null
  inventory: {
    gpus: GpuDevice[]
    network: {
      hostname: string | null
      private_ip: string | null
      external_ip: string | null
      geolocation: string | null
    }
    agent_version: string
  } | null
}

export interface NodeListResponse {
  nodes: NodeRecord[]
  summary: {
    total: number
    healthy: number
    total_gpus: number
    total_vram_gib: number
    free_vram_gib: number
    hourly_cost_usd: number | null
  }
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
  rtt_ms_p50: number | null
  jitter_ms: number | null
  packet_loss_pct: number | null
  tcp_mbps_single_stream: number | null
  sustained_mbps: number | null
  connection_success: boolean | null
  error: string | null
}

export interface TopologyResponse {
  matrix: {
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
  rationale: string[]
  warnings: string[]
  disconnect_impact: string | null
}