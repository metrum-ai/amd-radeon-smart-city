// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * TypeScript interfaces for all WebSocket message schemas.
 * Matches the payload definitions in docs/plan.md §12.
 */

// ---------------------------------------------------------------------------
// /ws/alerts — new CRITICAL alerts and status updates
// ---------------------------------------------------------------------------

export interface AlertMessage {
  type: "alert";
  data: {
    alert_id: string;
    timestamp: string; // ISO8601
    stream_id: number;
    zone_id: string;
    zone_name: string;
    location_name: string;
    lat: number;
    lon: number;
    person_count: number;
    threshold: number;
    severity: "SAFE" | "CRITICAL";
    trend?: string;
    description: string;
    is_auto_popup: boolean;
    status: string;
    violation_type: "crowd_density" | "vehicle_intrusion" | "restricted_zone_entry" | null;
  };
}

// ---------------------------------------------------------------------------
// /ws/counts — crowd count updates (1–2 Hz)
// ---------------------------------------------------------------------------

export interface CountUpdateMessage {
  type: "count_update";
  data: {
    stream_id: number;
    zone_id: string;
    zone_name: string;
    location_name: string;
    lat: number;
    lon: number;
    person_count: number;
    severity: "SAFE" | "CRITICAL";
    trend?: string;
    timestamp: string; // ISO8601
  };
}

// ---------------------------------------------------------------------------
// /ws/heatmap/{stream_id} — heatmap frames (5 Hz)
// ---------------------------------------------------------------------------

export interface HeatmapMessage {
  type: "heatmap";
  stream_id: number;
  image_b64: string;
  timestamp: string; // ISO8601
}

// ---------------------------------------------------------------------------
// /ws/stream/{stream_id} — MJPEG stream frames (up to 10 FPS)
// ---------------------------------------------------------------------------

export interface StreamFrameMessage {
  type: "frame";
  stream_id: number;
  image_b64: string;
  person_count: number;
  severity: "SAFE" | "CRITICAL";
  timestamp: string; // ISO8601
}

// ---------------------------------------------------------------------------
// /ws/telemetry — GPU telemetry (every 5s)
// ---------------------------------------------------------------------------

export interface GPUStats {
  gpu_id: number;
  utilization_pct: number;
  vram_used_gb: number;
  vram_total_gb: number;
  temperature_c: number;
}

export interface TelemetryMessage {
  type: "telemetry";
  data: {
    gpus: GPUStats[];
    system_power_watts: number | null;
    timestamp: string; // ISO8601
  };
}

// ---------------------------------------------------------------------------
// Keepalive
// ---------------------------------------------------------------------------

export interface PingMessage {
  type: "ping";
}

export interface PongMessage {
  type: "pong";
}

// ---------------------------------------------------------------------------
// Union type for all inbound WebSocket messages
// ---------------------------------------------------------------------------

export type WsMessage =
  | AlertMessage
  | CountUpdateMessage
  | HeatmapMessage
  | StreamFrameMessage
  | TelemetryMessage
  | PingMessage
  | PongMessage;
