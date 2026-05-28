// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * useMapData - fetches live camera map data and subscribes to
 * WebSocket crowd-count updates.
 *
 * Optimizations (Vercel Best Practices):
 * - Stable WS handler via useCallback to prevent handler churn
 * - useTransition for non-blocking count updates
 * - Ref-based WS to avoid reconnection on unrelated prop changes
 * - Exponential backoff reconnection for port-forwarded environments
 */

import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { API_BASE, WS_BASE } from "../lib/runtimeConfig";

export interface CameraMarker {
  stream_id: number;
  location_name: string;
  lat: number;
  lon: number;
  severity: "SAFE" | "CRITICAL";
  zone_type: string;
  violation_type: string | null;
  person_count: number;
  threshold: number;
  is_auto_popup: boolean;
  stream_status: string;
  location_tag?: string | null;
  preview_video_url?: string | null;
  webrtc_whep_url?: string | null;
  marker_color?: string | null;
  marker_radius?: number | null;
  marker_pulse?: boolean | null;
  tooltip_width?: number | null;
  tooltip_aspect_ratio?: string | null;
}

export interface MapData {
  cameras: CameraMarker[];
  timestamp: string;
  active_critical_count: number;
  active_vehicle_intrusions: number;
  active_restricted_violations: number;
}

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_RECONNECT_DELAY = 1_000;

export function useMapData(refreshMs = 10_000, cityLabel?: string) {
  const [mapData, setMapData] = useState<MapData | null>(null);
  const [loading, setLoading] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const [, startTransition] = useTransition();

  const prevCityRef = useRef(cityLabel);
  const abortRef = useRef<AbortController | null>(null);
  const hasDataRef = useRef(false);

  const fetchMapData = useCallback(async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    // On city switch, clear stale data so React unmounts old
    // CamCard/WhepPlayer components immediately.
    if (prevCityRef.current !== cityLabel) {
      prevCityRef.current = cityLabel;
      setMapData(null);
      hasDataRef.current = false;
    }

    // Only flash the loading spinner when we have nothing to show yet
    // (initial load or city switch). Periodic refreshes update silently
    // so existing camera feeds stay mounted and keep streaming.
    if (!hasDataRef.current) {
      setLoading(true);
    }
    try {
      const params = new URLSearchParams();
      if (cityLabel) params.set("city", cityLabel);
      const res = await fetch(
        `${API_BASE}/api/v1/dashboard/map?${params.toString()}`,
        { signal: ac.signal },
      );
      if (res.ok) {
        const data: MapData = await res.json();
        if (!ac.signal.aborted) {
          setMapData(data);
          hasDataRef.current = true;
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
    } finally {
      if (!ac.signal.aborted) setLoading(false);
    }
  }, [cityLabel]);

  // Initial + periodic REST refresh (fallback when WS fails)
  useEffect(() => {
    void fetchMapData();
    const id = setInterval(() => void fetchMapData(), refreshMs);
    return () => {
      clearInterval(id);
      abortRef.current?.abort();
    };
  }, [fetchMapData, refreshMs]);

  // Stable WS message handler using useCallback
  const handleWsMessage = useCallback(
    (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data as string);
        if (msg.type === "count_update") {
          startTransition(() => {
            setMapData((prev) => {
              if (!prev) return prev;
              const updated = prev.cameras.map((cam) =>
                cam.stream_id === msg.data.stream_id
                  ? {
                      ...cam,
                      person_count: msg.data.person_count,
                      severity: msg.data.severity as "SAFE" | "CRITICAL",
                    }
                  : cam
              );
              return { ...prev, cameras: updated };
            });
          });
        }
      } catch {
        // ignore malformed frames
      }
    },
    [startTransition]
  );

  // WebSocket with exponential backoff reconnection
  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/counts`);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
      };

      ws.onmessage = handleWsMessage;

      ws.onerror = () => {
        // Will trigger onclose
      };

      ws.onclose = () => {
        if (!mountedRef.current) return;
        // Exponential backoff reconnect
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY);
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, [handleWsMessage]);

  return { mapData, loading, refetch: fetchMapData };
}
