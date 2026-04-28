// Created by Metrum AI for AMD

/**
 * useAlerts - fetches paginated alerts from the REST API and subscribes
 * to real-time alert pushes via WebSocket.
 *
 * Returns raw alerts array - display staggering is handled by the
 * consuming component (AlertStream) for proper one-at-a-time rendering.
 *
 * Includes exponential backoff reconnection for port-forwarded environments.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, WS_BASE } from "../lib/runtimeConfig";

export interface Alert {
  alert_id: string;
  timestamp: string;
  stream_id: number;
  zone_id: string;
  zone_name: string;
  zone_type: string;
  location_name: string;
  lat: number;
  lon: number;
  person_count: number;
  threshold: number;
  severity: "SAFE" | "CRITICAL";
  violation_type: string | null;
  description: string;
  is_auto_popup: boolean;
  status: string;
}

interface UseAlertsOptions {
  severity?: string;
  city?: string;
  locationNames?: Set<string>;
  limit?: number;
  autoRefreshMs?: number;
}

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_RECONNECT_DELAY = 1_000;

export function useAlerts(opts: UseAlertsOptions = {}) {
  const { severity, city, locationNames, limit = 50, autoRefreshMs = 0 } = opts;
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);
  const reconnectTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  // Stable refs to avoid WS reconnection on prop changes
  const limitRef = useRef(limit);
  limitRef.current = limit;
  const locationNamesRef = useRef(locationNames);
  locationNamesRef.current = locationNames;

  const fetchAlerts = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: String(limit) });
      if (severity && severity !== "all") {
        params.set("severity", severity.toUpperCase());
      }
      if (city) params.set("city", city);
      const res = await fetch(`${API_BASE}/api/v1/alerts?${params}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAlerts(data.items ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [severity, city, limit]);

  // Initial fetch
  useEffect(() => {
    void fetchAlerts();
  }, [fetchAlerts]);

  // Auto-refresh (fallback when WS fails)
  useEffect(() => {
    if (!autoRefreshMs) return;
    const id = setInterval(() => void fetchAlerts(), autoRefreshMs);
    return () => clearInterval(id);
  }, [fetchAlerts, autoRefreshMs]);

  // Stable WS message handler - appends new alerts directly
  const handleWsMessage = useCallback((ev: MessageEvent) => {
    try {
      const msg = JSON.parse(ev.data as string);
      if (msg.type === "alert") {
        const alert = msg.data as Alert;
        const names = locationNamesRef.current;
        if (names && names.size > 0 && !names.has(alert.location_name)) {
          return;
        }
        setAlerts((prev) => {
          const exists = prev.some((a) => a.alert_id === alert.alert_id);
          if (exists) return prev;
          return [alert, ...prev].slice(0, limitRef.current);
        });
      }
    } catch {
      // ignore malformed frames
    }
  }, []);

  // WebSocket with exponential backoff reconnection
  useEffect(() => {
    mountedRef.current = true;

    const connect = () => {
      if (!mountedRef.current) return;

      const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
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

  return { alerts, loading, error, refetch: fetchAlerts };
}
