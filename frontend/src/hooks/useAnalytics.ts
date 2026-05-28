// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

/**
 * useAnalytics — fetches live KPIs, weekly heatmap, and trend data.
 *
 * All data comes from the backend API.  The backend always returns
 * real or simulated-from-live data, so no static mocks are needed here.
 */

import { useCallback, useEffect, useState } from "react";
import type { TimeRange } from "../features/app/appSlice";
import { API_BASE } from "../lib/runtimeConfig";
const KPI_REFRESH_MS = 10_000;
const HEATMAP_REFRESH_MS = 30_000;
const TREND_REFRESH_MS = 15_000;

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface KPIData {
  active_critical_zones: number;
  peak_crowd_count: number;
  peak_crowd_location: string;
  streams_live: number;
  streams_total: number;
  detection_latency_p95: number;
}

export interface TrendZone {
  zone_id: string;
  zone_name: string;
  location_name: string;
  data: number[];
}

export interface TrendData {
  zones: TrendZone[];
  labels: string[];
  range: string;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAnalytics(timeRange: TimeRange, cityLabel?: string) {
  const [kpis, setKpis] = useState<KPIData | null>(null);
  const [heatmap, setHeatmap] = useState<number[][] | null>(null);
  const [trends, setTrends] = useState<TrendData | null>(null);

  // -- KPIs (refresh every 10 s) -------------------------------------------
  const fetchKpis = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (cityLabel) params.set("city", cityLabel);
      const res = await fetch(
        `${API_BASE}/api/v1/analytics/kpis?${params.toString()}`,
      );
      if (res.ok) {
        const data = (await res.json()) as KPIData;
        setKpis(data);
      }
    } catch {
      // retain previous value on transient error
    }
  }, [cityLabel]);

  useEffect(() => {
    void fetchKpis();
    const id = setInterval(() => void fetchKpis(), KPI_REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchKpis]);

  // -- Weekly heatmap (refresh every 30 s) ----------------------------------
  const fetchHeatmap = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (cityLabel) params.set("city", cityLabel);
      const res = await fetch(
        `${API_BASE}/api/v1/analytics/heatmap-weekly?${params.toString()}`,
      );
      if (res.ok) {
        const data = (await res.json()) as { grid: number[][] };
        setHeatmap(data.grid);
      }
    } catch {
      // retain previous value
    }
  }, [cityLabel]);

  useEffect(() => {
    void fetchHeatmap();
    const id = setInterval(() => void fetchHeatmap(), HEATMAP_REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchHeatmap]);

  // -- Trends (re-fetch on range change + interval) -------------------------
  const fetchTrends = useCallback(async () => {
    try {
      const params = new URLSearchParams({ range: timeRange });
      if (cityLabel) params.set("city", cityLabel);
      const res = await fetch(
        `${API_BASE}/api/v1/analytics/trends?${params.toString()}`,
      );
      if (res.ok) {
        const data = (await res.json()) as TrendData;
        setTrends(data);
      }
    } catch {
      // retain previous value
    }
  }, [cityLabel, timeRange]);

  useEffect(() => {
    void fetchTrends();
    const id = setInterval(() => void fetchTrends(), TREND_REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchTrends]);

  return { kpis, heatmap, trends };
}
