// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "../lib/runtimeConfig";

export interface HotspotCity {
  label: string;
  center: [number, number];
  zoom: number;
}

export interface HotspotStyle {
  color: string;
  radius: number;
  pulse: boolean;
}

export interface HotspotTooltip {
  width: number;
  aspect_ratio: string;
}

export interface HotspotSubLocation {
  zone_id: string;
  label: string;
}

export interface HotspotMainLocation {
  id: string;
  label: string;
  center: [number, number];
  zoom: number;
  sub_locations: HotspotSubLocation[];
}

export interface HotspotConfig {
  cities: HotspotCity[];
  main_locations: HotspotMainLocation[];
  styles: Record<string, HotspotStyle>;
  tooltip: HotspotTooltip;
}

const FALLBACK_CONFIG: HotspotConfig = {
  cities: [
    { label: "Austin Downtown, TX", center: [30.2672, -97.7431], zoom: 13 },
    { label: "Dallas Downtown, TX", center: [32.7767, -96.797], zoom: 13 },
    { label: "Houston Downtown, TX", center: [29.7604, -95.3698], zoom: 13 },
  ],
  main_locations: [
    {
      id: "austin_downtown",
      label: "Austin Downtown, TX",
      center: [30.2672, -97.7431],
      zoom: 13,
      sub_locations: [
        { zone_id: "congress_plaza_main", label: "Congress Avenue Main Plaza" },
        {
          zone_id: "congress_vehicle_exclusion",
          label: "Congress Vehicle Exclusion Zone",
        },
      ],
    },
    {
      id: "dallas_downtown",
      label: "Dallas Downtown, TX",
      center: [32.7767, -96.797],
      zoom: 13,
      sub_locations: [
        { zone_id: "cam24_main", label: "Main Street & Akard" },
        { zone_id: "cam27_main", label: "Deep Ellum Main Strip" },
      ],
    },
    {
      id: "houston_downtown",
      label: "Houston Downtown, TX",
      center: [29.7604, -95.3698],
      zoom: 13,
      sub_locations: [
        { zone_id: "cam37_main", label: "Discovery Green Park" },
        { zone_id: "cam45_main", label: "Buffalo Bayou Park East" },
      ],
    },
  ],
  styles: {
    critical: { color: "#ED1C24", radius: 10, pulse: true },
    safe: { color: "#22C55E", radius: 6, pulse: false },
  },
  tooltip: { width: 220, aspect_ratio: "16/9" },
};

export function useHotspotConfig() {
  const [config, setConfig] = useState<HotspotConfig>(FALLBACK_CONFIG);

  const fetchConfig = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/dashboard/hotspot-config`);
      if (!res.ok) return;
      const data = (await res.json()) as HotspotConfig;
      if (Array.isArray(data.cities) && data.cities.length > 0) {
        setConfig(data);
      }
    } catch {
      // Fallback config remains active.
    }
  }, []);

  useEffect(() => {
    void fetchConfig();
  }, [fetchConfig]);

  return { config };
}
