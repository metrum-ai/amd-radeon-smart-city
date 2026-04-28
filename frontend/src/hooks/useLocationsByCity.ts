// Created by Metrum AI for AMD

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "../lib/runtimeConfig";

export interface CityLocation {
  zone_id: string;
  label: string;
}

const CITY_ID_MAP: Record<string, string> = {
  "Austin Downtown, TX": "austin_downtown",
  "Dallas Downtown, TX": "dallas_downtown",
  "Houston Downtown, TX": "houston_downtown",
};

export function useLocationsByCity(cityLabel: string): CityLocation[] {
  const [locations, setLocations] = useState<CityLocation[]>([]);

  const cityId = CITY_ID_MAP[cityLabel] ?? "";

  const fetch_ = useCallback(async () => {
    if (!cityId) return;
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/dashboard/locations?city_id=${encodeURIComponent(cityId)}`
      );
      if (!res.ok) return;
      const data = (await res.json()) as Record<string, CityLocation[]>;
      setLocations(data[cityId] ?? []);
    } catch {
      // Keep existing locations on transient failure
    }
  }, [cityId]);

  useEffect(() => {
    setLocations([]);
    void fetch_();
  }, [fetch_]);

  return locations;
}
