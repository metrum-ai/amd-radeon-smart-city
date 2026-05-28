// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { createContext, useContext } from "react";
import type {
  CircleMarker as LeafletCircleMarker,
  Map as LeafletMapInstance,
} from "leaflet";

export const LeafletMapContext = createContext<LeafletMapInstance | null>(null);
export const LeafletMarkerContext =
  createContext<LeafletCircleMarker | null>(null);

export function useLeafletMap() {
  return useContext(LeafletMapContext);
}
