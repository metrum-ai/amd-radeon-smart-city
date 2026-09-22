// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

const browserOrigin =
  typeof window !== "undefined" ? window.location.origin : "";

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? browserOrigin;

export const WS_BASE = API_BASE.replace(/^http/, "ws");

// Empty by default: LAN deployments use ICE host candidates, no STUN needed.
// Set VITE_ICE_SERVERS (JSON array) at build time for cross-NAT segments.
function parseIceServers(): RTCIceServer[] {
  const raw = import.meta.env.VITE_ICE_SERVERS;
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    console.warn("VITE_ICE_SERVERS is not valid JSON; ignoring it.");
    return [];
  }
}

export const ICE_SERVERS: RTCIceServer[] = parseIceServers();
