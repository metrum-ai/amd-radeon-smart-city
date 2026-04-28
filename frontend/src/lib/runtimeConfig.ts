// Created by Metrum AI for AMD

const browserOrigin =
  typeof window !== "undefined" ? window.location.origin : "";

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? browserOrigin;

export const WS_BASE = API_BASE.replace(/^http/, "ws");
