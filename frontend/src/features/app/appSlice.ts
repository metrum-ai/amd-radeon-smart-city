// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { createSlice, PayloadAction } from "@reduxjs/toolkit";

export type Severity = "critical" | "safe";
export type AppMode = "plan" | "ops";
export type TimeRange = "day" | "week" | "month" | "quarter";

export interface CameraData {
  stream_id: number;
  lat: number;
  lon: number;
  name: string;
  count: number;
  sev: Severity;
  previewUrl?: string;
  webrtcWhepUrl?: string;
  locationTag?: string;
  markerColor?: string;
  markerRadius?: number;
  markerPulse?: boolean;
  tooltipWidth?: number;
  tooltipAspectRatio?: string;
}

export const RANGE_LABELS: Record<TimeRange, string> = {
  day: "Today",
  week: "7 Days",
  month: "30 Days",
  quarter: "90 Days",
};

export const RANGE_DAYS: Record<TimeRange, number> = {
  day: 1,
  week: 7,
  month: 30,
  quarter: 90,
};

interface AppState {
  theme: "dark" | "light";
  mode: AppMode;
  severityFilter: "all" | Severity;
  timeRange: TimeRange;
  selectedMainLocationId: string;
  miniPreviewCam: CameraData | null;
}

const initialState: AppState = {
  theme: "dark",
  mode: "ops",
  severityFilter: "all",
  timeRange: "month",
  selectedMainLocationId: "austin_downtown",
  miniPreviewCam: null,
};

const appSlice = createSlice({
  name: "app",
  initialState,
  reducers: {
    toggleTheme(state) {
      state.theme = state.theme === "dark" ? "light" : "dark";
    },
    setMode(state, action: PayloadAction<AppMode>) {
      state.mode = action.payload;
    },
    setSeverityFilter(state, action: PayloadAction<"all" | Severity>) {
      state.severityFilter = action.payload;
    },
    setTimeRange(state, action: PayloadAction<TimeRange>) {
      state.timeRange = action.payload;
    },
    setSelectedMainLocationId(state, action: PayloadAction<string>) {
      state.selectedMainLocationId = action.payload;
    },
    setMiniPreviewCam(state, action: PayloadAction<CameraData | null>) {
      state.miniPreviewCam = action.payload;
    },
  },
});

export const {
  toggleTheme,
  setMode,
  setSeverityFilter,
  setTimeRange,
  setSelectedMainLocationId,
  setMiniPreviewCam,
} = appSlice.actions;

export default appSlice.reducer;
