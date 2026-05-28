// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import type { RootState } from "../../store";

export const selectTheme = (state: RootState) => state.app.theme;
export const selectMode = (state: RootState) => state.app.mode;
export const selectSeverityFilter = (state: RootState) =>
  state.app.severityFilter;
export const selectTimeRange = (state: RootState) => state.app.timeRange;
export const selectMainLocationId = (state: RootState) =>
  state.app.selectedMainLocationId;
export const selectMiniPreviewCam = (state: RootState) =>
  state.app.miniPreviewCam;
