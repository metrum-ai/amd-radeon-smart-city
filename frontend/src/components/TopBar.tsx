// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { useCallback } from "react";
import { useAppSelector, useAppDispatch } from "../store";
import { selectMode } from "../features/app/appSelectors";
import { setMode } from "../features/app/appSlice";
import type { AppMode } from "../features/app/appSlice";
import amdCorpLogo from "../assets/AMD_WhiteLogo_DarkTheme.png";
import "../styles/components/TopBar.css";

export default function TopBar() {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectMode);

  const handleModeSwitch = useCallback(
    (m: AppMode) => {
      if (m !== mode) dispatch(setMode(m));
    },
    [mode, dispatch],
  );

  return (
    <header className="topbar">
      <div className="topbar__left">
        <img src={amdCorpLogo} alt="AMD" className="brand-corp-logo" />
      </div>

      <span className="topbar__center">Smart City Analytics Platform</span>

      <div className="topbar__right">
        <div className="mode-switch">
          <button
            className={`mode-btn ${mode === "plan" ? "active" : ""}`}
            onClick={() => handleModeSwitch("plan")}
          >
            Analytics
          </button>
          <button
            className={`mode-btn ${mode === "ops" ? "active" : ""}`}
            onClick={() => handleModeSwitch("ops")}
          >
            Live Operations
          </button>
        </div>
      </div>
    </header>
  );
}
