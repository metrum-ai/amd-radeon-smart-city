// Created by Metrum AI for AMD

import { lazy, Suspense, useState, useCallback } from "react";
import { useAppSelector } from "./store";
import { selectMode } from "./features/app/appSelectors";
import TopBar from "./components/TopBar";
import MapPanel from "./components/MapPanel";
import MetricsPanel from "./components/MetricsPanel";

// Lazy load heavy panels for better initial load (Task 316, 317)
const CameraPanel = lazy(() => import("./components/CameraPanel"));
const PlanningPanel = lazy(() => import("./components/PlanningPanel"));

function PanelSkeleton() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        color: "var(--text-dim)",
        fontFamily: "var(--font)",
        fontSize: "var(--fs-sm)",
      }}
    >
      Loading...
    </div>
  );
}

export default function App() {
  const mode = useAppSelector(selectMode);
  const [gpuSidebarOpen, setGpuSidebarOpen] = useState(true);

  const handleToggleSidebar = useCallback(() => {
    setGpuSidebarOpen((p) => !p);
  }, []);

  return (
    <>
      <TopBar />
      <div className={`main${gpuSidebarOpen ? "" : " main--sidebar-collapsed"}`}>
        <MapPanel />
        <div className="content-panel">
          <div className="content-panel__main">
            <Suspense fallback={<PanelSkeleton />}>
              {mode === "ops" ? <CameraPanel /> : <PlanningPanel />}
            </Suspense>
          </div>
        </div>
        <MetricsPanel open={gpuSidebarOpen} onToggle={handleToggleSidebar} />
      </div>
    </>
  );
}
