// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import {
  useCallback, useMemo,
} from "react";
import gpuImg from "../assets/AMDRadeonAIPROR9700S.webp";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import MetricGraph from "./MetricGraph";
import {
  usePrometheusQuery,
  QUERIES,
  MB_TO_GB,
} from "../hooks/usePrometheusQuery";
import "../styles/components/MetricsPanel.css";

const GPU_IMG = gpuImg;

const SPECS = [
  { key: "Ray Accel", val: "64" },
  { key: "AI Accel", val: "128" },
  { key: "CUs", val: "64" },
  { key: "VRAM", val: "32 GB" },
] as const;

const MB_TO_GB_OPT: { transform: typeof MB_TO_GB } = {
  transform: MB_TO_GB,
};

interface MetricsPanelProps {
  open: boolean;
  onToggle: () => void;
}

export default function MetricsPanel({
  open,
  onToggle,
}: MetricsPanelProps) {
  const gpuCompute = usePrometheusQuery(QUERIES.gpuCompute);
  const gpuMemory = usePrometheusQuery(
    QUERIES.gpuMemory, MB_TO_GB_OPT,
  );
  const gpuTemp = usePrometheusQuery(QUERIES.gpuTemp);
  const gpuPower = usePrometheusQuery(QUERIES.gpuPower);
  const cpuUtil = usePrometheusQuery(QUERIES.cpuUtil);
  const sysMem = usePrometheusQuery(QUERIES.sysMem);

  const activeStreamsQ = usePrometheusQuery(QUERIES.activeStreams);
  const streamFpsTotalQ = usePrometheusQuery(QUERIES.streamFpsTotal);

  const _asArr = activeStreamsQ.series["all"];
  const totalStreams = _asArr != null ? (_asArr[_asArr.length - 1] ?? 0) : 0;
  const _fpArr = streamFpsTotalQ.series["all"];
  const liveFps = _fpArr != null ? (_fpArr[_fpArr.length - 1] ?? 0) : 0;

  const allQueries = useMemo(() => [
    gpuCompute, gpuMemory, gpuTemp,
    gpuPower, cpuUtil, sysMem,
  ], [gpuCompute, gpuMemory, gpuTemp,
    gpuPower, cpuUtil, sysMem]);

  const anyConnected = allQueries.some(
    (q) => q.connected === true,
  );
  const allChecked = allQueries.every(
    (q) => q.connected !== null,
  );

  const handleToggle = useCallback(() => {
    onToggle();
  }, [onToggle]);

  return (
    <div
      className={
        "metrics-panel" +
        (open ? "" : " metrics-panel--collapsed")
      }
    >
      {/* Collapsed overlay */}
      {!open && (
        <div
          className="metrics-panel__collapsed-inner"
          onClick={handleToggle}
        >
          <span className="metrics-panel__collapsed-label">
            AMD Radeon&#8482; AI PRO R9700S
          </span>
          <span className="metrics-panel__collapsed-chevron">
            &#8249;
          </span>
        </div>
      )}

      {/* Open content */}
      {open && (
        <>
          <div className="metrics-panel__header">
            <div
              className="metrics-panel__close-btn"
              onClick={handleToggle}
            >
              &#8250;
            </div>
          </div>

          <div className="metrics-hero-showcase">
            <div className="metrics-hero-img">
              <img
                src={GPU_IMG}
                alt="AMD Radeon AI PRO R9700S"
                loading="eager"
              />
            </div>
            <div className="metrics-hero-identity">
              <h2 className="metrics-hero-name">
                AMD Radeon&#8482; AI PRO R9700S
              </h2>
            </div>
            <div className="spec-strip">
              {SPECS.map((s) => (
                <div key={s.key} className="spec-pill">
                  <span className="spec-pill__val">
                    {s.val}
                  </span>
                  <span className="spec-pill__key">
                    {s.key}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="metric-stat-row">
            <div className="metric-stat-card">
              <span className="metric-stat-card__label">
                Total Streams
              </span>
              <AnimatedNumber
                value={totalStreams}
                className="metric-stat-card__val mono"
                fitWidth={105}
                maxFontSize={32}
                minFontSize={18}
              />
            </div>
            <div className="metric-stat-card">
              <span className="metric-stat-card__label">
                Total<br />FPS
              </span>
              <AnimatedNumber
                value={liveFps}
                format={(v) =>
                  Math.round(v).toLocaleString()
                }
                className="metric-stat-card__val mono"
                fitWidth={105}
                maxFontSize={32}
                minFontSize={18}
              />
            </div>
          </div>

          <div className="metrics-graphs">
            {allChecked && !anyConnected && (
              <div className="metrics-offline-banner">
                <span className="metrics-offline-banner__title">
                  Prometheus not reachable
                </span>
                <span className="metrics-offline-banner__sub">
                  Expecting metrics at localhost:9090
                </span>
              </div>
            )}

            <MetricGraph
              label="GPU Compute"
              series={gpuCompute.series}
              timestamps={gpuCompute.timestamps}
              unit="%"
              color="#ED1C24"
            />
            <MetricGraph
              label="GPU Memory"
              series={gpuMemory.series}
              timestamps={gpuMemory.timestamps}
              unit="GB"
              color="#FACC15"
            />
            <MetricGraph
              label="GPU Temperature"
              series={gpuTemp.series}
              timestamps={gpuTemp.timestamps}
              unit={"\u00B0C"}
              color="#22C55E"
            />
            <MetricGraph
              label="GPU Power"
              series={gpuPower.series}
              timestamps={gpuPower.timestamps}
              unit="W"
              color="#F26522"
            />
            <MetricGraph
              label="CPU Utilization"
              series={cpuUtil.series}
              timestamps={cpuUtil.timestamps}
              unit="%"
              color="#EC4899"
            />
            <MetricGraph
              label="System Memory"
              series={sysMem.series}
              timestamps={sysMem.timestamps}
              unit="GB"
              color="#A855F7"
            />
          </div>

          <div className="metrics-panel__footer">
            <span
              className={
                "metrics-live-dot" +
                (anyConnected
                  ? " metrics-live-dot--on"
                  : "")
              }
            />
            <span
              className={
                "metrics-live-label" +
                (anyConnected
                  ? " metrics-live-label--on"
                  : "")
              }
            >
              {anyConnected ? "Live" : "Offline"}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
