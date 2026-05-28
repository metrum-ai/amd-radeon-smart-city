// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { memo, useCallback, useMemo, useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { useAppSelector, useAppDispatch } from "../store";
import {
  selectMainLocationId,
  selectTimeRange,
} from "../features/app/appSelectors";
import {
  setTimeRange,
  setSelectedMainLocationId,
  RANGE_LABELS,
} from "../features/app/appSlice";
import type { TimeRange } from "../features/app/appSlice";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import { useAnalytics } from "../hooks/useAnalytics";
import type { TrendZone } from "../hooks/useAnalytics";
import { useHotspotConfig } from "../hooks/useHotspotConfig";
import { useLocationsByCity } from "../hooks/useLocationsByCity";
import { API_BASE } from "../lib/runtimeConfig";
import { AgentStepper } from "./AgentStepper";
import type { AgentTraceStep } from "./AgentStepper";
import "../styles/components/PlanningPanel.css";

const TIME_RANGE_KEYS: TimeRange[] = ["day", "week", "month"];
const MONO = "Geist Mono Variable";
// Zone colors: first three zones get these; any extras get grey
const ZONE_COLORS = ["#ED1C24", "#F26522", "#007C97", "#8B5CF6", "#22C55E"];


function EChart({
  option,
  height,
}: {
  option: echarts.EChartsOption;
  height: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const instRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const inst = echarts.init(ref.current, undefined, { renderer: "canvas" });
    instRef.current = inst;
    const ro = new ResizeObserver(() => inst.resize());
    ro.observe(ref.current);
    return () => {
      ro.disconnect();
      inst.dispose();
    };
  }, []);

  useEffect(() => {
    instRef.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={ref} style={{ height }} />;
}

function LiveBadge() {
  return (
    <span className="live-badge">
      <span className="live-badge__dot" />
      LIVE
    </span>
  );
}

function PatternHeatmap({ heatmap }: { heatmap: number[][] }) {
  const option = useMemo<echarts.EChartsOption>(() => {
    const heatmapData: number[][] = [];
    for (let d = 0; d < 7; d++)
      for (let h = 0; h < 24; h++)
        heatmapData.push([h, d, Math.round(heatmap[d]?.[h] ?? 0)]);
    const axisColor = "#8A8B8E";
    return {
      tooltip: {
        position: "top",
        backgroundColor: "rgba(14,14,17,.92)",
        borderColor: "#1E1E22",
        textStyle: { color: "#F0F0F0", fontFamily: "Klavika", fontSize: 11 },
        formatter: (params: unknown) => {
          const p = params as { value: number[] };
          const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
          return `<span style="font-family:${MONO};font-weight:600">${p.value[2]}%</span> density<br/>${days[p.value[1]]} ${p.value[0]}:00`;
        },
      },
      grid: { left: 44, right: 14, top: 14, bottom: 28 },
      xAxis: {
        type: "category",
        data: Array.from({ length: 24 }, (_, i) => `${i}h`),
        splitArea: { show: false },
        axisLabel: {
          fontSize: 9,
          fontFamily: MONO,
          color: axisColor,
          interval: 3,
        },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: "category",
        data: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        axisLabel: {
          fontSize: 9,
          fontFamily: "Klavika Condensed",
          color: axisColor,
        },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      visualMap: {
        min: 0,
        max: 100,
        show: false,
        inRange: {
          color: [
            "#1A2332",
            "#134E5E",
            "#1B8A7A",
            "#E8742A",
            "#ED1C24",
          ],
        },
      },
      series: [
        {
          type: "heatmap",
          data: heatmapData,
          itemStyle: {
            borderWidth: 2,
            borderColor: "#151518",
            borderRadius: 3,
          },
          emphasis: {
            itemStyle: {
              borderColor: "#F0F0F0",
              borderWidth: 1,
              shadowBlur: 6,
              shadowColor: "rgba(237,28,36,0.25)",
            },
          },
        },
      ],
      backgroundColor: "transparent",
    };
  }, [heatmap]);

  return (
    <div className="chart-card">
      <div className="chart-card__head">
        <div className="chart-card__titles">
          <span className="chart-card__title">Weekly Crowd Pattern</span>
          <span className="chart-sub">Average density by hour &amp; day</span>
        </div>
        <LiveBadge />
      </div>
      <EChart option={option} height={188} />
      <div className="heatmap-legend">
        <span>Low</span>
        <div className="legend-gradient" />
        <span>Critical</span>
      </div>
    </div>
  );
}

const ZoneFilter = memo(function ZoneFilter({
  zones,
  selected,
  onChange,
}: {
  zones: TrendZone[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const allSelected = selected.size === 0;
  const lowerQ = query.toLowerCase();
  const filtered = query
    ? zones.filter((z) => z.zone_name.toLowerCase().includes(lowerQ))
    : zones;

  const toggle = useCallback(
    (id: string) => {
      const next = new Set(selected);
      next.delete("__none__");
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      if (next.size === 0) {
        onChange(new Set(["__none__"]));
      } else if (next.size === zones.length) {
        onChange(new Set());
      } else {
        onChange(next);
      }
    },
    [selected, onChange, zones.length],
  );

  const selectAll = useCallback(() => onChange(new Set()), [onChange]);
  const clearAll = useCallback(
    () => onChange(new Set(["__none__"])),
    [onChange],
  );

  const isNone = selected.has("__none__");
  const visibleCount = isNone ? 0 : selected.size;
  const label = allSelected
    ? "All locations"
    : isNone
      ? "None selected"
      : `${visibleCount} of ${zones.length}`;

  return (
    <div className="zone-filter" ref={wrapRef}>
      <button
        className="zone-filter__trigger"
        onClick={() => setOpen((p) => !p)}
        type="button"
      >
        {label}
        <span className={`zone-filter__caret${open ? " is-open" : ""}`}>
          &#9662;
        </span>
      </button>
      {open && (
        <div className="zone-filter__dropdown">
          <input
            className="zone-filter__search"
            placeholder="Search zones..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <div className="zone-filter__actions">
            <button type="button" onClick={selectAll}>
              All
            </button>
            <button type="button" onClick={clearAll}>
              Clear
            </button>
          </div>
          <div className="zone-filter__list">
            {filtered.map((z, i) => {
              const checked =
                allSelected || (!isNone && selected.has(z.zone_id));
              const dotColor =
                ZONE_COLORS[zones.indexOf(z) % ZONE_COLORS.length];
              return (
                <label key={z.zone_id} className="zone-filter__item">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(z.zone_id)}
                  />
                  <span
                    className="zone-filter__dot"
                    style={{ background: dotColor }}
                  />
                  <span className="zone-filter__name">{z.zone_name}</span>
                </label>
              );
            })}
            {filtered.length === 0 && (
              <span className="zone-filter__empty">No matches</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

function TrendChart({
  zones,
  labels,
  timeRange,
}: {
  zones: TrendZone[];
  labels: string[];
  timeRange: TimeRange;
}) {
  const [selectedZones, setSelectedZones] = useState<Set<string>>(
    new Set(),
  );

  const visibleZones = useMemo(() => {
    if (selectedZones.size === 0) return zones;
    if (selectedZones.has("__none__")) return [];
    return zones.filter((z) => selectedZones.has(z.zone_id));
  }, [zones, selectedZones]);

  const option = useMemo<echarts.EChartsOption>(() => {
    const N = labels.length;
    const axisColor = "#8A8B8E";
    const gridColor = "rgba(30,30,34,.8)";

    const series = visibleZones.map((z) => {
      const idx = zones.indexOf(z);
      const color = ZONE_COLORS[idx % ZONE_COLORS.length];
      const rgb = color.replace("#", "");
      const r = parseInt(rgb.slice(0, 2), 16);
      const g = parseInt(rgb.slice(2, 4), 16);
      const b = parseInt(rgb.slice(4, 6), 16);
      return {
        name: z.zone_name,
        type: "line" as const,
        data: z.data,
        lineStyle: { color, width: 1.5 },
        itemStyle: { color },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: `rgba(${r},${g},${b},0.06)` },
            { offset: 0.7, color: `rgba(${r},${g},${b},0.01)` },
            { offset: 1, color: `rgba(${r},${g},${b},0)` },
          ]),
        },
        symbol: "none",
        smooth: 0.35,
      };
    });

    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(14,14,17,.92)",
        borderColor: "#1E1E22",
        textStyle: { color: "#F0F0F0", fontFamily: "Klavika", fontSize: 11 },
        axisPointer: {
          type: "cross",
          lineStyle: { color: "#2A2A2F", type: "dashed" },
          crossStyle: { color: "#2A2A2F" },
          label: { show: false },
        },
      },
      legend: { show: false },
      grid: { left: 36, right: 8, top: 16, bottom: 32 },
      xAxis: {
        type: "category",
        data: labels,
        axisLabel: {
          fontSize: 9,
          fontFamily: MONO,
          color: axisColor,
          interval: N <= 7 ? 0 : N <= 14 ? 1 : 4,
          rotate: N > 14 ? -30 : 0,
        },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        axisLabel: { fontSize: 9, fontFamily: MONO, color: axisColor },
        splitLine: { lineStyle: { color: gridColor, type: "dashed" } },
      },
      series,
      animation: true,
      animationDuration: 400,
      animationEasingUpdate: "cubicOut",
      backgroundColor: "transparent",
    };
  }, [visibleZones, zones, labels]);

  return (
    <div className="chart-card">
      <div className="chart-card__head">
        <div className="chart-card__titles">
          <span className="chart-card__title">Count Trend</span>
          <span className="chart-sub">
            {timeRange === "day" ? "Hourly counts" : "Daily peak"} -{" "}
            {RANGE_LABELS[timeRange]}
          </span>
        </div>
        <ZoneFilter
          zones={zones}
          selected={selectedZones}
          onChange={setSelectedZones}
        />
        <LiveBadge />
      </div>
      <EChart option={option} height={188} />
      <div className="trend-legend">
        {visibleZones.map((z) => {
          const idx = zones.indexOf(z);
          return (
            <div key={z.zone_id} className="tl-item">
              <div
                className="tl-dot"
                style={{
                  background: ZONE_COLORS[idx % ZONE_COLORS.length],
                }}
              />
              {z.zone_name}
            </div>
          );
        })}
      </div>
    </div>
  );
}

type ReportStatus = "idle" | "pending" | "processing" | "completed" | "failed";

function AiReport({
  cityLabel,
}: {
  cityLabel: string;
}) {
  const today = new Date();
  const weekAgo = new Date(today);
  weekAgo.setDate(today.getDate() - 7);
  const monthAgo = new Date(today);
  monthAgo.setMonth(today.getMonth() - 1);
  const todayIso = today.toISOString().slice(0, 10);
  const weekAgoIso = weekAgo.toISOString().slice(0, 10);
  const monthAgoIso = monthAgo.toISOString().slice(0, 10);
  const [fromDate, setFromDate] = useState(weekAgoIso);
  const [tillDate, setTillDate] = useState(todayIso);

  const subLocations = useLocationsByCity(cityLabel);

  const [reportZoneId, setReportZoneId] = useState("");

  // Reset and seed zone when city or location list changes
  useEffect(() => {
    setReportZoneId(subLocations[0]?.zone_id ?? "");
  }, [subLocations]);

  const [reportStatus, setReportStatus] = useState<ReportStatus>("idle");
  const [reportId, setReportId] = useState<string | null>(null);
  const [reportText, setReportText] = useState<string>("");
  const [reportError, setReportError] = useState<string | null>(null);
  const [agentTrace, setAgentTrace] = useState<AgentTraceStep[]>([]);

  const renderedReportHtml = useMemo(() => {
    if (!reportText) return "";
    const raw = marked.parse(reportText, { breaks: true, async: false }) as string;
    const clean = DOMPurify.sanitize(raw, {
      ALLOWED_TAGS: [
        "p", "br", "strong", "em", "b", "i", "u",
        "ul", "ol", "li",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "thead", "tbody", "tr", "th", "td",
        "code", "pre", "blockquote", "hr", "a",
      ],
      ALLOWED_ATTR: ["href", "class"],
      FORCE_BODY: true,
    });
    // Force noopener on all links after sanitization
    return clean.replace(/<a href=/g, '<a rel="noopener noreferrer" target="_blank" href=');
  }, [reportText]);

  const isRunning =
    reportStatus === "pending" || reportStatus === "processing";
  const canGenerate =
    fromDate.length > 0 && tillDate.length > 0 && reportZoneId.length > 0;

  const fetchFinalTrace = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports/${id}/trace`);
      if (!res.ok) return;
      const trace = (await res.json()) as AgentTraceStep[];
      if (trace.length > 0) setAgentTrace(trace);
    } catch {
      // Trace is supplemental — ignore failures
    }
  };

  const pollReportStatus = async (newReportId: string) => {
    for (let i = 0; i < 120; i++) {
      const res = await fetch(
        `${API_BASE}/api/v1/reports/${newReportId}/status`
      );
      if (!res.ok) {
        setReportStatus("failed");
        setReportError("Unable to fetch report job status.");
        return;
      }
      const data = (await res.json()) as {
        status: "pending" | "processing" | "completed" | "failed";
        error?: string | null;
        content?: string | null;
        agent_trace?: AgentTraceStep[] | null;
      };
      if (data.agent_trace && data.agent_trace.length > 0) {
        setAgentTrace(data.agent_trace);
      }
      if (data.status === "completed") {
        setReportStatus("completed");
        setReportError(null);
        if (data.content) setReportText(data.content);
        await fetchFinalTrace(newReportId);
        return;
      }
      if (data.status === "failed") {
        setReportStatus("failed");
        setReportError(data.error ?? "Report generation failed.");
        return;
      }
      setReportStatus(data.status);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    setReportStatus("failed");
    setReportError("Report generation timed out. Try again.");
  };

  const handleGenerateReport = async () => {
    if (!canGenerate) return;
    setReportError(null);
    setReportText("");
    setAgentTrace([]);
    setReportStatus("pending");
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zone_id: reportZoneId,
          from_date: fromDate,
          to_date: tillDate,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(
          text || `Failed to submit report generation (${res.status})`
        );
      }
      const data = (await res.json()) as { report_id: string };
      setReportId(data.report_id);
      await pollReportStatus(data.report_id);
    } catch (err) {
      setReportStatus("failed");
      setReportError(
        err instanceof Error ? err.message : "Failed to generate report."
      );
    }
  };

  const handleDownload = () => {
    if (!reportId) return;
    window.open(`${API_BASE}/api/v1/reports/${reportId}/download`, "_blank");
  };

  const showTrace = agentTrace.length > 0 || isRunning;
  const showReport = reportStatus === "completed" && reportText;

  return (
    <div className="report-card">
      <div className="report-card__head">
        AI Incident Summary
        <span className="ai-badge">AI GENERATED</span>
      </div>

      {/* Location selector — zones for the currently selected city */}
      <div className="report-location-row">
        <label className="report-field report-field--grow">
          <span className="report-field-label">Location</span>
          <select
            className="report-select"
            value={reportZoneId}
            onChange={(e) => setReportZoneId(e.target.value)}
            disabled={isRunning}
          >
            {subLocations.map((sub) => (
              <option key={sub.zone_id} value={sub.zone_id}>
                {sub.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {/* Date range + generate */}
      <div className="report-date-range">
        <label className="report-date-field">
          <span className="report-date-label">From</span>
          <input
            type="date"
            className="report-date-input"
            value={fromDate}
            min={monthAgoIso}
            max={todayIso}
            onChange={(e) => setFromDate(e.target.value)}
          />
        </label>
        <label className="report-date-field">
          <span className="report-date-label">Till</span>
          <input
            type="date"
            className="report-date-input"
            value={tillDate}
            min={monthAgoIso}
            max={todayIso}
            onChange={(e) => setTillDate(e.target.value)}
          />
        </label>
        <button
          className="report-generate-btn"
          onClick={() => void handleGenerateReport()}
          disabled={!canGenerate || isRunning}
        >
          {isRunning ? "Generating…" : "Generate Report"}
        </button>
      </div>

      <div className="report-body">
        {reportError && (
          <p className="report-error">
            <strong>Error:</strong> {reportError}
          </p>
        )}

        {/* Agent stepper — visible while running and after (collapses auto) */}
        {showTrace && (
          <AgentStepper steps={agentTrace} isRunning={isRunning} />
        )}

        {/* Report content — shown after trace auto-collapses */}
        {showReport ? (
          <div
            className="report-markdown"
            dangerouslySetInnerHTML={{ __html: renderedReportHtml }}
          />
        ) : !reportError && !isRunning && reportStatus === "idle" ? (
          <p className="report-placeholder">
            Select a city and location, then generate an AI summary.
          </p>
        ) : null}
      </div>

      <div className="report-actions">
        <button
          className="export-btn export-btn--primary"
          onClick={handleDownload}
          disabled={!reportId || reportStatus !== "completed"}
        >
          Download PDF Report
        </button>
      </div>
    </div>
  );
}

export default function PlanningPanel() {
  const dispatch = useAppDispatch();
  const timeRange = useAppSelector(selectTimeRange);
  const selectedMainLocationId = useAppSelector(selectMainLocationId);
  const { config: hotspotConfig } = useHotspotConfig();
  const mainLocations = hotspotConfig.main_locations;
  const selectedMainLocation = useMemo(
    () => mainLocations.find((location) => location.id === selectedMainLocationId),
    [mainLocations, selectedMainLocationId]
  );
  const selectedCityLabel = selectedMainLocation?.label ?? "";
  const { kpis, heatmap, trends } = useAnalytics(timeRange, selectedCityLabel);

  useEffect(() => {
    const locationIds = new Set(mainLocations.map((location) => location.id));
    if (!locationIds.has(selectedMainLocationId)) {
      dispatch(setSelectedMainLocationId(mainLocations[0]?.id ?? ""));
    }
  }, [dispatch, mainLocations, selectedMainLocationId]);

  const kpiCards = [
    {
      val: kpis?.active_critical_zones ?? 0,
      unit: "",
      label: "Active Critical Zones",
      subtext: kpis?.peak_crowd_location || "—",
      cardClass: (kpis?.active_critical_zones ?? 0) > 0 ? "kpi-card--warn" : "",
    },
    {
      val: kpis?.peak_crowd_count ?? 0,
      unit: "",
      label: "Peak Crowd Count",
      subtext: kpis?.peak_crowd_location || "—",
      cardClass: (kpis?.peak_crowd_count ?? 0) >= 200 ? "kpi-card--alert" : "",
    },
    {
      val: kpis?.streams_live ?? 0,
      unit: (kpis?.streams_total ?? 0) > (kpis?.streams_live ?? 0)
        ? `/${kpis?.streams_total ?? 0}`
        : "",
      label: "Streams Live",
      subtext: (kpis?.streams_total ?? 0) > (kpis?.streams_live ?? 0)
        ? `${(kpis?.streams_total ?? 0) - (kpis?.streams_live ?? 0)} unavailable`
        : "all streams active",
      cardClass: "",
    },
    {
      val: Math.round(kpis?.detection_latency_p95 ?? 0),
      unit: "ms",
      label: "Detection Latency P95",
      subtext: (kpis?.detection_latency_p95 ?? 0) <= 100
        ? "within 100ms SLA"
        : "exceeds SLA",
      cardClass: (kpis?.detection_latency_p95 ?? 0) <= 100
        ? "kpi-card--ok"
        : "kpi-card--warn",
    },
  ];

  return (
    <div className="plan-panel">
      <div className="plan-header">
        <span className="panel-title">Safety Analytics</span>
        <span className="panel-subtitle">
          AI-driven crowd insights - {RANGE_LABELS[timeRange]}
        </span>
        <span style={{ flex: 1 }} />
        <div className="time-range-group">
          {TIME_RANGE_KEYS.map((k) => (
            <button
              key={k}
              className={`tr-btn ${timeRange === k ? "active" : ""}`}
              onClick={() => dispatch(setTimeRange(k))}
            >
              {RANGE_LABELS[k]}
            </button>
          ))}
        </div>
      </div>

      <div className="kpi-strip">
        {kpiCards.map((kpi) => (
          <div key={kpi.label} className={`kpi-card ${kpi.cardClass}`}>
            <div className="kpi-val">
              <AnimatedNumber value={kpi.val} />
              {kpi.unit && <small> {kpi.unit}</small>}
            </div>
            <div className="kpi-label">{kpi.label}</div>
          </div>
        ))}
      </div>

      <div className="analytics-body">
        <div className="analytics-left">
          <PatternHeatmap heatmap={heatmap ?? []} />
          <TrendChart zones={trends?.zones ?? []} labels={trends?.labels ?? []} timeRange={timeRange} />
        </div>
        <div className="analytics-right">
          <AiReport cityLabel={selectedCityLabel} />
        </div>
      </div>
    </div>
  );
}
