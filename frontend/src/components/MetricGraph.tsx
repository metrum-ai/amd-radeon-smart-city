// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { memo, useEffect, useRef, useMemo } from "react";
import * as echarts from "echarts";
import { HISTORY_LEN } from "../hooks/usePrometheusQuery";
import { AnimatedNumber } from "../hooks/AnimatedNumber";

const GPU_COLORS = [
  "#ED1C24", "#F26522", "#FACC15", "#22C55E", "#A855F7",
];

const BORDER_COLOR = "#2E2E35";
const DIM_COLOR = "#70737A";
const TEXT_COLOR = "#B0B2B5";
const BG_COLOR = "#08080A";

function seriesColor(color: string, idx: number): string {
  if (idx === 0) return color;
  const available = GPU_COLORS.filter(
    (c) => c.toLowerCase() !== color.toLowerCase(),
  );
  return available[(idx - 1) % available.length];
}

function fmtNum(v: number): string {
  if (isNaN(v)) return "--";
  return v % 1 ? v.toFixed(1) : String(Math.round(v));
}

interface MetricGraphProps {
  label: string;
  series: Record<string, (number | null)[]>;
  timestamps: (number | null)[];
  unit: string;
  color: string;
  maxVal?: number;
}

const X_LABELS: string[] = Array.from(
  { length: HISTORY_LEN },
  (_, i) => {
    if (i === 0) return "-40s";
    if (i === Math.floor(HISTORY_LEN / 2)) return "-20s";
    if (i === HISTORY_LEN - 1) return "now";
    return "";
  },
);

function buildYAxisOption(
  maxVal: number | undefined,
): echarts.YAXisComponentOption {
  const base: echarts.YAXisComponentOption = {
    type: "value",
    min: 0,
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: {
      show: true,
      fontSize: 7,
      fontFamily: "Inter, sans-serif",
      color: DIM_COLOR,
      margin: 4,
      formatter: (v: number) => String(Math.round(v)),
    },
    splitLine: {
      show: true,
      lineStyle: { color: BORDER_COLOR, width: 0.5 },
    },
  };
  if (maxVal !== undefined) {
    base.max = maxVal;
    base.interval = maxVal / 4;
  }
  return base;
}

function buildStructuralOption(
  color: string,
  maxVal: number | undefined,
  label: string,
): echarts.EChartsOption {
  return {
    animation: true,
    animationDuration: 300,
    animationEasing: "linear",
    grid: {
      left: 28, right: 4, top: 4, bottom: 16,
      containLabel: false,
    },
    xAxis: {
      type: "category",
      data: X_LABELS,
      boundaryGap: false,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        show: true,
        interval: (_idx: number, value: string) =>
          value !== "",
        fontSize: 7,
        fontFamily: "Inter, sans-serif",
        color: DIM_COLOR,
        margin: 4,
      },
      splitLine: { show: false },
    },
    yAxis: buildYAxisOption(maxVal),
    tooltip: {
      trigger: "axis",
      backgroundColor: BG_COLOR,
      borderColor: BORDER_COLOR,
      borderWidth: 1,
      padding: [5, 8],
      textStyle: {
        fontSize: 10,
        fontFamily: "Inter, sans-serif",
        color: TEXT_COLOR,
      },
      axisPointer: {
        type: "line",
        lineStyle: {
          color: DIM_COLOR, width: 0.5, type: "dashed",
        },
      },
    },
    series: [{
      type: "line",
      name: label,
      data: [],
      smooth: false,
      symbol: "circle",
      symbolSize: 4,
      showSymbol: true,
      lineStyle: {
        color, width: 1.5,
        join: "round" as const,
        cap: "round" as const,
      },
      itemStyle: {
        color, borderWidth: 0, opacity: 0.5,
      },
      emphasis: {
        itemStyle: { opacity: 1, borderWidth: 0 },
        scale: 1.8,
      },
      areaStyle: { color: color + "10" },
    }],
  };
}

function MetricGraphInner({
  label,
  series,
  timestamps,
  unit,
  color,
  maxVal,
}: MetricGraphProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const instRef = useRef<echarts.ECharts | null>(null);
  const prevKeysRef = useRef<string>("");
  const timestampsRef = useRef(timestamps);
  timestampsRef.current = timestamps;
  const unitRef = useRef(unit);
  unitRef.current = unit;

  const keysStr = useMemo(() => {
    return Object.keys(series).sort().join(",");
  }, [series]);

  const keys = useMemo(
    () => keysStr ? keysStr.split(",") : [],
    [keysStr],
  );

  const hasData = keys.length > 0 &&
    keys.some((k) => series[k]?.some((v) => v !== null));

  const avgCurrent = useMemo(() => {
    const vals = keys
      .map((k) => series[k]?.[series[k]?.length - 1])
      .filter(
        (v): v is number =>
          v !== null && v !== undefined,
      );
    if (vals.length === 0) return null;
    return (
      Math.round(
        (vals.reduce((a, b) => a + b, 0) / vals.length) *
        10,
      ) / 10
    );
  }, [keys, series]);

  useEffect(() => {
    if (!chartRef.current) return;
    const inst = echarts.init(chartRef.current, undefined, {
      renderer: "canvas",
    });
    instRef.current = inst;

    const base = buildStructuralOption(color, maxVal, label);
    inst.setOption(base);

    const ro = new ResizeObserver(() => inst.resize());
    ro.observe(chartRef.current);
    return () => {
      ro.disconnect();
      inst.dispose();
      instRef.current = null;
    };
  }, [color, maxVal, label]);

  useEffect(() => {
    const inst = instRef.current;
    if (!inst) return;

    const structureChanged = keysStr !== prevKeysRef.current;
    prevKeysRef.current = keysStr;

    const isMulti = keys.length > 1;

    if (structureChanged || isMulti) {
      const fullSeries: echarts.SeriesOption[] = keys.map(
        (k, idx) => {
          const c = isMulti
            ? seriesColor(color, idx)
            : color;
          return {
            id: `s-${k}`,
            type: "line" as const,
            color: c,
            name: isMulti ? `GPU ${k}` : label,
            data: series[k] || [],
            smooth: false,
            symbol: "circle",
            symbolSize: 4,
            showSymbol: true,
            lineStyle: {
              color: c, width: 1.5,
              join: "round" as const,
              cap: "round" as const,
            },
            itemStyle: {
              color: c, borderWidth: 0, opacity: 0.5,
            },
            emphasis: {
              itemStyle: {
                opacity: 1, borderWidth: 0,
              },
              scale: 1.8,
            },
            areaStyle: { color: c + "10" },
          };
        },
      );

      inst.setOption(
        {
          series: fullSeries,
          tooltip: {
            formatter: (
              params: echarts.DefaultLabelFormatterCallbackParams[],
            ) => buildTooltip(
              params, timestampsRef, unitRef, keys,
            ),
          },
        },
        structureChanged
          ? { replaceMerge: ["series"] }
          : { lazyUpdate: true },
      );
    } else {
      const k = keys[0];
      const data = k ? series[k] || [] : [];
      inst.setOption(
        {
          series: [{ data }],
          tooltip: {
            formatter: (
              params: echarts.DefaultLabelFormatterCallbackParams[],
            ) => buildTooltip(
              params, timestampsRef, unitRef, keys,
            ),
          },
        },
        { lazyUpdate: true },
      );
    }
  }, [series, keysStr]);

  return (
    <div className="metric-graph">
      <div className="metric-graph__head">
        <span className="metric-graph__label">
          {label}
        </span>
        {hasData && avgCurrent !== null ? (
          <span
            className="metric-graph__val"
            style={{ color }}
          >
            <AnimatedNumber
              value={avgCurrent}
              format={fmtNum}
            />
            <span className="metric-graph__unit">
              {unit}
            </span>
          </span>
        ) : (
          <span className="metric-graph__val-empty">
            --
          </span>
        )}
      </div>
      <div className="metric-graph__card">
        <div
          ref={chartRef}
          className="metric-graph__chart"
        />
      </div>
      {keys.length > 1 && hasData && (
        <div className="metric-graph__legend">
          {keys.map((k, i) => {
            const c = seriesColor(color, i);
            return (
              <div
                key={k}
                className="metric-graph__legend-item"
              >
                <span
                  className="metric-graph__legend-dot"
                  style={{ background: c }}
                />
                <span
                  className="metric-graph__legend-text"
                >
                  GPU {k}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function buildTooltip(
  params: echarts.DefaultLabelFormatterCallbackParams[],
  timestampsRef: React.RefObject<(number | null)[]>,
  unitRef: React.RefObject<string>,
  keys: string[],
): string {
  if (!Array.isArray(params) || params.length === 0) {
    return "";
  }
  const idx = params[0].dataIndex as number;
  const ts = timestampsRef.current?.[idx];
  const u = unitRef.current;
  let timeStr = "--:--:--";
  if (ts) {
    const d = new Date(ts);
    timeStr = d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  let html =
    `<div style="font-size:8px;color:${DIM_COLOR};` +
    `margin-bottom:3px;letter-spacing:0.04em">` +
    `${timeStr}</div>`;

  const isMulti = keys.length > 1;
  for (const p of params) {
    const val = p.value as number | null;
    const c = p.color as string;
    html +=
      `<div style="font-size:10px;font-weight:600;` +
      `color:${c};display:flex;` +
      `align-items:center;gap:4px">`;
    if (isMulti) {
      html +=
        `<span style="width:6px;height:6px;` +
        `border-radius:3px;background:${c};` +
        `display:inline-block;flex-shrink:0"></span>`;
      html +=
        `<span style="font-size:8px;color:` +
        `${TEXT_COLOR};font-weight:500">` +
        `${p.seriesName}</span>`;
    }
    const display =
      val !== null && val !== undefined
        ? (val % 1 ? val.toFixed(1) : val) + (u ?? "")
        : "--";
    html += `<span>${display}</span></div>`;
  }
  return html;
}

const MetricGraph = memo(MetricGraphInner);
export default MetricGraph;
