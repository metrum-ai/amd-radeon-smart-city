// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { useEffect, useState, useCallback } from "react";

export type AgentTraceStep = {
  agent: string;
  tool?: string | null;
  summary: string;
  timestamp: string;
  input_preview?: string | null;
};

type Props = {
  steps: AgentTraceStep[];
  isRunning: boolean;
};

const AGENT_META: Record<
  string,
  { color: string; label: string; abbr: string }
> = {
  InvestigatorAgent: {
    color: "var(--amd-teal)",
    label: "Investigator",
    abbr: "INV",
  },
  SOPAdvisorAgent: {
    color: "var(--amd-orange)",
    label: "SOP Advisor",
    abbr: "SOP",
  },
  OrchestratorAgent: {
    color: "var(--sev-critical)",
    label: "Orchestrator",
    abbr: "ORC",
  },
};

const SUMMARY_TRUNCATE = 120;

function ExpandableSummary({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const toggle = useCallback(() => setExpanded((e) => !e), []);
  const isLong = text.length > SUMMARY_TRUNCATE;

  return (
    <p className="stepper-summary">
      {isLong && !expanded ? `${text.slice(0, SUMMARY_TRUNCATE)}…` : text}
      {isLong && (
        <button className="stepper-expand-btn" onClick={toggle}>
          {expanded ? " show less" : " show more"}
        </button>
      )}
    </p>
  );
}

function agentMeta(agent: string) {
  return (
    AGENT_META[agent] ?? {
      color: "var(--border-light)",
      label: agent,
      abbr: agent.slice(0, 3).toUpperCase(),
    }
  );
}

export function AgentStepper({ steps, isRunning }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  // Auto-collapse 1.4 s after the last step lands and run is done
  useEffect(() => {
    if (!isRunning && steps.length > 0) {
      const t = setTimeout(() => setCollapsed(true), 1400);
      return () => clearTimeout(t);
    }
  }, [isRunning, steps.length]);

  // Reset to expanded when a new run starts
  useEffect(() => {
    if (isRunning) setCollapsed(false);
  }, [isRunning]);

  if (steps.length === 0 && !isRunning) return null;

  // Derive what's currently running from the last dispatched step
  const lastStep = steps[steps.length - 1];
  const activeLabel = isRunning
    ? lastStep
      ? `${agentMeta(lastStep.agent).label} running…`
      : "Starting…"
    : `${steps.length} steps`;

  return (
    <div className="agent-stepper">
      <button
        className="stepper-header"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        <span className="stepper-title">Agent Execution Trace</span>
        <span className="stepper-count mono">{activeLabel}</span>
        {isRunning && <span className="stepper-live-dot" />}
        <span className={`stepper-chevron${collapsed ? " is-collapsed" : ""}`}>
          ▾
        </span>
      </button>

      {!collapsed && (
        <div className="stepper-body">
          {steps.map((step, idx) => {
            const meta = agentMeta(step.agent);
            const isLast = idx === steps.length - 1 && !isRunning;
            return (
              <div
                key={`${step.agent}-${step.timestamp}-${idx}`}
                className={`stepper-step${isLast ? " stepper-step--last" : ""}`}
              >
                <div className="stepper-rail">
                  <div
                    className="stepper-node"
                    style={{ borderColor: meta.color, color: meta.color }}
                  >
                    {meta.abbr}
                  </div>
                  {!isLast && (
                    <div
                      className="stepper-connector"
                      style={{ background: meta.color }}
                    />
                  )}
                </div>
                <div className="stepper-info">
                  <div className="stepper-info-head">
                    <span
                      className="stepper-agent-name"
                      style={{ color: meta.color }}
                    >
                      {meta.label}
                    </span>
                    {step.tool && (
                      <span className="stepper-tool-pill">{step.tool}</span>
                    )}
                    <span className="stepper-ts mono">
                      {new Date(step.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </span>
                  </div>
                  <ExpandableSummary text={step.summary} />
                </div>
              </div>
            );
          })}

          {isRunning && (
            <div className="stepper-step stepper-step--active">
              <div className="stepper-rail">
                <div className="stepper-node stepper-node--spinner">
                  <span className="stepper-spinner" />
                </div>
              </div>
              <div className="stepper-info">
                <span className="stepper-processing-label">
                  Processing&hellip;
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
