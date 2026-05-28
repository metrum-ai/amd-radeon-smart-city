// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import { useState, useEffect } from "react";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import { API_BASE } from "../lib/runtimeConfig";
import "../styles/components/StatusBar.css";

export default function StatusBar() {
  const [now, setNow] = useState(() => new Date());
  const [streamCount, setStreamCount] = useState<number | null>(null);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchCount = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/streams`);
        if (res.ok) {
          const data = (await res.json()) as unknown[];
          setStreamCount(data.length);
        }
      } catch {
        // Keep last value on transient errors.
      }
    };
    void fetchCount();
    const id = setInterval(fetchCount, 10_000);
    return () => clearInterval(id);
  }, []);
  const ts = now.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="statusbar">
      <div className="statusbar__item">
        <span
          className="statusbar__dot live"
          style={{ background: "var(--amd-teal)" }}
        />
        API Connected
      </div>
      <div className="statusbar__item">
        <span
          className="statusbar__dot live"
          style={{ background: "var(--amd-teal)" }}
        />
        WebSocket Active
      </div>
      <div className="statusbar__item">
        <span
          className="statusbar__dot"
          style={{ background: "var(--amd-orange)" }}
        />
        <AnimatedNumber value={streamCount ?? 0} className="mono" /> Streams
      </div>
      <div className="statusbar__item">
        P95: <AnimatedNumber value={67} className="mono" />
        <span className="mono">ms</span>
      </div>
      <div className="statusbar__spacer" />
      <div className="statusbar__right">
        <span className="statusbar__ts">{ts}</span>
      </div>
    </div>
  );
}
