// Created by Metrum AI for AMD

import { useState, useEffect } from "react";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import "../styles/components/StatusBar.css";

export default function StatusBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
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
        <AnimatedNumber value={64} className="mono" /> /{" "}
        <AnimatedNumber value={100} className="mono" /> Streams
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
