// Created by Metrum AI for AMD

import {
  memo,
  useRef,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useAppSelector, useAppDispatch } from "../store";
import { selectMainLocationId, selectSeverityFilter } from "../features/app/appSelectors";
import { setSeverityFilter } from "../features/app/appSlice";
import type { Severity } from "../features/app/appSlice";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import { useHotspotConfig } from "../hooks/useHotspotConfig";
import { useMapData } from "../hooks/useMapData";
import { API_BASE } from "../lib/runtimeConfig";

const PAGE_SIZE = 6;
const STREAM_LIMIT = 100;

import "../styles/components/CameraPanel.css";

type StreamStats = {
  id: number;
  fps: number;
  latency_ms: number;
  status: string;
};

type CameraFeed = {
  streamId: number;
  name: string;
  personCount: number;
  severity: "SAFE" | "CRITICAL";
  pairedWhepUrl: string | null;
  fps: number | null;
  latencyMs: number | null;
  status: string;
};

const FILL_VIDEO: React.CSSProperties = {
  display: "block",
  width: "100%",
  height: "100%",
  objectFit: "cover",
};

// STUN servers for WebRTC NAT traversal (Bug 310)
const ICE_SERVERS: RTCIceServer[] = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

/**
 * Direct WHEP WebRTC player for MediaMTX stream endpoints.
 */
const WhepPlayer = memo(function WhepPlayer({
  whepUrl,
  style,
  onConnectionChange,
}: {
  whepUrl: string;
  style?: React.CSSProperties;
  onConnectionChange?: (connected: boolean) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let aborted = false;
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
    pcRef.current = pc;

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    pc.ontrack = (ev) => {
      if (video.srcObject !== ev.streams[0]) {
        video.srcObject = ev.streams[0];
      }
    };

    pc.onconnectionstatechange = () => {
      const connected = pc.connectionState === "connected";
      onConnectionChange?.(connected);
    };

    const url = new URL(whepUrl, window.location.origin).toString();

    (async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        if (aborted) return;

        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: offer.sdp,
        });

        if (!res.ok || aborted) return;

        const answerSdp = await res.text();
        await pc.setRemoteDescription({
          type: "answer",
          sdp: answerSdp,
        });
      } catch {
        onConnectionChange?.(false);
      }
    })();

    return () => {
      aborted = true;
      pc.close();
      pcRef.current = null;
      video.srcObject = null;
    };
  }, [whepUrl, onConnectionChange]);

  return <video ref={videoRef} autoPlay muted playsInline style={style} />;
});

const CamCard = memo(function CamCard({
  feed,
}: {
  feed: CameraFeed;
}) {
  const sev = feed.severity === "CRITICAL" ? "critical" : "safe";
  const count = feed.personCount;
  const sevLabel = sev === "critical" ? "CRITICAL" : "SAFE";
  const sevClass = sev === "critical" ? "sev-critical" : "";

  return (
    <div className={`cam-card ${sevClass}`}>
      <div className="cam-card__videos cam-card__videos--paired">
        {feed.pairedWhepUrl ? (
          <WhepPlayer whepUrl={feed.pairedWhepUrl} style={FILL_VIDEO} />
        ) : (
          <div className="cam-card__unavailable">Stream unavailable</div>
        )}
        <div className="cam-card__panel-labels">
          <span className="cam-card__panel-label">YOLO</span>
          <span className="cam-card__panel-label">DENSITY</span>
        </div>
        <span className={`cam-card__sev-badge cam-card__sev-badge--${sev}`}>
          {sevLabel}
        </span>
      </div>
      <div className="cam-card__data">
        <span className="cam-card__location">{feed.name}</span>
        <AnimatedNumber value={count} className="cam-card__stat mono" />
        <span className="cam-card__stat mono">
          {feed.fps != null ? `${feed.fps.toFixed(1)}fps` : "-- fps"}
        </span>
        <span className="cam-card__stat mono">
          {feed.latencyMs != null ? `${Math.round(feed.latencyMs)}ms` : "-- ms"}
        </span>
        {sev === "critical" && <span className="cam-card__alert mono">ALERT</span>}
      </div>
    </div>
  );
});

const SEV_OPTIONS: {
  key: "all" | Severity;
  label: string;
  color?: string;
}[] = [
    { key: "all", label: "All" },
    { key: "critical", label: "Critical", color: "var(--sev-critical)" },
    { key: "safe", label: "Safe", color: "var(--sev-safe)" },
  ];

export default function CameraPanel() {
  const dispatch = useAppDispatch();
  const severityFilter = useAppSelector(selectSeverityFilter);
  const selectedMainLocationId = useAppSelector(selectMainLocationId);
  const [page, setPage] = useState(0);
  const { config } = useHotspotConfig();
  const selectedCityLabel = useMemo(
    () => config.main_locations.find((loc) => loc.id === selectedMainLocationId)?.label,
    [config.main_locations, selectedMainLocationId],
  );
  const { mapData, loading: mapLoading } = useMapData(10_000, selectedCityLabel);

  // Reset to page 0 whenever the active city changes.
  useEffect(() => {
    setPage(0);
  }, [selectedMainLocationId]);
  const [streamStatsById, setStreamStatsById] = useState<Record<number, StreamStats>>({});

  useEffect(() => {
    let cancelled = false;

    const fetchStreams = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/streams`);
        if (!res.ok) return;
        const data = (await res.json()) as StreamStats[];
        if (cancelled) return;
        const byId = data.reduce<Record<number, StreamStats>>((acc, item) => {
          acc[item.id] = item;
          return acc;
        }, {});
        setStreamStatsById(byId);
      } catch {
        // Keep last successful stats on transient backend/network failures.
      }
    };

    void fetchStreams();
    const id = window.setInterval(fetchStreams, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const allCameras = useMemo(
    () =>
      [...(mapData?.cameras ?? [])]
        .sort((a, b) => a.stream_id - b.stream_id)
        .slice(0, STREAM_LIMIT),
    [mapData?.cameras],
  );

  const handleSevClick = useCallback(
    (sev: "all" | Severity) => {
      dispatch(setSeverityFilter(severityFilter === sev ? "all" : sev));
      setPage(0);
    },
    [severityFilter, dispatch],
  );

  const feedsWithLive = allCameras.map((camera) => {
    const stats = streamStatsById[camera.stream_id];
    const displaySev = camera.severity === "CRITICAL" ? "critical" : "safe";
    return {
      feed: {
        streamId: camera.stream_id,
        name: camera.location_name || `cam${camera.stream_id}`,
        personCount: camera.person_count,
        severity: camera.severity,
        pairedWhepUrl: camera.webrtc_whep_url ?? null,
        fps: stats?.fps ?? null,
        latencyMs: stats?.latency_ms ?? null,
        status: stats?.status ?? camera.stream_status,
      } satisfies CameraFeed,
      displaySev,
    };
  });

  const filtered = feedsWithLive.filter(
    ({ displaySev }) => severityFilter === "all" || displaySev === severityFilter,
  );
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageSlice = filtered.slice(
    safePage * PAGE_SIZE,
    safePage * PAGE_SIZE + PAGE_SIZE,
  );

  const totalCritical = feedsWithLive.filter(
    (f) => f.displaySev === "critical"
  ).length;
  const totalSafe = feedsWithLive.filter(
    (f) => f.displaySev === "safe"
  ).length;

  return (
    <div className="cam-panel">
      <div className="cam-header">
        <span className="panel-title">Camera Feed Alerts</span>
        <span style={{ flex: 1 }} />
        <div className="sev-chip-group">
          {SEV_OPTIONS.map((opt) => {
            const count =
              opt.key === "all"
                ? feedsWithLive.length
                : opt.key === "critical"
                  ? totalCritical
                  : totalSafe;
            const isActive = severityFilter === opt.key;
            return (
              <button
                key={opt.key}
                className={`sev-chip sev-chip--${opt.key} ${isActive ? "sev-chip--active" : ""}`}
                onClick={() => handleSevClick(opt.key)}
              >
                {opt.color && (
                  <span
                    className={`sev-chip__dot ${opt.key === "critical" ? "sev-chip__dot--pulse" : ""}`}
                    style={{ background: opt.color }}
                  />
                )}
                <span className="sev-chip__label">{opt.label}</span>
                <span className="sev-chip__count">{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="cam-body">
        {mapLoading && !mapData ? (
          <div className="cam-loading">
            <span className="cam-loading__spinner" />
            <span className="cam-loading__label">Loading feeds…</span>
          </div>
        ) : (
          <div className="cam-grid" key={selectedMainLocationId}>
            {pageSlice.map(({ feed }) => (
              <CamCard key={feed.streamId} feed={feed} />
            ))}
          </div>
        )}
      </div>
      {totalPages > 1 && (
        <div className="cam-pagination">
          <button
            className="cam-pagination__btn"
            disabled={safePage === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            Prev
          </button>
          <span className="cam-pagination__info mono">
            {safePage + 1} / {totalPages}
          </span>
          <button
            className="cam-pagination__btn"
            disabled={safePage >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
