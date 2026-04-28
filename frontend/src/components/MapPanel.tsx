// Created by Metrum AI for AMD

import { useEffect, useCallback, useMemo, useState, useRef, memo, useId } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import Place from "@mui/icons-material/Place";
import "leaflet/dist/leaflet.css";
import { useAppSelector, useAppDispatch } from "../store";
import {
  selectMode,
  selectMainLocationId,
  selectTimeRange,
} from "../features/app/appSelectors";
import {
  setSelectedMainLocationId,
  RANGE_LABELS,
} from "../features/app/appSlice";
import type { CameraData } from "../features/app/appSlice";
import { AnimatedNumber } from "../hooks/AnimatedNumber";
import { SEV_COLORS, SEV_RADIUS } from "../data";
import { useHotspotConfig } from "../hooks/useHotspotConfig";
import { useMapData } from "../hooks/useMapData";
import { useAlerts } from "../hooks/useAlerts";
import "../styles/components/MapPanel.css";

const GOOGLE_TILES =
  "https://mt{s}.google.com/vt/lyrs=r&x={x}&y={y}&z={z}";
const GOOGLE_ATTRIBUTION = "Google Maps";

function MapInvalidator() {
  const map = useMap();
  const mode = useAppSelector(selectMode);

  useEffect(() => {
    const timer = setTimeout(() => map.invalidateSize(), 400);
    return () => clearTimeout(timer);
  }, [mode, map]);

  useEffect(() => {
    const container = map.getContainer();
    if (!container || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => map.invalidateSize());
    ro.observe(container);
    return () => ro.disconnect();
  }, [map]);

  return null;
}

function FlyTo({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 1.2 });
  }, [center, zoom, map]);
  return null;
}

function useZoom(initial: number) {
  const [zoom, setZoom] = useState(initial);
  useMapEvents({ zoomend: (e) => setZoom(e.target.getZoom()) });
  return zoom;
}

function zoomScale(zoom: number): number {
  if (zoom >= 16) return 0.7;
  if (zoom >= 14) return 1.0;
  if (zoom >= 12) return 1.4;
  return 1.8;
}

const FILL_VIDEO: React.CSSProperties = {
  display: "block",
  width: "100%",
  height: "100%",
  objectFit: "cover",
};

// No external STUN — the deployment is LAN/on-prem; ICE-TCP via the nginx
// stream proxy uses host candidates only.  External STUN would require an
// internet round-trip that blocks ICE gathering for 5–8 s on isolated networks.
const ICE_SERVERS: RTCIceServer[] = [];

// Maximum ms to wait for ICE gathering before sending the WHEP offer.
// Host candidates appear in < 50 ms on LAN; the cap prevents an indefinite
// wait if the environment is unusual.
const ICE_GATHER_TIMEOUT_MS = 1_500;

// ----------------------------------------------------------------
// WHEP Connection Pool
//
// Persists RTCPeerConnection sessions across Leaflet tooltip
// mount/unmount cycles so re-hovering a marker shows video
// instantly.  Pre-connects in the background for visible markers
// so even the *first* hover is near-instant.
// ----------------------------------------------------------------

const POOL_RETRY_LIMIT = 4;
const POOL_RETRY_BASE_MS = 2_500;

interface _PoolEntry {
  pc: RTCPeerConnection;
  streamReady: Promise<MediaStream>;
  failed: boolean;
  onFailed: (() => void) | null;
}

const _whepPool = new Map<string, _PoolEntry>();

function _poolConnect(whepUrl: string): _PoolEntry {
  const existing = _whepPool.get(whepUrl);
  if (existing && !existing.failed) return existing;

  if (existing) {
    existing.pc.close();
    _whepPool.delete(whepUrl);
  }

  const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });
  let resolveStream!: (s: MediaStream) => void;
  let rejectStream!: (e: Error) => void;
  const streamReady = new Promise<MediaStream>((res, rej) => {
    resolveStream = res;
    rejectStream = rej;
  });

  const entry: _PoolEntry = { pc, streamReady, failed: false, onFailed: null };
  _whepPool.set(whepUrl, entry);

  // Video-only — the analytics pipeline does not publish audio tracks.
  // A single video transceiver halves ICE candidate work vs. audio+video.
  pc.addTransceiver("video", { direction: "recvonly" });

  pc.ontrack = (ev) => resolveStream(ev.streams[0]);

  pc.onconnectionstatechange = () => {
    const s = pc.connectionState;
    if (s === "failed" || s === "closed") {
      entry.failed = true;
      rejectStream(new Error("connection_failed"));
      entry.onFailed?.();
    }
  };

  // Some browsers only transition iceConnectionState to "failed" without
  // ever setting connectionState to "failed".  Handle both.
  pc.oniceconnectionstatechange = () => {
    if (pc.iceConnectionState === "failed") {
      entry.failed = true;
      rejectStream(new Error("ice_failed"));
      entry.onFailed?.();
    }
  };

  const url = new URL(whepUrl, window.location.origin).toString();
  (async () => {
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      // Wait for ICE gathering to complete (or hit the cap) so that all
      // local host candidates are included in the SDP we send to MediaMTX.
      // Without this, the offer arrives with zero candidates and negotiation
      // silently stalls on connections that need them.
      if (pc.iceGatheringState !== "complete") {
        await Promise.race([
          new Promise<void>((resolve) => {
            const onState = () => {
              if (pc.iceGatheringState === "complete") {
                pc.removeEventListener("icegatheringstatechange", onState);
                resolve();
              }
            };
            pc.addEventListener("icegatheringstatechange", onState);
          }),
          new Promise<void>((resolve) =>
            setTimeout(resolve, ICE_GATHER_TIMEOUT_MS),
          ),
        ]);
      }

      // Use pc.localDescription.sdp — it contains the candidates gathered
      // above, unlike the original offer.sdp snapshot which had none.
      const sdpToSend = pc.localDescription?.sdp ?? offer.sdp;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: sdpToSend,
      });
      if (!res.ok) {
        entry.failed = true;
        pc.close();
        rejectStream(new Error(`whep_${res.status}`));
        return;
      }
      const answerSdp = await res.text();
      await pc.setRemoteDescription({
        type: "answer",
        sdp: answerSdp,
      });
    } catch (err) {
      entry.failed = true;
      pc.close();
      rejectStream(
        err instanceof Error ? err : new Error("whep_error"),
      );
    }
  })();

  return entry;
}

/** Tear down every connection in the pool. */
function _poolCleanAll() {
  for (const [, entry] of _whepPool) entry.pc.close();
  _whepPool.clear();
}

// ----------------------------------------------------------------
// usePooledStream - resolves a cached or new MediaStream for a URL
// ----------------------------------------------------------------

function usePooledStream(whepUrl: string | undefined) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!whepUrl) {
      setStream(null);
      setFailed(false);
      return;
    }

    let cancelled = false;
    let retryTimer: number | null = null;
    let attempts = 0;
    let currentEntry: _PoolEntry | null = null;

    const tryConnect = () => {
      if (cancelled) return;
      const entry = _poolConnect(whepUrl);
      currentEntry = entry;
      entry.streamReady
        .then((s) => {
          if (!cancelled) {
            // If any track has ended (PC renegotiated / disconnected since
            // the stream was first captured), evict and fast-retry rather
            // than handing a dead stream to the video element.
            if (s.getTracks().some((t) => t.readyState === "ended")) {
              _whepPool.delete(whepUrl);
              currentEntry = null;
              attempts += 1;
              if (attempts <= POOL_RETRY_LIMIT) {
                retryTimer = window.setTimeout(tryConnect, 200);
              } else {
                setFailed(true);
              }
              return;
            }
            setStream(s);
            setFailed(false);
            entry.onFailed = () => {
              if (cancelled) return;
              setStream(null);
              _whepPool.delete(whepUrl);
              currentEntry = null;
              attempts += 1;
              if (attempts <= POOL_RETRY_LIMIT) {
                const delay =
                  POOL_RETRY_BASE_MS * Math.pow(1.5, attempts - 1);
                retryTimer = window.setTimeout(tryConnect, delay);
              } else {
                setFailed(true);
              }
            };
          }
        })
        .catch(() => {
          if (cancelled) return;
          if (currentEntry) currentEntry.onFailed = null;
          currentEntry = null;
          attempts += 1;
          if (attempts <= POOL_RETRY_LIMIT) {
            const delay =
              POOL_RETRY_BASE_MS * Math.pow(1.5, attempts - 1);
            retryTimer = window.setTimeout(tryConnect, delay);
          } else {
            setFailed(true);
          }
        });
    };

    tryConnect();

    return () => {
      cancelled = true;
      if (retryTimer !== null) clearTimeout(retryTimer);
      if (currentEntry) currentEntry.onFailed = null;
    };
  }, [whepUrl]);

  return { stream, failed };
}

// ----------------------------------------------------------------
// TooltipVideo - wires a pooled MediaStream into a <video> element
// ----------------------------------------------------------------

const TOOLTIP_SPINNER_WRAP: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  pointerEvents: "none",
};

const TooltipVideo = memo(function TooltipVideo({
  whepUrl,
  width,
  aspectRatio,
}: {
  whepUrl?: string;
  width?: number;
  aspectRatio?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { stream, failed } = usePooledStream(whepUrl);
  const [playing, setPlaying] = useState(false);

  // Attach the cached MediaStream to the <video> DOM node.
  // autoPlay alone is unreliable when srcObject is set programmatically
  // inside Leaflet's tooltip DOM — explicit play() is required.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !stream) {
      if (videoRef.current) videoRef.current.srcObject = null;
      setPlaying(false);
      return;
    }
    video.srcObject = stream;
    const onPlay = () => setPlaying(true);
    video.addEventListener("playing", onPlay);
    // "canplay" fires earlier than "playing" on some browsers and covers
    // the case where the video resumes without a full play/pause cycle.
    video.addEventListener("canplay", onPlay);
    video.play().catch(() => {
      // Autoplay was blocked or stream not ready yet — the "playing" event
      // will still fire once the browser allows it.
    });
    return () => {
      video.removeEventListener("playing", onPlay);
      video.removeEventListener("canplay", onPlay);
      video.srcObject = null;
    };
  }, [stream]);

  // Stuck-frame detector: for live RTSP streams currentTime advances
  // continuously.  If it hasn't moved for 3 × 2 s = 6 s, the stream is
  // frozen — evict the pool entry and trigger a reconnect.
  useEffect(() => {
    if (!playing || !whepUrl) return;
    const video = videoRef.current;
    if (!video) return;
    let lastTime = video.currentTime;
    let stuckCount = 0;
    const timer = setInterval(() => {
      if (!videoRef.current) return;
      const cur = videoRef.current.currentTime;
      if (cur === lastTime) {
        if (++stuckCount >= 3) {
          clearInterval(timer);
          const poolEntry = _whepPool.get(whepUrl);
          if (poolEntry && !poolEntry.failed) {
            poolEntry.failed = true;
            poolEntry.onFailed?.();
          }
        }
      } else {
        lastTime = cur;
        stuckCount = 0;
      }
    }, 2_000);
    return () => clearInterval(timer);
  }, [playing, whepUrl]);

  useEffect(() => {
    setPlaying(false);
  }, [whepUrl]);

  const containerStyle = useMemo(
    () => ({
      width: width ?? 220,
      aspectRatio: aspectRatio ?? "16/9",
      background: "#111",
      borderRadius: 3,
      overflow: "hidden",
      marginTop: 6,
      position: "relative" as const,
    }),
    [width, aspectRatio],
  );

  if (!whepUrl) return null;

  return (
    <div style={containerStyle}>
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={FILL_VIDEO}
      />
      {!playing && !failed && (
        <div style={TOOLTIP_SPINNER_WRAP}>
          <span className="map-loading-spinner" />
        </div>
      )}
      {failed && (
        <div style={TOOLTIP_SPINNER_WRAP}>
          <span style={{ color: "#888", fontSize: 11 }}>No signal</span>
        </div>
      )}
    </div>
  );
});

/**
 * HotspotMarkers - memoized for performance (Task 315, 325).
 * Debounces marker updates via useMapData's useTransition.
 */
const HotspotMarkers = memo(function HotspotMarkers() {
  const zoom = useZoom(13);
  const scale = zoomScale(zoom);
  const { config } = useHotspotConfig();
  const [activeTooltipStreamId, setActiveTooltipStreamId] = useState<
    number | null
  >(null);
  const selectedMainLocationId = useAppSelector(selectMainLocationId);
  const selectedMainLocation = useMemo(
    () => config.main_locations.find((loc) => loc.id === selectedMainLocationId),
    [config.main_locations, selectedMainLocationId]
  );
  const { mapData } = useMapData(10_000, selectedMainLocation?.label);

  const cameras: CameraData[] = useMemo(() => {
    if (!mapData?.cameras.length) return [];
    return mapData.cameras.map((c) => ({
      stream_id: c.stream_id,
      lat: c.lat,
      lon: c.lon,
      name: c.location_name,
      count: c.person_count,
      sev: (c.severity === "CRITICAL" ? "critical" : "safe") as "critical" | "safe",
      previewUrl: c.preview_video_url ?? undefined,
      webrtcWhepUrl: c.webrtc_whep_url ?? undefined,
      locationTag: c.location_tag ?? undefined,
      markerColor: c.marker_color ?? undefined,
      markerRadius: c.marker_radius ?? undefined,
      markerPulse: c.marker_pulse ?? undefined,
      tooltipWidth: c.tooltip_width ?? undefined,
      tooltipAspectRatio: c.tooltip_aspect_ratio ?? undefined,
    }));
  }, [mapData?.cameras]);

  useEffect(() => {
    setActiveTooltipStreamId(null);
    _poolCleanAll();
  }, [selectedMainLocationId]);

  useEffect(() => () => _poolCleanAll(), []);

  return (
    <>
      {cameras.map((cam, i) => {
        const baseR = SEV_RADIUS[cam.sev];
        const styleCfg = config.styles[cam.sev];
        const r = Math.round((cam.markerRadius ?? styleCfg?.radius ?? baseR) * scale);
        const color = cam.markerColor ?? styleCfg?.color ?? SEV_COLORS[cam.sev];
        const isCrit = cam.sev === "critical";
        const pulse = cam.markerPulse ?? styleCfg?.pulse ?? isCrit;
        return (
          <CircleMarker
            key={`marker-${cam.stream_id}`}
            center={[cam.lat, cam.lon]}
            radius={r}
            pathOptions={{
              color,
              fillColor: color,
              fillOpacity: 0.5,
              weight: 2,
              opacity: 0.9,
              className: pulse ? "hotspot-pulse" : "",
            }}
            eventHandlers={{
              mouseover: () => {
                if (!cam.webrtcWhepUrl) return;
                // Warm only the marker the user is approaching.  Do not
                // pre-open readers for the whole city; that overloads MediaMTX
                // and causes every tooltip to degrade together.
                _poolConnect(cam.webrtcWhepUrl).streamReady.catch(() => {
                  // The active TooltipVideo owns visible failure state.
                });
              },
              tooltipopen: () => setActiveTooltipStreamId(cam.stream_id),
              tooltipclose: () => {
                setActiveTooltipStreamId((id) =>
                  id === cam.stream_id ? null : id,
                );
              },
            }}
          >
            <Tooltip direction="top" offset={[0, -8]}>
              <div className="hotspot-tooltip">
                <b>{cam.name}</b>
                <div className="hotspot-tooltip__row">
                  Count: <AnimatedNumber value={cam.count} className="mono" />
                  <span style={{ marginLeft: 8 }}>
                    Severity:{" "}
                    <span style={{ color }}>{cam.sev.toUpperCase()}</span>
                  </span>
                </div>
                <TooltipVideo
                  whepUrl={
                    activeTooltipStreamId === cam.stream_id
                      ? cam.webrtcWhepUrl
                      : undefined
                  }
                  width={cam.tooltipWidth}
                  aspectRatio={cam.tooltipAspectRatio}
                />
              </div>
            </Tooltip>
          </CircleMarker>
        );
      })}
    </>
  );
});


interface AlertItem {
  id: number;
  text: string;
  sev: string;
  time: string;
  exiting?: boolean;
}

const MAX_VISIBLE_ALERTS = 4;
const ALERT_STAGGER_MS = 1200;

/**
 * AlertStream - displays alerts one-at-a-time with 1.2s gap (Bug 311).
 * Queues incoming alerts and renders them sequentially.
 */
interface AlertStreamProps {
  cityLabel?: string;
  cityLocationNames?: Set<string>;
}

const AlertStream = memo(function AlertStream({
  cityLabel,
  cityLocationNames,
}: AlertStreamProps) {
  const [displayAlerts, setDisplayAlerts] = useState<AlertItem[]>([]);
  const idRef = useRef(0);
  const seenAlertIdsRef = useRef<Set<string>>(new Set());
  const pendingQueueRef = useRef<AlertItem[]>([]);
  const isProcessingRef = useRef(false);
  const processingTimerRef = useRef<number | null>(null);

  // Clear display and seen-set when city changes
  const prevCityRef = useRef(cityLabel);
  useEffect(() => {
    if (prevCityRef.current !== cityLabel) {
      prevCityRef.current = cityLabel;
      seenAlertIdsRef.current = new Set();
      pendingQueueRef.current = [];
      setDisplayAlerts([]);
    }
  }, [cityLabel]);

  const { alerts: liveAlerts } = useAlerts({
    severity: "CRITICAL",
    city: cityLabel,
    locationNames: cityLocationNames,
    limit: 10,
    autoRefreshMs: 15_000,
  });

  // Add a single alert to the display with exit animation for overflow
  const pushAlert = useCallback((alert: AlertItem) => {
    setDisplayAlerts((prev) => {
      const next = [alert, ...prev];
      if (next.length > MAX_VISIBLE_ALERTS) {
        // Mark the oldest for exit, then remove after animation
        const withExit = next.map((a, i) =>
          i >= MAX_VISIBLE_ALERTS ? { ...a, exiting: true } : a
        );
        setTimeout(() => {
          setDisplayAlerts((p) => p.slice(0, MAX_VISIBLE_ALERTS));
        }, 350);
        return withExit.slice(0, MAX_VISIBLE_ALERTS + 1);
      }
      return next;
    });
  }, []);

  // Process pending queue one item at a time
  const processQueue = useCallback(() => {
    if (pendingQueueRef.current.length === 0) {
      isProcessingRef.current = false;
      return;
    }

    isProcessingRef.current = true;
    const nextAlert = pendingQueueRef.current.shift();
    if (nextAlert) {
      pushAlert(nextAlert);
    }

    // Schedule next item
    processingTimerRef.current = window.setTimeout(() => {
      processQueue();
    }, ALERT_STAGGER_MS);
  }, [pushAlert]);

  // Enqueue new alerts from live data
  useEffect(() => {
    if (liveAlerts.length === 0) return;

    // Find alerts we haven't seen yet
    const newAlerts: AlertItem[] = [];
    for (const alert of liveAlerts) {
      if (!seenAlertIdsRef.current.has(alert.alert_id)) {
        seenAlertIdsRef.current.add(alert.alert_id);
        newAlerts.push({
          id: ++idRef.current,
          text: alert.description || `Alert at ${alert.location_name}`,
          sev: alert.severity === "CRITICAL" ? "critical" : "safe",
          time: new Date(alert.timestamp).toLocaleTimeString("en-US", {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          }),
        });
      }
    }

    // Add new alerts to pending queue (newest first)
    if (newAlerts.length > 0) {
      pendingQueueRef.current.push(...newAlerts.reverse());
      // Limit pending queue size
      if (pendingQueueRef.current.length > 10) {
        pendingQueueRef.current = pendingQueueRef.current.slice(-10);
      }
      // Start processing if not already
      if (!isProcessingRef.current) {
        processQueue();
      }
    }
  }, [liveAlerts, processQueue]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (processingTimerRef.current) {
        clearTimeout(processingTimerRef.current);
      }
    };
  }, []);

  return (
    <div className="alert-stream">
      {displayAlerts.map((a) => (
        <div
          key={a.id}
          className={`alert-toast alert-toast--${a.sev} ${a.exiting ? "alert-toast--exit" : ""}`}
        >
          <div className="alert-toast__body">
            <span className="alert-toast__text">{a.text}</span>
            <span className="alert-toast__time mono">{a.time}</span>
          </div>
        </div>
      ))}
    </div>
  );
});

export default function MapPanel() {
  const mode = useAppSelector(selectMode);
  const timeRange = useAppSelector(selectTimeRange);
  const selectedMainLocationId = useAppSelector(selectMainLocationId);
  const dispatch = useAppDispatch();
  const { config } = useHotspotConfig();
  const locations = config.main_locations;
  const selectedIdx = Math.max(
    0,
    locations.findIndex((l) => l.id === selectedMainLocationId)
  );
  const loc = locations[selectedIdx] ?? {
    center: config.cities[0]?.center ?? [30.2672, -97.7431],
    zoom: config.cities[0]?.zoom ?? 13,
  };
  const selectedCityLabel = locations[selectedIdx]?.label;
  const { mapData: cityMapData, loading: mapLoading } = useMapData(10_000, selectedCityLabel);
  const cityStreamCount = cityMapData?.cameras.length ?? 0;
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const dropdownId = useId();

  useEffect(() => {
    if (!dropdownOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [dropdownOpen]);

  useEffect(() => {
    if (locations.length === 0) return;
    const exists = locations.some((location) =>
      location.id === selectedMainLocationId
    );
    if (!exists) {
      dispatch(setSelectedMainLocationId(locations[0].id));
    }
  }, [dispatch, locations, selectedMainLocationId]);


  return (
    <div className="map-panel">
      <div className="map-inner">
        <MapContainer
          center={loc.center}
          zoom={loc.zoom}
          zoomControl={false}
          attributionControl={false}
          style={{ width: "100%", height: "100%" }}
        >
          <TileLayer
            attribution={GOOGLE_ATTRIBUTION}
            url={GOOGLE_TILES}
            subdomains="0123"
            maxZoom={20}
          />
          <MapInvalidator />
          <FlyTo center={loc.center} zoom={loc.zoom} />
          <HotspotMarkers />
        </MapContainer>
      </div>

      <div
        ref={dropdownRef}
        className={`map-city-select${dropdownOpen ? " map-city-select--open" : ""}`}
        role="combobox"
        aria-expanded={dropdownOpen}
        aria-haspopup="listbox"
        aria-controls={dropdownId}
        onClick={() => setDropdownOpen((o) => !o)}
      >
        <Place sx={{ fontSize: 13, color: "var(--text-muted)" }} />
        <span className="map-city-select__value">
          {locations[selectedIdx]?.label ?? "Select city"}
        </span>
        <span className={`map-city-select__chevron${dropdownOpen ? " map-city-select__chevron--open" : ""}`}>
          ▾
        </span>
        {dropdownOpen && (
          <ul
            id={dropdownId}
            className="map-city-select__menu"
            role="listbox"
          >
            {locations.map((l, i) => (
              <li
                key={l.id}
                role="option"
                aria-selected={i === selectedIdx}
                className={`map-city-select__option${i === selectedIdx ? " map-city-select__option--active" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  dispatch(setSelectedMainLocationId(l.id));
                  setDropdownOpen(false);
                }}
              >
                {l.label}
              </li>
            ))}
          </ul>
        )}
      </div>
      {mapLoading && (
        <div className="map-loading-overlay">
          <span className="map-loading-spinner" />
        </div>
      )}

      <div className="map-info-overlay">
        <span className="info-chip">
          {mode === "plan"
            ? `Zone Analysis - ${RANGE_LABELS[timeRange]}`
            : `Live City Map - ${cityStreamCount} Stream${cityStreamCount !== 1 ? "s" : ""}`}
        </span>
      </div>
      {mode === "ops" && (
        <AlertStream
          cityLabel={selectedCityLabel}
          cityLocationNames={
            cityMapData
              ? new Set(cityMapData.cameras.map((c) => c.location_name))
              : undefined
          }
        />
      )}
    </div>
  );
}
