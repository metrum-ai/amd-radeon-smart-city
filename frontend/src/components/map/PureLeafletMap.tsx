// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

import {
  useContext,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { flushSync } from "react-dom";
import { createRoot, type Root } from "react-dom/client";
import L, {
  type CircleMarker as LeafletCircleMarker,
  type CircleMarkerOptions,
  type Direction,
  type LatLngExpression,
  type LeafletEventHandlerFn,
  type Map as LeafletMapInstance,
  type PointExpression,
} from "leaflet";
import {
  LeafletMapContext,
  LeafletMarkerContext,
  useLeafletMap,
} from "./LeafletMapContext";

interface PureMapContainerProps {
  attributionControl: boolean;
  center: LatLngExpression;
  children: ReactNode;
  style: CSSProperties;
  zoom: number;
  zoomControl: boolean;
}

export function PureMapContainer({
  attributionControl,
  center,
  children,
  style,
  zoom,
  zoomControl,
}: PureMapContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const initialViewRef = useRef({ center, zoom });
  const [map, setMap] = useState<LeafletMapInstance | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const instance = L.map(container, {
      attributionControl,
      center: initialViewRef.current.center,
      zoom: initialViewRef.current.zoom,
      zoomControl,
    });
    setMap(instance);

    return () => {
      setMap(null);
      instance.remove();
    };
  }, [attributionControl, zoomControl]);

  return (
    <div ref={containerRef} style={style}>
      <LeafletMapContext.Provider value={map}>
        {map ? children : null}
      </LeafletMapContext.Provider>
    </div>
  );
}

interface PureTileLayerProps {
  attribution: string;
  maxZoom: number;
  subdomains: string | string[];
  url: string;
}

export function PureTileLayer({
  attribution,
  maxZoom,
  subdomains,
  url,
}: PureTileLayerProps) {
  const map = useLeafletMap();

  useEffect(() => {
    if (!map) return;
    const layer = L.tileLayer(url, {
      attribution,
      maxZoom,
      subdomains,
    });
    layer.addTo(map);
    return () => {
      layer.remove();
    };
  }, [attribution, map, maxZoom, subdomains, url]);

  return null;
}

interface PureCircleMarkerProps {
  center: LatLngExpression;
  children?: ReactNode;
  eventHandlers?: Record<string, LeafletEventHandlerFn>;
  pathOptions?: CircleMarkerOptions;
  radius: number;
}

export function PureCircleMarker({
  center,
  children,
  eventHandlers,
  pathOptions,
  radius,
}: PureCircleMarkerProps) {
  const map = useLeafletMap();
  const initialMarkerRef = useRef({ center, radius });
  const markerRef = useRef<LeafletCircleMarker | null>(null);
  const [marker, setMarker] = useState<LeafletCircleMarker | null>(null);

  useEffect(() => {
    if (!map) return;
    const instance = L.circleMarker(initialMarkerRef.current.center, {
      radius: initialMarkerRef.current.radius,
    });
    instance.addTo(map);
    markerRef.current = instance;
    setMarker(instance);

    return () => {
      setMarker(null);
      markerRef.current = null;
      instance.remove();
    };
  }, [map]);

  useEffect(() => {
    const instance = markerRef.current;
    if (!instance) return;
    instance.setLatLng(center);
  }, [center]);

  useEffect(() => {
    const instance = markerRef.current;
    if (!instance) return;
    instance.setRadius(radius);
  }, [radius]);

  useEffect(() => {
    const instance = markerRef.current;
    if (!instance || !pathOptions) return;
    instance.setStyle(pathOptions);
  }, [pathOptions]);

  useEffect(() => {
    const instance = markerRef.current;
    if (!instance || !eventHandlers) return;

    for (const [eventName, handler] of Object.entries(eventHandlers)) {
      instance.on(eventName, handler);
    }
    return () => {
      for (const [eventName, handler] of Object.entries(eventHandlers)) {
        instance.off(eventName, handler);
      }
    };
  }, [eventHandlers, marker]);

  return (
    <LeafletMarkerContext.Provider value={marker}>
      {marker ? children : null}
    </LeafletMarkerContext.Provider>
  );
}

interface PureTooltipProps {
  children: ReactNode;
  direction: Direction;
  offset: PointExpression;
}

export function PureTooltip({ children, direction, offset }: PureTooltipProps) {
  const marker = useContext(LeafletMarkerContext);
  const childrenRef = useRef(children);
  const rootRef = useRef<Root | null>(null);

  useEffect(() => {
    if (!marker) return;

    const container = document.createElement("div");
    const root = createRoot(container);
    const resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => marker.getTooltip()?.update());
    rootRef.current = root;
    flushSync(() => {
      root.render(childrenRef.current);
    });
    resizeObserver?.observe(container);
    marker.bindTooltip(container, {
      direction,
      offset,
    });
    marker.getTooltip()?.update();

    return () => {
      resizeObserver?.disconnect();
      marker.unbindTooltip();
      root.unmount();
      rootRef.current = null;
    };
  }, [direction, marker, offset]);

  useEffect(() => {
    childrenRef.current = children;
    if (!rootRef.current) return;
    flushSync(() => {
      rootRef.current?.render(children);
    });
    marker?.getTooltip()?.update();
  }, [children, marker]);

  return null;
}
