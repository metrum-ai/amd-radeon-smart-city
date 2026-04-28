# Created by Metrum AI for AMD

"""WebSocket hub: manages all real-time channel subscriptions."""

import asyncio
import base64
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from smart_city.observability.metrics import WEBSOCKET_CONNECTIONS

logger = logging.getLogger(__name__)

_FRAME_MIN_INTERVAL_S = 0.1  # 10 FPS max
_PING_INTERVAL_S = 30.0


class WebSocketHub:
    """Manages WebSocket subscriptions across multiple named channels.

    All broadcast methods serialise messages to JSON and handle dead
    connections gracefully (remove on disconnect/send error).
    """

    def __init__(self) -> None:
        """Initialise with empty subscriber registry."""
        self._channels: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._last_frame_time: Dict[int, float] = {}

    async def subscribe(self, ws: WebSocket, channel: str) -> None:
        """Register a WebSocket to a named channel.

        Args:
            ws: The WebSocket connection.
            channel: Channel name (e.g. "alerts", "counts").
        """
        self._channels[channel].add(ws)
        WEBSOCKET_CONNECTIONS.labels(channel=channel).set(
            len(self._channels[channel])
        )
        logger.debug("WS subscribed to '%s'. Total: %d",
                     channel, len(self._channels[channel]))

    async def unsubscribe(self, ws: WebSocket, channel: str) -> None:
        """Remove a WebSocket from a channel.

        Args:
            ws: The WebSocket to remove.
            channel: Channel name.
        """
        self._channels[channel].discard(ws)
        remaining = len(self._channels.get(channel, set()))
        WEBSOCKET_CONNECTIONS.labels(channel=channel).set(remaining)
        if not self._channels[channel]:
            del self._channels[channel]

    async def broadcast(self, channel: str, message: Dict[str, Any]) -> None:
        """Serialise and send a message to all channel subscribers.

        Dead connections are silently removed.

        Args:
            channel: Target channel name.
            message: Dict to serialise as JSON.
        """
        dead: Set[WebSocket] = set()
        for ws in list(self._channels.get(channel, [])):
            try:
                await ws.send_text(json.dumps(message, default=str))
            except WebSocketDisconnect:
                dead.add(ws)
            except (RuntimeError, OSError, ValueError, TypeError) as exc:
                # RuntimeError: send on already-closed transport
                # OSError: underlying socket gone
                # ValueError/TypeError: message could not be serialised
                logger.debug(
                    "WS send error on channel '%s': %s",
                    channel, exc, exc_info=True,
                )
                dead.add(ws)
        for ws in dead:
            self._channels[channel].discard(ws)

    async def broadcast_alert(self, alert) -> None:
        """Broadcast a CrowdAlert to the 'alerts' channel.

        Args:
            alert: CrowdAlert dataclass instance.
        """
        await self.broadcast(
            "alerts",
            {
                "type": "alert",
                "data": {
                    "alert_id": alert.alert_id,
                    "timestamp": alert.timestamp.isoformat(),
                    "stream_id": alert.stream_id,
                    "zone_id": alert.zone_id,
                    "zone_name": alert.zone_name,
                    "zone_type": alert.zone_type,
                    "location_name": alert.location_name,
                    "lat": alert.lat,
                    "lon": alert.lon,
                    "person_count": alert.person_count,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "violation_type": alert.violation_type,
                    "description": alert.description,
                    "is_auto_popup": alert.is_auto_popup,
                    "status": alert.status,
                },
            },
        )

    async def broadcast_count_update(
        self,
        stream_id: int,
        zone_id: str,
        zone_name: str,
        location_name: str,
        lat: float,
        lon: float,
        count: int,
        severity: str,
    ) -> None:
        """Broadcast a crowd count update to the 'counts' channel.

        Args:
            stream_id: Source stream identifier.
            zone_id: Zone identifier.
            zone_name: Zone display name.
            location_name: Location display name.
            lat: Latitude.
            lon: Longitude.
            count: Current person count.
            severity: Current severity string.
        """
        from datetime import datetime, timezone

        await self.broadcast(
            "counts",
            {
                "type": "count_update",
                "data": {
                    "stream_id": stream_id,
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "location_name": location_name,
                    "lat": lat,
                    "lon": lon,
                    "person_count": count,
                    "severity": severity,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
            },
        )

    async def broadcast_heatmap(
        self, stream_id: int, heatmap_b64: str
    ) -> None:
        """Broadcast a heatmap frame to the stream-specific channel.

        Args:
            stream_id: Source stream identifier.
            heatmap_b64: Base64-encoded JPEG heatmap string.
        """
        from datetime import datetime, timezone

        channel = f"heatmap:{stream_id}"
        if channel not in self._channels:
            return
        await self.broadcast(
            channel,
            {
                "type": "heatmap",
                "stream_id": stream_id,
                "image_b64": heatmap_b64,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    async def publish_stream_frame(
        self,
        stream_id: int,
        frame_rgb: np.ndarray,
        heatmap: Optional[np.ndarray],
    ) -> None:
        """Encode a frame as JPEG and broadcast to the stream channel.

        Rate-limited to 10 FPS. Skips if no subscribers.

        Args:
            stream_id: Source stream identifier.
            frame_rgb: RGB numpy frame (H, W, 3).
            heatmap: Optional heatmap to overlay.
        """
        channel = f"stream:{stream_id}"
        if channel not in self._channels or not self._channels[channel]:
            return

        now = time.time()
        if now - self._last_frame_time.get(stream_id, 0) < _FRAME_MIN_INTERVAL_S:
            return
        self._last_frame_time[stream_id] = now

        if heatmap is not None:
            if heatmap.shape[:2] != frame_rgb.shape[:2]:
                heatmap = cv2.resize(
                    heatmap, (frame_rgb.shape[1], frame_rgb.shape[0])
                )
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(bgr, 0.6, heatmap, 0.4, 0)
        else:
            blended = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        from datetime import datetime, timezone

        await self.broadcast(
            channel,
            {
                "type": "frame",
                "stream_id": stream_id,
                "image_b64": b64,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    async def ping_all(self) -> None:
        """Send a ping to all active WebSocket connections."""
        for channel in list(self._channels.keys()):
            await self.broadcast(channel, {"type": "ping"})
