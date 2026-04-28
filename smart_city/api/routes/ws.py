# Created by Metrum AI for AMD

"""WebSocket connection endpoints."""
import logging
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

_VALID_CHANNELS = {"alerts", "counts"}


@router.websocket("/ws/{channel}")
async def ws_channel(websocket: WebSocket, channel: str) -> None:
    """Accept WebSocket connections for alerts or counts channels.

    Args:
        websocket: The incoming WebSocket connection.
        channel: Named channel: 'alerts' | 'counts'.
    """
    if channel not in _VALID_CHANNELS:
        await websocket.close(code=4004, reason="Unknown channel.")
        return

    hub = getattr(websocket.app.state, "ws_hub", None)
    if hub is None:
        await websocket.close(code=1011, reason="Hub unavailable.")
        return

    await websocket.accept()
    await hub.subscribe(websocket, channel)
    logger.debug("WS client connected: channel=%s", channel)
    try:
        while True:
            await websocket.receive_text()  # keep alive / ignore pings
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(websocket, channel)
        logger.debug("WS client disconnected: channel=%s", channel)


@router.websocket("/ws/stream/{stream_id}")
async def ws_stream(websocket: WebSocket, stream_id: int) -> None:
    """Accept WebSocket connections for per-stream video frames.

    Args:
        websocket: The incoming WebSocket connection.
        stream_id: Numeric stream identifier.
    """
    hub = getattr(websocket.app.state, "ws_hub", None)
    if hub is None:
        await websocket.close(code=1011, reason="Hub unavailable.")
        return

    channel = f"stream:{stream_id}"
    await websocket.accept()
    await hub.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(websocket, channel)


@router.websocket("/ws/heatmap/{stream_id}")
async def ws_heatmap(websocket: WebSocket, stream_id: int) -> None:
    """Accept WebSocket connections for per-stream heatmap frames.

    Args:
        websocket: The incoming WebSocket connection.
        stream_id: Numeric stream identifier.
    """
    hub = getattr(websocket.app.state, "ws_hub", None)
    if hub is None:
        await websocket.close(code=1011, reason="Hub unavailable.")
        return

    channel = f"heatmap:{stream_id}"
    await websocket.accept()
    await hub.subscribe(websocket, channel)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(websocket, channel)


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    """Accept WebSocket connections for GPU/system telemetry updates.

    Args:
        websocket: The incoming WebSocket connection.
    """
    hub = getattr(websocket.app.state, "ws_hub", None)
    if hub is None:
        await websocket.close(code=1011, reason="Hub unavailable.")
        return

    await websocket.accept()
    await hub.subscribe(websocket, "telemetry")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(websocket, "telemetry")
