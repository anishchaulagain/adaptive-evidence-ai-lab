"""Server-Sent Event encoding helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.streaming.events import StreamEvent

SSE_MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable proxy buffering, or progress stalls
}


def encode_event(event: StreamEvent) -> str:
    """Render an event as an SSE frame."""
    raise NotImplementedError


async def event_stream(events: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
    """Encode an async event source into an SSE byte stream."""
    raise NotImplementedError
