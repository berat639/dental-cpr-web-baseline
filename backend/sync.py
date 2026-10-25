"""Real-time synchronisation primitives for the local service.

The viewer (browser) and the service manager (Electron) both subscribe to one
Server-Sent Events stream. The service itself is the single source of truth:
every dataset/job/lifecycle change is published here, from request handlers on
the event loop *and* from the AI worker thread.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass

HEARTBEAT_SECONDS = 2.0
PROGRESS_MIN_INTERVAL = 0.25  # at most 4 progress events per job per second


def sse(event: str, data: object) -> str:
    """Encode one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[str]
    role: str


class EventBus:
    """Thread-safe fan-out of SSE frames to every connected client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_id = 0
        self._last_progress: dict[str, float] = {}
        self.last_viewer_seen = time.time()

    def subscribe(self, role: str) -> tuple[int, asyncio.Queue[str]]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._next_id += 1
            subscriber_id = self._next_id
            self._subscribers[subscriber_id] = _Subscriber(asyncio.get_running_loop(), queue, role)
            if role == "viewer":
                self.last_viewer_seen = time.time()
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: int) -> None:
        with self._lock:
            subscriber = self._subscribers.pop(subscriber_id, None)
            if subscriber and subscriber.role == "viewer":
                self.last_viewer_seen = time.time()

    def counts(self) -> dict[str, int]:
        with self._lock:
            roles = [subscriber.role for subscriber in self._subscribers.values()]
        return {"viewers": roles.count("viewer"), "managers": roles.count("manager")}

    def publish(self, event: str, data: object, *, roles: set[str] | None = None, throttle_key: str | None = None) -> bool:
        """Queue an event for subscribers. Returns False if it was throttled."""
        now = time.monotonic()
        with self._lock:
            if throttle_key is not None:
                if now - self._last_progress.get(throttle_key, 0.0) < PROGRESS_MIN_INTERVAL:
                    return False
                self._last_progress[throttle_key] = now
            targets = [s for s in self._subscribers.values() if roles is None or s.role in roles]
        frame = sse(event, data)
        for subscriber in targets:
            try:
                subscriber.loop.call_soon_threadsafe(self._put, subscriber.queue, frame)
            except RuntimeError:  # loop already closed during shutdown
                pass
        return True

    @staticmethod
    def _put(queue: asyncio.Queue[str], frame: str) -> None:
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            # A stalled client must never block the service; it resynchronises
            # from the `hello` snapshot when it reconnects.
            pass


class FolderPickBroker:
    """Routes a viewer's native folder-picker request to the Electron manager."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[dict[str, object]]] = {}

    def create(self, request_id: str) -> asyncio.Future[dict[str, object]]:
        future: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        return future

    def resolve(self, request_id: str, result: dict[str, object]) -> bool:
        future = self._pending.get(request_id)
        if not future or future.done():
            return False
        future.set_result(result)
        return True

    def discard(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
