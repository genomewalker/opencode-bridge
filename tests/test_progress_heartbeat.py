"""Regression test for the MCP tool-call idle-timeout fix.

Long fusion/conductor_fusion/room_run calls (multi-round xhigh panels) can run
past the client's MCP tool idle timeout with zero output, so call_tool now
sends a progress notification every 25s when the client opted in via a
progressToken. These tests exercise that heartbeat in isolation, without
waiting on real wall-clock time.
"""

import asyncio
from types import SimpleNamespace

import chitta_bridge.server as srv


class _FakeMeta:
    def __init__(self, token):
        self.progressToken = token


class _FakeCtx:
    def __init__(self, token, session):
        self.meta = _FakeMeta(token) if token is not None else None
        self.session = session


def _patch_ctx(monkeypatch, token, session=None):
    ctx = _FakeCtx(token, session)
    monkeypatch.setattr(type(srv.server), "request_context", property(lambda self: ctx))


def test_heartbeat_sends_progress_when_token_present(monkeypatch):
    calls = []
    fired = asyncio.Event()

    async def fake_send_progress(token, progress, total=None, message=None, related_request_id=None):
        calls.append((token, progress, message))
        fired.set()

    session = SimpleNamespace(send_progress_notification=fake_send_progress)
    _patch_ctx(monkeypatch, "tok-123", session)

    async def fake_sleep(_secs):
        await asyncio.sleep(0)

    monkeypatch.setattr(srv, "asyncio", SimpleNamespace(
        sleep=fake_sleep, create_task=asyncio.create_task, CancelledError=asyncio.CancelledError,
    ))

    async def slow_handler(_args):
        await fired.wait()
        return [srv.TextContent(type="text", text="done")]

    srv.REGISTRY["__test_slow__"] = SimpleNamespace(handler=slow_handler, hidden=True)
    try:
        result = asyncio.run(srv.call_tool("__test_slow__", {}))
    finally:
        del srv.REGISTRY["__test_slow__"]

    assert result[0].text == "done"
    assert calls, "expected at least one progress notification while the handler was still running"
    assert calls[0][0] == "tok-123"
    assert "still running" in calls[0][2]


def test_no_heartbeat_without_progress_token(monkeypatch):
    _patch_ctx(monkeypatch, None)

    async def fast_handler(_args):
        return [srv.TextContent(type="text", text="done")]

    srv.REGISTRY["__test_fast__"] = SimpleNamespace(handler=fast_handler, hidden=True)
    try:
        result = asyncio.run(srv.call_tool("__test_fast__", {}))
    finally:
        del srv.REGISTRY["__test_fast__"]

    assert result[0].text == "done"


def test_heartbeat_task_cancelled_after_completion(monkeypatch):
    session = SimpleNamespace(send_progress_notification=lambda *a, **kw: asyncio.sleep(0))
    _patch_ctx(monkeypatch, "tok-456", session)

    async def fake_sleep(_secs):
        await asyncio.sleep(3600)  # would hang the test (past the wait_for below) if not cancelled

    monkeypatch.setattr(srv, "asyncio", SimpleNamespace(
        sleep=fake_sleep, create_task=asyncio.create_task, CancelledError=asyncio.CancelledError,
    ))

    async def fast_handler(_args):
        return [srv.TextContent(type="text", text="done")]

    srv.REGISTRY["__test_fast2__"] = SimpleNamespace(handler=fast_handler, hidden=True)
    try:
        result = asyncio.run(asyncio.wait_for(srv.call_tool("__test_fast2__", {}), timeout=2))
    finally:
        del srv.REGISTRY["__test_fast2__"]

    assert result[0].text == "done"
