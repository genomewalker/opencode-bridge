"""Persistent Codex **app-server** transport (experimental, feature-flagged).

This is a *prototype* second transport for the Codex backend. Instead of
spawning ``codex exec`` per task (the default, in ``codex.py``), it drives the
persistent Codex **daemon** and speaks its JSON-RPC 2.0 protocol.

Transport (verified end-to-end): spawn the **standalone** codex binary
(``$CODEX_HOME/packages/standalone/current/codex``) as ``app-server --stdio`` and
speak **newline-delimited** JSON-RPC on its stdio. Confirmed:
``initialize`` -> ``initialized`` -> ``thread/start`` -> ``turn/start`` ->
``item/agentMessage/delta`` / ``item/completed`` -> ``turn/completed`` returns the
reply. Two prerequisites that were the real blockers: (1) the **standalone**
install must exist (the npm launcher's app-server does not self-serve), and (2) the
codex state DB must be healthy (a malformed DB triggers an interactive "Press Enter"
recovery that hangs a non-interactive server). The ``daemon``/``proxy`` subcommands
are a separate managed-daemon path and are NOT used here.

It is wired into ``CodexBridge.run_task`` behind ``CHITTA_CODEX_APP_SERVER=1``
for exactly one path. Any failure here must fall back to the exec path — this
module never raises out of ``run_turn``/``start_thread`` for a protocol issue;
it raises :class:`CodexAppServerError`, which the caller catches and downgrades.

Protocol (verified against codex-cli 0.144.0 schema
``ClientRequest.json`` / ``ServerNotification.json`` / ``ServerRequest.json``):

* handshake: request ``initialize`` -> await result -> notification
  ``initialized``.
* ``thread/start`` {model, cwd, sandbox, approvalPolicy} -> result.thread.id.
* ``turn/start`` {threadId, input:[{type:"text",text}], effort} -> ack; the
  reply arrives as notifications.
* notifications: ``item/agentMessage/delta`` (accumulate ``params.delta``),
  ``item/completed`` (``params.item.type == "agentMessage"`` carries the final
  ``text``), ``turn/completed`` ends the turn.
* approvals are *server->client requests* (they carry an ``id``):
  ``execCommandApproval`` / ``applyPatchApproval`` want ``{"decision":"approved"}``;
  ``item/commandExecution/requestApproval`` / ``item/fileChange/requestApproval``
  want ``{"decision":"accept"}``. Auto-accepted here (full-auto).

VERIFIED end-to-end through ``run_task`` (flag on): returns the reply via the
app-server in ~17s vs ~120-270s for exec on this box — the persistent process
avoids per-turn startup. Still off by default (``CHITTA_CODEX_APP_SERVER``) and
falls back to exec on any failure (missing standalone install, unhealthy DB, etc.).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


class CodexAppServerError(RuntimeError):
    """Any app-server transport failure. The caller falls back to exec."""


def _codex_pids() -> list:
    """PIDs of live codex processes (by exe basename, not cmdline — a cmdline
    match also matches the matcher)."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.path.basename(os.readlink(entry / "exe")) == "codex":
                pids.append(int(entry.name))
        except OSError:
            continue
    return pids


def clear_stale_shm(env: Optional[dict] = None) -> list:
    """Drop leftover SQLite ``-shm`` files in CODEX_HOME when no codex is alive.

    A killed codex (node hop, SIGKILL) leaves its ``-shm`` behind. On an NFS home
    the stale file makes every later open fail with SQLITE_PROTOCOL ("locking
    protocol"), so the app-server dies during ``initialize`` — and because that
    death is itself unclean it leaves a fresh stale ``-shm``, which makes the
    breakage permanent until someone clears it by hand.

    Only ``-shm`` is removed: it is a pure lock/index cache SQLite rebuilds from
    the ``-wal``. The ``-wal`` holds committed data and is never touched.
    """
    if _codex_pids():
        return []
    home = Path((env or os.environ).get("CODEX_HOME") or os.path.expanduser("~/.codex"))
    cleared = []
    for shm in home.glob("*.sqlite-shm"):
        try:
            shm.unlink()
            cleared.append(shm.name)
        except OSError:
            continue
    return cleared


def standalone_codex(env: Optional[dict] = None) -> Optional[str]:
    """Path to the STANDALONE codex binary that serves ``app-server --stdio``,
    or None if it isn't installed. The npm launcher's app-server does not
    self-serve, so without this the app-server path can't work — callers use
    this to skip it FAST (no spawn/timeout) and go straight to exec.
    """
    home = (env or os.environ).get("CODEX_HOME") or os.path.expanduser("~/.codex")
    sa = Path(home) / "packages" / "standalone" / "current" / "codex"
    return str(sa) if sa.exists() else None


# Server->client approval requests and the decision payload that accepts them.
# Shapes verified against *ApprovalResponse.json in the 0.144.0 schema.
_APPROVE_APPROVED = frozenset({"execCommandApproval", "applyPatchApproval"})
_APPROVE_ACCEPT = frozenset({
    "item/commandExecution/requestApproval",
    "item/fileChange/requestApproval",
})


def _auto_approval_result(method: str) -> dict:
    if method in _APPROVE_APPROVED:
        return {"decision": "approved"}
    if method in _APPROVE_ACCEPT:
        return {"decision": "accept"}
    # Unknown approval kind: best-effort accept so the turn keeps moving.
    return {"decision": "approved"}


class CodexAppServer:
    """Owns one persistent ``codex app-server`` process and its JSON-RPC loop."""

    def __init__(
        self,
        codex_bin: str,
        env: Optional[dict] = None,
        client_name: str = "chitta-bridge",
        client_version: str = "0.29",
        init_timeout: float = 8.0,  # app-server is experimental; fail fast to the exec fallback
    ) -> None:
        self._bin = codex_bin
        self._env = env
        self._client_name = client_name
        self._client_version = client_version
        self._init_timeout = init_timeout

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._pending: dict[int, asyncio.Future] = {}
        # turn state, keyed by threadId currently running a turn
        self._turns: dict[str, "_TurnState"] = {}
        self._next_id = 0
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def ensure_started(self) -> None:
        """Start (or respawn) the app-server and complete the handshake."""
        async with self._start_lock:
            if self._started and self._alive():
                return
            await self._spawn()

    def _server_bin(self) -> str:
        """Path to the binary that actually serves ``app-server --stdio``.

        Must be the STANDALONE codex binary — the npm launcher's ``app-server``
        does not self-serve (it delegates to the standalone install). Prefer
        ``$CODEX_HOME/packages/standalone/current/codex``; fall back to the
        configured bin if the standalone install isn't present (caller then
        falls back to exec).
        """
        return standalone_codex(self._env) or self._bin

    async def _spawn(self) -> None:
        # Tear down any dead remnants first.
        await self._teardown_locked()
        cleared = clear_stale_shm(self._env)
        if cleared:
            print(f"[codex app-server] cleared stale sqlite locks: {', '.join(cleared)}",
                  file=sys.stderr, flush=True)
        try:
            # `app-server --stdio` on the standalone binary: newline-delimited
            # JSON-RPC on stdio, verified end-to-end (initialize -> thread/start
            # -> turn/start -> turn/completed). Requires the standalone install
            # and a healthy state DB; on either problem the spawn/initialize
            # fails and the caller falls back to exec.
            self._proc = await asyncio.create_subprocess_exec(
                self._server_bin(), "app-server", "--stdio",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
                start_new_session=True,
                limit=2 ** 24,
            )
        except OSError as e:
            raise CodexAppServerError(f"failed to spawn app-server: {e}") from e

        self._pending.clear()
        self._turns.clear()
        self._reader_task = asyncio.ensure_future(self._read_loop())
        self._stderr_task = asyncio.ensure_future(self._drain_stderr())

        try:
            await self._request(
                "initialize",
                {"clientInfo": {"name": self._client_name,
                                "version": self._client_version}},
                timeout=self._init_timeout,
            )
        except Exception as e:
            await self._teardown_locked()
            raise CodexAppServerError(f"initialize failed: {e}") from e

        await self._notify("initialized", {})
        self._started = True

    async def _teardown_locked(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(CodexAppServerError("app-server shut down"))
        self._pending.clear()
        for ts in self._turns.values():
            if not ts.done.done():
                ts.done.set_exception(CodexAppServerError("app-server shut down"))
        self._turns.clear()
        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task
        self._reader_task = None
        self._stderr_task = None
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                self._proc.terminate()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            if self._proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    self._proc.kill()
        self._proc = None
        self._started = False

    async def aclose(self) -> None:
        async with self._start_lock:
            await self._teardown_locked()

    # -- wire I/O -----------------------------------------------------------

    async def _write(self, obj: dict) -> None:
        if not self._alive():
            raise CodexAppServerError("app-server not running")
        data = (json.dumps(obj) + "\n").encode()
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

    async def _notify(self, method: str, params: Any) -> None:
        await self._write({"method": method, "params": params})

    async def _request(self, method: str, params: Any, timeout: float) -> Any:
        rid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._write({"method": method, "id": rid, "params": params})
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise CodexAppServerError(
                f"{method} timed out after {timeout}s") from e
        finally:
            self._pending.pop(rid, None)

    async def _drain_stderr(self) -> None:
        try:
            while self._proc and self._proc.stderr:
                line = await self._proc.stderr.readline()
                if not line:
                    break
        except (asyncio.CancelledError, Exception):
            return

    async def _read_loop(self) -> None:
        proc = self._proc
        try:
            while proc and proc.stdout:
                try:
                    line = await proc.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    # Oversized frame; skip to next newline.
                    continue
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            return
        except Exception:
            return
        finally:
            # Reader exited => process died. Fail everything in flight.
            err = CodexAppServerError("app-server read loop ended (process died)")
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(err)
            for ts in list(self._turns.values()):
                if not ts.done.done():
                    ts.done.set_exception(err)

    def _dispatch(self, msg: dict) -> None:
        mid = msg.get("id")
        method = msg.get("method")

        # Server -> client request (has both id and method): approval etc.
        if method is not None and mid is not None:
            result = _auto_approval_result(method)
            asyncio.ensure_future(
                self._write({"id": mid, "result": result}))
            return

        # Response to one of our requests.
        if method is None and mid is not None:
            fut = self._pending.get(mid)
            if fut and not fut.done():
                if "error" in msg:
                    fut.set_exception(
                        CodexAppServerError(f"rpc error: {msg['error']}"))
                else:
                    fut.set_result(msg.get("result"))
            return

        # Notification.
        if method is not None:
            self._on_notification(method, msg.get("params") or {})

    def _on_notification(self, method: str, params: dict) -> None:
        if method == "item/agentMessage/delta":
            ts = self._turns.get(params.get("threadId"))
            if ts is not None:
                ts.deltas.append(params.get("delta", ""))
            return
        if method == "item/completed":
            item = params.get("item") or {}
            if item.get("type") == "agentMessage":
                ts = self._turns.get(params.get("threadId"))
                if ts is not None:
                    ts.final_message = item.get("text", "") or ts.final_message
            return
        if method == "turn/completed":
            ts = self._turns.get(params.get("threadId"))
            if ts is not None and not ts.done.done():
                ts.done.set_result(None)
            return
        if method == "error":
            # Fail any running turn so the caller can fall back.
            msg = params.get("message") or str(params)
            for ts in self._turns.values():
                if not ts.done.done():
                    ts.done.set_exception(CodexAppServerError(f"server error: {msg}"))

    # -- high-level API -----------------------------------------------------

    async def start_thread(
        self,
        model: Optional[str],
        cwd: str,
        sandbox: str = "danger-full-access",
        approval_policy: str = "never",
        timeout: float = 30.0,
    ) -> str:
        await self.ensure_started()
        params: dict = {"cwd": cwd, "sandbox": sandbox,
                        "approvalPolicy": approval_policy}
        if model:
            params["model"] = model
        result = await self._request("thread/start", params, timeout=timeout)
        try:
            return result["thread"]["id"]
        except (KeyError, TypeError) as e:
            raise CodexAppServerError(
                f"thread/start: unexpected result {result!r}") from e

    async def run_turn(
        self,
        thread_id: str,
        text: str,
        effort: Optional[str] = None,
        timeout: float = 300.0,
    ) -> str:
        if not self._alive():
            raise CodexAppServerError("app-server not running")
        ts = _TurnState()
        self._turns[thread_id] = ts
        try:
            params: dict = {
                "threadId": thread_id,
                "input": [{"type": "text", "text": text}],
            }
            if effort:
                params["effort"] = effort
            # turn/start returns an ack; the reply arrives via notifications.
            await self._request("turn/start", params, timeout=timeout)
            await asyncio.wait_for(ts.done, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise CodexAppServerError(
                f"turn did not complete within {timeout}s") from e
        finally:
            self._turns.pop(thread_id, None)
        reply = ts.final_message or "".join(ts.deltas)
        if not reply:
            raise CodexAppServerError("turn completed with no agent message")
        return reply

    async def run_once(
        self,
        text: str,
        cwd: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        sandbox: str = "danger-full-access",
        approval_policy: str = "never",
        timeout: float = 300.0,
    ) -> str:
        """Convenience: fresh thread + one turn -> reply text."""
        thread_id = await self.start_thread(
            model=model, cwd=cwd, sandbox=sandbox,
            approval_policy=approval_policy, timeout=min(timeout, 60.0))
        return await self.run_turn(
            thread_id, text, effort=effort, timeout=timeout)


class _TurnState:
    __slots__ = ("deltas", "final_message", "done")

    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.final_message: str = ""
        self.done: asyncio.Future = asyncio.get_event_loop().create_future()


# ---------------------------------------------------------------------------
# Module-level singleton (one app-server per bridge process).
# ---------------------------------------------------------------------------

_SINGLETON: Optional[CodexAppServer] = None
_SINGLETON_LOCK: Optional[asyncio.Lock] = None


def _singleton_lock() -> asyncio.Lock:
    global _SINGLETON_LOCK
    if _SINGLETON_LOCK is None:
        _SINGLETON_LOCK = asyncio.Lock()
    return _SINGLETON_LOCK


async def get_singleton(codex_bin: str, env: Optional[dict]) -> CodexAppServer:
    global _SINGLETON
    async with _singleton_lock():
        if _SINGLETON is None or not _SINGLETON._alive():
            if _SINGLETON is not None:
                with contextlib.suppress(Exception):
                    await _SINGLETON.aclose()
            _SINGLETON = CodexAppServer(codex_bin=codex_bin, env=env)
        return _SINGLETON
