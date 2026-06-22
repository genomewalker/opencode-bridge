"""Local LLM backend — Ollama / vLLM via OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import glob as _glob
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["GpuNodeDiscovery", "LocalSession", "LocalModelBridge"]

import re as _re
from pathlib import Path as _Path

_SAFE_ID_RE = _re.compile(r"^[a-zA-Z0-9_\-]+$")


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal."""
    if _Path(session_id).name != session_id:
        raise ValueError("Invalid session ID: path separators not allowed")
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session ID: must be alphanumeric, hyphens, underscores only")
    return session_id

# Default port for Ollama / vLLM (OpenAI-compatible)
_LOCAL_LLM_PORT = 11434

# URL cache files written by slurm-serve-ollama.sh.
_DEFAULT_URL_DIR = str(Path.home() / ".chitta-bridge" / "endpoints")
_OLLAMA_URL_GLOB = (
    os.environ.get("CHITTA_BRIDGE_URL_DIR", _DEFAULT_URL_DIR)
    + "/ollama-server-*.url"
)


class GpuNodeDiscovery:
    """Discover GPU nodes reachable via Slurm or direct hostname and probe for Ollama/vLLM."""

    _ENV_NODES_VAR = "CHITTA_BRIDGE_GPU_NODES"

    # Dead-host cooldown — class-level state shared across all callers
    _dead_hosts: dict = {}     # url → expiry monotonic time
    _host_failures: dict = {}  # url → consecutive failure count
    _FAIL_THRESHOLD = 2
    _COOLDOWN_SECS = 20

    @classmethod
    def _is_cooled_down(cls, url: str) -> bool:
        import time
        return time.monotonic() < cls._dead_hosts.get(url, 0)

    @classmethod
    def _record_failure(cls, url: str) -> None:
        import time
        cls._host_failures[url] = cls._host_failures.get(url, 0) + 1
        if cls._host_failures[url] >= cls._FAIL_THRESHOLD:
            cls._dead_hosts[url] = time.monotonic() + cls._COOLDOWN_SECS
            cls._host_failures[url] = 0

    @classmethod
    def _record_success(cls, url: str) -> None:
        cls._dead_hosts.pop(url, None)
        cls._host_failures.pop(url, None)

    @staticmethod
    def _probe_ollama(base_url: str, timeout: int = 4) -> Optional[list[str]]:
        """Return list of available model names at base_url, or None if unreachable."""
        tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        try:
            req = urllib.request.urlopen(tags_url, timeout=timeout)
            data = json.loads(req.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            GpuNodeDiscovery._record_success(base_url)
            return models
        except Exception:
            GpuNodeDiscovery._record_failure(base_url)
            return None

    @classmethod
    def _tailscale_peers(cls) -> list[str]:
        """Return Ollama/vLLM base_urls from Tailscale peers. Silent on any error."""
        if not shutil.which("tailscale"):
            return []
        try:
            import subprocess
            import json as _json
            out = subprocess.check_output(
                ["tailscale", "status", "--json"],
                timeout=6, stderr=subprocess.DEVNULL, text=True,
            )
            data = _json.loads(out)
            urls = []
            for peer in data.get("Peer", {}).values():
                ips = peer.get("TailscaleIPs") or []
                if not ips:
                    continue
                ip = ips[0]
                for port in (11434, 8000):  # Ollama, vLLM
                    urls.append(f"http://{ip}:{port}/v1")
            return urls
        except Exception:
            return []

    @classmethod
    def _cached_urls(cls) -> list[tuple[str, str]]:
        """Return list of (model_hint, base_url) from /tmp/ollama-server-*.url files."""
        results = []
        for path in _glob.glob(_OLLAMA_URL_GLOB):
            try:
                url = Path(path).read_text().strip()
                if url:
                    hint = Path(path).stem.removeprefix("ollama-server-")
                    results.append((hint, url))
            except OSError:
                pass
        return results

    @classmethod
    def _slurm_gpu_nodes(cls) -> list[str]:
        """Return hostnames of running Slurm jobs that allocated GPU resources."""
        if not shutil.which("squeue"):
            return []
        try:
            import subprocess
            out = subprocess.check_output(
                ["squeue", "--format=%T %N %b", "--noheader", "--me"],
                timeout=8,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            nodes = []
            for line in out.strip().splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "RUNNING" and "gpu" in parts[2].lower():
                    nodes.append(parts[1])
            return nodes
        except Exception:
            return []

    @classmethod
    def _env_nodes(cls) -> list[str]:
        """Return nodes from the CHITTA_BRIDGE_GPU_NODES env variable."""
        val = os.environ.get(cls._ENV_NODES_VAR, "")
        return [n.strip() for n in val.split(",") if n.strip()]

    @classmethod
    def discover(cls) -> list[dict]:
        """
        Return a list of reachable LLM endpoints:
          [{"base_url": "http://node:11434/v1", "node": "nodename", "models": [...], "source": "..."}]
        """
        seen: dict[str, dict] = {}  # base_url -> entry

        # 1. Cached URL files (highest priority — already health-checked at launch time)
        for hint, base_url in cls._cached_urls():
            models = cls._probe_ollama(base_url)
            if models is not None:
                node = base_url.split("//")[-1].split(":")[0]
                seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "cached"}

        # 2. Slurm running GPU jobs (my own jobs that allocated a GPU)
        for node in cls._slurm_gpu_nodes():
            base_url = f"http://{node}:{_LOCAL_LLM_PORT}/v1"
            if base_url not in seen:
                models = cls._probe_ollama(base_url)
                if models is not None:
                    seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "slurm"}

        # 3. Env-configured nodes
        for node in cls._env_nodes():
            base_url = f"http://{node}:{_LOCAL_LLM_PORT}/v1"
            if base_url not in seen:
                models = cls._probe_ollama(base_url)
                if models is not None:
                    seen[base_url] = {"base_url": base_url, "node": node, "models": models, "source": "env"}

        # 4. Tailscale peers
        for ts_url in cls._tailscale_peers():
            if ts_url not in seen and not cls._is_cooled_down(ts_url):
                models = cls._probe_ollama(ts_url)
                if models is not None:
                    node = ts_url.split("//")[-1].split(":")[0]
                    seen[ts_url] = {"base_url": ts_url, "node": node, "models": models, "source": "tailscale"}

        # 5. Localhost fallback
        local_url = f"http://localhost:{_LOCAL_LLM_PORT}/v1"
        if local_url not in seen:
            models = cls._probe_ollama(local_url)
            if models is not None:
                seen[local_url] = {"base_url": local_url, "node": "localhost", "models": models, "source": "local"}

        return list(seen.values())


@dataclass
class LocalSession:
    id: str
    endpoint: str          # e.g. http://node:11434/v1
    model: str
    messages: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    attached_claude_sessions: list = field(default_factory=list)


class LocalModelBridge:
    """Chat with local LLMs (Ollama/vLLM) running on GPU nodes via OpenAI-compatible API."""

    def __init__(self):
        self.sessions: dict[str, LocalSession] = {}
        self._active_id: Optional[str] = None
        self._start_time = datetime.now()
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _session_lock(self, sid: str) -> asyncio.Lock:
        lock = self._session_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[sid] = lock
        return lock

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, model: str, endpoint: str) -> str:
        _sanitize_session_id(session_id)
        if session_id in self.sessions:
            return f"Session '{session_id}' already exists."
        s = LocalSession(id=session_id, endpoint=endpoint.rstrip("/"), model=model)
        self.sessions[session_id] = s
        self._active_id = session_id
        return f"Started local session '{session_id}' → {endpoint} model={model}"

    def _active(self) -> Optional[LocalSession]:
        if self._active_id and self._active_id in self.sessions:
            return self.sessions[self._active_id]
        return None

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        self._active_id = session_id
        s = self.sessions[session_id]
        return f"Switched to local session '{session_id}' ({s.model} @ {s.endpoint})"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "No session to end."
        del self.sessions[sid]
        if self._active_id == sid:
            self._active_id = next(iter(self.sessions), None)
        return f"Ended local session '{sid}'."

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No local model sessions."
        lines = []
        for sid, s in self.sessions.items():
            marker = " [active]" if sid == self._active_id else ""
            lines.append(f"  {sid}{marker} — {s.model} @ {s.endpoint} ({len(s.messages)} messages)")
        return "\n".join(lines)

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "No session found."
        s = self.sessions[sid]
        msgs = s.messages[-last_n:]
        if not msgs:
            return "No messages yet."
        return "\n".join(f"[{m['role']}]: {m['content'][:300]}" for m in msgs)

    def get_config(self) -> str:
        s = self._active()
        if not s:
            return "No active local session."
        return f"Session: {s.id}\nEndpoint: {s.endpoint}\nModel: {s.model}\nMessages: {len(s.messages)}"

    def health_check(self) -> dict:
        uptime = int((datetime.now() - self._start_time).total_seconds())
        return {"status": "ok", "sessions": len(self.sessions), "uptime": uptime}

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        sid = session_id or self._active_id
        if not sid or sid not in self.sessions:
            return "Error: no active local session."
        async with self._session_lock(sid):
            s = self.sessions[sid]
            s.messages.append({"role": "user", "content": message})

            payload: dict = {
                "model": s.model,
                "messages": s.messages.copy(),
                "stream": False,
            }
            if system_prompt:
                payload["messages"] = [{"role": "system", "content": system_prompt}] + payload["messages"]

            try:
                reply = await asyncio.get_event_loop().run_in_executor(
                    None, self._post_completion, s.endpoint, payload
                )
            except Exception as e:
                s.messages.pop()  # roll back user message on error
                return f"Error calling local model: {e}"

            s.messages.append({"role": "assistant", "content": reply})
            return reply

    @staticmethod
    def _post_completion(endpoint: str, payload: dict, timeout: int = 300) -> str:
        """POST to /v1/chat/completions with retries for model-loading connection drops."""
        import http.client
        url = f"{endpoint}/chat/completions"
        data = json.dumps(payload).encode()
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(4):
            if attempt:
                import time
                time.sleep(10 * attempt)  # 10s, 20s, 30s back-off while model loads
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                # Retry on 500/502/503 (model loading, GPU contention)
                if e.code in (500, 502, 503):
                    last_exc = e
                    continue
                raise
            except (http.client.RemoteDisconnected, ConnectionResetError, urllib.error.URLError) as e:
                last_exc = e
                continue
        GpuNodeDiscovery._record_failure(endpoint)
        raise last_exc

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    @staticmethod
    def list_models_at(endpoint: str, timeout: int = 8) -> list[str]:
        tags_url = endpoint.rstrip("/").removesuffix("/v1") + "/api/tags"
        try:
            req = urllib.request.urlopen(tags_url, timeout=timeout)
            data = json.loads(req.read().decode())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []
