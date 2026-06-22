"""Codex backend: session management and task execution via the Codex CLI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal as _signal
import threading as _threading
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["CodexSession", "CodexJob", "CodexBridge"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CODEX_SANDBOX = "danger-full-access"
PERSISTED_SCHEMA_VERSION = 1

_STARTUP_WARNING_PREFIXES = (
    "WARNING: failed to clean up stale",
)

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


# ---------------------------------------------------------------------------
# Helpers (private to this module — duplicated from server.py to keep the
# module self-contained; server.py still owns the originals)
# ---------------------------------------------------------------------------

def _sanitize_session_id(session_id: str) -> str:
    if Path(session_id).name != session_id:
        raise ValueError("Invalid session ID: path separators not allowed")
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session ID: must be alphanumeric, hyphens, underscores only")
    return session_id


_path_write_locks: dict[str, _threading.Lock] = {}
_path_write_locks_mu = _threading.Lock()


def _path_write_lock(path: Path) -> _threading.Lock:
    try:
        key = os.path.realpath(str(path))
    except OSError:
        key = os.path.abspath(str(path))
    with _path_write_locks_mu:
        lock = _path_write_locks.get(key)
        if lock is None:
            lock = _threading.Lock()
            _path_write_locks[key] = lock
    return lock


_HARDENED_ATOMIC_WRITE_OK = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)


def _atomic_write_text_legacy(path: Path, content: str, encoding: str = "utf-8") -> None:
    import tempfile
    parent = path.parent
    fd, tmp = tempfile.mkstemp(dir=str(parent), prefix=".tmp.")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if not _HARDENED_ATOMIC_WRITE_OK:
        _atomic_write_text_legacy(path, content, encoding)
        return

    dir_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        dirfd = os.open(str(parent), dir_flags)
    except OSError as e:
        raise OSError(f"refused to write {path}: parent dir unsafe ({e})") from e

    try:
        target_name = path.name
        try:
            st = os.stat(target_name, dir_fd=dirfd, follow_symlinks=False)
        except FileNotFoundError:
            st = None
        if st is not None:
            import stat as _stat_mod
            if _stat_mod.S_ISLNK(st.st_mode) or st.st_nlink > 1:
                raise PermissionError(
                    f"refused to write {path}: target is a symlink or hardlink"
                )
        import tempfile
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=".tmp.")
        tmp_base = os.path.basename(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding=encoding) as fh:
                fh.write(content)
            os.rename(tmp_base, target_name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
        except Exception:
            try:
                os.unlink(tmp_base, dir_fd=dirfd)
            except OSError:
                pass
            raise
    finally:
        os.close(dirfd)


def _sync_kill_group(proc) -> None:
    if proc is None or getattr(proc, "returncode", None) is not None:
        return
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, _signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


_SECRET_EXACT = frozenset({
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY", "OPENAI_ORG_ID",
    "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_OAUTH",
    "HUGGINGFACE_TOKEN", "HF_TOKEN",
    "NPM_TOKEN", "PYPI_TOKEN", "TWINE_PASSWORD",
    "SLACK_TOKEN", "DISCORD_TOKEN",
    "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY",
    "DIGITALOCEAN_TOKEN",
    "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "DOCKER_PASSWORD", "DOCKER_AUTH_TOKEN",
})
_SECRET_PREFIXES = ("AWS_", "AZURE_", "GCP_")
_SECRET_SUFFIXES = (
    "_TOKEN", "_API_KEY", "_APIKEY", "_SECRET", "_PASSWORD",
    "_PASSWD", "_CREDENTIALS", "_CREDS", "_PRIVATE_KEY",
    "_ACCESS_KEY", "_ACCESS_TOKEN", "_AUTH", "_AUTH_TOKEN",
    "_SESSION_TOKEN",
)
_LLM_KEEP_RE = re.compile(
    r"^(ANTHROPIC_|OPENAI_|OPENROUTER_|GROQ_|GEMINI_|GOOGLE_|MISTRAL_|DEEPSEEK_"
    r"|XAI_|TOGETHER_|FIREWORKS_|CEREBRAS_|OLLAMA_|CHITTA_)|_API_KEY$"
)


def _scrub_env(env: Optional[dict] = None) -> dict:
    src = env if env is not None else os.environ
    out = {}
    for k, v in src.items():
        ku = k.upper()
        if ku in _SECRET_EXACT:
            continue
        if any(ku.startswith(p) for p in _SECRET_PREFIXES):
            continue
        if any(ku.endswith(s) for s in _SECRET_SUFFIXES):
            continue
        out[k] = v
    return out


def _llm_env() -> dict:
    out = _scrub_env(os.environ)
    for k, v in os.environ.items():
        if k not in out and _LLM_KEEP_RE.search(k.upper()):
            out[k] = v
    return out


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _chitta_sql(query: str, timeout: int = 5) -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["chitta", "sql_query", "--query", query],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def _get_ppid_chain() -> list[int]:
    pids = []
    pid = os.getpid()
    for _ in range(15):
        pids.append(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    pid = int(line.split()[1])
                    break
            else:
                break
        except OSError:
            break
        if pid <= 1:
            break
    return pids


def _get_claude_session_id() -> Optional[str]:
    pids = _get_ppid_chain()
    pid_list = ",".join(str(p) for p in pids)
    output = _chitta_sql(
        f"SELECT session_id FROM session_registry WHERE pid IN ({pid_list}) AND status='active' ORDER BY last_heartbeat DESC LIMIT 1"
    )
    if output:
        for line in output.splitlines():
            candidate = line.strip().strip("|").strip()
            if _UUID_RE.match(candidate):
                return candidate
    return os.environ.get("CLAUDE_SESSION_ID")


def _chitta_session_alive(claude_session_id: str) -> Optional[bool]:
    if not _UUID_RE.match(claude_session_id):
        return None
    output = _chitta_sql(
        f"SELECT COUNT(*) FROM session_registry WHERE session_id='{claude_session_id}' AND status='active'"
    )
    if output is None:
        return None
    for line in output.splitlines():
        candidate = line.strip().strip("|").strip()
        if candidate.isdigit():
            return int(candidate) > 0
    return None


def _strip_startup_warnings(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.startswith(_STARTUP_WARNING_PREFIXES)]
    return "\n".join(lines).strip()


def _migrate_persisted(data: dict, kind: str) -> dict:
    version = data.get("schema_version", 0)
    if version > PERSISTED_SCHEMA_VERSION:
        return data
    data["schema_version"] = PERSISTED_SCHEMA_VERSION
    if kind == "room" and "retry_counts" in data:
        old = data["retry_counts"]
        migrated: dict = {}
        for k, v in old.items():
            if k.startswith("r") and ":" in k:
                try:
                    int(k[1:k.index(":")])
                    name = k[k.index(":") + 1:]
                    migrated[name] = max(migrated.get(name, 0), v)
                    continue
                except ValueError:
                    pass
            migrated[k] = max(migrated.get(k, 0), v)
        data["retry_counts"] = migrated
    return data


def find_codex() -> Optional[Path]:
    import shutil
    paths = [
        Path.home() / ".codex" / "bin" / "codex",
        Path("/usr/local/bin/codex"),
        Path("/usr/bin/codex"),
    ]
    for p in paths:
        if p.exists():
            return p
    which = shutil.which("codex")
    if which:
        return Path(which)
    return None


CODEX_BIN = find_codex()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Config:
    model: str = "openai/gpt-5.3-codex"
    agent: str = "plan"
    variant: str = "medium"
    codex_model: str = DEFAULT_CODEX_MODEL
    codex_sandbox: str = DEFAULT_CODEX_SANDBOX

    @classmethod
    def load(cls) -> "Config":
        config = cls()
        config_path = Path.home() / ".chitta-bridge" / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                config.model = data.get("model", config.model)
                config.agent = data.get("agent", config.agent)
                config.variant = data.get("variant", config.variant)
                config.codex_model = data.get("codex_model", config.codex_model)
                config.codex_sandbox = data.get("codex_sandbox", config.codex_sandbox)
            except Exception:
                pass
        config.model = os.environ.get("OPENCODE_MODEL", config.model)
        config.agent = os.environ.get("OPENCODE_AGENT", config.agent)
        config.variant = os.environ.get("OPENCODE_VARIANT") or config.variant
        config.codex_model = os.environ.get("CODEX_MODEL", config.codex_model)
        config.codex_sandbox = os.environ.get("CODEX_SANDBOX", config.codex_sandbox)
        return config

    def save(self) -> None:
        config_dir = Path.home() / ".chitta-bridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        data = {
            "model": self.model,
            "agent": self.agent,
            "variant": self.variant,
            "codex_model": self.codex_model,
            "codex_sandbox": self.codex_sandbox,
        }
        with _path_write_lock(config_path):
            _atomic_write_text(config_path, json.dumps(data, indent=2))


@dataclass
class CodexSession:
    """Session for Codex backend."""
    id: str
    model: str
    sandbox: str = DEFAULT_CODEX_SANDBOX
    full_auto: bool = True
    codex_session_id: Optional[str] = None
    working_dir: Optional[str] = None
    claude_session_ids: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    schema_version: int = PERSISTED_SCHEMA_VERSION

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def save(self, path: Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "id": self.id,
            "model": self.model,
            "sandbox": self.sandbox,
            "full_auto": self.full_auto,
            "codex_session_id": self.codex_session_id,
            "working_dir": self.working_dir,
            "claude_session_ids": self.claude_session_ids,
            "created": self.created,
            "messages": [asdict(m) for m in self.messages],
        }
        with _path_write_lock(path):
            _atomic_write_text(path, json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CodexSession":
        data = _migrate_persisted(json.loads(path.read_text()), "codex_session")
        session = cls(
            id=data["id"],
            model=data["model"],
            sandbox=data.get("sandbox", DEFAULT_CODEX_SANDBOX),
            full_auto=data.get("full_auto", True),
            codex_session_id=data.get("codex_session_id"),
            working_dir=data.get("working_dir"),
            claude_session_ids=data.get("claude_session_ids", []),
            created=data.get("created", datetime.now().isoformat()),
            schema_version=data.get("schema_version", PERSISTED_SCHEMA_VERSION),
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


@dataclass
class CodexJob:
    """Background Codex task with persistent status tracking."""
    id: str
    task: str
    model: str
    working_dir: str
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"  # running | completed | failed | cancelled
    effort: Optional[str] = None
    sandbox: Optional[str] = None
    resume_from: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None
    result: Optional[str] = None
    codex_session_id: Optional[str] = None
    schema_version: int = PERSISTED_SCHEMA_VERSION

    def save(self, path: Path) -> None:
        with _path_write_lock(path):
            _atomic_write_text(path, json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CodexJob":
        data = _migrate_persisted(json.loads(path.read_text()), "codex_job")
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class CodexBridge:
    """Bridge for Codex CLI interactions with session management."""

    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.config = Config.load()
        self.sessions: dict[str, CodexSession] = {}
        self.active_session: Optional[str] = None
        self.sessions_dir = Path.home() / ".chitta-bridge" / "codex-sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._job_locks: dict[str, asyncio.Lock] = {}
        self._load_sessions()
        self.jobs: dict[str, CodexJob] = {}
        self._job_tasks: dict[str, "asyncio.Task"] = {}
        self.jobs_dir = Path.home() / ".chitta-bridge" / "codex-jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def _session_lock(self, sid: str) -> asyncio.Lock:
        lock = self._session_locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[sid] = lock
        return lock

    def _job_lock(self, jid: str) -> asyncio.Lock:
        lock = self._job_locks.get(jid)
        if lock is None:
            lock = asyncio.Lock()
            self._job_locks[jid] = lock
        return lock

    def _load_sessions(self) -> None:
        for path in self.sessions_dir.glob("*.json"):
            try:
                session = CodexSession.load(path)
                self.sessions[session.id] = session
            except Exception:
                pass

    def _load_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = CodexJob.load(path)
                if job.status == "running":
                    job.status = "failed"
                    job.result = "Server restarted while job was running"
                    job.finished = datetime.now().isoformat()
                    job.save(path)
                self.jobs[job.id] = job
            except Exception:
                pass

    async def _run_codex(
        self, *args, timeout: int = 120, stall_timeout: int = 120, cwd: Optional[str] = None
    ) -> tuple[str, int]:
        """Run codex CLI command with streaming stdout and stall detection."""
        if not CODEX_BIN:
            return "Codex not installed. Install from: https://github.com/openai/codex", 1

        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
                env=_llm_env(),
            )
            proc.stdin.close()

            stderr_task = asyncio.ensure_future(proc.stderr.read())

            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + timeout
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Timed out after {timeout}s", 1
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=read_timeout,
                    )
                except asyncio.TimeoutError:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s", 1
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            out = "".join(stdout_parts).strip()
            if proc.returncode == 0:
                err = _strip_startup_warnings(stderr_raw.decode(errors="replace")).strip()
            else:
                err = stderr_raw.decode(errors="replace").strip()
            output = out if out else err
            return output, proc.returncode or 0
        except asyncio.TimeoutError:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return "Command timed out", 1
        except asyncio.CancelledError:
            if proc:
                _sync_kill_group(proc)
                try:
                    await proc.wait()
                except Exception:
                    pass
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            raise
        except Exception as e:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}", 1

    async def _run_codex_exec_stdin(
        self,
        args: list,
        stdin_data: str,
        cwd: str,
        timeout: int = 300,
        stall_timeout: int = 180,
    ) -> tuple[str, int]:
        """Run a codex exec command with stdin data; returns (raw_output, returncode)."""
        if not CODEX_BIN:
            return "Codex not installed.", 1
        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=True,
                limit=2**20,
                env=_llm_env(),
            )
            proc.stdin.write(stdin_data.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            stderr_task = asyncio.ensure_future(proc.stderr.read())
            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + timeout
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Timed out after {timeout}s", 1
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=read_timeout)
                except asyncio.TimeoutError:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s", 1
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            out = "".join(stdout_parts)
            if proc.returncode != 0:
                err = stderr_raw.decode(errors="replace").strip()
                return err or out, proc.returncode or 1
            return out, 0
        except asyncio.CancelledError:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            raise
        except Exception as e:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}", 1

    @staticmethod
    def _parse_codex_jsonl(output: str) -> tuple[str, Optional[str]]:
        """Extract reply text and thread_id from Codex JSONL output.

        Handles both plain JSONL and JSON Text Sequences (RFC 7464) where each
        record is prefixed by the RS byte (\\x1e). The \\x1e is stripped before
        JSON parsing — silently swallowing it was causing empty replies.
        """
        reply_parts = []
        thread_id = None
        for line in output.split("\n"):
            line = line.lstrip("\x1e").strip()
            if not line or line.startswith("WARNING:") or line.startswith("ERROR:"):
                continue
            try:
                event = json.loads(line)
                if not thread_id and event.get("thread_id"):
                    thread_id = event["thread_id"]
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            reply_parts.append(text)
            except json.JSONDecodeError:
                continue
        return "\n".join(reply_parts), thread_id

    async def _run_rescue_background(self, job_id: str) -> None:
        """Coroutine that runs a rescue job in the background and updates its state."""
        job = self.jobs[job_id]
        job.started = datetime.now().isoformat()
        job.save(self.jobs_dir / f"{job_id}.json")

        model, effort = self._apply_codex_policy(job.model, job.effort)
        args = ["exec"]
        if job.resume_from:
            args.extend(["resume", job.resume_from])
        if model:
            args.extend(["--model", model])
        args.extend(["-c", f'model_reasoning_effort="{effort}"'])
        if job.sandbox:
            args.extend(["--sandbox", job.sandbox])
        args.extend(["--full-auto", "--json", "-"])

        try:
            output, code = await self._run_codex_exec_stdin(
                args, job.task, job.working_dir, timeout=1800, stall_timeout=120
            )
            reply, thread_id = self._parse_codex_jsonl(output)
            job.status = "completed" if code == 0 else "failed"
            job.result = reply or output or "(no output)"
            job.codex_session_id = thread_id
            job.finished = datetime.now().isoformat()
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.result = "(cancelled)"
            job.finished = datetime.now().isoformat()
        finally:
            job.save(self.jobs_dir / f"{job_id}.json")
            self._job_tasks.pop(job_id, None)

    async def start_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        sandbox: Optional[str] = None,
        full_auto: bool = True,
        working_dir: Optional[str] = None,
    ) -> str:
        session_id = _sanitize_session_id(session_id)
        model = model or self.config.codex_model
        sandbox = sandbox or self.config.codex_sandbox

        if not CODEX_BIN:
            return "Codex not installed — session not started. Install from: https://github.com/openai/codex"

        claude_session_id = await asyncio.to_thread(_get_claude_session_id)

        session = CodexSession(
            id=session_id,
            model=model,
            sandbox=sandbox,
            full_auto=full_auto,
            working_dir=working_dir or os.getcwd(),
            claude_session_ids=[claude_session_id] if claude_session_id else [],
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        result = f"Codex session '{session_id}' started\n  Model: {model}\n  Sandbox: {sandbox}"
        if full_auto:
            result += "\n  Mode: full-auto"
        if working_dir:
            result += f"\n  Working dir: {working_dir}"
        if claude_session_id:
            result += f"\n  Claude session: {claude_session_id}"
        return result

    def get_config(self) -> str:
        return f"""Codex configuration:
  Model: {self.config.codex_model}
  Sandbox: {self.config.codex_sandbox}

Set via:
  - ~/.chitta-bridge/config.json
  - CODEX_MODEL, CODEX_SANDBOX env vars
  - codex_configure tool"""

    def set_config(self, model: Optional[str] = None, sandbox: Optional[str] = None) -> str:
        changes = []
        if model:
            self.config.codex_model = model
            changes.append(f"model: {model}")
        if sandbox:
            self.config.codex_sandbox = sandbox
            changes.append(f"sandbox: {sandbox}")
        if changes:
            self.config.save()
            return "Codex configuration updated:\n  " + "\n  ".join(changes)
        return "No changes made."

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session. Use codex_start first."

        async with self._session_lock(sid):
            return await self._send_message_locked(sid, message, images)

    async def _send_message_locked(
        self, sid: str, message: str, images: Optional[list[str]] = None,
    ) -> str:
        session = self.sessions[sid]
        session.add_message("user", message)

        if session.codex_session_id:
            args = ["exec", "--skip-git-repo-check", "resume", session.codex_session_id]
        else:
            args = ["exec", "--skip-git-repo-check"]

        if session.model:
            args.extend(["--model", session.model])

        if session.sandbox == "danger-full-access":
            if not session.codex_session_id:
                args.append("--dangerously-bypass-approvals-and-sandbox")
        elif session.full_auto:
            args.append("--full-auto")
        elif not session.codex_session_id:
            args.extend(["--sandbox", session.sandbox])

        if images:
            for img in images:
                args.extend(["--image", img])

        args.append("--json")
        args.append("-")

        proc = None
        stderr_task = None
        try:
            proc = await asyncio.create_subprocess_exec(
                str(CODEX_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=session.working_dir,
                start_new_session=True,
            )
            proc.stdin.write(message.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            stderr_task = asyncio.ensure_future(proc.stderr.read())

            stdout_parts: list[str] = []
            deadline = asyncio.get_event_loop().time() + 300
            stall_timeout = 180
            first_line = True

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return "Timed out after 300s"
                read_timeout = remaining if first_line else min(stall_timeout, remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=read_timeout,
                    )
                except asyncio.TimeoutError:
                    _sync_kill_group(proc)
                    await proc.wait()
                    stderr_task.cancel()
                    return f"Model stalled — no output for {stall_timeout}s"
                if not line:
                    break
                stdout_parts.append(line.decode(errors="replace"))
                first_line = False

            try:
                stderr_raw = await asyncio.wait_for(stderr_task, timeout=5)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                stderr_raw = b""
            await proc.wait()

            output = "".join(stdout_parts)
            if proc.returncode != 0:
                err = stderr_raw.decode(errors="replace").strip()
                return f"Error: {err or output}"
        except asyncio.TimeoutError:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return "Command timed out"
        except asyncio.CancelledError:
            if proc:
                _sync_kill_group(proc)
                try:
                    await proc.wait()
                except Exception:
                    pass
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            raise
        except Exception as e:
            if proc:
                _sync_kill_group(proc)
                await proc.wait()
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            return f"Error: {e}"

        reply_parts = []
        for line in output.split("\n"):
            if not line or line.startswith("WARNING:"):
                continue
            try:
                event = json.loads(line)
                if not session.codex_session_id and event.get("thread_id"):
                    session.codex_session_id = event["thread_id"]
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            reply_parts.append(text)
            except json.JSONDecodeError:
                continue

        reply = "\n".join(reply_parts)
        if reply:
            session.add_message("assistant", reply)
            session.save(self.sessions_dir / f"{sid}.json")

        return reply or "No response received"

    @staticmethod
    def _cleanup_stale_arg0_dirs() -> None:
        """Remove stale codex-arg0* temp dirs that have no .lock file."""
        arg0_dir = Path.home() / ".codex" / "tmp" / "arg0"
        if not arg0_dir.is_dir():
            return
        for d in arg0_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("codex-arg0"):
                continue
            if not (d / ".lock").exists():
                try:
                    import shutil as _shutil
                    _shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass

    async def run_task(
        self,
        task: str,
        working_dir: Optional[str] = None,
        model: Optional[str] = None,
        full_auto: bool = True,
        effort: Optional[str] = None,
        sandbox: Optional[str] = None,
    ) -> str:
        """Run a one-off task without session management."""
        args = self._build_exec_args(model, effort, sandbox=sandbox, full_auto=full_auto)
        cwd = working_dir or os.getcwd()
        output, code = await self._run_codex_exec_stdin(args, task, cwd)
        self._cleanup_stale_arg0_dirs()
        if code != 0:
            return f"Error: {output}"
        reply, thread_id = self._parse_codex_jsonl(output)
        result = reply or output or "No response received"
        if thread_id:
            result += f"\n\n(Codex session: {thread_id} — resume with: codex resume {thread_id})"
        return result

    async def review_code(
        self,
        working_dir: Optional[str] = None,
        model: Optional[str] = None,
        mode: str = "normal",
        focus: Optional[str] = None,
        base: Optional[str] = None,
        effort: Optional[str] = None,
        background: bool = False,
        sandbox: Optional[str] = None,
    ) -> str:
        """Run Codex code review. mode='adversarial' pressure-tests design decisions."""
        model = model or self.config.codex_model
        cwd = working_dir or os.getcwd()

        if mode == "adversarial":
            focus_clause = f"\n\nSpecific focus area: {focus}" if focus else ""
            task = (
                "You are a senior adversarial code reviewer. Your job is NOT to find obvious bugs — "
                "it is to challenge the design decisions, architecture, and tradeoffs in this code.\n\n"
                "Review the uncommitted changes (or the full repo if no changes) and:\n"
                "1. Question whether the chosen approach was the right one at all\n"
                "2. Identify hidden assumptions that could break under load or edge cases\n"
                "3. Pressure-test failure modes: what happens when X fails, Y is slow, Z is empty?\n"
                "4. Challenge the architecture: would a different design be safer/simpler?\n"
                "5. Flag race conditions, data loss risks, rollback gaps, reliability holes\n"
                "6. Propose at least one alternative approach and explain the tradeoff"
                f"{focus_clause}\n\n"
                "Be direct and hard to satisfy. Do not praise good code — focus exclusively on risks."
            )
            if base:
                task += f"\n\nReview changes relative to base: {base}"
            if background:
                return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, sandbox=sandbox)
            output, code = await self._run_codex_exec_stdin(
                self._build_exec_args(model, effort, sandbox=sandbox), task, cwd, timeout=600
            )
            if code != 0:
                return f"Error: {output}"
            reply, thread_id = self._parse_codex_jsonl(output)
            result = reply or output or "Review complete"
            if thread_id:
                result += f"\n\n(Codex session: {thread_id} — resume with: codex resume {thread_id})"
            return result
        else:
            model, effort = self._apply_codex_policy(model, effort)
            args = ["exec", "review", "--model", model, "--json"]
            if base:
                args.extend(["--base", base])
            args.extend(["-c", f'model_reasoning_effort="{effort}"'])
            if sandbox:
                args.extend(["--sandbox", sandbox])
            if background:
                task = f"Run a code review{f' vs {base}' if base else ''}"
                return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, sandbox=sandbox)
            output, code = await self._run_codex(*args, cwd=cwd, timeout=600)
            if code != 0:
                return f"Error: {output}"
            return output or "Review complete"

    CODEX_EFFORTS = ("low", "medium", "high", "xhigh")

    @staticmethod
    def _apply_codex_policy(model: Optional[str], effort: Optional[str]) -> tuple:
        """Enforce defaults: reject Fast variants unless CODEX_ALLOW_FAST=1; default effort=high; validate effort enum."""
        if model and "fast" in model.lower() and os.environ.get("CODEX_ALLOW_FAST") != "1":
            raise ValueError(
                f"Refusing to use Fast Codex variant '{model}' implicitly. "
                "Set CODEX_ALLOW_FAST=1 to override, or pick a non-fast model."
            )
        effort = effort or "high"
        if effort not in CodexBridge.CODEX_EFFORTS:
            raise ValueError(
                f"Invalid Codex effort '{effort}'. Must be one of: {', '.join(CodexBridge.CODEX_EFFORTS)}."
            )
        return model, effort

    def _build_exec_args(
        self,
        model: Optional[str],
        effort: Optional[str],
        sandbox: Optional[str] = None,
        full_auto: bool = True,
    ) -> list:
        model, effort = self._apply_codex_policy(model, effort)
        args = ["exec", "--skip-git-repo-check"]
        if model:
            args.extend(["--model", model])
        args.extend(["-c", f'model_reasoning_effort="{effort}"'])
        effective_sandbox = sandbox or self.config.codex_sandbox
        if effective_sandbox == "danger-full-access":
            args.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            args.extend(["--sandbox", effective_sandbox])
            if full_auto:
                args.append("--full-auto")
        args.extend(["--json", "-"])
        return args

    async def _launch_rescue(
        self,
        task: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        cwd: Optional[str] = None,
        resume_from: Optional[str] = None,
        sandbox: Optional[str] = None,
    ) -> str:
        """Start a background rescue job and return its ID immediately."""
        job_id = f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
        job = CodexJob(
            id=job_id,
            task=task,
            model=model or self.config.codex_model,
            working_dir=cwd or os.getcwd(),
            effort=effort,
            sandbox=sandbox,
            resume_from=resume_from,
        )
        self.jobs[job_id] = job
        job.save(self.jobs_dir / f"{job_id}.json")
        t = asyncio.create_task(self._run_rescue_background(job_id))
        self._job_tasks[job_id] = t
        return (
            f"Rescue job started in background.\n"
            f"  Job ID: {job_id}\n"
            f"  Model: {job.model}{f', effort: {effort}' if effort else ''}"
            f"{f', resume: {resume_from}' if resume_from else ''}\n"
            f"  Use codex_job_status to check progress.\n"
            f"  Use codex_job_result {job_id} when done."
        )

    async def rescue(
        self,
        task: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        working_dir: Optional[str] = None,
        background: bool = True,
        resume_from: Optional[str] = None,
        fresh: bool = False,
        sandbox: Optional[str] = None,
    ) -> str:
        """Delegate a task to Codex. Supports background execution and session resume."""
        cwd = working_dir or os.getcwd()
        if not resume_from and not fresh:
            recent = [j for j in self.jobs.values() if j.codex_session_id and j.status == "completed"]
            if recent:
                latest = max(recent, key=lambda j: j.finished or "")
                resume_from = latest.codex_session_id

        if background:
            return await self._launch_rescue(task, model=model, effort=effort, cwd=cwd, resume_from=resume_from, sandbox=sandbox)

        args = self._build_exec_args(model or self.config.codex_model, effort, sandbox=sandbox)
        if resume_from:
            args = ["exec", "resume", resume_from] + args[1:]
        output, code = await self._run_codex_exec_stdin(args, task, cwd, timeout=600)
        if code != 0:
            return f"Error: {output}"
        reply, thread_id = self._parse_codex_jsonl(output)
        result = reply or output or "No response received"
        if thread_id:
            result += f"\n\n(Codex session: {thread_id} — resume with: codex rescue --resume {thread_id})"
        return result

    def job_status(self, job_id: Optional[str] = None) -> str:
        if not self.jobs:
            return "No rescue jobs found."
        if job_id:
            job_id = _sanitize_session_id(job_id) if re.fullmatch(r'[\w\-]+', job_id) else None
            if not job_id or job_id not in self.jobs:
                return f"Job '{job_id}' not found."
            job = self.jobs[job_id]
            lines = [
                f"Job: {job.id}",
                f"  Status:  {job.status}",
                f"  Task:    {job.task[:120]}{'…' if len(job.task) > 120 else ''}",
                f"  Model:   {job.model}{f' ({job.effort})' if job.effort else ''}",
                f"  Created: {job.created}",
            ]
            if job.started:
                lines.append(f"  Started: {job.started}")
            if job.finished:
                lines.append(f"  Finished: {job.finished}")
            if job.codex_session_id:
                lines.append(f"  Codex session: {job.codex_session_id}")
            return "\n".join(lines)

        jobs_sorted = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:10]
        lines = [f"Rescue Jobs ({len(self.jobs)} total, showing latest 10):"]
        for job in jobs_sorted:
            marker = {"running": "⏳", "completed": "✓", "failed": "✗", "cancelled": "⊘"}.get(job.status, "?")
            age = job.created[:19].replace("T", " ")
            lines.append(f"  {marker} {job.id}  [{job.status}]  {age}  {job.task[:60]}{'…' if len(job.task) > 60 else ''}")
        return "\n".join(lines)

    def job_result(self, job_id: Optional[str] = None) -> str:
        if not self.jobs:
            return "No rescue jobs found."
        if not job_id:
            completed = [j for j in self.jobs.values() if j.status == "completed"]
            if not completed:
                return "No completed jobs found."
            job = max(completed, key=lambda j: j.finished or "")
        else:
            if job_id not in self.jobs:
                return f"Job '{job_id}' not found."
            job = self.jobs[job_id]

        if job.status == "running":
            return f"Job '{job.id}' is still running. Check back with codex_job_status."
        lines = [
            f"Job: {job.id}  [{job.status}]",
            f"Task: {job.task[:200]}{'…' if len(job.task) > 200 else ''}",
            "",
            job.result or "(no output)",
        ]
        if job.codex_session_id:
            lines.append(f"\nCodex session: {job.codex_session_id}")
            lines.append(f"Resume in Codex: codex resume {job.codex_session_id}")
        return "\n".join(lines)

    def job_cancel(self, job_id: Optional[str] = None) -> str:
        if not job_id:
            running = [j for j in self.jobs.values() if j.status == "running"]
            if not running:
                return "No running jobs to cancel."
            if len(running) > 1:
                return f"Multiple running jobs: {', '.join(j.id for j in running)}. Specify job_id."
            job_id = running[0].id

        if job_id not in self.jobs:
            return f"Job '{job_id}' not found."
        job = self.jobs[job_id]
        if job.status != "running":
            return f"Job '{job_id}' is not running (status: {job.status})."

        task = self._job_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return f"Cancellation requested for job '{job_id}'."
        job.status = "cancelled"
        job.finished = datetime.now().isoformat()
        job.save(self.jobs_dir / f"{job_id}.json")
        return f"Job '{job_id}' marked as cancelled."

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No Codex sessions found."
        lines = ["Codex Sessions:"]
        for sid, session in self.sessions.items():
            active = " (active)" if sid == self.active_session else ""
            msg_count = len(session.messages)
            mode = "full-auto" if session.full_auto else session.sandbox
            cc_ids = f", claude={','.join(session.claude_session_ids)}" if session.claude_session_ids else ""
            lines.append(f"  - {sid}: {session.model} [{mode}], {msg_count} messages{cc_ids}{active}")
        return "\n".join(lines)

    def attach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Codex session not found."
        session = self.sessions[sid]
        if claude_session_id not in session.claude_session_ids:
            session.claude_session_ids.append(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
        return f"Attached Claude session '{claude_session_id}' to Codex session '{sid}'."

    def detach_claude_session(self, session_id: str, claude_session_id: str) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "Codex session not found."
        session = self.sessions[sid]
        if claude_session_id in session.claude_session_ids:
            session.claude_session_ids.remove(claude_session_id)
            session.save(self.sessions_dir / f"{sid}.json")
            return f"Detached Claude session '{claude_session_id}' from '{sid}'."
        return f"Claude session '{claude_session_id}' was not attached to '{sid}'."

    def end_unattached(self) -> str:
        """End all Codex sessions with no live Claude Code session IDs."""
        targets = []
        for sid, s in self.sessions.items():
            if not s.claude_session_ids:
                targets.append(sid)
            else:
                statuses = [_chitta_session_alive(csid) for csid in s.claude_session_ids]
                if any(st is True or st is None for st in statuses):
                    continue
                targets.append(sid)
        if not targets:
            return "All Codex sessions have live attached Claude Code IDs — nothing to end."
        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None
        return f"Ended {len(targets)} unattached Codex session(s): {', '.join(targets)}"

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session."
        session = self.sessions[sid]
        mode = "full-auto" if session.full_auto else session.sandbox
        lines = [f"Codex Session: {sid}", f"Model: {session.model}, Mode: {mode}", "---"]
        for msg in session.messages[-last_n:]:
            role = "You" if msg.role == "user" else "Codex"
            lines.append(f"\n**{role}:**\n{msg.content}")
        return "\n".join(lines)

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Codex session '{session_id}' not found."
        self.active_session = session_id
        session = self.sessions[session_id]
        mode = "full-auto" if session.full_auto else session.sandbox
        return f"Active Codex session: '{session_id}' ({session.model}, {mode})"

    def set_model(self, model: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session."
        session = self.sessions[sid]
        old_model = session.model
        session.model = model
        session.save(self.sessions_dir / f"{sid}.json")
        return f"Codex model changed: {old_model} -> {model}"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active Codex session to end."
        del self.sessions[sid]
        session_path = self.sessions_dir / f"{sid}.json"
        if session_path.exists():
            session_path.unlink()
        if self.active_session == sid:
            self.active_session = None
        return f"Codex session '{sid}' ended."

    def end_all(self, session_ids: Optional[list] = None, exclude_model: Optional[str] = None) -> str:
        """End all Codex sessions, or only the sessions named in session_ids."""
        if session_ids:
            candidates = [_sanitize_session_id(s) for s in session_ids if s in self.sessions]
            not_found = [s for s in session_ids if s not in self.sessions]
        else:
            candidates = list(self.sessions.keys())
            not_found = []

        if exclude_model:
            targets = [s for s in candidates if self.sessions[s].model != exclude_model]
            skipped = [s for s in candidates if self.sessions[s].model == exclude_model]
        else:
            targets = candidates
            skipped = []

        if not targets:
            msg = "No matching Codex sessions to end."
            if skipped:
                msg += f" Kept {len(skipped)} session(s) with model '{exclude_model}'."
            if not_found:
                msg += f" Not found: {', '.join(not_found)}"
            return msg

        for sid in targets:
            del self.sessions[sid]
            path = self.sessions_dir / f"{sid}.json"
            if path.exists():
                path.unlink()
            if self.active_session == sid:
                self.active_session = None

        lines = [f"Ended {len(targets)} Codex session(s): {', '.join(targets)}"]
        if skipped:
            lines.append(f"Kept {len(skipped)} session(s) with model '{exclude_model}': {', '.join(skipped)}")
        if not_found:
            lines.append(f"Not found: {', '.join(not_found)}")
        return "\n".join(lines)

    def health_check(self) -> dict:
        uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        return {
            "status": "ok" if CODEX_BIN else "codex not found",
            "codex_installed": CODEX_BIN is not None,
            "sessions": len(self.sessions),
            "uptime": uptime_seconds,
        }
