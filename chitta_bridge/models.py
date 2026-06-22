"""
Data models shared across the chitta-bridge package.

Extracted from server.py so other modules can import them without pulling in
the full MCP server dependency tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat as _stat_mod
import tempfile
import threading as _threading
from dataclasses import asdict, dataclass, field, fields as dc_fields
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["Message", "Session", "CodexSession", "CodexJob"]

# ---------------------------------------------------------------------------
# Schema version (mirrors server.py constant)
# ---------------------------------------------------------------------------
PERSISTED_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Internal write helpers (inlined from server.py to avoid circular imports)
# ---------------------------------------------------------------------------
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
    parent = path.parent
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=str(parent),
        prefix="." + path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        try:
            os.chmod(tmp.name, _stat_mod.S_IMODE(os.stat(path).st_mode))
        except FileNotFoundError:
            pass
        os.replace(tmp.name, path)
    except Exception:
        try:
            os.unlink(tmp.name)
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
            if _stat_mod.S_ISLNK(st.st_mode):
                raise OSError(f"refused to write {path}: target is a symlink")
            if st.st_nlink > 1:
                raise OSError(
                    f"refused to write {path}: target has st_nlink={st.st_nlink} "
                    "(hardlink clobber risk)"
                )

        tmp_name = f".{target_name}.{os.getpid()}.{_threading.get_ident()}.tmp"
        try:
            os.unlink(tmp_name, dir_fd=dirfd)
        except FileNotFoundError:
            pass

        file_flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        data = content.encode(encoding)
        tmp_fd = os.open(tmp_name, file_flags, 0o600, dir_fd=dirfd)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
                f.flush()
                if st is not None:
                    os.fchmod(f.fileno(), _stat_mod.S_IMODE(st.st_mode))
                os.fsync(f.fileno())
            os.rename(tmp_name, target_name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
        except Exception:
            try:
                os.unlink(tmp_name, dir_fd=dirfd)
            except OSError:
                pass
            raise
    finally:
        os.close(dirfd)


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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

DEFAULT_CODEX_SANDBOX = "danger-full-access"


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    """Local LLM session (Ollama/vLLM via OpenAI-compatible API)."""
    id: str
    endpoint: str
    model: str
    messages: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    attached_claude_sessions: list = field(default_factory=list)


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
    messages: list[Message] = field(default_factory=list)
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
