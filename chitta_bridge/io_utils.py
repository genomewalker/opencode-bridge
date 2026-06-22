"""
I/O utilities for chitta_bridge: path safety, atomic writes, env scrubbing.
"""

import os
import re
import hashlib
import stat as _stat_mod
import signal as _signal
import tempfile
import threading as _threading
from pathlib import Path
from typing import Optional

__all__ = [
    "_sanitize_session_id",
    "_reject_sensitive_path",
    "_blocked_read_path",
    "_path_write_lock",
    "_atomic_write_text",
    "_content_hash",
    "_atomic_write_text_legacy",
    "_sync_kill_group",
    "_scrub_env",
    "_llm_env",
]

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal."""
    if Path(session_id).name != session_id:
        raise ValueError("Invalid session ID: path separators not allowed")
    if not _SAFE_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid session ID: must be alphanumeric, hyphens, underscores only")
    return session_id


# ── Path-safety denylist (shared by file/symbol patch + write/edit tools) ──────

_SENSITIVE_SYSTEM_PREFIXES = (
    "/etc", "/boot", "/sys", "/proc", "/usr", "/dev", "/root",
)
_SENSITIVE_HOME_DIRS = frozenset({
    ".ssh", ".aws", ".gnupg", ".config", ".kube", ".docker",
})
_SENSITIVE_HOME_FILES = frozenset({
    ".gitconfig", ".netrc", ".git-credentials", ".pypirc", ".npmrc",
})
_CREDENTIAL_BASENAMES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
})


def _reject_sensitive_path(path: Path) -> Optional[str]:
    """Return a block message if writing `path` is forbidden, else None.

    Blocks system dirs (/etc, /boot, /sys, /proc, /usr, /dev, /root),
    most of /var (except /var/tmp), and credential/config locations under
    the user's home: ~/.ssh, ~/.aws, ~/.gnupg, ~/.config, ~/.kube, ~/.docker,
    ~/.gitconfig, ~/.netrc, ~/.npmrc, ~/.pypirc, ~/.git-credentials,
    plus any file whose basename starts with "credential" or matches a
    known SSH private-key name.
    """
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return f"(blocked: cannot resolve path — {path})"
    s = str(resolved)

    for prefix in _SENSITIVE_SYSTEM_PREFIXES:
        if s == prefix or s.startswith(prefix + "/"):
            return f"(blocked: system path — {resolved})"

    if s.startswith("/var/") and not s.startswith("/var/tmp/") and s != "/var/tmp":
        return f"(blocked: system path — {resolved})"

    home = str(Path.home())
    if s == home or s.startswith(home + "/"):
        rel = s[len(home):].lstrip("/")
        first = rel.split("/", 1)[0] if rel else ""
        if first in _SENSITIVE_HOME_DIRS:
            return f"(blocked: sensitive home dir — {resolved})"
        if first in _SENSITIVE_HOME_FILES:
            return f"(blocked: sensitive config file — {resolved})"

    name = resolved.name.lower()
    if name.startswith("credential") or name in _CREDENTIAL_BASENAMES:
        return f"(blocked: credential file — {resolved})"

    return None


def _blocked_read_path(path: Path) -> Optional[str]:
    """Return a block message if reading `path` is forbidden, else None.

    Read-side guard for room tools (read_file, pdf_read, doc_read): blocks
    kernel pseudo-filesystems, shadow files, and credential dotpaths.
    """
    str_path = str(path)
    blocked_prefixes = ("/proc", "/sys", "/dev")
    blocked_exact = ("/etc/shadow", "/etc/gshadow", "/etc/master.passwd")
    blocked_dotpaths = (
        "/.ssh/", "/.gnupg/", "/.aws/", "/.azure/", "/.gcloud/",
        "/.config/gh/", "/.docker/config.json", "/.kube/config",
        "/.netrc", "/.env", "/.npmrc",
    )
    if any(str_path.startswith(b) for b in blocked_prefixes):
        return f"(blocked: cannot read {path})"
    if str_path in blocked_exact:
        return f"(blocked: cannot read {path})"
    if any(bp in str_path for bp in blocked_dotpaths):
        return f"(blocked: sensitive file — {path})"
    return None


# ── Atomic write + per-path locks (concurrent patch/write safety) ──────────────

_path_write_locks: dict[str, _threading.Lock] = {}
_path_write_locks_mu = _threading.Lock()


def _path_write_lock(path: Path) -> _threading.Lock:
    """Return a lock for `path`. Keys are canonicalized via realpath so aliases
    (symlinks, `..` segments) coalesce onto the same lock."""
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


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write `content` to `path` with TOCTOU defenses.

    Hardened path (Linux / modern POSIX):
      - Opens the immediate parent with O_DIRECTORY|O_NOFOLLOW so a
        parent-swap to a symlink after the caller's denylist check fails
        with ELOOP.
      - Creates the temp file via openat(dirfd, O_CREAT|O_EXCL|O_NOFOLLOW)
        with mode 0600 — a final-component symlink cannot redirect our open.
      - Rejects an existing target that is a symlink OR has st_nlink > 1
        (hardlink-clobber defense).
      - Renames via renameat(dirfd, dirfd) for atomic, relative-resolution-
        safe swap.

    Fallback (Windows or platforms without dir_fd / O_NOFOLLOW): the legacy
    tempfile + os.replace path. Aliased/symlinked destinations are still
    caught by the caller's `_reject_sensitive_path` check; this is a race
    window, not a silent bypass.
    """
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

        # Hardlink / symlink clobber check on existing target.
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
                # Preserve the target's mode (exec bits, group perms) — the
                # rename would otherwise leave the file at the temp's 0600.
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


def _content_hash(text: str) -> str:
    """SHA-256 prefix of UTF-8 content — used for write-concurrency guards."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _atomic_write_text_legacy(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Fallback for platforms without dir_fd / O_NOFOLLOW support."""
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


# ── Subprocess process-group termination (shell grandchildren safety) ──────────


def _sync_kill_group(proc) -> None:
    """SIGKILL a subprocess and its process group. Drop-in for ``proc.kill()``.

    Requires the process was spawned with ``start_new_session=True``; without
    that, the group contains only ``proc`` itself and this degrades cleanly
    to the single-process kill.
    """
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


# ── Secret-denylist environment scrub (subprocess credential exposure) ────────

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
_SECRET_PREFIXES = (
    "AWS_", "AZURE_", "GCP_",
)
_SECRET_SUFFIXES = (
    "_TOKEN", "_API_KEY", "_APIKEY", "_SECRET", "_PASSWORD",
    "_PASSWD", "_CREDENTIALS", "_CREDS", "_PRIVATE_KEY",
    "_ACCESS_KEY", "_ACCESS_TOKEN", "_AUTH", "_AUTH_TOKEN",
    "_SESSION_TOKEN",
)


def _scrub_env(env: Optional[dict] = None) -> dict:
    """Return a copy of `env` (default: os.environ) with likely secrets dropped.

    Drops exact matches in `_SECRET_EXACT`, keys starting with
    `AWS_|AZURE_|GCP_`, and keys ending with secret-ish suffixes. Keeps
    safe operational vars (PATH, HOME, USER, LANG, TERM, TMPDIR, CONDA_*,
    SLURM_*, OLLAMA_*) so tools that depend on them still work.
    """
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


_LLM_KEEP_RE = re.compile(
    r"^(ANTHROPIC_|OPENAI_|OPENROUTER_|GROQ_|GEMINI_|GOOGLE_|MISTRAL_|DEEPSEEK_"
    r"|XAI_|TOGETHER_|FIREWORKS_|CEREBRAS_|OLLAMA_|CHITTA_)|_API_KEY$"
)


def _llm_env() -> dict:
    """Env for spawned LLM CLIs (opencode/codex/claude).

    Scrubs unrelated secrets (AWS, GitHub, Slack tokens, ...) but keeps
    model-provider keys the CLIs need to authenticate.
    """
    out = _scrub_env(os.environ)
    for k, v in os.environ.items():
        if k not in out and _LLM_KEEP_RE.search(k.upper()):
            out[k] = v
    return out
