"""chitta_bridge.soul — chittad daemon client for memory recall and storage."""

import json
import os
import socket
from pathlib import Path
from typing import Any, Optional

__all__ = ["SoulClient"]


class SoulClient:
    """Connect to chittad daemon for memory recall and storage."""

    @staticmethod
    def _djb2_hash(s: str) -> int:
        h = 5381
        for c in s:
            h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
        return h

    @classmethod
    def _socket_path(cls) -> str:
        home = os.environ.get("HOME", "")
        mind_path = os.path.join(home, ".claude", "mind")
        hash_val = cls._djb2_hash(mind_path)
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg and os.access(xdg, os.W_OK):
            base = os.path.join(xdg, "chitta")
        elif os.access(f"/run/user/{os.getuid()}", os.W_OK):
            base = os.path.join(f"/run/user/{os.getuid()}", "chitta")
        elif home:
            base = os.path.join(home, ".cache", "chitta")
        else:
            base = "/tmp"
        os.makedirs(base, mode=0o700, exist_ok=True)
        return os.path.join(base, f"chitta-{hash_val}.sock")

    @classmethod
    def _call(cls, method: str, arguments: dict, timeout: float = 5.0) -> Optional[str]:
        path = cls._socket_path()
        if not os.path.exists(path):
            return None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(path)
            req = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": method, "arguments": arguments},
            })
            sock.sendall((req + "\n").encode())
            response = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response += chunk
                if b"\n" in response:
                    break
            sock.close()
            data = json.loads(response.decode().strip())
            result = data.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "")
            return str(result)
        except Exception:
            return None

    @classmethod
    def recall(cls, query: str, limit: int = 5, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if realm:
            args["realm"] = realm
        return cls._call("recall", args)

    @classmethod
    def smart_context(cls, task: str, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"task": task}
        if realm:
            args["realm"] = realm
        return cls._call("smart_context", args, timeout=10.0)

    @classmethod
    def remember(cls, content: str, kind: str = "episode",
                 tags: str = "", confidence: float = 0.8,
                 realm: Optional[str] = None,
                 source_type: str = "observation", producer: str = "",
                 evidence_pointer: str = "") -> Optional[str]:
        # Memory hygiene: fold provenance into tags so it survives the existing
        # daemon store unchanged (no schema migration needed). source_type is
        # always recorded; producer/evidence_pointer only when supplied.
        _hyg = [f"src:{source_type}"]
        if producer:
            _hyg.append(f"by:{producer}")
        if evidence_pointer:
            _hyg.append(f"ev:{evidence_pointer}")
        tags = ",".join([t for t in (tags, *_hyg) if t])
        args: dict[str, Any] = {"content": content, "type": kind, "confidence": confidence}
        if tags:
            args["tags"] = tags
        if realm:
            args["realm"] = realm
        return cls._call("remember", args, timeout=60.0)

    @staticmethod
    def _recall_age_days(text: Optional[str]) -> Optional[float]:
        """Best-effort: extract the first ISO/epoch timestamp in recall output and
        return its age in days. Returns None if no parseable timestamp present.
        """
        if not text:
            return None
        import re as _re
        import time as _time
        from datetime import datetime as _dt
        m = _re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text)
        if m:
            try:
                ts = _dt.fromisoformat(m.group(0).replace(" ", "T")).timestamp()
                return round((_time.time() - ts) / 86400.0, 2)
            except ValueError:
                return None
        m2 = _re.search(r"\b(1[0-9]{9})\b", text)
        if m2:
            return round((_time.time() - float(m2.group(1))) / 86400.0, 2)
        return None

    @classmethod
    def hybrid_recall(cls, query: str, limit: int = 5, realm: Optional[str] = None) -> Optional[str]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if realm:
            args["realm"] = realm
        return cls._call("hybrid_recall", args)

    @classmethod
    def _call_json(cls, method: str, arguments: dict, timeout: float = 5.0) -> Optional[dict]:
        """Like _call but returns the full structured result dict (not just text)."""
        path = cls._socket_path()
        if not os.path.exists(path):
            return None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(path)
            req = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": method, "arguments": arguments},
            })
            sock.sendall((req + "\n").encode())
            response = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response += chunk
                if b"\n" in response:
                    break
            sock.close()
            data = json.loads(response.decode().strip())
            return data.get("result", {})
        except Exception:
            return None

    @classmethod
    def find_symbol_location(cls, filepath: str, symbol: str) -> Optional[tuple[int, int]]:
        """Use chitta tree-sitter index to locate a symbol.

        Returns (line_start, line_end) 1-based inclusive, or None if not found/unavailable.
        """
        result = cls._call_json("find_symbol", {"name": symbol})
        if not result:
            return None
        symbols = result.get("symbols", [])
        if not symbols:
            return None
        fp = str(Path(filepath).resolve())
        for sym in symbols:
            sym_file = str(Path(sym.get("file", "")).resolve())
            if sym_file == fp and sym.get("name") == symbol:
                return (int(sym["line_start"]), int(sym["line_end"]))
        # Fallback: if only one candidate returned, use it regardless of path
        # (handles cases where file path differs by symlink / relative form)
        if len(symbols) == 1:
            s = symbols[0]
            return (int(s["line_start"]), int(s["line_end"]))
        return None

    @classmethod
    def learn_codebase(cls, path: str, project: Optional[str] = None) -> Optional[str]:
        """Trigger tree-sitter re-index for a path. Short timeout — best-effort."""
        args: dict[str, Any] = {"path": path}
        if project:
            args["project"] = project
        return cls._call("learn_codebase", args, timeout=3.0)

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls._socket_path())
