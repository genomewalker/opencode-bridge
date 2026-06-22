"""Module discovery helpers: backend inference and model shorthand resolution."""

import re
from pathlib import Path
from typing import Optional

__all__ = [
    "_discover_claude_shorthands",
    "_discover_codex_shorthands",
    "_discover_local_endpoints",
    "_normalize_participant_shorthands",
    "_infer_backend",
]

_BACKEND_RULES: list[tuple[tuple[str, ...], str]] = [
    # Anthropic → claude
    (("claude", "opus", "sonnet", "haiku", "fable"), "claude"),
    # OpenAI → codex  (o1/o3/o4 require exact match or clear suffix to avoid false positives
    # on participant names like "o3-planning"; gpt- prefix is unambiguous)
    (("gpt-", "chatgpt", "codex", "text-davinci", "text-embedding"), "codex"),
    (("o1-", "o3-", "o4-", "o1mini", "o3mini"), "codex"),  # versioned OpenAI models

    # Open-source / local weights → local
    (("llama", "qwen", "mistral", "mixtral", "phi", "deepseek", "falcon", "vicuna",
      "orca", "gemma", "starcoder", "codellama", "yi-", "nous-", "wizardcoder",
      "openchat", "zephyr", "tinyllama", "stablelm", "internlm", "baichuan",
      "solar", "neural-chat"), "local"),
]

_CLAUDE_MODEL_CACHE: "dict[str, str] | None" = None
_CODEX_MODEL_CACHE: "dict[str, str] | None" = None


def _discover_claude_shorthands() -> "dict[str, str]":
    """Query Anthropic REST /v1/models and build family→model-id shorthand map.

    Reads primaryApiKey from ~/.claude/config.json (same credential claude -p uses).
    Response is ordered newest-first; first hit per family = latest.
    Falls back to hard-coded defaults if unreachable.
    """
    global _CLAUDE_MODEL_CACHE
    if _CLAUDE_MODEL_CACHE is not None:
        return _CLAUDE_MODEL_CACHE
    _FALLBACK: "dict[str, str]" = {
        "opus":     "claude-opus-4-8",
        "opus-4.8": "claude-opus-4-8",
        "opus-4-8": "claude-opus-4-8",
        "sonnet":   "claude-sonnet-4-6",
        "haiku":    "claude-haiku-4-5",
        "fable":    "claude-fable-5",
        "fable5":   "claude-fable-5",
    }
    try:
        import json as _json
        import urllib.request as _ur
        cfg_path = Path.home() / ".claude" / "config.json"
        if not cfg_path.exists():
            _CLAUDE_MODEL_CACHE = _FALLBACK
            return _CLAUDE_MODEL_CACHE
        api_key = _json.loads(cfg_path.read_text()).get("primaryApiKey", "")
        if not api_key:
            _CLAUDE_MODEL_CACHE = _FALLBACK
            return _CLAUDE_MODEL_CACHE
        req = _ur.Request(
            "https://api.anthropic.com/v1/models?limit=100",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        with _ur.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read())
        out: "dict[str, str]" = {}
        for model in data.get("data", []):
            mid = model.get("id", "")
            m = re.match(r'claude-([a-z]+)-([\d].*)', mid)
            if not m:
                continue
            family, ver = m.group(1), m.group(2)
            if family not in out:          # first = latest (newest-first list)
                out[family] = mid
                dot_ver = ver.replace("-", ".")
                out[f"{family}-{dot_ver}"] = mid
                out[f"{family}-{ver}"]     = mid
        if "fable" in out:
            ver_num = out["fable"].rsplit("-", 1)[-1]
            out[f"fable{ver_num}"] = out["fable"]
        _CLAUDE_MODEL_CACHE = out if out else _FALLBACK
    except Exception:
        _CLAUDE_MODEL_CACHE = _FALLBACK
    return _CLAUDE_MODEL_CACHE


def _discover_codex_shorthands() -> "dict[str, str]":
    """Read ~/.codex/config.toml and build GPT model shorthands.

    Exposes 'gpt' → default model from config. Falls back to gpt-5.5.
    """
    global _CODEX_MODEL_CACHE
    if _CODEX_MODEL_CACHE is not None:
        return _CODEX_MODEL_CACHE
    _FALLBACK: "dict[str, str]" = {"gpt": "gpt-5.5"}
    try:
        cfg_path = Path.home() / ".codex" / "config.toml"
        if not cfg_path.exists():
            _CODEX_MODEL_CACHE = _FALLBACK
            return _CODEX_MODEL_CACHE
        try:
            import tomllib as _toml
        except ImportError:
            import tomli as _toml  # type: ignore
        with open(cfg_path, "rb") as fh:
            cfg = _toml.load(fh)
        default_model = cfg.get("model", "gpt-5.5")
        out: "dict[str, str]" = {"gpt": default_model}
        # slug alias: "gpt5.5" → "gpt-5.5"
        slug = default_model.replace("-", "").replace(".", "")
        out[slug] = default_model
        _CODEX_MODEL_CACHE = out
    except Exception:
        _CODEX_MODEL_CACHE = _FALLBACK
    return _CODEX_MODEL_CACHE


def _discover_local_endpoints() -> list[tuple[str, str]]:
    """Return (model_hint, base_url) pairs from ~/.chitta-bridge/endpoints/ollama-server-*.url files."""
    import glob as _glob
    import os
    _dir = os.environ.get("CHITTA_BRIDGE_URL_DIR", str(Path.home() / ".chitta-bridge" / "endpoints"))
    results = []
    for path in _glob.glob(f"{_dir}/ollama-server-*.url"):
        try:
            url = Path(path).read_text().strip()
            if url:
                hint = Path(path).stem.removeprefix("ollama-server-")
                results.append((hint, url))
        except OSError:
            pass
    return results


def _normalize_participant_shorthands(plist: list) -> list:
    """Accept 'backend:model[:effort]' shorthand strings alongside participant dicts."""
    claude_sh = _discover_claude_shorthands()
    codex_sh  = _discover_codex_shorthands()
    out = []
    for p in plist or []:
        if isinstance(p, dict):
            out.append(p)
            continue
        s = str(p)
        parts = s.split(":")
        if parts[0] in ("codex", "claude", "local") and len(parts) > 1:
            backend = parts[0]
            if backend == "local":
                # local:model:tag[:effort] — model may contain colons (e.g. gemma4:26b)
                # effort is the last segment only if it matches a known effort keyword
                _EFFORT_KEYS = {"low", "medium", "high", "xhigh", "max"}
                rest = parts[1:]
                effort = rest[-1].lower() if len(rest) > 1 and rest[-1].lower() in _EFFORT_KEYS else None
                model = ":".join(rest[:-1] if effort else rest)
                d = {"name": s, "backend": "local", "model": model}
                if effort:
                    d["effort"] = effort
                # Resolve base_url from cached endpoint files if available
                for hint, url in _discover_local_endpoints():
                    if hint == model or model.startswith(hint):
                        d["base_url"] = url
                        break
                out.append(d)
                continue
            model = parts[1]
            if backend == "claude":
                model = claude_sh.get(model.lower(), model)
            elif backend == "codex":
                model = codex_sh.get(model.lower(), model)
            d = {"name": s, "backend": backend, "model": model}
            if len(parts) > 2:
                d["effort"] = parts[2].lower()
            out.append(d)
        else:
            try:
                inferred = _infer_backend(s)
            except ValueError:
                inferred = "claude"
            if inferred == "claude":
                model_id = claude_sh.get(s.lower(), s)
            elif inferred == "codex":
                model_id = codex_sh.get(s.lower(), s)
            else:
                model_id = s
            out.append({"name": s, "backend": inferred, "model": model_id})
    return out


def _infer_backend(participant_name: str, model: Optional[str] = None) -> str:
    """Infer the backend from participant name or model string.

    Checks model first (more specific), then participant name.
    Raises ValueError if the backend cannot be determined unambiguously.
    """
    # Exact-match short OpenAI model names that can't safely use startswith
    _CODEX_EXACT = {"o1", "o3", "o4", "o1-mini", "o3-mini", "o4-mini"}
    for probe in (model, participant_name):
        if not probe:
            continue
        low = probe.lower().strip()
        if low in _CODEX_EXACT:
            return "codex"
        for prefixes, backend in _BACKEND_RULES:
            if any(low.startswith(p) or low == p.rstrip("-") for p in prefixes):
                return backend
    raise ValueError(
        f"Cannot infer backend for participant '{participant_name}'"
        f"{f' (model={model!r})' if model else ''}. "
        "Set backend explicitly to one of: claude, codex, local"
    )
