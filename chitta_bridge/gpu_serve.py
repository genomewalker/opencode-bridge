"""
chitta-gpu — Start/stop/discover Ollama on SLURM GPU nodes.

Installed as a CLI entry point via chitta-bridge:
    chitta-gpu start [model] [options]
    chitta-gpu stop [model]
    chitta-gpu status
    chitta-gpu env [model]      # print export statements (for eval)

When used interactively:
    eval "$(chitta-gpu start qwen3-coder)"
    claude --model qwen3-coder

The last lines of 'start' output are shell exports, so piping through eval
sets ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN in the calling shell.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3-coder"
DEFAULT_PORT = 11434
DEFAULT_PARTITION = "compregular"
DEFAULT_GRES = "gpu:1"
DEFAULT_MEM = "48G"
DEFAULT_TIME = "08:00:00"
URL_FILE_PATTERN = "/tmp/ollama-server-{model}.url"
JOB_NAME_PATTERN = "ollama-{model}"


def _url_file(model: str) -> Path:
    return Path(URL_FILE_PATTERN.format(model=model))


def _job_name(model: str) -> str:
    return JOB_NAME_PATTERN.format(model=model)


def _find_jobs(model: str | None = None) -> list[dict]:
    """Find running Ollama SLURM jobs. If model is None, find all ollama-* jobs."""
    if not shutil.which("squeue"):
        return []
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--format=%i %j %T %N %b %M"],
            timeout=10, stderr=subprocess.DEVNULL, text=True,
        )
        prefix = _job_name(model) if model else "ollama-"
        jobs = []
        for line in out.strip().splitlines():
            parts = line.split(None, 5)
            if len(parts) >= 4 and parts[1].startswith(prefix):
                jobs.append({
                    "job_id": parts[0], "name": parts[1],
                    "state": parts[2], "node": parts[3],
                    "gres": parts[4] if len(parts) > 4 else "",
                    "time": parts[5] if len(parts) > 5 else "",
                })
        return jobs
    except Exception:
        return []


def _probe_ollama(base_url: str, timeout: int = 4) -> list[str] | None:
    """Return model names at base_url, or None if unreachable."""
    # Strip /v1 suffix if present — Ollama tags endpoint is at /api/tags, not /v1/api/tags
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        clean = clean[:-3]
    tags_url = clean + "/api/tags"
    try:
        resp = urllib.request.urlopen(tags_url, timeout=timeout)
        data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return None


def _cached_endpoints() -> list[dict]:
    """Return endpoints from /tmp/ollama-server-*.url files."""
    results = []
    for path in glob.glob("/tmp/ollama-server-*.url"):
        try:
            url = Path(path).read_text().strip()
            if url:
                model_hint = Path(path).stem.removeprefix("ollama-server-")
                results.append({"model": model_hint, "url": url, "path": path})
        except OSError:
            pass
    return results


def _ollama_bin() -> str:
    """Find ollama binary."""
    custom = os.environ.get("OLLAMA_BIN")
    if custom and Path(custom).is_file():
        return custom
    local = Path.home() / ".local" / "bin" / "ollama"
    if local.is_file():
        return str(local)
    found = shutil.which("ollama")
    if found:
        return found
    return "ollama"


def _parse_url(url: str) -> tuple[str, int]:
    """Extract (host, port) from a URL like http://node:11434 or http://node:11434/v1."""
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.hostname or "localhost", p.port or DEFAULT_PORT


def _print_exports(node: str, port: int, model: str):
    """Print shell export statements."""
    base_url = f"http://{node}:{port}"
    print(f"export ANTHROPIC_BASE_URL={base_url}")
    print(f"export ANTHROPIC_AUTH_TOKEN=ollama")
    print(f"# Run: claude --model {model}")


def cmd_start(args):
    model = args.model
    port = args.port
    job_name = _job_name(model)
    url_file = _url_file(model)
    ollama = _ollama_bin()

    # Check if model is already available on any reachable endpoint
    # 1. Cached URL files
    for ep in _cached_endpoints():
        available = _probe_ollama(ep["url"])
        if available is not None and model in available:
            node, ep_port = _parse_url(ep["url"])
            print(f"# {model} already available on {node}", file=sys.stderr)
            print(f"# Models: {', '.join(available)}", file=sys.stderr)
            url_file.write_text(ep["url"])
            _print_exports(node, ep_port, model)
            return

    # 2. Any running SLURM GPU jobs (even if named differently)
    for j in _find_jobs(model=None):
        if j["state"] == "RUNNING" and j["node"]:
            node = j["node"]
            base_url = f"http://{node}:{port}"
            available = _probe_ollama(base_url)
            if available is not None:
                if model in available:
                    print(f"# {model} already available on {node} (job {j['job_id']})", file=sys.stderr)
                    url_file.write_text(base_url)
                    _print_exports(node, port, model)
                    return
                # Server is running but model not loaded — pull it
                print(f"# Ollama on {node} (job {j['job_id']}) — pulling {model}...", file=sys.stderr)
                subprocess.run([ollama, "pull", model], env={**os.environ, "OLLAMA_HOST": f"{node}:{port}"})
                url_file.write_text(base_url)
                _print_exports(node, port, model)
                return

    # 3. Localhost
    local_url = f"http://localhost:{port}"
    available = _probe_ollama(local_url)
    if available is not None and model in available:
        print(f"# {model} already available locally", file=sys.stderr)
        url_file.write_text(local_url)
        _print_exports("localhost", port, model)
        return

    has_slurm = shutil.which("sbatch") is not None

    if has_slurm:
        _start_slurm(args, model, port, job_name, url_file, ollama)
    else:
        _start_local(args, model, port, url_file, ollama)


def _start_local(args, model: str, port: int, url_file: Path, ollama: str):
    """Start Ollama locally (no SLURM available)."""
    # Check if already running locally
    local_url = f"http://localhost:{port}"
    models = _probe_ollama(local_url)
    if models is not None:
        print(f"# Ollama already running locally", file=sys.stderr)
        if models:
            print(f"# Models: {', '.join(models)}", file=sys.stderr)
        if model not in (models or []):
            print(f"# Pulling {model}...", file=sys.stderr)
            subprocess.run([ollama, "pull", model], stderr=subprocess.DEVNULL)
        url_file.write_text(local_url)
        _print_exports("localhost", port, model)
        return

    # Start ollama serve in background
    print(f"# Starting Ollama locally (no SLURM detected)...", file=sys.stderr)
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    subprocess.Popen(
        [ollama, "serve"],
        stdout=open(f"/tmp/ollama-local-{model}.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    # Wait for it to come up
    print("# Waiting for Ollama to be ready...", file=sys.stderr)
    for _ in range(30):
        if _probe_ollama(local_url) is not None:
            break
        time.sleep(2)
    else:
        print("# Ollama didn't start. Check /tmp/ollama-local-{model}.log", file=sys.stderr)
        sys.exit(1)

    # Pull model
    print(f"# Pulling {model}...", file=sys.stderr)
    subprocess.run([ollama, "pull", model])

    url_file.write_text(local_url)
    print(f"# Ollama ready locally", file=sys.stderr)
    _print_exports("localhost", port, model)


def _start_slurm(args, model: str, port: int, job_name: str, url_file: Path, ollama: str):
    """Start Ollama on a SLURM GPU node."""
    print(f"# Submitting: {job_name} (model={model}, gres={args.gres}, mem={args.mem})", file=sys.stderr)

    wrap_script = f"""\
export OLLAMA_HOST=0.0.0.0:{port}
export OLLAMA_MODELS=${{OLLAMA_MODELS:-$HOME/.ollama/models}}
{ollama} serve &
SERVE_PID=$!
sleep 5
{ollama} pull {model} 2>&1 || true
echo "http://$(hostname):{port}" > {url_file}
wait $SERVE_PID
"""

    try:
        result = subprocess.check_output(
            ["sbatch", "--parsable",
             f"--job-name={job_name}",
             f"--partition={args.partition}",
             f"--gres={args.gres}",
             f"--mem={args.mem}",
             f"--time={args.time}",
             f"--output=/tmp/ollama-{model}-%j.log",
             f"--wrap={wrap_script}"],
            timeout=30, text=True, stderr=subprocess.PIPE,
        )
        job_id = result.strip()
        print(f"# Submitted job {job_id}", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: sbatch failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    # Wait for job to start
    print("# Waiting for job to start...", file=sys.stderr)
    node = None
    for _ in range(60):
        jobs = _find_jobs(model)
        for j in jobs:
            if j["state"] == "RUNNING" and j["node"]:
                node = j["node"]
                break
        if node:
            break
        time.sleep(5)

    if not node:
        print("# Timed out waiting for job. Check: squeue --me", file=sys.stderr)
        sys.exit(1)

    print(f"# Job running on {node}", file=sys.stderr)

    # Wait for Ollama to respond
    print("# Waiting for Ollama to be ready...", file=sys.stderr)
    base_url = f"http://{node}:{port}"
    for _ in range(60):
        models = _probe_ollama(base_url)
        if models is not None:
            print(f"# Ollama ready. Models: {', '.join(models) or '(pulling...)'}", file=sys.stderr)
            url_file.write_text(base_url)
            _print_exports(node, port, model)
            return
        time.sleep(5)

    # Ollama not ready yet but job is running — export anyway
    print("# Ollama not responding yet (may still be pulling model)", file=sys.stderr)
    print(f"# Check: curl {base_url}/api/tags", file=sys.stderr)
    print(f"# Logs: /tmp/ollama-{model}-*.log", file=sys.stderr)
    url_file.write_text(base_url)
    _print_exports(node, port, model)


def cmd_stop(args):
    model = args.model
    stopped = False

    # Stop SLURM jobs
    jobs = _find_jobs(model)
    for j in jobs:
        subprocess.run(["scancel", j["job_id"]], check=False)
        print(f"Cancelled job {j['job_id']} ({j['name']}) on {j['node']}", file=sys.stderr)
        stopped = True

    # Stop local ollama if no SLURM and it's running locally
    if not stopped and not shutil.which("sbatch"):
        local_url = f"http://localhost:{DEFAULT_PORT}"
        if _probe_ollama(local_url) is not None:
            subprocess.run(["pkill", "-f", "ollama serve"], check=False)
            print("Stopped local Ollama process.", file=sys.stderr)
            stopped = True

    if not stopped:
        print(f"No running Ollama found for {model}.", file=sys.stderr)

    url_file = _url_file(model)
    url_file.unlink(missing_ok=True)
    print("unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN")


def cmd_status(args):
    # Running jobs
    jobs = _find_jobs(model=None)
    if not jobs:
        print("No Ollama SLURM jobs running.", file=sys.stderr)
    else:
        print("Running Ollama jobs:", file=sys.stderr)
        for j in jobs:
            print(f"  {j['job_id']}  {j['name']:<25} {j['state']:<10} {j['node']:<20} {j['gres']}", file=sys.stderr)

    # Cached endpoints
    cached = _cached_endpoints()
    if cached:
        print("\nCached endpoints:", file=sys.stderr)
        for ep in cached:
            models = _probe_ollama(ep["url"])
            status = f"models: {', '.join(models)}" if models else "unreachable"
            print(f"  {ep['model']:<20} {ep['url']:<40} ({status})", file=sys.stderr)

    # If there's an active endpoint, print exports for the first one
    for ep in cached:
        if _probe_ollama(ep["url"]) is not None:
            node, port = _parse_url(ep["url"])
            _print_exports(node, port, ep["model"])
            return


def cmd_env(args):
    model = args.model
    url_file = _url_file(model)

    # Check cached URL file for this specific model
    if url_file.is_file():
        base_url = url_file.read_text().strip()
        if base_url and _probe_ollama(base_url) is not None:
            node, port = _parse_url(base_url)
            _print_exports(node, port, model)
            return

    # Probe all cached endpoints — model may be available on a server started for another model
    for ep in _cached_endpoints():
        models = _probe_ollama(ep["url"])
        if models and model in models:
            node, port = _parse_url(ep["url"])
            _print_exports(node, port, model)
            return

    # Check running SLURM jobs and probe their nodes
    for j in _find_jobs(model=None):
        if j["state"] == "RUNNING" and j["node"]:
            base_url = f"http://{j['node']}:{args.port}"
            models = _probe_ollama(base_url)
            if models and (model in models or not models):
                _print_exports(j["node"], args.port, model)
                return

    print(f"# No active endpoint for {model}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="chitta-gpu",
        description="Manage Ollama on SLURM GPU nodes for Claude Code",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start Ollama on a GPU node")
    p_start.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p_start.add_argument("--time", default=DEFAULT_TIME)
    p_start.add_argument("--partition", default=DEFAULT_PARTITION)
    p_start.add_argument("--gres", default=DEFAULT_GRES)
    p_start.add_argument("--mem", default=DEFAULT_MEM)
    p_start.add_argument("--port", type=int, default=DEFAULT_PORT)

    # stop
    p_stop = sub.add_parser("stop", help="Stop Ollama SLURM job")
    p_stop.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    # status
    sub.add_parser("status", help="Show running Ollama jobs and endpoints")

    # env
    p_env = sub.add_parser("env", help="Print shell exports for an active endpoint")
    p_env.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    p_env.add_argument("--port", type=int, default=DEFAULT_PORT)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "env": cmd_env}[args.command](args)


if __name__ == "__main__":
    main()
