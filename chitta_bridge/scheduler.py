"""
chitta-bridge native scheduler daemon.

Four asyncio tasks inside the aiohttp server:
  tick_loop          1-second DB poll → enqueue due jobs
  run_worker_loop    bounded executor dispatch (semaphore ≤ WORKER_SLOTS)
  reload_loop        jobs.yaml mtime+hash polling every 5 s
  slurm_reconcile    sacct polling for SLURM-backed running jobs

State split:
  scheduler.db (SQLite WAL)  — operational: last-run, failure counts, next_due
  chittad                    — semantic: summaries, arxiv hits, audit findings
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml
from croniter import croniter
from filelock import FileLock, Timeout
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

log = logging.getLogger("chitta.scheduler")

# ── Paths & tunables ──────────────────────────────────────────────────────────

SCHEDULER_DB    = Path.home() / ".chitta-bridge" / "scheduler.db"
SCHEDULER_LOCK  = Path.home() / ".chitta-bridge" / "scheduler.lock"
JOBS_YAML       = Path.home() / ".chitta-bridge" / "jobs.yaml"
RUNS_DIR        = Path.home() / ".chitta-bridge" / "job-runs"
WORKER_SLOTS    = 3   # max concurrent subprocesses (HPC login-node constraint)
MAX_CHAIN_DEPTH = 5
MAX_AUTO_DISABLE_FAILURES = 5
RELOAD_INTERVAL = 5.0   # seconds between jobs.yaml polls
SLURM_POLL_INTERVAL = 30.0

# ── Schema ────────────────────────────────────────────────────────────────────

class NotifyConfig(BaseModel):
    success: list[str] = []
    failure: list[str] = []

class ScheduleConfig(BaseModel):
    cron: str
    timezone: str = "UTC"
    jitter_seconds: int = 0
    catchup: bool = False

class ExecutorConfig(BaseModel):
    type: str  # codex | claude | room | workflow | slurm
    prompt: str = ""
    timeout_seconds: int = 1800
    cwd: Optional[str] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    participants: list[str] = []
    script: Optional[str] = None
    sbatch_args: list[str] = []

class OutputConfig(BaseModel):
    memory_tags: list[str] = []
    slack: NotifyConfig = NotifyConfig()

class ConditionsConfig(BaseModel):
    max_concurrent: int = 1
    skip_if_running: bool = True
    cost_ceiling_usd: float = 10.0

class OnResultConfig(BaseModel):
    when: str  # simpleeval expression on JobResult fields
    trigger: str  # job id to fire
    with_data: dict[str, Any] = {}

class JobConfig(BaseModel):
    id: str
    enabled: bool = True
    description: str = ""
    schedule: ScheduleConfig
    executor: ExecutorConfig
    output: OutputConfig = OutputConfig()
    conditions: ConditionsConfig = ConditionsConfig()
    on_result: list[OnResultConfig] = []

class JobsConfig(BaseModel):
    version: int = 1
    jobs: list[JobConfig] = []

# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class JobResult:
    status: str  # success | failure | timeout | abandoned
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    items: list[Any] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    json_data: Any = None

@dataclass
class RunRequest:
    job_id: str
    scheduled_ts: float
    run_id: str
    chain_depth: int = 0

# ── State store ───────────────────────────────────────────────────────────────

class StateStore:
    def __init__(self, path: Path = SCHEDULER_DB) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS job_state (
                    job_id          TEXT PRIMARY KEY,
                    enabled         INTEGER NOT NULL DEFAULT 1,
                    disabled_reason TEXT,
                    failure_count   INTEGER NOT NULL DEFAULT 0,
                    last_started    REAL,
                    last_finished   REAL,
                    last_status     TEXT,
                    next_due        REAL,
                    config_hash     TEXT,
                    updated_at      REAL NOT NULL DEFAULT (unixepoch())
                );
                CREATE TABLE IF NOT EXISTS job_runs (
                    run_id          TEXT PRIMARY KEY,
                    job_id          TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'running',
                    executor_kind   TEXT NOT NULL,
                    slurm_jobid     TEXT,
                    scheduled_at    REAL NOT NULL,
                    started_at      REAL NOT NULL,
                    finished_at     REAL,
                    summary         TEXT,
                    metrics_json    TEXT,
                    artifacts_json  TEXT,
                    error           TEXT,
                    idempotency_key TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id      TEXT PRIMARY KEY,
                    source_run_id TEXT,
                    target_job_id TEXT NOT NULL,
                    payload_json  TEXT,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    created_at    REAL NOT NULL DEFAULT (unixepoch())
                );
                CREATE INDEX IF NOT EXISTS idx_runs_job
                    ON job_runs(job_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_state_next
                    ON job_state(next_due) WHERE enabled=1;
            """)

    def upsert_job(self, job_id: str, config_hash: str, next_due: float) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO job_state (job_id, config_hash, next_due, updated_at)
                VALUES (?, ?, ?, unixepoch())
                ON CONFLICT(job_id) DO UPDATE SET
                    config_hash = excluded.config_hash,
                    next_due    = CASE
                        WHEN job_state.config_hash != excluded.config_hash
                             THEN excluded.next_due
                        ELSE job_state.next_due
                    END,
                    updated_at  = unixepoch()
            """, (job_id, config_hash, next_due))

    def get_due(self, now: float) -> list[tuple[str, float]]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT js.job_id, js.next_due
                FROM   job_state js
                WHERE  js.enabled  = 1
                  AND  js.next_due IS NOT NULL
                  AND  js.next_due <= ?
                  AND  NOT EXISTS (
                      SELECT 1 FROM job_runs jr
                      WHERE  jr.job_id = js.job_id AND jr.status = 'running'
                  )
            """, (now,)).fetchall()
        return [(r["job_id"], r["next_due"]) for r in rows]

    def create_run(self, run_id: str, job_id: str, executor_kind: str,
                   scheduled_at: float, idem_key: str) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO job_runs
                        (run_id, job_id, executor_kind, status,
                         scheduled_at, started_at, idempotency_key)
                    VALUES (?, ?, ?, 'running', ?, ?, ?)
                """, (run_id, job_id, executor_kind, scheduled_at, time.time(), idem_key))
                conn.execute("UPDATE job_state SET last_started=? WHERE job_id=?",
                             (time.time(), job_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def finish_run(self, run_id: str, job_id: str,
                   result: JobResult, next_due: Optional[float]) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                UPDATE job_runs SET
                    status=?, finished_at=?, summary=?,
                    metrics_json=?, artifacts_json=?, error=?
                WHERE run_id=?
            """, (
                result.status, now, result.summary[:2000],
                json.dumps(result.metrics), json.dumps(result.artifacts),
                result.summary if result.status != "success" else None,
                run_id,
            ))
            conn.execute("""
                UPDATE job_state SET
                    last_finished=?, last_status=?,
                    failure_count = CASE ? WHEN 'success' THEN 0
                                         ELSE failure_count + 1 END,
                    next_due=?
                WHERE job_id=?
            """, (now, result.status, result.status, next_due, job_id))

    def set_slurm_jobid(self, run_id: str, slurm_jobid: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE job_runs SET slurm_jobid=? WHERE run_id=?",
                         (slurm_jobid, run_id))

    def get_stale_subprocess_runs(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM job_runs
                WHERE status='running'
                  AND executor_kind IN ('codex','claude','workflow','room')
            """).fetchall()
        return [dict(r) for r in rows]

    def get_running_slurm(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM job_runs
                WHERE status='running' AND executor_kind='slurm'
                  AND slurm_jobid IS NOT NULL
            """).fetchall()
        return [dict(r) for r in rows]

    def abandon_run(self, run_id: str, job_id: str, next_due: Optional[float]) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute("""
                UPDATE job_runs SET status='abandoned', finished_at=?,
                    error='bridge restart — subprocess reaped'
                WHERE run_id=?
            """, (now, run_id))
            conn.execute("""
                UPDATE job_state SET last_status='abandoned',
                    last_finished=?, next_due=?
                WHERE job_id=?
            """, (now, next_due, job_id))

    def disable(self, job_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute("""
                UPDATE job_state SET enabled=0, disabled_reason=? WHERE job_id=?
            """, (reason, job_id))

    def enable(self, job_id: str) -> None:
        with self._conn() as conn:
            conn.execute("""
                UPDATE job_state
                SET enabled=1, disabled_reason=NULL, failure_count=0
                WHERE job_id=?
            """, (job_id,))

    def list_all(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT js.*,
                    (SELECT count(*) FROM job_runs jr WHERE jr.job_id=js.job_id) AS run_count,
                    (SELECT jr2.summary FROM job_runs jr2 WHERE jr2.job_id=js.job_id
                     ORDER BY jr2.started_at DESC LIMIT 1) AS last_summary
                FROM job_state js ORDER BY js.job_id
            """).fetchall()
        return [dict(r) for r in rows]

    def history(self, job_id: str, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM job_runs WHERE job_id=?
                ORDER BY started_at DESC LIMIT ?
            """, (job_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_failure_count(self, job_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT failure_count FROM job_state WHERE job_id=?", (job_id,)
            ).fetchone()
        return row["failure_count"] if row else 0

# ── Cron helpers ──────────────────────────────────────────────────────────────

def _next_due(cfg: ScheduleConfig, after: Optional[float] = None) -> float:
    import pytz
    tz = pytz.timezone(cfg.timezone)
    base = after or time.time()
    nxt = float(croniter(cfg.cron, base, hash_values=True).get_next())
    if cfg.jitter_seconds > 0:
        nxt += random.uniform(0, cfg.jitter_seconds)
    return nxt

def _config_hash(cfg: JobConfig) -> str:
    raw = json.dumps(cfg.model_dump(), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# ── Chain predicate evaluation ────────────────────────────────────────────────

def _eval_predicate(expr: str, result: JobResult) -> bool:
    try:
        from simpleeval import EvalWithCompoundTypes
        evaluator = EvalWithCompoundTypes(names={
            "status":   result.status,
            "summary":  result.summary,
            "metrics":  result.metrics,
            "items":    result.items,
        })
        return bool(evaluator.eval(expr))
    except Exception as e:
        log.warning("chain predicate eval failed: %s — %s", expr, e)
        return False

# ── Slack notify helper ───────────────────────────────────────────────────────

async def _slack_notify(channel: str, text: str,
                         slack_fn: Optional[Callable] = None) -> None:
    if slack_fn is None:
        log.info("slack[%s]: %s", channel, text[:200])
        return
    try:
        await slack_fn(channel=channel, message=text)
    except Exception as e:
        log.warning("slack notify failed: %s", e)

# ── Executors ─────────────────────────────────────────────────────────────────

async def _run_subprocess(cmd: list[str], cwd: str, prompt: str,
                           timeout: int, run_dir: Path) -> JobResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or None,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return JobResult(status="timeout",
                             summary=f"timed out after {timeout}s")
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        text = stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[:500]
            return JobResult(status="failure", summary=err or text[:500],
                             artifacts=[str(stdout_path), str(stderr_path)])
        return JobResult(status="success",
                         summary=text[:2000],
                         artifacts=[str(stdout_path)])
    except Exception as e:
        return JobResult(status="failure", summary=str(e))


async def _execute_job(cfg: JobConfig, run_id: str,
                        bridge_tools: dict,
                        semaphore: asyncio.Semaphore) -> JobResult:
    ex = cfg.executor
    run_dir = RUNS_DIR / run_id
    kind = ex.type

    if kind == "codex":
        codex_bin = bridge_tools.get("codex_bin", "codex")
        cmd = [codex_bin, "exec", "--skip-git-repo-check", "--full-auto",
               "--json", "-"]
        if ex.model:
            cmd += ["--model", ex.model]
        if ex.effort:
            cmd += ["-c", f'model_reasoning_effort="{ex.effort}"']
        async with semaphore:
            return await _run_subprocess(cmd, ex.cwd or os.getcwd(),
                                          ex.prompt, ex.timeout_seconds, run_dir)

    elif kind == "claude":
        claude_bin = bridge_tools.get("claude_bin", "claude")
        cmd = [claude_bin, "-p", "--output-format", "json"]
        if ex.model:
            cmd += ["--model", ex.model]
        if ex.effort:
            cmd += ["--effort", ex.effort]
        async with semaphore:
            return await _run_subprocess(cmd, ex.cwd or os.getcwd(),
                                          ex.prompt, ex.timeout_seconds, run_dir)

    elif kind == "room":
        room_mgr = bridge_tools.get("room_manager")
        if room_mgr is None:
            return JobResult(status="failure", summary="room_manager not available")
        rid = f"sched-{cfg.id}-{run_id[:8]}"
        participants = ex.participants or ["codex:gpt-5.5:medium"]
        async with semaphore:
            try:
                await room_mgr.create(rid, cfg.description or cfg.id, participants)
                from datetime import datetime as _dt
                room_mgr.rooms[rid].messages.append({
                    "name": "MODERATOR", "content": ex.prompt,
                    "ts": _dt.now().isoformat(),
                })
                summary = await room_mgr.run_rounds(rid, rounds=1)
                return JobResult(status="success", summary=summary[:2000])
            except Exception as e:
                return JobResult(status="failure", summary=str(e))

    elif kind == "workflow":
        # Fire-and-forget via bridge HTTP API (avoids circular import)
        http_url = bridge_tools.get("bridge_url", "http://localhost:7681")
        token = bridge_tools.get("bridge_token", "")
        script = ex.script or f"""
export const meta = {{name: 'sched-{cfg.id}', description: '{cfg.id}'}}
phase('Run')
const result = await agent({json.dumps(ex.prompt)})
return result
"""
        import aiohttp as _ah
        async with semaphore:
            try:
                async with _ah.ClientSession() as sess:
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    async with sess.post(
                        f"{http_url}/mcp",
                        json={"method": "tools/call", "params": {
                            "name": "advanced",
                            "arguments": {"tool": "codex_run",
                                          "arguments": {"task": ex.prompt,
                                                        "model": ex.model or "gpt-5.5",
                                                        "effort": ex.effort or "medium"}},
                        }},
                        headers=headers,
                        timeout=_ah.ClientTimeout(total=ex.timeout_seconds),
                    ) as resp:
                        data = await resp.json()
                        text = str(data.get("result", ""))[:2000]
                        return JobResult(status="success", summary=text)
            except Exception as e:
                return JobResult(status="failure", summary=str(e))

    elif kind == "slurm":
        try:
            sbatch_args = ex.sbatch_args or ["--time=01:00:00", "--mem=4G"]
            script_content = f"#!/bin/bash\n{ex.prompt}\n"
            script_path = run_dir / "job.sh"
            run_dir.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_content)
            result = subprocess.run(
                ["sbatch"] + sbatch_args + [str(script_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return JobResult(status="failure", summary=result.stderr.strip())
            # Extract SLURM job id: "Submitted batch job 12345"
            slurm_jobid = result.stdout.strip().split()[-1]
            # Return a pending result; slurm_reconcile will finish it
            return JobResult(
                status="slurm_submitted",
                summary=f"SLURM job {slurm_jobid} submitted",
                metrics={"slurm_jobid": slurm_jobid},
            )
        except Exception as e:
            return JobResult(status="failure", summary=str(e))

    return JobResult(status="failure", summary=f"unknown executor type: {kind}")


# ── Chitta memory write ───────────────────────────────────────────────────────

def _store_in_chitta(job_id: str, result: JobResult,
                      tags: list[str], chitta_fn: Optional[Callable]) -> None:
    if chitta_fn is None or not result.summary:
        return
    try:
        chitta_fn(
            content=f"[scheduler:{job_id}] {result.summary}",
            kind="episode",
            tags=",".join(["scheduler", job_id] + tags),
            confidence=0.8 if result.status == "success" else 0.5,
        )
    except Exception as e:
        log.warning("chitta store failed for %s: %s", job_id, e)


# ── Main scheduler service ────────────────────────────────────────────────────

class SchedulerService:
    def __init__(
        self,
        jobs_yaml: Path = JOBS_YAML,
        db_path: Path = SCHEDULER_DB,
        bridge_tools: Optional[dict] = None,
        slack_fn: Optional[Callable] = None,
        chitta_fn: Optional[Callable] = None,
    ) -> None:
        self.jobs_yaml = jobs_yaml
        self.db = StateStore(db_path)
        self.bridge_tools = bridge_tools or {}
        self.slack_fn = slack_fn
        self.chitta_fn = chitta_fn

        self._configs: dict[str, JobConfig] = {}
        self._yaml_mtime: float = 0.0
        self._yaml_hash: str = ""
        self._run_queue: asyncio.Queue[RunRequest] = asyncio.Queue(maxsize=64)
        self._semaphore = asyncio.Semaphore(WORKER_SLOTS)
        self._tasks: list[asyncio.Task] = []
        self._lock_file: Optional[FileLock] = None
        self._active = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        # Single-instance guard
        self._lock_file = FileLock(str(SCHEDULER_LOCK), timeout=0)
        try:
            self._lock_file.acquire()
        except Timeout:
            log.warning("scheduler: another instance holds the lock — scheduler disabled")
            return

        self._active = True
        self._recover_stale_runs()
        await self._reload_yaml()

        self._tasks = [
            asyncio.create_task(self._tick_loop(),     name="sched-tick"),
            asyncio.create_task(self._worker_loop(),   name="sched-worker"),
            asyncio.create_task(self._reload_loop(),   name="sched-reload"),
            asyncio.create_task(self._slurm_loop(),    name="sched-slurm"),
        ]
        log.info("scheduler: started with %d jobs", len(self._configs))

    async def stop(self) -> None:
        self._active = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._lock_file and self._lock_file.is_locked:
            self._lock_file.release()
        log.info("scheduler: stopped")

    # ── Crash recovery ────────────────────────────────────────────────────────

    def _recover_stale_runs(self) -> None:
        # Subprocess runs: bridge died → processes reaped → mark abandoned
        for row in self.db.get_stale_subprocess_runs():
            jid = row["job_id"]
            cfg = self._configs.get(jid)
            next_due = _next_due(cfg.schedule) if cfg else None
            self.db.abandon_run(row["run_id"], jid, next_due)
            log.info("scheduler: abandoned stale %s run %s", jid, row["run_id"])
        # SLURM runs: re-attach via sacct (handled in slurm_loop)

    # ── YAML reload ───────────────────────────────────────────────────────────

    async def _reload_loop(self) -> None:
        while self._active:
            await asyncio.sleep(RELOAD_INTERVAL)
            try:
                await self._reload_yaml()
            except Exception as e:
                log.error("scheduler reload error: %s", e)

    async def _reload_yaml(self) -> None:
        if not self.jobs_yaml.exists():
            return
        stat = self.jobs_yaml.stat()
        # Security: refuse non-0600
        mode = stat.st_mode & 0o777
        if mode & 0o077:
            log.error("scheduler: %s is not mode 600 — refusing to load", self.jobs_yaml)
            return
        raw = self.jobs_yaml.read_text()
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if h == self._yaml_hash:
            return
        try:
            data = yaml.safe_load(raw)
            cfg = JobsConfig.model_validate(data or {})
        except Exception as e:
            log.error("scheduler: jobs.yaml parse error: %s", e)
            return

        new_ids = {j.id for j in cfg.jobs}
        old_ids = set(self._configs)

        for jcfg in cfg.jobs:
            ch = _config_hash(jcfg)
            nxt = _next_due(jcfg.schedule)
            self.db.upsert_job(jcfg.id, ch, nxt)
            self._configs[jcfg.id] = jcfg

        # Remove dropped jobs from in-memory config (don't delete DB history)
        for removed in old_ids - new_ids:
            self._configs.pop(removed, None)

        self._yaml_hash = h
        self._yaml_mtime = stat.st_mtime
        log.info("scheduler: reloaded %d jobs (hash %s)", len(cfg.jobs), h)

    # ── Tick loop ─────────────────────────────────────────────────────────────

    async def _tick_loop(self) -> None:
        while self._active:
            now = time.time()
            try:
                for job_id, scheduled_ts in self.db.get_due(now):
                    cfg = self._configs.get(job_id)
                    if cfg is None or not cfg.enabled:
                        continue
                    run_id = str(uuid.uuid4())
                    idem_key = f"{job_id}:{scheduled_ts:.0f}"
                    if self.db.create_run(
                        run_id, job_id, cfg.executor.type, scheduled_ts, idem_key
                    ):
                        await self._enqueue(RunRequest(job_id, scheduled_ts, run_id))
                        # Advance next_due immediately to prevent re-firing
                        nxt = _next_due(cfg.schedule)
                        self.db.upsert_job(job_id, _config_hash(cfg), nxt)
            except Exception as e:
                log.error("tick_loop error: %s", e)
            await asyncio.sleep(1.0)

    async def _enqueue(self, req: RunRequest) -> None:
        try:
            self._run_queue.put_nowait(req)
        except asyncio.QueueFull:
            log.warning("scheduler: run queue full, dropping %s", req.job_id)

    # ── Worker loop ───────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        while self._active:
            try:
                req = await asyncio.wait_for(self._run_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            asyncio.create_task(self._run_job(req))

    async def _run_job(self, req: RunRequest) -> None:
        cfg = self._configs.get(req.job_id)
        if cfg is None:
            log.warning("scheduler: job %s not found", req.job_id)
            return

        log.info("scheduler: running %s (run_id=%s)", req.job_id, req.run_id)
        result = await _execute_job(cfg, req.run_id, self.bridge_tools, self._semaphore)

        # SLURM submitted — record jobid, don't finish yet
        if result.status == "slurm_submitted":
            slurm_jobid = str(result.metrics.get("slurm_jobid", ""))
            self.db.set_slurm_jobid(req.run_id, slurm_jobid)
            log.info("scheduler: SLURM job %s submitted for %s", slurm_jobid, req.job_id)
            return

        nxt = _next_due(cfg.schedule)
        self.db.finish_run(req.run_id, req.job_id, result, nxt)

        # Auto-disable after too many consecutive failures
        if result.status != "success":
            fc = self.db.get_failure_count(req.job_id)
            if fc >= MAX_AUTO_DISABLE_FAILURES:
                reason = f"auto-disabled after {fc} consecutive failures"
                self.db.disable(req.job_id, reason)
                log.warning("scheduler: %s — %s", req.job_id, reason)

        # Notify
        await self._notify(cfg, result, req.run_id)

        # Store in chitta
        _store_in_chitta(req.job_id, result, cfg.output.memory_tags, self.chitta_fn)

        # Chain
        if req.chain_depth < MAX_CHAIN_DEPTH:
            await self._eval_chains(cfg, result, req.chain_depth)

    async def _notify(self, cfg: JobConfig, result: JobResult, run_id: str) -> None:
        channels = (cfg.output.slack.success if result.status == "success"
                    else cfg.output.slack.failure)
        if not channels:
            return
        icon = "✅" if result.status == "success" else "❌"
        text = (f"{icon} *{cfg.id}* — `{result.status}`\n"
                f"{result.summary[:400]}")
        for ch in channels:
            await _slack_notify(ch, text, self.slack_fn)

    async def _eval_chains(self, cfg: JobConfig, result: JobResult,
                            depth: int) -> None:
        for rule in cfg.on_result:
            if not _eval_predicate(rule.when, result):
                continue
            target = self._configs.get(rule.trigger)
            if target is None:
                log.warning("chain: target job %s not found", rule.trigger)
                continue
            run_id = str(uuid.uuid4())
            idem_key = f"chain:{cfg.id}:{rule.trigger}:{time.time():.0f}"
            if self.db.create_run(run_id, rule.trigger, target.executor.type,
                                   time.time(), idem_key):
                await self._enqueue(RunRequest(rule.trigger, time.time(),
                                               run_id, depth + 1))
                log.info("scheduler: chain %s → %s (depth %d)",
                         cfg.id, rule.trigger, depth + 1)

    # ── SLURM reconcile ───────────────────────────────────────────────────────

    async def _slurm_loop(self) -> None:
        while self._active:
            await asyncio.sleep(SLURM_POLL_INTERVAL)
            try:
                await self._reconcile_slurm()
            except Exception as e:
                log.error("slurm_reconcile error: %s", e)

    async def _reconcile_slurm(self) -> None:
        running = self.db.get_running_slurm()
        if not running:
            return
        jobids = [r["slurm_jobid"] for r in running]
        try:
            proc = await asyncio.create_subprocess_exec(
                "sacct", "-j", ",".join(jobids),
                "--format=JobID,State,ExitCode", "--noheader", "--parsable2",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except Exception:
            return

        states: dict[str, str] = {}
        for line in stdout.decode().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2:
                states[parts[0]] = parts[1]

        for row in running:
            state = states.get(row["slurm_jobid"], "")
            if state in ("COMPLETED",):
                result = JobResult(status="success",
                                   summary=f"SLURM job {row['slurm_jobid']} completed")
            elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                result = JobResult(status="failure",
                                   summary=f"SLURM job {row['slurm_jobid']}: {state}")
            else:
                continue  # still running
            cfg = self._configs.get(row["job_id"])
            nxt = _next_due(cfg.schedule) if cfg else None
            self.db.finish_run(row["run_id"], row["job_id"], result, nxt)
            if cfg:
                await self._notify(cfg, result, row["run_id"])
                _store_in_chitta(row["job_id"], result,
                                  cfg.output.memory_tags, self.chitta_fn)

    # ── Manual control (for MCP tools) ────────────────────────────────────────

    async def run_now(self, job_id: str, dry_run: bool = False) -> str:
        cfg = self._configs.get(job_id)
        if cfg is None:
            return f"job '{job_id}' not found"
        if dry_run:
            chains = [f"  if ({r.when}) → trigger {r.trigger}"
                      for r in cfg.on_result]
            return (f"dry-run: {job_id}\n"
                    f"  executor: {cfg.executor.type}\n"
                    f"  prompt: {cfg.executor.prompt[:200]}\n"
                    f"  notify success: {cfg.output.slack.success}\n"
                    f"  notify failure: {cfg.output.slack.failure}\n"
                    + ("\n".join(chains) if chains else "  no chains"))
        run_id = str(uuid.uuid4())
        idem_key = f"manual:{job_id}:{time.time():.0f}"
        if self.db.create_run(run_id, job_id, cfg.executor.type, time.time(), idem_key):
            await self._enqueue(RunRequest(job_id, time.time(), run_id))
            return f"queued: {job_id} (run_id={run_id})"
        return f"could not queue {job_id} — already running?"

    def pause(self, job_id: str) -> str:
        self.db.disable(job_id, "manually paused")
        return f"paused: {job_id}"

    def resume(self, job_id: str) -> str:
        self.db.enable(job_id)
        cfg = self._configs.get(job_id)
        if cfg:
            nxt = _next_due(cfg.schedule)
            self.db.upsert_job(job_id, _config_hash(cfg), nxt)
        return f"resumed: {job_id}"

    def list_jobs(self) -> str:
        rows = self.db.list_all()
        if not rows:
            return "No jobs in scheduler.db. Add jobs to ~/.chitta-bridge/jobs.yaml"
        lines = []
        for r in rows:
            status = "⏸ paused" if not r["enabled"] else "✅ active"
            last = r.get("last_status") or "never run"
            nxt = (datetime.fromtimestamp(r["next_due"]).strftime("%Y-%m-%d %H:%M")
                   if r.get("next_due") else "—")
            cfg = self._configs.get(r["job_id"])
            desc = cfg.description if cfg else ""
            lines.append(
                f"**{r['job_id']}** {status} | next: {nxt} | last: {last} | "
                f"runs: {r.get('run_count',0)} | failures: {r['failure_count']}"
                + (f"\n  _{desc}_" if desc else "")
            )
            if r.get("disabled_reason"):
                lines.append(f"  ⚠️ {r['disabled_reason']}")
        return "\n".join(lines)

    def job_history(self, job_id: str) -> str:
        rows = self.db.history(job_id)
        if not rows:
            return f"No history for '{job_id}'"
        lines = [f"# History: {job_id}\n"]
        for r in rows:
            ts = datetime.fromtimestamp(r["started_at"]).strftime("%Y-%m-%d %H:%M:%S")
            dur = (f"{r['finished_at']-r['started_at']:.0f}s"
                   if r.get("finished_at") else "running")
            lines.append(f"[{ts}] {r['status']} ({dur}) — {(r.get('summary') or '')[:120]}")
        return "\n".join(lines)
