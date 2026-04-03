---
description: Delegate a task to a background Codex job with rescue, resume, effort control, and job tracking.
---

# Rescue

Delegate a task to Codex running in the background. Supports session resume, effort control, and full job lifecycle.

## Usage

Start a background rescue:
```
Use mcp__chitta_bridge__codex_rescue with task="investigate why the tests started failing"
```

With model and effort:
```
Use mcp__chitta_bridge__codex_rescue with task="fix the flaky integration test" model="gpt-5.4-mini" effort="high"
```

Resume a previous session:
```
Use mcp__chitta_bridge__codex_rescue with task="apply the top fix from the last run" resume_from="SESSION_ID"
```

Start fresh (ignore previous sessions):
```
Use mcp__chitta_bridge__codex_rescue with task="redesign the database connection" fresh=true
```

## Job Management

- `mcp__chitta_bridge__codex_job_status` — check progress of all or a specific job
- `mcp__chitta_bridge__codex_job_result` — get final output + Codex session ID for `codex resume`
- `mcp__chitta_bridge__codex_job_cancel` — cancel a running job
