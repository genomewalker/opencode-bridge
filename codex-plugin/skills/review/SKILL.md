---
description: Run Codex code review via chitta-bridge. Supports normal and adversarial mode, branch comparison, background execution.
---

# Code Review

Run a code review on the current repository using chitta-bridge's enhanced review.

## Modes

- **Normal**: Standard code review (same as `codex exec review`)
- **Adversarial**: Challenges design decisions, architecture, tradeoffs, failure modes, and hidden assumptions. Does not praise good code — focuses exclusively on risks.

## Usage

Review current changes:
```
Use mcp__chitta_bridge__codex_review
```

Adversarial review with focus:
```
Use mcp__chitta_bridge__codex_review with mode="adversarial" and focus="race conditions and data loss"
```

Branch review vs main:
```
Use mcp__chitta_bridge__codex_review with base="main"
```

Background review (returns job ID):
```
Use mcp__chitta_bridge__codex_review with background=true
```

Then check with `mcp__chitta_bridge__codex_job_status` and `mcp__chitta_bridge__codex_job_result`.
