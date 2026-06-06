---
description: Create multi-model discussion rooms where AI agents debate topics, then optionally execute the design via a Claude Code Workflow.
---

# Discussion Rooms

Multi-model rooms for design, planning, and adversarial review.

## Pattern 1 — Room only (discussion, no code change)

```
# 1. Create the room — use shorthands, not hardcoded versions
advanced(tool="room_create", arguments={
  topic: "...",
  participants: ["claude:<latest-opus>:xhigh", "codex:<latest-gpt>:xhigh"],
  rounds: 3
})

# 2. ALWAYS pass prompt= — room produces 0 responses without it
advanced(tool="room_run", arguments={
  room_id: "room-xxxxxxxx",
  prompt: "<full design brief here>",
  rounds: 3
})

# 3. Follow-up (same rule — prompt= required every time)
advanced(tool="room_run", arguments={
  room_id: "room-xxxxxxxx",
  prompt: "Synthesize into a concrete implementation spec.",
  rounds: 1
})
```

## Pattern 2 — Room designs → Workflow executes

Rooms **design**. Workflows **execute**. Never describe a workflow in prose — call `Workflow()`.

```
# 1. Room debate → get the design
room_create + room_run(prompt="<full brief>")

# 2. Synthesize room output into a Workflow JS script

# 3. Actually execute it — this is the step that makes changes
Workflow(script="""
  export const meta = { name: 'fix-xyz', description: '...' }
  phase('Implement')
  await agent('Apply the design: ...')
""")
```

## Participant shorthands — check live model list, don't hardcode versions

Use the shorthand format `backend:model:effort`. To see current models:

```
advanced(tool="list", action="list")
```

Format:
- `claude:<model-id>:xhigh` — Claude with extended thinking
- `codex:<model-id>:xhigh` — Codex/GPT with extended reasoning
- `codex:<model-id>:medium` — Codex medium effort

Effort levels: codex=low/medium/high/xhigh · claude=low/medium/xhigh/max
**NOTE: `high` is NOT valid for claude backends — use `xhigh`**

## CRITICAL: room_run always needs prompt=

```
# WRONG — produces 0 responses
room_run(room_id="room-xxx")

# CORRECT
room_run(room_id="room-xxx", prompt="Your full question here")
```
