---
description: Create multi-model discussion rooms where AI agents debate topics, then optionally execute the design via a Claude Code Workflow.
---

# Discussion Rooms

Multi-model rooms for design, planning, and adversarial review. Two usage patterns:

## Pattern 1 — Room only (discussion, no code change)

```
# 1. Create the room
advanced(tool="room_create", arguments={
  topic: "...",
  participants: ["claude:claude-opus-4-8:xhigh", "codex:gpt-5.5:xhigh"],
  rounds: 3
})

# 2. Run with the full brief in prompt= — REQUIRED, room produces 0 responses without it
advanced(tool="room_run", arguments={
  room_id: "room-xxxxxxxx",
  prompt: "<full design brief here>",
  rounds: 3
})

# 3. Follow-up (inject into live room)
advanced(tool="room_run", arguments={
  room_id: "room-xxxxxxxx",
  prompt: "Synthesize into a concrete implementation spec.",
  rounds: 1
})
```

## Pattern 2 — Room designs → Workflow executes (the correct pattern for code changes)

```
# 1. Create room + run debate to get the design
advanced(tool="room_create", ...) → advanced(tool="room_run", prompt="...")

# 2. Read the output, synthesize into a Workflow JS script

# 3. Execute the Workflow — this is what actually makes the changes
Workflow(script="""
  export const meta = { name: 'fix-xyz', description: '...' }
  phase('Implement')
  await agent('Apply the design: ...', { label: 'impl' })
""")
```

The room **designs**. The Workflow **executes**. Never describe the workflow in prose — actually call `Workflow()`.

## CRITICAL: room_run always needs prompt=

```
# WRONG — produces 0 responses
room_run(room_id="room-xxx")

# CORRECT — prompt= injects the message and triggers inference
room_run(room_id="room-xxx", prompt="Your full question here")
```

## Participant shorthands

```
claude:claude-opus-4-8:xhigh   — Opus 4.8 extended thinking
claude:claude-opus-4-7:xhigh   — Opus 4.7 extended thinking
codex:gpt-5.5:xhigh            — GPT-5.5 extended reasoning
codex:gpt-5.5:medium           — GPT-5.5 medium effort
```

Effort: codex=low/medium/high/xhigh · claude=low/medium/xhigh/max (NOT 'high' for claude)

## Effort guidelines

- `xhigh` for architecture, gap analysis, adversarial design review
- `medium` for follow-up synthesis, terse spec extraction
- Never use `high` for claude backends — invalid, use `xhigh`
