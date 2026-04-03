---
description: Create multi-model discussion rooms where AI agents debate topics, with optional soul context and challenge rounds.
---

# Discussion Rooms

Create rooms where multiple AI models discuss a topic in parallel rounds. Each participant can have a soul (system prompt, tools, memory namespace).

## Usage

Create a room:
```
Use mcp__chitta_bridge__room_create with room_id="design-review" topic="Review the caching strategy" and participants (JSON array of name/backend/model/soul objects)
```

Run discussion rounds:
```
Use mcp__chitta_bridge__room_run with room_id="design-review" rounds=2 challenge=true
```

Read transcript:
```
Use mcp__chitta_bridge__room_read with room_id="design-review"
```

## Participant Backends

- `"backend": "opencode"` — cloud models via OpenCode (GPT, Claude, Gemini)
- `"backend": "local"` — local models via Ollama (gemma4:26b, qwen3:30b-a3b, etc.)
- `"backend": "codex"` — Codex models

## Challenge Mode

Set `challenge=true` to auto-extract claims between rounds and inject challenge prompts. Participants must disagree with at least one claim and provide evidence.
