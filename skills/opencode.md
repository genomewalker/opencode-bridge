# OpenCode Discussion

Collaborative discussion with OpenCode models (GPT-5, Claude, Gemini). Sessions persist across messages.

## Usage

```
/opencode [command] [args]
```

## Commands

| Command | Description |
|---------|-------------|
| `/opencode` | Start/continue session |
| `/opencode plan <task>` | Plan with plan agent |
| `/opencode ask <question>` | Ask anything |
| `/opencode review <file>` | Review code |
| `/opencode models` | List models |
| `/opencode model <name>` | Switch model |
| `/opencode agent <name>` | Switch agent |
| `/opencode config` | Show configuration |
| `/opencode set model <name>` | Set default model |
| `/opencode set agent <name>` | Set default agent |
| `/opencode end` | End session |

## Instructions

### Starting a Session

When user says `/opencode` or `/opencode start`:
1. Call `opencode_start(session_id="discuss-{timestamp}")`
2. Report: "Connected to OpenCode ({model}). Ready."

### Planning

When user says `/opencode plan <task>`:
1. Call `opencode_plan(task=<task>)`
2. Relay the response

### Asking

When user says `/opencode ask <question>`:
1. Call `opencode_discuss(message=<question>)`
2. Relay the response

### Code Review

When user says `/opencode review <file>`:
1. Call `opencode_review(code_or_file=<file>)`
2. Relay the findings

### Configuration

When user says `/opencode config`:
1. Call `opencode_config()`
2. Show current model and agent

When user says `/opencode set model <name>`:
1. Call `opencode_configure(model=<name>)`
2. Confirm the change

When user says `/opencode set agent <name>`:
1. Call `opencode_configure(agent=<name>)`
2. Confirm the change

### Follow-ups

After initial connection, messages like these should be sent as follow-ups:
- "what do you think about..."
- "how would you implement..."
- "can you explain..."

Call `opencode_discuss(message=<user message>)` and relay response.

### Session Management

- `/opencode models` → `opencode_models()`
- `/opencode model <name>` → `opencode_model(model=<name>)`
- `/opencode agent <name>` → `opencode_agent(agent=<name>)`
- `/opencode end` → `opencode_end()`

## Example Flow

```
User: /opencode
Claude: Connected to OpenCode (openai/gpt-5.2-codex, plan agent). Ready.

User: Let's plan an RLM-inspired hierarchical retrieval system
Claude: [calls opencode_plan, relays response]

User: What about the filtering stage?
Claude: [calls opencode_discuss, relays response]

User: /opencode model github-copilot/claude-opus-4.5
Claude: Model changed to github-copilot/claude-opus-4.5

User: /opencode set model openai/gpt-5.2-codex
Claude: Default model set to openai/gpt-5.2-codex (persisted)

User: /opencode end
Claude: Session ended.
```

## Available Models

Popular models:
- `openai/gpt-5.2-codex` - Best for code
- `openai/gpt-5.1-codex-max` - Longer context
- `github-copilot/claude-opus-4.5` - Claude
- `github-copilot/gpt-5.2` - GPT-5.2

Use `/opencode models` for full list.

## Agents

- `plan` - Planning mode (default)
- `build` - Implementation mode
- `explore` - Exploration/research
- `general` - General purpose
