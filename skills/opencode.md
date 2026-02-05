# OpenCode Discussion

Collaborative discussion with OpenCode models (GPT-5, Claude, Gemini). Sessions persist across messages. Auto-detects discussion domains and frames OpenCode as a specialized expert.

## Usage

```
/opencode [command] [args]
```

## Commands

| Command | Description |
|---------|-------------|
| `/opencode` | Start/continue session |
| `/opencode plan <task>` | Plan with plan agent |
| `/opencode ask <question>` | Ask anything (auto-detects domain) |
| `/opencode review <file>` | Review code |
| `/opencode models` | List models |
| `/opencode model <name>` | Switch model |
| `/opencode agent <name>` | Switch agent |
| `/opencode config` | Show configuration |
| `/opencode set model <name>` | Set default model |
| `/opencode set agent <name>` | Set default agent |
| `/opencode end` | End session |

## Domain Detection

When you send a message via `opencode_discuss`, the system auto-detects the discussion domain and frames OpenCode as a specialized expert:

| Domain | Example triggers |
|--------|-----------------|
| Architecture | "microservice", "system design", "event driven" |
| Debugging | "bug", "root cause", "stack trace" |
| Performance | "optimize", "bottleneck", "cache miss" |
| Security | "vulnerability", "SQL injection", "OWASP" |
| Testing | "unit test", "coverage", "TDD" |
| DevOps | "deploy", "CI/CD pipeline", "kubernetes" |
| Database | "schema", "query optimization", "migration" |
| API Design | "REST API", "versioning", "endpoint" |
| Frontend | "React", "component", "SSR", "accessibility" |
| Algorithms | "dynamic programming", "time complexity" |
| Code Quality | "refactor", "SOLID", "technical debt" |
| Planning | "roadmap", "MVP", "user story" |
| General | fallback when nothing else matches |

Override detection with `domain` parameter: `opencode_discuss(message="...", domain="security")`.

The response includes the detected domain and confidence: `[Domain: Architecture] [Confidence: 92%]`.

Follow-up messages in an existing session get a lighter prompt that preserves the collaborative framing without repeating the full setup.

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
2. Note the detected domain in your relay
3. Relay the response

To force a specific domain: `opencode_discuss(message=<question>, domain="security")`

### Code Review

When user says `/opencode review <file>`:
1. Call `opencode_review(code_or_file=<file>)`
2. Relay the findings

Note: Code review bypasses the companion system and uses specialized review prompts.

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

Call `opencode_discuss(message=<user message>)` and relay response. Follow-ups automatically get a lighter prompt.

### Session Management

- `/opencode models` → `opencode_models()`
- `/opencode model <name>` → `opencode_model(model=<name>)`
- `/opencode agent <name>` → `opencode_agent(agent=<name>)`
- `/opencode end` → `opencode_end()`

## Example Flow

```
User: /opencode
Claude: Connected to OpenCode (openai/gpt-5.2-codex, plan agent). Ready.

User: Should we use event sourcing for our order system?
Claude: [calls opencode_discuss]
       [Domain: Architecture & System Design] [Confidence: 92%]
       [OpenCode responds as a distributed systems architect]

User: What about the security implications?
Claude: [calls opencode_discuss — follow-up, lighter prompt]
       [Domain: Security & Threat Modeling] [Confidence: 76%]

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
