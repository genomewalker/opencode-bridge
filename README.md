# OpenCode Bridge

MCP server for continuous discussion sessions with OpenCode and Codex. Collaborate with GPT-5, Claude, Gemini, o3, and other models through Claude Code.

## Quick Start

```bash
# 1. Install
uv pip install git+https://github.com/genomewalker/opencode-bridge.git

# 2. Register with Claude Code
opencode-bridge-install

# 3. Use in Claude Code
# The tools are now available - Claude will use them automatically
```

## Features

- **Dual backend support**: OpenCode and Codex CLI
- **Continuous sessions**: Conversation history persists across messages
- **Multiple models**: OpenCode (GPT-5.x, Claude, Gemini) + Codex (o3, o4-mini, gpt-4.1)
- **Agent support**: plan, build, explore, general agents (OpenCode)
- **Agentic execution**: Full-auto mode with sandboxed file operations (Codex)
- **Variant control**: Set reasoning effort (minimal to max)
- **File/image attachment**: Share code files and images for context
- **Session continuity**: Conversations continue across tool calls

## Installation

### With uv (recommended)

```bash
uv pip install git+https://github.com/genomewalker/opencode-bridge.git
```

### With pip

```bash
pip install git+https://github.com/genomewalker/opencode-bridge.git
```

### From source

```bash
git clone https://github.com/genomewalker/opencode-bridge.git
cd opencode-bridge
pip install -e .
```

## Register with Claude Code

```bash
# Install (registers MCP server)
opencode-bridge-install

# Verify
claude mcp list

# Uninstall
opencode-bridge-uninstall
```

## OpenCode Tools

| Tool | Description |
|------|-------------|
| `opencode_start` | Start a new session |
| `opencode_discuss` | Send a message |
| `opencode_plan` | Start planning discussion |
| `opencode_brainstorm` | Open-ended brainstorming |
| `opencode_review` | Review code |
| `opencode_models` | List available models |
| `opencode_agents` | List available agents |
| `opencode_model` | Change session model |
| `opencode_agent` | Change session agent |
| `opencode_variant` | Change reasoning effort |
| `opencode_config` | Show current configuration |
| `opencode_configure` | Set defaults (persisted) |
| `opencode_history` | Show conversation history |
| `opencode_sessions` | List all sessions |
| `opencode_switch` | Switch to another session |
| `opencode_end` | End current session |
| `opencode_health` | Server health check |

## Codex Tools

| Tool | Description |
|------|-------------|
| `codex_start` | Start a new Codex session |
| `codex_discuss` | Send a message to Codex |
| `codex_run` | Run a one-off task (stateless) |
| `codex_review` | Run code review on repository |
| `codex_model` | Change session model |
| `codex_config` | Show Codex configuration |
| `codex_configure` | Set Codex defaults (persisted) |
| `codex_history` | Show conversation history |
| `codex_sessions` | List all Codex sessions |
| `codex_switch` | Switch to another session |
| `codex_end` | End current session |
| `codex_health` | Codex health check |

## Available Models

### OpenCode

| Provider | Models |
|----------|--------|
| openai | gpt-5.2-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini |
| github-copilot | claude-opus-4.5, claude-sonnet-4.5, gpt-5, gemini-2.5-pro |
| opencode | gpt-5-nano (free), glm-4.7-free, grok-code |

Run `opencode models` to see all available models.

### Codex

| Model | Description |
|-------|-------------|
| o3 | Default, high capability |
| o4-mini | Faster, lower cost |
| gpt-4.1 | Alternative option |

## Configuration

### Environment variables

```bash
# OpenCode
export OPENCODE_MODEL="openai/gpt-5.2-codex"
export OPENCODE_AGENT="plan"
export OPENCODE_VARIANT="medium"

# Codex
export CODEX_MODEL="o3"
export CODEX_SANDBOX="workspace-write"
```

### Config file

`~/.opencode-bridge/config.json`:
```json
{
  "model": "openai/gpt-5.2-codex",
  "agent": "plan",
  "variant": "medium",
  "codex_model": "o3",
  "codex_sandbox": "workspace-write"
}
```

### OpenCode Variants (reasoning effort)

`minimal` -> `low` -> `medium` -> `high` -> `xhigh` -> `max`

Higher variants use more reasoning tokens for complex tasks.

### Codex Sandbox Modes

| Mode | Description |
|------|-------------|
| `read-only` | Can only read files |
| `workspace-write` | Can write to workspace (default) |
| `danger-full-access` | Full filesystem access (use with caution) |

The `full_auto` option (default: true) enables low-friction execution with `workspace-write` sandbox.

## Requirements

- Python 3.10+
- [OpenCode CLI](https://opencode.ai) installed (for opencode_* tools)
- [Codex CLI](https://github.com/openai/codex) installed (for codex_* tools)
- Claude Code

## License

MIT
