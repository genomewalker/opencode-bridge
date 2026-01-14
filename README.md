# OpenCode Bridge

MCP server for continuous discussion sessions with OpenCode. Collaborate with GPT-5, Claude, Gemini, and other models through Claude Code.

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

- **Continuous sessions**: Conversation history persists across messages
- **Multiple models**: Access all OpenCode models (GPT-5.x, Claude, Gemini, etc.)
- **Agent support**: plan, build, explore, general agents
- **Variant control**: Set reasoning effort (minimal → max)
- **File attachment**: Share code files for review
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

## Available Models

| Provider | Models |
|----------|--------|
| openai | gpt-5.2-codex, gpt-5.1-codex-max, gpt-5.1-codex-mini |
| github-copilot | claude-opus-4.5, claude-sonnet-4.5, gpt-5, gemini-2.5-pro |
| opencode | gpt-5-nano (free), glm-4.7-free, grok-code |

Run `opencode models` to see all available models.

## MCP Tools

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

## Configuration

### Environment variables

```bash
export OPENCODE_MODEL="openai/gpt-5.2-codex"
export OPENCODE_AGENT="plan"
export OPENCODE_VARIANT="medium"
```

### Config file

`~/.opencode-bridge/config.json`:
```json
{
  "model": "openai/gpt-5.2-codex",
  "agent": "plan",
  "variant": "medium"
}
```

### Variants (reasoning effort)

`minimal` → `low` → `medium` → `high` → `xhigh` → `max`

Higher variants use more reasoning tokens for complex tasks.

## Requirements

- Python 3.10+
- [OpenCode CLI](https://opencode.ai) installed
- Claude Code

## License

MIT
