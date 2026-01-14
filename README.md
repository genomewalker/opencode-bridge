# OpenCode Bridge

MCP server for continuous discussion sessions with OpenCode. Collaborate with GPT-5, Claude, Gemini, and other models through Claude Code.

## Features

- **Continuous sessions**: Conversation history persists across messages
- **Multiple models**: Access all OpenCode models (GPT-5.x, Claude Opus 4.5, Gemini 3, etc.)
- **Agent support**: Use plan, build, explore, or general agents
- **File attachment**: Share code files for review
- **Persistent config**: Set your preferred defaults

## Installation

```bash
# With uv (recommended)
uv pip install git+https://github.com/genomewalker/opencode-bridge.git

# With pip
pip install git+https://github.com/genomewalker/opencode-bridge.git

# From source
cd opencode-bridge
pip install -e .
```

## Configuration

### Set default model

```bash
# Via environment variable
export OPENCODE_MODEL="openai/gpt-5.2-codex"
export OPENCODE_AGENT="plan"

# Via config file (~/.opencode-bridge/config.json)
{
  "model": "openai/gpt-5.2-codex",
  "agent": "plan"
}
```

### Available models

```
openai/gpt-5.2-codex          # Best for code
openai/gpt-5.1-codex-max      # Longer context
github-copilot/claude-opus-4.5 # Claude via Copilot
github-copilot/gpt-5.2        # GPT-5.2 via Copilot
```

Run `opencode models` to see all available models.

## Usage with Claude Code

### Register as MCP server

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "opencode-bridge": {
      "command": "opencode-bridge"
    }
  }
}
```

### Use the skill

```
/opencode                     Start a session
/opencode plan <task>         Plan something
/opencode ask <question>      Ask anything
/opencode review <file>       Review code
/opencode model <name>        Switch model
/opencode end                 End session
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `opencode_models` | List available models |
| `opencode_agents` | List available agents |
| `opencode_start` | Start a new session |
| `opencode_discuss` | Send a message |
| `opencode_plan` | Start planning with plan agent |
| `opencode_brainstorm` | Open-ended brainstorming |
| `opencode_review` | Review code |
| `opencode_model` | Change model for session |
| `opencode_agent` | Change agent for session |
| `opencode_config` | Show current configuration |
| `opencode_configure` | Set default model/agent |
| `opencode_history` | Show conversation history |
| `opencode_sessions` | List all sessions |
| `opencode_switch` | Switch to another session |
| `opencode_end` | End current session |

## Requirements

- Python 3.10+
- OpenCode CLI installed (`~/.opencode/bin/opencode`)
- mcp>=1.0.0

## License

MIT
