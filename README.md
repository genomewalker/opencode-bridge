# Chitta Bridge

MCP server for multi-model AI discussions — connect Claude Code to any AI backend: cloud providers, agentic CLIs, and local GPU models.

## Quick Start

```bash
# 1. Install
uv pip install git+https://github.com/genomewalker/chitta-bridge.git

# 2. Register with Claude Code
chitta-bridge-install

# 3. Use in Claude Code
# The tools are now available - Claude will use them automatically
```

## Features

- **Multiple backends**: OpenCode, Codex CLI, and local GPU models (Ollama/vLLM)
- **Continuous sessions**: Conversation history persists across messages
- **Session warmup**: background ping captures session ID — subsequent calls skip cold start
- **Multiple models**: OpenCode (GPT-5.x, Claude, Gemini) + Codex (o3, o4-mini, gpt-4.1)
- **Agent support**: plan, build, explore, general agents (OpenCode)
- **Agentic execution**: Full-auto mode with sandboxed file operations (Codex)
- **Variant control**: Set reasoning effort (minimal to max)
- **File/image attachment**: Share code files and images for context
- **Session continuity**: Conversations continue across tool calls
- **Discussion rooms**: async multi-agent roundtables — any mix of backends respond in parallel, see the full thread, synthesize into one answer

## Installation

### With uv (recommended)

```bash
uv pip install git+https://github.com/genomewalker/chitta-bridge.git
```

### With pip

```bash
pip install git+https://github.com/genomewalker/chitta-bridge.git
```

### From source

```bash
git clone https://github.com/genomewalker/chitta-bridge.git
cd chitta-bridge
pip install -e .
```

## Register with Claude Code

```bash
# Install (registers MCP server)
chitta-bridge-install

# Verify
claude mcp list

# Uninstall
chitta-bridge-uninstall
```

## OpenCode Backend

| Tool | Description |
|------|-------------|
| `opencode_start` | Start a new session (auto-warms up, captures session ID) |
| `opencode_discuss` | Send a message |
| `opencode_plan` | Start planning discussion |
| `opencode_brainstorm` | Open-ended brainstorming |
| `opencode_review` | Review code |
| `opencode_ping` | Check if model is reachable |
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

## Discussion Rooms

Async multi-agent roundtable — any mix of backends (OpenCode, Codex, local GPU models, Claude) post in parallel each round, each seeing the full thread before responding.

```python
# Create a room with 3 participants (including a local GPU model)
room_create(
    room_id="my-room",
    topic="What's the best way to design a cache invalidation strategy?",
    participants='[
        {"name":"Codex-GPT54","backend":"codex","session_id":"existing-codex-session"},
        {"name":"Gemini","backend":"opencode","session_id":"existing-gemini-session"},
        {"name":"Llama","backend":"local","model":"llama3.3:70b","base_url":"http://gpunode01:11434/v1"}
    ]'
)

# Run 2 rounds (all participants respond in parallel each round)
room_run(room_id="my-room", rounds=2)

# Add a participant mid-discussion
room_add_participant(room_id="my-room", participant='{"name":"Claude","backend":"claude"}')

# Read the full transcript
room_read(room_id="my-room")
```

| Tool | Description |
|------|-------------|
| `room_create` | Create a discussion room with named participants |
| `room_add_participant` | Add a participant to an existing room |
| `room_run` | Run N rounds — all participants respond in parallel |
| `room_read` | Read the full transcript |
| `room_synthesize` | Distill the transcript — consensus, disagreements, best answer, open questions |

### Synthesis

After running a room, distill the full discussion into a single answer. Any backend can act as synthesizer — Claude (default), local GPU model, OpenCode, or Codex.

```python
room_synthesize(room_id="my-room")

# Use a local model as synthesizer
room_synthesize(
    room_id="my-room",
    synthesizer='{"name":"Qwen3","backend":"local","model":"qwen3:30b-a3b","base_url":"http://gpunode01:11434/v1"}'
)
```

## Local Models (GPU Nodes)

Chat with local LLMs (Ollama / vLLM) running on GPU nodes — via Slurm auto-discovery or direct hostname.

```bash
# 1. Start Ollama on a Slurm GPU node (writes URL to /tmp/ollama-server-<model>.url)
slurm-serve-ollama.sh llama3.3:70b

# 2. Discover available nodes and models
local_discover()

# 3. Start a session (auto-discovers endpoint if omitted)
local_start(session_id="llm1", model="llama3.3:70b")

# 4. Chat
local_discuss(message="Explain cache invalidation strategies")

# Or specify node explicitly
local_start(session_id="llm2", model="qwen3:30b-a3b", endpoint="http://gpunode01:11434/v1")
```

### Discovery order

1. `/tmp/ollama-server-*.url` cache files (written by `slurm-serve-ollama.sh`)
2. Your running Slurm GPU jobs (`squeue --me`)
3. `CHITTA_BRIDGE_GPU_NODES=node1,node2` environment variable
4. `localhost:11434` fallback

| Tool | Description |
|------|-------------|
| `local_discover` | Find GPU nodes with Ollama/vLLM running |
| `local_start` | Start a session (auto-discovers endpoint) |
| `local_discuss` | Chat with the local model |
| `local_models` | List models available at an endpoint |
| `local_sessions` | List active local sessions |
| `local_switch` | Switch active session |
| `local_end` | End a session |
| `local_history` | Show conversation history |
| `local_health` | Health check |

## Web Search

Search the web and fetch pages directly from Claude Code — no API key needed (DuckDuckGo).

```python
# Search
web_search(query="ancient metagenomics DNA damage authentication")

# Fetch a page
web_fetch(url="https://example.com/article", max_chars=12000)
```

| Tool | Description |
|------|-------------|
| `web_search` | Search via DuckDuckGo — returns titles, URLs, snippets |
| `web_fetch` | Fetch a web page as plain text (HTML stripped) |

## Soul Memory (chittad)

Bidirectional memory bridge to the cc-soul daemon. Room discussions automatically pull relevant memories as context, and syntheses are stored back as episodes.

```python
# Check if soul is running
soul_status()

# Recall memories
soul_recall(query="cache invalidation strategies", limit=5)

# Store a memory
soul_remember(content="Room discussion concluded X is better than Y", kind="episode")

# Smart context (memories + code symbols + graph)
soul_context(task="refactor authentication middleware")
```

| Tool | Description |
|------|-------------|
| `soul_recall` | Search memories by query |
| `soul_remember` | Store a new memory |
| `soul_context` | Smart context assembly (memories + symbols + graph) |
| `soul_status` | Check if chittad is available |

Discussion rooms automatically:
- **Inject soul context** at creation — participants see relevant memories
- **Store synthesis back** — room conclusions become soul episodes

## Codex Backend

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

`~/.chitta-bridge/config.json`:
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
- [OpenCode CLI](https://opencode.ai) for `opencode_*` tools
- [Codex CLI](https://github.com/openai/codex) for `codex_*` tools
- Ollama or vLLM on a GPU node for `local_*` tools
- Claude Code

## License

MIT
