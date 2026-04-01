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

Async multi-agent roundtable with **agent souls** — participants get persistent identity, memory, tools, and structured challenge rounds.

### Basic Room

```python
room_create(
    room_id="my-room",
    topic="What's the best way to design a cache invalidation strategy?",
    participants='[
        {"name":"Codex","backend":"codex","session_id":"codex-1"},
        {"name":"Gemini","backend":"opencode","session_id":"gemini-1"},
        {"name":"Llama","backend":"local","model":"qwen2.5:32b","base_url":"http://gpunode:11434/v1"}
    ]'
)

room_run(room_id="my-room", rounds=2)
room_synthesize(room_id="my-room")
```

### Soul-Powered Room

Each participant can have a **soul** — a system prompt, memory namespace, tools, challenge bias, and response format:

```python
room_create(
    room_id="expert-panel",
    topic="How should we authenticate ancient DNA from permafrost?",
    participants='[
        {"name":"Paleogenomicist","backend":"local","model":"qwen2.5:32b",
         "base_url":"http://gpunode:11434/v1",
         "soul":{
           "system_prompt":"You are a senior paleogenomicist with 15+ years experience...",
           "realm":"agent:paleogenomicist",
           "tools":["recall","remember","web_search","smart_context"],
           "max_tool_turns":2,
           "challenge_bias":0.7,
           "response_format":"### Key Points\\n### Tools & Thresholds\\n### Caveats"
         }},
        {"name":"Bioinformatician","backend":"local","model":"phi4:14b",
         "base_url":"http://gpunode:11434/v1",
         "soul":{
           "system_prompt":"You are a computational biologist specializing in pipelines...",
           "realm":"agent:bioinformatician",
           "tools":["recall","remember","smart_context"],
           "challenge_bias":0.4
         }}
    ]'
)

# Challenge mode: between rounds, a moderator extracts claims and
# forces participants to disagree, provide evidence, and refine
room_run(room_id="expert-panel", rounds=2, challenge=true)
room_synthesize(room_id="expert-panel")
```

### Soul Features

| Feature | Description |
|---------|-------------|
| `system_prompt` | Agent identity, expertise, personality |
| `realm` | Chitta memory namespace — per-agent persistent memory |
| `tools` | Available tools (see Agent Tools below) |
| `max_tool_turns` | Max tool-use iterations per response (default 3) |
| `max_rounds` | Max discussion rounds, 0 = unlimited |
| `challenge_bias` | 0 = agreeable, 1 = devil's advocate |
| `response_format` | Structured output template |

### Challenge Rounds

When `challenge=true`, a moderator automatically:
1. Extracts substantive claims from the previous round
2. Injects a challenge prompt requiring each participant to disagree with at least one claim
3. Forces evidence-based refinement instead of polite agreement

### GPU Contention Handling

When multiple local models share the same GPU endpoint, rooms automatically run participants **sequentially** to avoid model-swap thrashing. Different endpoints run in parallel.

### Room Tools

| Tool | Description |
|------|-------------|
| `room_create` | Create a discussion room with named participants and optional souls |
| `room_add_participant` | Add a participant to an existing room |
| `room_run` | Run N rounds with optional challenge mode |
| `room_read` | Read the full transcript |
| `room_synthesize` | Distill the transcript — consensus, disagreements, best answer, open questions |

### Agent Tools

Tools available to soul-powered room participants via mediated XML tool calling. Assign a subset per agent via the `tools` field.

**Memory (core)**

| Tool | Description |
|------|-------------|
| `recall` | Semantic vector search over agent's memory realm |
| `remember` | Store an insight or fact in agent's memory realm |
| `smart_context` | Task-aware context assembly (memories + code symbols + graph) |

**Memory (extended)**

| Tool | Description |
|------|-------------|
| `recall_keyword` | BM25 keyword search — best when exact terms are known |
| `recall_temporal` | Search memories from a specific time range (since/until) |
| `hybrid_recall` | Combined vector + BM25 search — best general-purpose recall |
| `5w_search` | Structured who/what/when/where/why search |
| `forget` | Remove a memory by similarity match |

**Web**

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo search, returns titles + URLs + snippets |
| `web_fetch` | Fetch a URL as plain text (HTML stripped, max 8000 chars) |

**File operations**

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers (offset/limit, capped at 500 lines) |
| `write_file` | Create or overwrite a file (auto-creates parent dirs) |
| `edit_file` | Targeted string replacement with context display |
| `glob` | Find files by glob pattern, sorted by modification time |
| `grep` | Regex search over file contents with context lines |

**Shell**

| Tool | Description |
|------|-------------|
| `bash` | Execute a shell command (sandboxed, 60s timeout, dangerous commands blocked) |

**Code intelligence (via chitta)**

| Tool | Description |
|------|-------------|
| `read_function` | Read a function's source code by name |
| `read_symbol` | Look up any code symbol (class, function, variable) |
| `search_symbols` | Search for code symbols matching a query |
| `codebase_overview` | High-level overview of codebase structure |

**Task tracking**

| Tool | Description |
|------|-------------|
| `todo_add` | Add a task to the agent's personal todo list |
| `todo_list` | List current todo items |
| `todo_done` | Mark a todo item as complete |

### Synthesis

After running a room, distill the full discussion into a single answer. Any backend can act as synthesizer — Claude (default), local GPU model, OpenCode, or Codex.

```python
room_synthesize(room_id="my-room")

# Use a local model as synthesizer
room_synthesize(
    room_id="my-room",
    synthesizer='{"name":"Qwen3","backend":"local","model":"qwen3:30b-a3b","base_url":"http://gpunode:11434/v1"}'
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

Bidirectional memory bridge to the cc-soul daemon with **realm-scoped** memory. Each room participant can have its own memory namespace, and room discussions automatically pull relevant memories as context.

```python
# Check if soul is running
soul_status()

# Recall memories (global or realm-scoped)
soul_recall(query="cache invalidation strategies", limit=5)

# Store a memory
soul_remember(content="Room discussion concluded X is better than Y", kind="episode")

# Smart context (memories + code symbols + graph)
soul_context(task="refactor authentication middleware")
```

| Tool | Description |
|------|-------------|
| `soul_recall` | Search memories by query (supports realm scoping) |
| `soul_remember` | Store a new memory (supports realm scoping) |
| `soul_context` | Smart context assembly (memories + symbols + graph) |
| `soul_status` | Check if chittad is available |

Discussion rooms automatically:
- **Seed agent realms** on first turn — identity and topic stored for future recall
- **Inject soul context** at creation — participants see relevant memories (code symbols filtered)
- **Store contributions back** — each agent's response stored in their realm
- **Store synthesis back** — room conclusions become soul episodes
- **Hybrid recall** — vector + BM25 keyword matching for better memory retrieval

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
