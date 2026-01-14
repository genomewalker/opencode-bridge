#!/usr/bin/env python3
"""
OpenCode Bridge - MCP server for continuous OpenCode sessions.

Features:
- Continuous discussion sessions with conversation history
- Access to all OpenCode models (GPT-5, Claude, Gemini, etc.)
- Agent support (plan, build, explore, general)
- Session continuation
- File attachment for code review

Configuration:
- OPENCODE_MODEL: Default model (e.g., openai/gpt-5.2-codex)
- OPENCODE_AGENT: Default agent (plan, build, explore, general)
- ~/.opencode-bridge/config.json: Persistent config
"""

import os
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Default configuration
DEFAULT_MODEL = "openai/gpt-5.2-codex"
DEFAULT_AGENT = "plan"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    agent: str = DEFAULT_AGENT

    @classmethod
    def load(cls) -> "Config":
        config = cls()

        # Load from config file
        config_path = Path.home() / ".opencode-bridge" / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                config.model = data.get("model", config.model)
                config.agent = data.get("agent", config.agent)
            except Exception:
                pass

        # Environment variables override config file
        config.model = os.environ.get("OPENCODE_MODEL", config.model)
        config.agent = os.environ.get("OPENCODE_AGENT", config.agent)

        return config

    def save(self):
        config_dir = Path.home() / ".opencode-bridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps({
            "model": self.model,
            "agent": self.agent
        }, indent=2))


def find_opencode() -> Optional[Path]:
    """Find opencode binary."""
    # Check common locations
    paths = [
        Path.home() / ".opencode" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
        Path("/usr/bin/opencode"),
    ]
    for p in paths:
        if p.exists():
            return p
    # Check PATH
    which = shutil.which("opencode")
    if which:
        return Path(which)
    return None


OPENCODE_BIN = find_opencode()


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Session:
    id: str
    model: str
    agent: str
    opencode_session_id: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str):
        self.messages.append(Message(role=role, content=content))

    def save(self, path: Path):
        data = {
            "id": self.id,
            "model": self.model,
            "agent": self.agent,
            "opencode_session_id": self.opencode_session_id,
            "created": self.created,
            "messages": [asdict(m) for m in self.messages]
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Session":
        data = json.loads(path.read_text())
        session = cls(
            id=data["id"],
            model=data["model"],
            agent=data.get("agent", "build"),
            opencode_session_id=data.get("opencode_session_id"),
            created=data.get("created", datetime.now().isoformat())
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


class OpenCodeBridge:
    def __init__(self):
        self.config = Config.load()
        self.sessions: dict[str, Session] = {}
        self.active_session: Optional[str] = None
        self.sessions_dir = Path.home() / ".opencode-bridge" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.available_models: list[str] = []
        self.available_agents: list[str] = []
        self._load_sessions()

    def _load_sessions(self):
        for path in self.sessions_dir.glob("*.json"):
            try:
                session = Session.load(path)
                self.sessions[session.id] = session
            except Exception:
                pass

    def _run_opencode(self, *args, timeout: int = 300) -> tuple[str, int]:
        """Run opencode CLI command and return output."""
        if not OPENCODE_BIN:
            return "OpenCode not installed. Install from: https://opencode.ai", 1

        cmd = [str(OPENCODE_BIN)] + list(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout or result.stderr
            return output.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "Command timed out", 1
        except Exception as e:
            return f"Error: {e}", 1

    async def list_models(self, provider: Optional[str] = None) -> str:
        """List available models from OpenCode."""
        args = ["models"]
        if provider:
            args.append(provider)

        output, code = self._run_opencode(*args)
        if code != 0:
            return f"Error listing models: {output}"

        self.available_models = [line.strip() for line in output.split("\n") if line.strip()]

        # Group by provider
        providers: dict[str, list[str]] = {}
        for model in self.available_models:
            if "/" in model:
                prov, name = model.split("/", 1)
            else:
                prov, name = "other", model
            providers.setdefault(prov, []).append(name)

        lines = ["Available models:"]
        for prov in sorted(providers.keys()):
            lines.append(f"\n**{prov}:**")
            for name in sorted(providers[prov]):
                full = f"{prov}/{name}"
                lines.append(f"  - {full}")

        return "\n".join(lines)

    async def list_agents(self) -> str:
        """List available agents from OpenCode."""
        output, code = self._run_opencode("agent", "list")
        if code != 0:
            return f"Error listing agents: {output}"

        # Parse agent names from output
        agents = []
        for line in output.split("\n"):
            line = line.strip()
            if line and "(" in line:
                name = line.split("(")[0].strip()
                agents.append(name)

        self.available_agents = agents
        return "Available agents:\n" + "\n".join(f"  - {a}" for a in agents)

    async def start_session(
        self,
        session_id: str,
        model: Optional[str] = None,
        agent: Optional[str] = None
    ) -> str:
        # Use config defaults if not specified
        model = model or self.config.model
        agent = agent or self.config.agent

        session = Session(
            id=session_id,
            model=model,
            agent=agent
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        return f"Session '{session_id}' started\n  Model: {model}\n  Agent: {agent}"

    def get_config(self) -> str:
        """Get current configuration."""
        return f"""Current configuration:
  Model: {self.config.model}
  Agent: {self.config.agent}

Set via:
  - ~/.opencode-bridge/config.json
  - OPENCODE_MODEL and OPENCODE_AGENT env vars
  - opencode_configure tool"""

    def set_config(self, model: Optional[str] = None, agent: Optional[str] = None) -> str:
        """Update and persist configuration."""
        changes = []
        if model:
            self.config.model = model
            changes.append(f"model: {model}")
        if agent:
            self.config.agent = agent
            changes.append(f"agent: {agent}")

        if changes:
            self.config.save()
            return f"Configuration updated:\n  " + "\n  ".join(changes)
        return "No changes made."

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None
    ) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session. Use opencode_start first."

        session = self.sessions[sid]
        session.add_message("user", message)

        # Build command
        args = ["run"]
        args.extend(["--model", session.model])
        args.extend(["--agent", session.agent])

        # Continue session if we have an opencode session ID
        if session.opencode_session_id:
            args.extend(["--session", session.opencode_session_id])

        # Attach files
        if files:
            for f in files:
                args.extend(["--file", f])

        # Add the message
        args.append(message)

        output, code = self._run_opencode(*args)

        if code != 0:
            return f"Error: {output}"

        # Try to extract session ID from output for continuation
        # OpenCode outputs session info that we can parse
        reply = output.strip()
        if reply:
            session.add_message("assistant", reply)
            session.save(self.sessions_dir / f"{sid}.json")

        return reply or "No response received"

    async def plan(
        self,
        task: str,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None
    ) -> str:
        """Start a planning discussion using the plan agent."""
        sid = session_id or self.active_session

        # If no active session, create one for planning
        if not sid or sid not in self.sessions:
            sid = f"plan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="plan")

        # Switch to plan agent if not already
        session = self.sessions[sid]
        if session.agent != "plan":
            session.agent = "plan"
            session.save(self.sessions_dir / f"{sid}.json")

        return await self.send_message(task, sid, files)

    async def brainstorm(
        self,
        topic: str,
        session_id: Optional[str] = None
    ) -> str:
        """Open-ended brainstorming discussion."""
        sid = session_id or self.active_session

        if not sid or sid not in self.sessions:
            sid = f"brainstorm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="build")

        prompt = f"""Let's brainstorm about: {topic}

Please provide:
1. Key considerations and trade-offs
2. Multiple approaches or solutions
3. Pros and cons of each approach
4. Your recommended approach and why"""

        return await self.send_message(prompt, sid)

    async def review_code(
        self,
        code_or_file: str,
        focus: str = "correctness, efficiency, and potential bugs",
        session_id: Optional[str] = None
    ) -> str:
        """Review code for issues and improvements."""
        sid = session_id or self.active_session

        if not sid or sid not in self.sessions:
            sid = f"review-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="build")

        # Check if it's a file path
        files = None
        if Path(code_or_file).exists():
            files = [code_or_file]
            prompt = f"Please review the attached code file, focusing on: {focus}"
        else:
            prompt = f"""Please review this code, focusing on: {focus}

```
{code_or_file}
```"""

        return await self.send_message(prompt, sid, files)

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No sessions found."

        lines = ["Sessions:"]
        for sid, session in self.sessions.items():
            active = " (active)" if sid == self.active_session else ""
            msg_count = len(session.messages)
            lines.append(f"  - {sid}: {session.model} [{session.agent}], {msg_count} messages{active}")
        return "\n".join(lines)

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        lines = [f"Session: {sid}", f"Model: {session.model}, Agent: {session.agent}", "---"]

        for msg in session.messages[-last_n:]:
            role = "You" if msg.role == "user" else "OpenCode"
            lines.append(f"\n**{role}:**\n{msg.content}")

        return "\n".join(lines)

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        self.active_session = session_id
        session = self.sessions[session_id]
        return f"Active session: '{session_id}' ({session.model}, {session.agent})"

    def set_model(self, model: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_model = session.model
        session.model = model
        session.save(self.sessions_dir / f"{sid}.json")

        return f"Model changed: {old_model} -> {model}"

    def set_agent(self, agent: str, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_agent = session.agent
        session.agent = agent
        session.save(self.sessions_dir / f"{sid}.json")

        return f"Agent changed: {old_agent} -> {agent}"

    def end_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session to end."

        del self.sessions[sid]
        session_path = self.sessions_dir / f"{sid}.json"
        if session_path.exists():
            session_path.unlink()

        if self.active_session == sid:
            self.active_session = None

        return f"Session '{sid}' ended."


# MCP Server setup
bridge = OpenCodeBridge()
server = Server("opencode-bridge")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="opencode_models",
            description="List available models from OpenCode (GPT-5, Claude, Gemini, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Filter by provider (openai, github-copilot, anthropic)"
                    }
                }
            }
        ),
        Tool(
            name="opencode_agents",
            description="List available agents (plan, build, explore, general)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_start",
            description="Start a new discussion session with OpenCode",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Unique identifier for this session"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (default: openai/gpt-5.2-codex)"
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent to use: plan, build, explore, general (default: plan)"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_discuss",
            description="Send a message to OpenCode. Use for code review, architecture, brainstorming.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File paths to attach for context"
                    }
                },
                "required": ["message"]
            }
        ),
        Tool(
            name="opencode_plan",
            description="Start a planning discussion with the plan agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "What to plan"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant file paths"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="opencode_brainstorm",
            description="Open-ended brainstorming on a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to brainstorm about"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="opencode_review",
            description="Review code for issues and improvements",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_or_file": {
                        "type": "string",
                        "description": "Code snippet or file path"
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to focus on (default: correctness, efficiency, bugs)"
                    }
                },
                "required": ["code_or_file"]
            }
        ),
        Tool(
            name="opencode_model",
            description="Change the model for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "New model"}
                },
                "required": ["model"]
            }
        ),
        Tool(
            name="opencode_agent",
            description="Change the agent for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "New agent (plan, build, explore, general)"}
                },
                "required": ["agent"]
            }
        ),
        Tool(
            name="opencode_history",
            description="Get conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of messages (default: 20)"}
                }
            }
        ),
        Tool(
            name="opencode_sessions",
            description="List all sessions",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_switch",
            description="Switch to a different session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to switch to"}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_end",
            description="End the current session",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_config",
            description="Get current configuration (default model and agent)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_configure",
            description="Set default model and/or agent (persisted)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Default model"},
                    "agent": {"type": "string", "description": "Default agent"}
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "opencode_models":
            result = await bridge.list_models(arguments.get("provider"))
        elif name == "opencode_agents":
            result = await bridge.list_agents()
        elif name == "opencode_start":
            result = await bridge.start_session(
                session_id=arguments["session_id"],
                model=arguments.get("model", "openai/gpt-5.2-codex"),
                agent=arguments.get("agent", "plan")
            )
        elif name == "opencode_discuss":
            result = await bridge.send_message(
                message=arguments["message"],
                files=arguments.get("files")
            )
        elif name == "opencode_plan":
            result = await bridge.plan(
                task=arguments["task"],
                files=arguments.get("files")
            )
        elif name == "opencode_brainstorm":
            result = await bridge.brainstorm(arguments["topic"])
        elif name == "opencode_review":
            result = await bridge.review_code(
                code_or_file=arguments["code_or_file"],
                focus=arguments.get("focus", "correctness, efficiency, and potential bugs")
            )
        elif name == "opencode_model":
            result = bridge.set_model(arguments["model"])
        elif name == "opencode_agent":
            result = bridge.set_agent(arguments["agent"])
        elif name == "opencode_history":
            result = bridge.get_history(last_n=arguments.get("last_n", 20))
        elif name == "opencode_sessions":
            result = bridge.list_sessions()
        elif name == "opencode_switch":
            result = bridge.set_active(arguments["session_id"])
        elif name == "opencode_end":
            result = bridge.end_session()
        elif name == "opencode_config":
            result = bridge.get_config()
        elif name == "opencode_configure":
            result = bridge.set_config(
                model=arguments.get("model"),
                agent=arguments.get("agent")
            )
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def main():
    import asyncio

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    asyncio.run(run())


if __name__ == "__main__":
    main()
