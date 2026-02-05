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
import asyncio
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability


# File size thresholds
SMALL_FILE = 500        # lines
MEDIUM_FILE = 1500      # lines
LARGE_FILE = 5000       # lines

# Language detection by extension
LANG_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript/React",
    ".jsx": "JavaScript/React", ".go": "Go", ".rs": "Rust", ".java": "Java",
    ".c": "C", ".cpp": "C++", ".h": "C/C++ Header", ".hpp": "C++ Header",
    ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".kt": "Kotlin", ".scala": "Scala", ".sh": "Shell", ".bash": "Bash",
    ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".yaml": "YAML", ".yml": "YAML", ".json": "JSON", ".toml": "TOML",
    ".xml": "XML", ".md": "Markdown", ".r": "R", ".lua": "Lua",
    ".zig": "Zig", ".nim": "Nim", ".ex": "Elixir", ".erl": "Erlang",
    ".clj": "Clojure", ".hs": "Haskell", ".ml": "OCaml", ".vue": "Vue",
    ".svelte": "Svelte", ".dart": "Dart", ".proto": "Protocol Buffers",
}


_file_info_cache: dict[str, dict] = {}

MAX_READ_SIZE = 10 * 1024 * 1024  # 10MB - above this, estimate lines from size


def get_file_info(filepath: str) -> dict:
    """Get metadata about a file: size, lines, language, etc. Results are cached per path."""
    filepath = str(Path(filepath).resolve())
    if filepath in _file_info_cache:
        return _file_info_cache[filepath]

    p = Path(filepath)
    if not p.is_file():
        return {}
    try:
        stat = p.stat()
        ext = p.suffix.lower()

        # Count lines efficiently: stream for large files, estimate for huge ones
        if stat.st_size > MAX_READ_SIZE:
            # Estimate: ~40 bytes per line for code files
            line_count = stat.st_size // 40
        else:
            # Stream line counting without loading full content into memory
            line_count = 0
            with open(p, "r", errors="replace") as f:
                for _ in f:
                    line_count += 1

        result = {
            "path": filepath,
            "name": p.name,
            "size_bytes": stat.st_size,
            "size_human": _human_size(stat.st_size),
            "lines": line_count,
            "language": LANG_MAP.get(ext, ext.lstrip(".").upper() if ext else "Unknown"),
            "ext": ext,
            "category": (
                "small" if line_count <= SMALL_FILE
                else "medium" if line_count <= MEDIUM_FILE
                else "large" if line_count <= LARGE_FILE
                else "very large"
            ),
        }
        _file_info_cache[filepath] = result
        return result
    except Exception:
        return {"path": filepath, "name": p.name}


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.0f}{unit}" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def build_file_context(file_paths: list[str]) -> str:
    """Build a context block describing attached files."""
    if not file_paths:
        return ""
    infos = [info for f in file_paths if (info := get_file_info(f))]
    if not infos:
        return ""

    parts = ["## Attached Files\n"]
    for info in infos:
        line = f"- **{info.get('name', '?')}**"
        details = []
        if "language" in info:
            details.append(info["language"])
        if "lines" in info:
            details.append(f"{info['lines']} lines")
        if "size_human" in info:
            details.append(info["size_human"])
        if "category" in info:
            details.append(info["category"])
        if details:
            line += f" ({', '.join(details)})"
        parts.append(line)

    total_lines = sum(i.get("lines", 0) for i in infos)
    if total_lines > LARGE_FILE:
        parts.append(f"\n> Total: {total_lines} lines across {len(infos)} file(s) — this is a large review.")
        parts.append("> Focus on the most critical issues first. Use a structured, section-by-section approach.")

    return "\n".join(parts)


def build_review_prompt(file_infos: list[dict], focus: str) -> str:
    """Build an adaptive review prompt based on file size and type."""
    total_lines = sum(i.get("lines", 0) for i in file_infos)

    # Base review instructions
    prompt_parts = [f"Please review the attached code, focusing on: **{focus}**\n"]

    # Add file context
    if file_infos:
        prompt_parts.append("### Files to review:")
        for info in file_infos:
            prompt_parts.append(f"- {info.get('name', '?')} ({info.get('language', '?')}, {info.get('lines', '?')} lines)")
        prompt_parts.append("")

    # Adapt strategy to file size
    if total_lines > LARGE_FILE:
        prompt_parts.append("""### Review Strategy (Large File)
This is a large codebase review. Use this structured approach:

1. **Architecture Overview**: Describe the overall structure, main components, and data flow
2. **Critical Issues**: Security vulnerabilities, bugs, race conditions, memory leaks
3. **Design Concerns**: Architectural problems, tight coupling, missing abstractions
4. **Code Quality**: Naming, duplication, complexity hotspots (focus on the worst areas)
5. **Key Recommendations**: Top 5 most impactful improvements, prioritized

Do NOT try to comment on every line. Focus on patterns and the most impactful findings.""")
    elif total_lines > MEDIUM_FILE:
        prompt_parts.append("""### Review Strategy (Medium File)
Provide a structured review:

1. **Summary**: What does this code do? Overall assessment
2. **Issues Found**: Bugs, security concerns, edge cases, error handling gaps
3. **Design Feedback**: Structure, patterns, abstractions
4. **Specific Suggestions**: Concrete improvements with code examples where helpful""")
    else:
        prompt_parts.append("""### Review Guidelines
Provide a thorough review covering:
- Correctness and edge cases
- Error handling
- Code clarity and naming
- Any security concerns
- Concrete suggestions for improvement""")

    return "\n".join(prompt_parts)


def build_message_prompt(message: str, file_paths: list[str]) -> str:
    """Build a smart prompt that includes file context and instructions."""
    parts = []

    # Add file context if files are attached
    user_files = [f for f in file_paths if not Path(f).name.startswith("opencode_msg_")]
    if user_files:
        file_context = build_file_context(user_files)
        if file_context:
            parts.append(file_context)
            parts.append("")

        total_lines = sum(get_file_info(f).get("lines", 0) for f in user_files)
        if total_lines > LARGE_FILE:
            parts.append("**Note:** Large file(s) attached. Read through the full content carefully before responding. "
                         "If asked to analyze or review, use a structured section-by-section approach.")
            parts.append("")

    parts.append("## Request")
    parts.append("Respond to the user's request in the attached message file. "
                 "Read all attached files completely before responding.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Domain Detection & Companion System
# ---------------------------------------------------------------------------

@dataclass
class DomainProfile:
    """Defines a domain of expertise with persona, frameworks, and approach."""
    id: str
    name: str
    keywords: list[str]
    phrases: list[str]
    file_indicators: list[str]  # file extensions or name patterns
    expert_persona: str
    thinking_frameworks: list[str]
    key_questions: list[str]
    structured_approach: list[str]
    agent_hint: str  # suggested opencode agent


DOMAIN_REGISTRY: dict[str, DomainProfile] = {}


def _register(*profiles: DomainProfile):
    for p in profiles:
        DOMAIN_REGISTRY[p.id] = p


_register(
    DomainProfile(
        id="architecture",
        name="Architecture & System Design",
        keywords=["architecture", "microservice", "monolith", "scalab", "distributed",
                  "component", "module", "layer", "decouple", "coupling", "cohesion",
                  "event", "queue", "broker", "gateway", "proxy", "load balancer"],
        phrases=["system design", "event driven", "event sourcing", "service mesh",
                 "domain driven", "hexagonal architecture", "clean architecture",
                 "micro frontend", "message bus", "data pipeline", "cqrs"],
        file_indicators=[".proto", ".yaml", ".yml", ".tf", ".hcl"],
        expert_persona=(
            "a senior distributed systems architect who has designed systems serving "
            "millions of users. You think in terms of components, boundaries, data flow, "
            "and failure modes. You've seen both over-engineered and under-engineered "
            "systems and know when each approach is appropriate."
        ),
        thinking_frameworks=["C4 model (context, containers, components, code)",
                             "CAP theorem", "DDD (bounded contexts, aggregates)",
                             "CQRS/Event Sourcing trade-offs",
                             "Twelve-Factor App principles"],
        key_questions=["What are the key quality attributes (latency, throughput, availability)?",
                       "Where are the domain boundaries?",
                       "What data consistency model fits here?",
                       "What happens when a component fails?",
                       "How will this evolve in 6-12 months?"],
        structured_approach=["Clarify requirements and constraints",
                             "Identify components and their responsibilities",
                             "Define interfaces and data flow",
                             "Analyze trade-offs and failure modes",
                             "Recommend with rationale"],
        agent_hint="plan",
    ),
    DomainProfile(
        id="debugging",
        name="Debugging & Troubleshooting",
        keywords=["bug", "error", "crash", "fail", "exception", "traceback",
                  "stacktrace", "debug", "breakpoint", "segfault", "panic",
                  "hang", "freeze", "corrupt", "unexpected", "wrong"],
        phrases=["root cause", "stack trace", "doesn't work", "stopped working",
                 "race condition", "deadlock", "memory leak", "null pointer",
                 "off by one", "regression", "flaky test", "intermittent failure"],
        file_indicators=[".log", ".dump", ".core"],
        expert_persona=(
            "a seasoned debugger who has tracked down the most elusive bugs — race "
            "conditions, heisenbugs, memory corruption, off-by-one errors hidden for "
            "years. You are methodical, hypothesis-driven, and never jump to conclusions."
        ),
        thinking_frameworks=["Five Whys (root cause analysis)",
                             "Scientific method (hypothesize, test, refine)",
                             "Binary search / bisection for isolating changes",
                             "Rubber duck debugging"],
        key_questions=["When did it start happening? What changed?",
                       "Is it reproducible? Under what conditions?",
                       "What are the exact symptoms vs. expected behavior?",
                       "Have we ruled out environment differences?",
                       "What is the minimal reproduction case?"],
        structured_approach=["Reproduce and isolate the issue",
                             "Form hypotheses ranked by likelihood",
                             "Gather evidence: logs, traces, state inspection",
                             "Narrow down via elimination",
                             "Fix, verify, and prevent regression"],
        agent_hint="build",
    ),
    DomainProfile(
        id="performance",
        name="Performance & Optimization",
        keywords=["performance", "optimize", "bottleneck", "latency", "throughput",
                  "cache", "profil", "benchmark", "slow", "fast", "speed",
                  "memory", "cpu", "io", "bandwidth", "concurren"],
        phrases=["cache miss", "hot path", "time complexity", "space complexity",
                 "p99 latency", "tail latency", "garbage collection", "connection pool",
                 "query plan", "flame graph", "load test"],
        file_indicators=[".perf", ".prof", ".bench"],
        expert_persona=(
            "a performance engineer who obsesses over microseconds and memory allocations. "
            "You profile before optimizing, know that premature optimization is the root of "
            "all evil, and always ask 'what does the data say?' before recommending changes."
        ),
        thinking_frameworks=["Amdahl's Law", "Little's Law",
                             "USE method (Utilization, Saturation, Errors)",
                             "Roofline model", "Big-O analysis with practical constants"],
        key_questions=["What is the actual bottleneck (CPU, memory, I/O, network)?",
                       "Do we have profiling data or benchmarks?",
                       "What's the target performance? Current baseline?",
                       "What are the hot paths?",
                       "What trade-offs are acceptable (memory vs speed, complexity vs perf)?"],
        structured_approach=["Measure current performance with profiling/benchmarks",
                             "Identify the bottleneck — do not guess",
                             "Propose targeted optimizations",
                             "Estimate impact and trade-offs",
                             "Measure again after changes"],
        agent_hint="build",
    ),
    DomainProfile(
        id="security",
        name="Security & Threat Modeling",
        keywords=["security", "vulnerab", "auth", "token", "encrypt", "hash",
                  "ssl", "tls", "cors", "csrf", "xss", "injection", "sanitiz",
                  "permission", "privilege", "secret", "credential"],
        phrases=["sql injection", "cross site", "threat model", "attack surface",
                 "zero trust", "defense in depth", "least privilege",
                 "owasp top 10", "security audit", "penetration test",
                 "access control", "input validation"],
        file_indicators=[".pem", ".key", ".cert", ".env"],
        expert_persona=(
            "a senior application security engineer who thinks like an attacker but "
            "builds like a defender. You know the OWASP Top 10 by heart, understand "
            "cryptographic primitives, and always consider the full threat model."
        ),
        thinking_frameworks=["STRIDE threat modeling",
                             "OWASP Top 10",
                             "Defense in depth",
                             "Zero trust architecture",
                             "Principle of least privilege"],
        key_questions=["What is the threat model? Who are the adversaries?",
                       "What data is sensitive and how is it protected?",
                       "Where are the trust boundaries?",
                       "What authentication and authorization model is in use?",
                       "Are there known CVEs in dependencies?"],
        structured_approach=["Identify assets and threat actors",
                             "Map the attack surface",
                             "Enumerate threats (STRIDE)",
                             "Assess risk (likelihood x impact)",
                             "Recommend mitigations prioritized by risk"],
        agent_hint="plan",
    ),
    DomainProfile(
        id="testing",
        name="Testing & Quality Assurance",
        keywords=["test", "assert", "mock", "stub", "fixture", "coverage",
                  "spec", "suite", "expect", "verify", "tdd", "bdd"],
        phrases=["unit test", "integration test", "end to end", "test coverage",
                 "test driven", "edge case", "boundary condition", "test pyramid",
                 "property based", "mutation testing", "snapshot test",
                 "regression test"],
        file_indicators=["_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js",
                         ".spec.ts", "_spec.rb"],
        expert_persona=(
            "a testing specialist who believes tests are living documentation. You "
            "understand the test pyramid, know when to mock and when not to, and "
            "write tests that catch real bugs without being brittle."
        ),
        thinking_frameworks=["Test pyramid (unit → integration → e2e)",
                             "FIRST principles (Fast, Independent, Repeatable, Self-validating, Timely)",
                             "Arrange-Act-Assert pattern",
                             "Equivalence partitioning & boundary value analysis"],
        key_questions=["What behavior are we verifying?",
                       "What are the edge cases and boundary conditions?",
                       "Is this a unit, integration, or e2e concern?",
                       "What should we mock vs. use real implementations?",
                       "How will we know if this test is catching real bugs?"],
        structured_approach=["Identify what behavior to test",
                             "Determine test level (unit/integration/e2e)",
                             "Design test cases covering happy path and edge cases",
                             "Write clear, maintainable assertions",
                             "Review for brittleness and false confidence"],
        agent_hint="build",
    ),
    DomainProfile(
        id="devops",
        name="DevOps & Infrastructure",
        keywords=["deploy", "pipeline", "container", "docker", "kubernetes", "k8s",
                  "terraform", "ansible", "helm", "ci", "cd", "infra", "cloud",
                  "aws", "gcp", "azure", "monitoring", "alert", "observ"],
        phrases=["ci/cd pipeline", "infrastructure as code", "blue green deployment",
                 "canary release", "rolling update", "auto scaling",
                 "service discovery", "container orchestration",
                 "gitops", "platform engineering"],
        file_indicators=[".tf", ".hcl", "Dockerfile", ".yml", ".yaml",
                         "Jenkinsfile", ".github"],
        expert_persona=(
            "a senior DevOps/platform engineer who has managed production infrastructure "
            "at scale. You think in terms of reliability, repeatability, and observability. "
            "You know that every manual step is a future incident."
        ),
        thinking_frameworks=["DORA metrics (deployment frequency, lead time, MTTR, change failure rate)",
                             "Infrastructure as Code principles",
                             "SRE golden signals (latency, traffic, errors, saturation)",
                             "GitOps workflow"],
        key_questions=["What is the deployment target (cloud, on-prem, hybrid)?",
                       "What are the reliability requirements (SLOs)?",
                       "How do we roll back if something goes wrong?",
                       "What observability do we have?",
                       "What is the blast radius of a bad deploy?"],
        structured_approach=["Assess current infrastructure and deployment process",
                             "Identify gaps in reliability and automation",
                             "Design pipeline and infrastructure changes",
                             "Plan rollout with rollback strategy",
                             "Define success metrics and alerts"],
        agent_hint="plan",
    ),
    DomainProfile(
        id="database",
        name="Database & Data Modeling",
        keywords=["database", "schema", "table", "column", "index", "query",
                  "sql", "nosql", "migration", "join", "foreign key", "primary key",
                  "transaction", "acid", "normali", "partition", "shard", "replica"],
        phrases=["query optimization", "execution plan", "database migration",
                 "data model", "schema design", "query plan", "n+1 query",
                 "connection pool", "read replica", "write ahead log",
                 "eventual consistency"],
        file_indicators=[".sql", ".prisma", ".migration"],
        expert_persona=(
            "a database architect with deep expertise in both relational and NoSQL systems. "
            "You think about data access patterns first, schema second. You've tuned queries "
            "from minutes to milliseconds and know when denormalization is the right call."
        ),
        thinking_frameworks=["Normal forms (1NF through BCNF) and when to denormalize",
                             "ACID vs BASE trade-offs",
                             "Index design (B-tree, hash, composite, covering)",
                             "CAP theorem applied to data stores"],
        key_questions=["What are the primary access patterns (reads vs writes)?",
                       "What consistency guarantees are needed?",
                       "How much data and what growth rate?",
                       "What are the query performance requirements?",
                       "How will the schema evolve?"],
        structured_approach=["Understand access patterns and data relationships",
                             "Design schema to match access patterns",
                             "Plan indexing strategy",
                             "Consider partitioning/sharding needs",
                             "Design migration path from current state"],
        agent_hint="build",
    ),
    DomainProfile(
        id="api_design",
        name="API Design",
        keywords=["api", "endpoint", "rest", "graphql", "grpc", "webhook",
                  "pagination", "versioning", "rate limit", "openapi", "swagger",
                  "request", "response", "payload", "header", "status code"],
        phrases=["rest api", "api design", "api versioning", "breaking change",
                 "backward compatible", "content negotiation", "hateoas",
                 "api gateway", "graphql schema", "api contract"],
        file_indicators=[".openapi", ".swagger", ".graphql", ".gql", ".proto"],
        expert_persona=(
            "a senior API designer who has built APIs used by thousands of developers. "
            "You think about developer experience, consistency, evolvability, and "
            "backward compatibility. You know REST deeply but aren't dogmatic about it."
        ),
        thinking_frameworks=["REST maturity model (Richardson)",
                             "API-first design",
                             "Consumer-driven contracts",
                             "Robustness principle (be liberal in what you accept)"],
        key_questions=["Who are the API consumers (internal, external, both)?",
                       "What operations does the API need to support?",
                       "How will we handle versioning and breaking changes?",
                       "What authentication and rate limiting model?",
                       "What error format and status code conventions?"],
        structured_approach=["Identify resources and operations",
                             "Design URL structure and HTTP methods",
                             "Define request/response schemas",
                             "Plan versioning and error handling",
                             "Document with examples"],
        agent_hint="plan",
    ),
    DomainProfile(
        id="frontend",
        name="Frontend & UI",
        keywords=["react", "vue", "svelte", "angular", "component", "render",
                  "state", "hook", "prop", "css", "style", "dom", "browser",
                  "responsive", "animation", "accessibility", "a11y", "ssr"],
        phrases=["server side rendering", "client side rendering", "state management",
                 "component library", "design system", "web vitals",
                 "progressive enhancement", "single page app", "hydration",
                 "code splitting", "lazy loading"],
        file_indicators=[".tsx", ".jsx", ".vue", ".svelte", ".css", ".scss", ".less"],
        expert_persona=(
            "a senior frontend architect who cares deeply about user experience, "
            "accessibility, and performance. You've built design systems and know "
            "that the best code is the code that makes users productive and happy."
        ),
        thinking_frameworks=["Component composition patterns",
                             "Unidirectional data flow",
                             "Web Core Vitals (LCP, FID, CLS)",
                             "Progressive enhancement",
                             "WCAG accessibility guidelines"],
        key_questions=["What is the target user experience?",
                       "What rendering strategy fits (SSR, CSR, ISR, SSG)?",
                       "How will we manage state (local, global, server)?",
                       "What are the accessibility requirements?",
                       "What are the performance budgets?"],
        structured_approach=["Clarify UX requirements and constraints",
                             "Choose rendering and state management strategy",
                             "Design component hierarchy",
                             "Plan for accessibility and performance",
                             "Define testing approach (visual, interaction, a11y)"],
        agent_hint="build",
    ),
    DomainProfile(
        id="algorithms",
        name="Algorithms & Data Structures",
        keywords=["algorithm", "complexity", "sort", "search", "graph", "tree",
                  "heap", "hash", "array", "linked list", "stack", "queue",
                  "recursive", "dynamic", "greedy", "backtrack"],
        phrases=["time complexity", "space complexity", "dynamic programming",
                 "divide and conquer", "binary search", "breadth first",
                 "depth first", "shortest path", "minimum spanning",
                 "sliding window", "two pointer"],
        file_indicators=[],
        expert_persona=(
            "a computer scientist who loves elegant solutions and rigorous analysis. "
            "You think in terms of invariants, complexity classes, and correctness proofs. "
            "You know that the right data structure often matters more than the algorithm."
        ),
        thinking_frameworks=["Big-O analysis (time and space)",
                             "Problem reduction (what known problem does this map to?)",
                             "Invariant-based reasoning",
                             "Amortized analysis"],
        key_questions=["What are the input constraints (size, range, distribution)?",
                       "What are the performance requirements?",
                       "Is there a known algorithm or pattern that applies?",
                       "Can we trade space for time (or vice versa)?",
                       "What edge cases must we handle?"],
        structured_approach=["Understand the problem and constraints",
                             "Identify applicable patterns or known algorithms",
                             "Design solution with correctness argument",
                             "Analyze time and space complexity",
                             "Consider optimizations and edge cases"],
        agent_hint="build",
    ),
    DomainProfile(
        id="code_quality",
        name="Code Quality & Refactoring",
        keywords=["refactor", "clean", "readab", "maintainab", "solid", "dry",
                  "smell", "debt", "pattern", "antipattern", "principle",
                  "naming", "abstraction", "duplication"],
        phrases=["code smell", "technical debt", "design pattern", "code review",
                 "clean code", "single responsibility", "dependency injection",
                 "separation of concerns", "boy scout rule",
                 "strangler fig", "legacy code"],
        file_indicators=[],
        expert_persona=(
            "a pragmatic software craftsperson who values readability over cleverness. "
            "You refactor with purpose, not for its own sake. You know that good code "
            "is code your teammates can understand and modify with confidence."
        ),
        thinking_frameworks=["SOLID principles (applied pragmatically)",
                             "Refactoring patterns (Fowler)",
                             "Code smells catalog",
                             "Connascence (coupling analysis)"],
        key_questions=["What problem is the current design causing?",
                       "Is this refactoring worth the risk and effort?",
                       "What's the minimal change that improves the situation?",
                       "How do we refactor safely (tests as safety net)?",
                       "Will this be clearer to the next person reading it?"],
        structured_approach=["Identify the pain point or code smell",
                             "Ensure adequate test coverage before refactoring",
                             "Apply incremental, safe transformations",
                             "Verify behavior preservation after each step",
                             "Review for clarity and simplicity"],
        agent_hint="build",
    ),
    DomainProfile(
        id="planning",
        name="Project Planning & Product",
        keywords=["plan", "roadmap", "milestone", "sprint", "epic", "story",
                  "requirement", "scope", "prioriti", "estimate", "mvp",
                  "feature", "deadline", "backlog", "stakeholder"],
        phrases=["user story", "acceptance criteria", "definition of done",
                 "minimum viable", "project plan", "technical spec",
                 "request for comments", "design doc", "product requirement",
                 "scope creep"],
        file_indicators=[],
        expert_persona=(
            "a seasoned tech lead who bridges engineering and product. You break down "
            "ambiguous problems into concrete, shippable increments. You know that the "
            "best plan is one the team actually follows."
        ),
        thinking_frameworks=["User story mapping",
                             "RICE prioritization (Reach, Impact, Confidence, Effort)",
                             "MoSCoW prioritization",
                             "Incremental delivery (thin vertical slices)"],
        key_questions=["What is the user problem we're solving?",
                       "What is the smallest thing we can ship to learn?",
                       "What are the dependencies and risks?",
                       "How will we know this succeeded?",
                       "What can we defer without losing value?"],
        structured_approach=["Define the problem and success criteria",
                             "Break down into shippable increments",
                             "Identify dependencies, risks, and unknowns",
                             "Prioritize by value and effort",
                             "Define first concrete next steps"],
        agent_hint="plan",
    ),
    DomainProfile(
        id="general",
        name="General Discussion",
        keywords=[],
        phrases=[],
        file_indicators=[],
        expert_persona=(
            "a knowledgeable senior engineer with broad experience across the stack. "
            "You think clearly, communicate precisely, and always consider the broader "
            "context before diving into details."
        ),
        thinking_frameworks=["First principles thinking",
                             "Trade-off analysis",
                             "Systems thinking"],
        key_questions=["What are we trying to achieve?",
                       "What are the constraints?",
                       "What are the trade-offs?"],
        structured_approach=["Understand the question and context",
                             "Consider multiple perspectives",
                             "Analyze trade-offs",
                             "Provide a clear recommendation"],
        agent_hint="plan",
    ),
)


@dataclass
class DomainDetection:
    """Result of domain detection."""
    primary: DomainProfile
    confidence: int  # 0-100
    secondary: Optional[DomainProfile] = None
    secondary_confidence: int = 0


def detect_domain(
    message: str,
    file_paths: Optional[list[str]] = None,
) -> DomainDetection:
    """Score message against all domains and return best match.

    Scoring rules:
    - keyword match: +1 per keyword found
    - phrase match: +2 per phrase found  (phrases are more specific)
    - file indicator: +1.5 per matching file extension/pattern
    """
    text = message.lower()
    scores: dict[str, float] = {}

    for domain_id, profile in DOMAIN_REGISTRY.items():
        if domain_id == "general":
            continue  # general is the fallback
        score = 0.0

        for kw in profile.keywords:
            if kw in text:
                score += 1

        for phrase in profile.phrases:
            if phrase in text:
                score += 2

        if file_paths:
            for fp in file_paths:
                fp_lower = fp.lower()
                name_lower = Path(fp).name.lower()
                for indicator in profile.file_indicators:
                    ind = indicator.lower()
                    if fp_lower.endswith(ind) or ind == name_lower or ind in fp_lower:
                        score += 1.5

        if score > 0:
            scores[domain_id] = score

    if not scores:
        return DomainDetection(
            primary=DOMAIN_REGISTRY["general"],
            confidence=50,
        )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_id, best_score = ranked[0]

    # Confidence: scale relative to number of matches.
    # A score of 5+ is very confident; 1 is low.
    confidence = min(99, int(40 + best_score * 12))

    result = DomainDetection(
        primary=DOMAIN_REGISTRY[best_id],
        confidence=confidence,
    )

    # Cross-domain detection: secondary if >60% of primary
    if len(ranked) > 1:
        second_id, second_score = ranked[1]
        if second_score >= best_score * 0.6:
            result.secondary = DOMAIN_REGISTRY[second_id]
            result.secondary_confidence = min(99, int(40 + second_score * 12))

    return result


def build_companion_prompt(
    message: str,
    files: Optional[list[str]] = None,
    domain_override: Optional[str] = None,
    is_followup: bool = False,
) -> tuple[str, DomainDetection]:
    """Assemble a domain-aware companion prompt.

    Returns (prompt_text, domain_detection).
    """
    # Detect or override domain
    if domain_override and domain_override in DOMAIN_REGISTRY:
        profile = DOMAIN_REGISTRY[domain_override]
        detection = DomainDetection(primary=profile, confidence=99)
    else:
        detection = detect_domain(message, files)
        profile = detection.primary

    # Follow-up: lightweight prompt
    if is_followup:
        parts = [
            "## Continuing Our Discussion",
            "",
            message,
            "",
            "Remember: challenge assumptions, consider alternatives, be explicit about trade-offs.",
        ]
        return "\n".join(parts), detection

    # --- Full initial prompt ---
    parts = []

    # File context
    user_files = [f for f in (files or []) if not Path(f).name.startswith("opencode_msg_")]
    if user_files:
        file_context = build_file_context(user_files)
        if file_context:
            parts.append("## Context")
            parts.append(file_context)
            parts.append("")

    # Cross-domain note
    cross = ""
    if detection.secondary:
        cross = f" This also touches on **{detection.secondary.name}**, so weave in that perspective where relevant."

    # Discussion setup
    parts.append("## Discussion Setup")
    parts.append(
        f"You are {profile.expert_persona}{cross}\n"
        f"I'm bringing you a question about **{profile.name}**, "
        "and I want us to think through it together as peers."
    )
    parts.append("")

    # Frameworks
    parts.append(f"### Analytical Toolkit")
    for fw in profile.thinking_frameworks:
        parts.append(f"- {fw}")
    parts.append("")

    # Key questions
    parts.append("### Key Questions to Consider")
    for q in profile.key_questions:
        parts.append(f"- {q}")
    parts.append("")

    # Collaborative ground rules
    parts.append("## Collaborative Ground Rules")
    parts.append("- Think out loud, share your reasoning")
    parts.append("- Challenge questionable assumptions — including mine")
    parts.append("- Lay out trade-offs explicitly: what we gain, what we lose")
    parts.append("- Propose at least one alternative I haven't considered")
    parts.append("")

    # Structured approach
    parts.append(f"## Approach")
    for i, step in enumerate(profile.structured_approach, 1):
        parts.append(f"{i}. {step}")
    parts.append("")

    # The question
    parts.append("## The Question")
    parts.append(message)
    parts.append("")

    # Synthesize
    parts.append("## Synthesize")
    parts.append("1. Your recommendation with rationale")
    parts.append("2. Key trade-offs")
    parts.append("3. Risks or blind spots")
    parts.append("4. Open questions worth exploring")

    return "\n".join(parts), detection


# Default configuration
DEFAULT_MODEL = "openai/gpt-5.2-codex"
DEFAULT_AGENT = "plan"
DEFAULT_VARIANT = "medium"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    agent: str = DEFAULT_AGENT
    variant: str = DEFAULT_VARIANT

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
                config.variant = data.get("variant", config.variant)
            except Exception:
                pass

        # Environment variables override config file
        config.model = os.environ.get("OPENCODE_MODEL", config.model)
        config.agent = os.environ.get("OPENCODE_AGENT", config.agent)
        config.variant = os.environ.get("OPENCODE_VARIANT") or config.variant

        return config

    def save(self):
        config_dir = Path.home() / ".opencode-bridge"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        data = {"model": self.model, "agent": self.agent, "variant": self.variant}
        config_path.write_text(json.dumps(data, indent=2))


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
    variant: str = DEFAULT_VARIANT
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
            "variant": self.variant,
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
            agent=data.get("agent", DEFAULT_AGENT),
            variant=data.get("variant", DEFAULT_VARIANT),
            opencode_session_id=data.get("opencode_session_id"),
            created=data.get("created", datetime.now().isoformat())
        )
        for m in data.get("messages", []):
            session.messages.append(Message(**m))
        return session


class OpenCodeBridge:
    def __init__(self):
        self.start_time = datetime.now()
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

    async def _run_opencode(self, *args, timeout: int = 300) -> tuple[str, int]:
        """Run opencode CLI command and return output (async)."""
        global OPENCODE_BIN
        # Lazy retry: if binary wasn't found at startup, try again
        if not OPENCODE_BIN:
            OPENCODE_BIN = find_opencode()
        if not OPENCODE_BIN:
            return "OpenCode not installed. Install from: https://opencode.ai", 1

        try:
            proc = await asyncio.create_subprocess_exec(
                str(OPENCODE_BIN), *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=b''),
                timeout=timeout
            )
            # Combine stdout+stderr so errors aren't silently lost
            out = stdout.decode(errors="replace").strip()
            err = stderr.decode(errors="replace").strip()
            output = out if out else err
            # If both exist and return code indicates error, include stderr
            if out and err and proc.returncode:
                output = f"{out}\n\nStderr:\n{err}"
            return output, proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Command timed out after {timeout}s", 1
        except Exception as e:
            return f"Error: {e}", 1

    async def list_models(self, provider: Optional[str] = None) -> str:
        """List available models from OpenCode."""
        args = ["models"]
        if provider:
            args.append(provider)

        output, code = await self._run_opencode(*args)
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
        output, code = await self._run_opencode("agent", "list")
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
        agent: Optional[str] = None,
        variant: Optional[str] = None
    ) -> str:
        # Use config defaults if not specified
        model = model or self.config.model
        agent = agent or self.config.agent
        variant = variant or self.config.variant

        session = Session(
            id=session_id,
            model=model,
            agent=agent,
            variant=variant
        )
        self.sessions[session_id] = session
        self.active_session = session_id
        session.save(self.sessions_dir / f"{session_id}.json")

        result = f"Session '{session_id}' started\n  Model: {model}\n  Agent: {agent}"
        if variant:
            result += f"\n  Variant: {variant}"
        return result

    def get_config(self) -> str:
        """Get current configuration."""
        return f"""Current configuration:
  Model: {self.config.model}
  Agent: {self.config.agent}
  Variant: {self.config.variant}

Set via:
  - ~/.opencode-bridge/config.json
  - OPENCODE_MODEL, OPENCODE_AGENT, OPENCODE_VARIANT env vars
  - opencode_configure tool"""

    def set_config(self, model: Optional[str] = None, agent: Optional[str] = None, variant: Optional[str] = None) -> str:
        """Update and persist configuration."""
        changes = []
        if model:
            self.config.model = model
            changes.append(f"model: {model}")
        if agent:
            self.config.agent = agent
            changes.append(f"agent: {agent}")
        if variant:
            self.config.variant = variant
            changes.append(f"variant: {variant}")

        if changes:
            self.config.save()
            return "Configuration updated:\n  " + "\n  ".join(changes)
        return "No changes made."

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        files: Optional[list[str]] = None,
        domain_override: Optional[str] = None,
        _raw: bool = False,
    ) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session. Use opencode_start first."

        session = self.sessions[sid]
        session.add_message("user", message)
        # Save immediately so user messages aren't lost if OpenCode fails
        session.save(self.sessions_dir / f"{sid}.json")

        # Always write message to temp file to avoid shell escaping issues
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, prefix='opencode_msg_'
        )
        temp_file.write(message)
        temp_file.close()
        files = (files or []) + [temp_file.name]

        # Build prompt: companion system unless _raw is set
        domain_info = ""
        if _raw:
            run_prompt = build_message_prompt(message, files)
        else:
            is_followup = len(session.messages) > 1
            run_prompt, detection = build_companion_prompt(
                message, files, domain_override=domain_override,
                is_followup=is_followup,
            )
            domain_info = f"[Domain: {detection.primary.name}] [Confidence: {detection.confidence}%]"
            if detection.secondary:
                domain_info += f" [Also: {detection.secondary.name} ({detection.secondary_confidence}%)]"

        args = ["run", run_prompt]

        args.extend(["--model", session.model])
        args.extend(["--agent", session.agent])

        # Add variant if specified
        if session.variant:
            args.extend(["--variant", session.variant])

        # Continue session if we have an opencode session ID
        if session.opencode_session_id:
            args.extend(["--session", session.opencode_session_id])

        # Attach files
        if files:
            for f in files:
                args.extend(["--file", f])

        # Use JSON format to get session ID
        args.extend(["--format", "json"])

        # Scale timeout based on attached file size
        user_files = [f for f in files if not Path(f).name.startswith("opencode_msg_")]
        total_lines = sum(get_file_info(f).get("lines", 0) for f in user_files)
        # Base 300s, +60s per 1000 lines above threshold, capped at 900s
        timeout = min(900, 300 + max(0, (total_lines - MEDIUM_FILE) * 60 // 1000))

        output, code = await self._run_opencode(*args, timeout=timeout)

        # Cleanup temp file
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass

        if code != 0:
            return f"Error: {output}"

        # Parse JSON events for session ID and text
        reply_parts = []
        for line in output.split("\n"):
            if not line:
                continue
            try:
                event = json.loads(line)
                if not session.opencode_session_id and "sessionID" in event:
                    session.opencode_session_id = event["sessionID"]
                if event.get("type") == "text":
                    text = event.get("part", {}).get("text", "")
                    if text:
                        reply_parts.append(text)
            except json.JSONDecodeError:
                continue

        reply = "".join(reply_parts)
        if reply:
            session.add_message("assistant", reply)

        # Save if we got a reply or captured a new session ID
        if reply or session.opencode_session_id:
            session.save(self.sessions_dir / f"{sid}.json")

        response = reply or "No response received"
        if domain_info:
            response = f"{domain_info}\n\n{response}"
        return response

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
        """Open-ended brainstorming discussion — routes through companion system."""
        sid = session_id or self.active_session

        if not sid or sid not in self.sessions:
            sid = f"brainstorm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            await self.start_session(sid, agent="build")

        return await self.send_message(f"Let's brainstorm about: {topic}", sid)

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

        # Check if it's a file path (could be multiple, comma or space separated)
        files = None
        file_paths = []

        # Try splitting by comma first, then check each part
        candidates = [c.strip() for c in code_or_file.replace(",", " ").split() if c.strip()]
        for candidate in candidates:
            if Path(candidate).is_file():
                file_paths.append(candidate)

        if file_paths:
            files = file_paths
            file_infos = [get_file_info(f) for f in file_paths]
            file_infos = [i for i in file_infos if i]
            prompt = build_review_prompt(file_infos, focus)

            # Increase timeout for large files
            total_lines = sum(i.get("lines", 0) for i in file_infos)
            if total_lines > LARGE_FILE:
                # Use variant=high for large reviews if not already high+
                session = self.sessions[sid]
                if session.variant in ("minimal", "low", "medium"):
                    prompt += "\n\n> *Auto-escalated to thorough review due to file size.*"
        else:
            # Inline code snippet
            prompt = f"""Please review this code, focusing on: **{focus}**

```
{code_or_file}
```

Provide:
- Issues found (bugs, edge cases, security)
- Design feedback
- Concrete improvement suggestions"""

        return await self.send_message(prompt, sid, files, _raw=True)

    def list_sessions(self) -> str:
        if not self.sessions:
            return "No sessions found."

        lines = ["Sessions:"]
        for sid, session in self.sessions.items():
            active = " (active)" if sid == self.active_session else ""
            msg_count = len(session.messages)
            variant_str = f", variant={session.variant}" if session.variant else ""
            lines.append(f"  - {sid}: {session.model} [{session.agent}{variant_str}], {msg_count} messages{active}")
        return "\n".join(lines)

    def get_history(self, session_id: Optional[str] = None, last_n: int = 20) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        variant_str = f", Variant: {session.variant}" if session.variant else ""
        lines = [f"Session: {sid}", f"Model: {session.model}, Agent: {session.agent}{variant_str}", "---"]

        for msg in session.messages[-last_n:]:
            role = "You" if msg.role == "user" else "OpenCode"
            lines.append(f"\n**{role}:**\n{msg.content}")

        return "\n".join(lines)

    def set_active(self, session_id: str) -> str:
        if session_id not in self.sessions:
            return f"Session '{session_id}' not found."
        self.active_session = session_id
        session = self.sessions[session_id]
        variant_str = f", variant={session.variant}" if session.variant else ""
        return f"Active session: '{session_id}' ({session.model}, {session.agent}{variant_str})"

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

    def set_variant(self, variant: Optional[str], session_id: Optional[str] = None) -> str:
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session."

        session = self.sessions[sid]
        old_variant = session.variant or "none"
        session.variant = variant
        session.save(self.sessions_dir / f"{sid}.json")

        new_variant = variant or "none"
        return f"Variant changed: {old_variant} -> {new_variant}"

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

    def export_session(self, session_id: Optional[str] = None, format: str = "markdown") -> str:
        """Export a session as markdown or JSON."""
        sid = session_id or self.active_session
        if not sid or sid not in self.sessions:
            return "No active session to export."

        session = self.sessions[sid]

        if format == "json":
            data = {
                "id": session.id,
                "model": session.model,
                "agent": session.agent,
                "variant": session.variant,
                "created": session.created,
                "messages": [asdict(m) for m in session.messages]
            }
            return json.dumps(data, indent=2)

        # Markdown format
        lines = [
            f"# Session: {session.id}",
            f"**Model:** {session.model} | **Agent:** {session.agent} | **Variant:** {session.variant}",
            f"**Created:** {session.created}",
            f"**Messages:** {len(session.messages)}",
            "",
            "---",
            "",
        ]
        for msg in session.messages:
            role = "User" if msg.role == "user" else "OpenCode"
            lines.append(f"## {role}")
            lines.append(f"*{msg.timestamp}*\n")
            lines.append(msg.content)
            lines.append("\n---\n")

        return "\n".join(lines)

    def health_check(self) -> dict:
        """Return server health status."""
        uptime_seconds = int((datetime.now() - self.start_time).total_seconds())
        return {
            "status": "ok",
            "sessions": len(self.sessions),
            "uptime": uptime_seconds
        }


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
                    },
                    "variant": {
                        "type": "string",
                        "description": "Model variant for reasoning effort: minimal, low, medium, high, xhigh, max (default: medium)"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="opencode_discuss",
            description="Send a message to OpenCode. Use for code review, architecture, brainstorming. "
                        "Auto-detects discussion domain and frames OpenCode as a specialized expert. "
                        "Use 'domain' to override detection.",
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
                    },
                    "domain": {
                        "type": "string",
                        "description": "Override auto-detected domain",
                        "enum": ["architecture", "debugging", "performance", "security",
                                 "testing", "devops", "database", "api_design",
                                 "frontend", "algorithms", "code_quality", "planning",
                                 "general"]
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
            description="Review code for issues and improvements. Supports large files with adaptive review strategies. Can accept multiple file paths (space or comma separated).",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_or_file": {
                        "type": "string",
                        "description": "Code snippet, file path, or multiple file paths (space/comma separated)"
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
            name="opencode_variant",
            description="Change the model variant (reasoning effort) for the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "variant": {"type": "string", "description": "New variant: minimal, low, medium, high, xhigh, max"}
                },
                "required": ["variant"]
            }
        ),
        Tool(
            name="opencode_history",
            description="Get conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (default: active session)"},
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
            description="Get current configuration (default model, agent, variant)",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="opencode_configure",
            description="Set default model, agent, and/or variant (persisted)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Default model"},
                    "agent": {"type": "string", "description": "Default agent"},
                    "variant": {"type": "string", "description": "Default variant: minimal, low, medium, high, xhigh, max"}
                }
            }
        ),
        Tool(
            name="opencode_export",
            description="Export a session transcript as markdown or JSON",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session to export (default: active)"},
                    "format": {"type": "string", "description": "Export format: markdown or json (default: markdown)", "enum": ["markdown", "json"]}
                }
            }
        ),
        Tool(
            name="opencode_health",
            description="Health check: returns server status, session count, and uptime",
            inputSchema={"type": "object", "properties": {}}
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
                model=arguments.get("model"),
                agent=arguments.get("agent"),
                variant=arguments.get("variant")
            )
        elif name == "opencode_discuss":
            result = await bridge.send_message(
                message=arguments["message"],
                files=arguments.get("files"),
                domain_override=arguments.get("domain"),
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
        elif name == "opencode_variant":
            result = bridge.set_variant(arguments["variant"])
        elif name == "opencode_history":
            result = bridge.get_history(
                session_id=arguments.get("session_id"),
                last_n=arguments.get("last_n", 20)
            )
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
                agent=arguments.get("agent"),
                variant=arguments.get("variant")
            )
        elif name == "opencode_export":
            result = bridge.export_session(
                session_id=arguments.get("session_id"),
                format=arguments.get("format", "markdown")
            )
        elif name == "opencode_health":
            health = bridge.health_check()
            result = f"Status: {health['status']}\nSessions: {health['sessions']}\nUptime: {health['uptime']}s"
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


def main():
    import asyncio

    async def run():
        init_options = InitializationOptions(
            server_name="opencode-bridge",
            server_version="0.1.0",
            capabilities=ServerCapabilities(tools=ToolsCapability())
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, init_options)

    asyncio.run(run())


if __name__ == "__main__":
    main()
