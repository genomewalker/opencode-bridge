"""Progress deadline: a turn can be perfectly lively and still be getting nowhere.

The idle watchdog catches a turn that goes *quiet*. It cannot catch a livelocked
one -- a retry storm, an A->B->A edit thrash, a poll that never drains -- because
those emit events the whole time. Two orthogonal detectors close that gap:

  STALL  no monotone progress (tokens / items / text) for ``progress_timeout``,
         while events keep arriving.
  CYCLE  the same effectful action, against the same file state, returning the
         same result, ``cycle_repeats`` times inside ``cycle_window``.

A pure-reasoning panelist can trip neither: it runs no effectful actions, so CYCLE
has nothing to look at, and its token counters rise, so STALL cannot fire. That is
why progress is measured from token/text counters and never from the working tree
-- tying it to the tree is what would kill the rooms that touch no files.

The event field names below are not from the docs; they were read off real events
from both backends. Notably codex's wire format is camelCase
(``tokenUsage.total.outputTokens``) while its Rust structs are snake_case: keying
on ``output_tokens`` matches nothing and silently disables STALL.
"""

import hashlib
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

PROGRESS_TIMEOUT = 900.0   # no monotone progress this long, while events flow => stalled
CYCLE_WINDOW = 600.0       # sliding window for repeat detection
CYCLE_REPEATS = 3          # identical action+state+result this many times => looping

# on = kill, shadow = log the verdict only, off = disabled.
MODE = os.environ.get("CHITTA_WATCHDOG", "on").lower()

# A rerun of an identical command differs only in its clock, pids and durations.
# Without scrubbing those, no repeated command ever *looks* repeated and CYCLE is dead.
_NOISE = re.compile(r"0x[0-9a-fA-F]+|\d+")


def _sha(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode(errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


class ProgressWatch:
    """Backend-neutral. Fed by the ``feed_*`` adapters, polled by ``verdict``."""

    def __init__(self, now: float,
                 progress_timeout: float = PROGRESS_TIMEOUT,
                 cycle_window: float = CYCLE_WINDOW,
                 cycle_repeats: int = CYCLE_REPEATS) -> None:
        self.progress_timeout = progress_timeout
        self.cycle_window = cycle_window
        self.cycle_repeats = cycle_repeats
        self.key: tuple = ()
        self.t_prog = now
        self.hist: list[tuple[str, float]] = []   # [(digest, t)] inside the window
        self.files: dict[str, str] = {}           # path -> sha of last written content
        self.suspend_until = 0.0
        # Monotone accumulators; the STALL key is built from these.
        self.items = 0
        self.chars = 0
        self.tokens = 0
        self._pending: dict[str, tuple[str, str]] = {}   # tool_use_id -> (name, payload)

    # -- signals ---------------------------------------------------------
    def progress(self, now: float) -> None:
        """A strict increase in the accumulator tuple resets the stall clock.

        Monotone rather than "changed": a hash that merely changes can be reset
        forever by text that oscillates, which is the very thing we're hunting.
        """
        key = (self.items, self.tokens, self.chars)
        if key > self.key:
            self.key, self.t_prog = key, now

    def wrote(self, path: str, content: str) -> None:
        """Shadow file map, built from the edits the backend already reports.

        This is what makes `test -> edit -> test` three *different* actions rather
        than one repeated three times, and it is why we never have to shell out to
        `git status` at every action boundary (which the room's plan proposed --
        it costs a subprocess per action and has no answer for non-git dirs).
        """
        self.files[path] = _sha(content)

    def action(self, name: str, payload: str, result: str, now: float) -> None:
        """A completed *effectful* action: command, file change, MCP tool call."""
        state = _sha(*(f"{p}={s}" for p, s in sorted(self.files.items())))
        digest = _sha(name, payload, state, _NOISE.sub("#", result))
        self.hist = [(d, t) for d, t in self.hist if now - t < self.cycle_window]
        self.hist.append((digest, now))
        self.t_prog = now   # doing something effectful is progress, whatever else

    def blocked(self, now: float) -> None:
        """An externally imposed wait (rate limit). Not our stall to punish."""
        self.suspend_until = now + self.progress_timeout

    def verdict(self, now: float) -> Optional[str]:
        v = self._verdict(now)
        if v is None or MODE == "off":
            return None
        if MODE == "shadow":
            log.warning("watchdog (shadow, would kill): %s", v)
            self.t_prog = now          # don't re-fire on every wake
            self.hist.clear()
            return None
        return v

    def _verdict(self, now: float) -> Optional[str]:
        if MODE == "off":
            return None
        if now < self.suspend_until:
            self.t_prog = now
            return None
        if now - self.t_prog > self.progress_timeout:
            return f"no progress in {self.progress_timeout:.0f}s (stalled)"
        live = [d for d, t in self.hist if now - t < self.cycle_window]
        for d in set(live):
            n = live.count(d)
            if n >= self.cycle_repeats:
                return (f"same action+state+result {n}x in "
                        f"{self.cycle_window:.0f}s (looping)")
        return None

    def left(self, now: float) -> float:
        """Seconds until STALL could next fire; folds into the caller's sleep.

        CYCLE is only reachable when an action arrives, and the caller already
        wakes at least every idle_timeout, so a cycle is caught within one wake.

        Never returns <= 0 while disabled: the callers feed this straight into a
        min() for their sleep, and a negative timeout there is a busy loop.
        """
        if MODE == "off":
            return float("inf")
        return max(self.suspend_until, self.t_prog + self.progress_timeout) - now

    # -- backend adapters ------------------------------------------------
    def feed_codex(self, method: str, params: dict, now: float) -> None:
        if method == "item/agentMessage/delta":
            self.chars += len(params.get("delta") or "")
        elif method == "thread/tokenUsage/updated":
            total = (params.get("tokenUsage") or {}).get("total") or {}
            self.tokens = max(self.tokens,
                              (total.get("outputTokens") or 0)
                              + (total.get("reasoningOutputTokens") or 0))
        elif method == "item/completed":
            item = params.get("item") or {}
            kind = item.get("type")
            self.items += 1
            if kind == "commandExecution":
                self.action("cmd",
                            f"{item.get('command')}\x00{item.get('cwd')}",
                            f"{item.get('exitCode')}\x00{item.get('aggregatedOutput')}",
                            now)
            elif kind == "fileChange":
                for ch in item.get("changes") or []:
                    self.wrote(str(ch.get("path")), str(ch.get("content") or ch.get("diff") or ""))
                self.action("edit", str(item.get("changes")), str(item.get("status")), now)
            elif kind == "mcpToolCall":
                self.action("mcp",
                            f"{item.get('server')}\x00{item.get('tool')}\x00{item.get('arguments')}",
                            f"{item.get('status')}\x00{item.get('result')}",
                            now)
            elif kind == "agentMessage":
                self.chars += len(item.get("text") or "")
        self.progress(now)

    def feed_claude(self, data: dict, now: float) -> None:
        t = data.get("type")
        if t == "system":
            # Hooks fire constantly and prove only that the process breathes; counting
            # them as progress would keep a wedged turn alive forever.
            if data.get("subtype") == "thinking_tokens":
                self.tokens = max(self.tokens, data.get("estimated_tokens") or 0)
        elif t == "rate_limit_event":
            self.blocked(now)
        elif t == "assistant":
            self.items += 1
            for b in (data.get("message") or {}).get("content") or []:
                bt = b.get("type")
                if bt in ("text", "thinking"):
                    self.chars += len(b.get(bt) or "")
                elif bt == "tool_use":
                    name, inp = b.get("name") or "", b.get("input") or {}
                    path = inp.get("file_path")
                    if path and name in ("Edit", "Write", "NotebookEdit"):
                        self.wrote(str(path),
                                   str(inp.get("new_string") or inp.get("content") or ""))
                    self._pending[str(b.get("id"))] = (name, repr(sorted(inp.items())))
        elif t == "user":
            self.items += 1
            for b in (data.get("message") or {}).get("content") or []:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                pend = self._pending.pop(str(b.get("tool_use_id")), None)
                if pend is not None:
                    self.action(pend[0], pend[1], str(b.get("content")), now)
        self.progress(now)
