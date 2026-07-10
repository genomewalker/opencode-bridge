#!/usr/bin/env python3
"""Decision-bet experiment: does the [cited:N]/[asserted] grounding tag actually
change synthesis output? (Raised by the tier2 fusion room's opus judge.)

Runs the SAME synthesis prompt over one saved room's transcript twice — once with
the grounding tags present, once with them stripped — and diffs the two syntheses.
If the outputs are ~identical, the tag is a no-op and hardening it is wasted work.

Usage:
    python .scripts/ab_grounding_tag.py <room_id> [--model claude-opus-4-8]

Reads ~/.chitta-bridge/rooms/<room_id>.json. Needs a reachable `claude` CLI.
This is a manual experiment (real model calls), not a unit test — hence .scripts/.
"""
import argparse
import asyncio
import difflib
import re
from pathlib import Path

from chitta_bridge.rooms import RoomManager, DiscussionRoom
from chitta_bridge.backends.codex import CodexBridge
from chitta_bridge.backends.local import LocalModelBridge

_TAG_RE = re.compile(r" \[(?:cited:\d+|asserted: no citations|grounded:\d+ citations)\]")

SYNTH_PROMPT = (
    "You are a neutral synthesizer reviewing a multi-agent discussion. Produce: "
    "(1) the double-confirmed claims, (2) the strongest minority reading, "
    "(3) the single load-bearing unverified assumption. Be concise.\n\n{transcript}"
)


async def _synthesize(rm: RoomManager, transcript: str, model: str) -> str:
    prompt = SYNTH_PROMPT.format(transcript=transcript)
    return await rm._run_claude_p(prompt, model=model, timeout=900)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("room_id")
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()

    path = Path.home() / ".chitta-bridge" / "rooms" / f"{args.room_id}.json"
    if not path.exists():
        raise SystemExit(f"no such room: {path}")

    rm = RoomManager(CodexBridge(), LocalModelBridge())
    room = DiscussionRoom.load(path)

    tagged = rm._build_annotated_transcript(room)
    stripped = _TAG_RE.sub("", tagged)
    if tagged == stripped:
        raise SystemExit("transcript has no grounding tags — pick a room with scored turns")

    print(f"# A/B grounding-tag test on {args.room_id} (model={args.model})")
    print(f"tagged transcript: {len(tagged)} chars · stripped: {len(stripped)} chars\n")

    tagged_out, stripped_out = await asyncio.gather(
        _synthesize(rm, tagged, args.model),
        _synthesize(rm, stripped, args.model),
    )

    ratio = difflib.SequenceMatcher(None, tagged_out, stripped_out).ratio()
    print(f"## similarity(tagged, stripped) = {ratio:.3f}  "
          f"({'NO-OP: tag does not move output' if ratio > 0.9 else 'tag CHANGES output'})\n")
    print("=" * 70, "\nWITH TAGS:\n", "=" * 70, "\n", tagged_out, sep="")
    print("\n", "=" * 70, "\nTAGS STRIPPED:\n", "=" * 70, "\n", stripped_out, sep="")


if __name__ == "__main__":
    asyncio.run(main())
