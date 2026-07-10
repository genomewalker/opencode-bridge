#!/usr/bin/env python3
"""Decision-bet experiment (v2): does the [cited:N]/[asserted] grounding tag
actually change synthesis output, or are we hardening a no-op?

v1 was confounded — claude -p re-verified against the live repo (both syntheses
cited a commit newer than the transcript). This version removes every confound:

  * CONTROLLED synthetic transcript — a neutral, uncheckable factual fork
    (fictional treaty date). The cited side and the asserted side claim OPPOSITE
    answers, neither externally verifiable, so the ONLY signal distinguishing
    them is the grounding tag.
  * Tool-free + run from an empty temp dir, so the model cannot read anything.
  * VARIANCE BASELINE — run each condition twice. Compare cross-condition
    similarity (tagged vs stripped) against within-condition similarity
    (tagged vs tagged). Tag effect is real only if cross << within.
  * DIRECTIONAL read — does the synthesis endorse the CITED answer (1847) or the
    ASSERTED one (1852)? If the tag steers, tagged runs should favor 1847 more.

Usage: python .scripts/ab_grounding_tag.py [--model claude-opus-4-8]
Manual experiment (real model calls), not a unit test — hence .scripts/.
"""
import argparse
import asyncio
import difflib
import os
import re
import tempfile

# Cited side = 1847, asserted side = 1852. Same prose in both conditions;
# only the trailing tags differ between TAGGED and STRIPPED.
_TURNS = [
    ("Alaric", "The Zorvath Treaty was signed in 1847. The Meridian Archive scan "
               "(https://archive.example/zorvath), DOI 10.4021/zorvath.1847, and "
               "arXiv:1901.04477 all date the signing to the spring of 1847.", 3),
    ("Brfinal", "No — the Zorvath Treaty was signed in 1852. The 1847 date confuses "
                "the treaty with the earlier Zorvath Accord; the binding signatures "
                "were not affixed until 1852.", 0),
    ("Alaric2", "1847 stands. The archival scan reproduces the signature page with a "
                "1847 watermark; the 1852 claim rests on no primary source.", 2),
]
_QUESTION = "Was the Zorvath Treaty signed in 1847 or 1852? Give the single most defensible date and one sentence why."


def _transcript(tagged: bool) -> str:
    lines = ["# Discussion: Zorvath Treaty signing date", ""]
    for name, text, score in _TURNS:
        tag = ""
        if tagged:
            tag = f" [cited:{score}]" if score > 0 else " [asserted: no citations]"
        lines.append(f"**{name}:**{tag} {text}\n")
    return "\n".join(lines)


_PROMPT = (
    "You are a neutral synthesizer. Messages may be tagged [cited:N] (N citation-shaped "
    "references, unverified) or [asserted: no citations]. Given only the discussion below, "
    "answer the question. Be concise (<=4 sentences).\n\nQUESTION: " + _QUESTION + "\n\n{t}"
)


def _valid(out: str) -> bool:
    """A usable synthesis: not an error/empty, and actually picks a date."""
    return bool(out) and not out.startswith("[error") and ("1847" in out or "1852" in out)


async def _run(model: str, tagged: bool, tries: int = 3) -> str:
    from chitta_bridge.rooms import RoomManager
    from chitta_bridge.backends.codex import CodexBridge
    from chitta_bridge.backends.local import LocalModelBridge
    rm = RoomManager(CodexBridge(), LocalModelBridge())
    # allowed_tools=[] → `--allowedTools ""` (deny all); cwd is an empty temp dir.
    out = ""
    for _ in range(tries):  # retry: a timed-out/empty run must not corrupt the baseline
        out = await rm._run_claude_p(_PROMPT.format(t=_transcript(tagged)),
                                     model=model, timeout=600, allowed_tools=[])
        if _valid(out):
            return out
    return out  # return last (invalid) so caller can see + exclude it


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _endorses(text: str) -> str:
    # crude: which date does the answer land on
    has47, has52 = "1847" in text, "1852" in text
    first = min((text.find("1847") if has47 else 10**9),
                (text.find("1852") if has52 else 10**9))
    if not has47 and not has52:
        return "?"
    # endorsed = the date that appears in the first (headline) sentence
    head = text[:max(0, first) + 120].lower()
    if "1847" in head and "1852" not in head:
        return "1847(cited)"
    if "1852" in head and "1847" not in head:
        return "1852(asserted)"
    return "1847(cited)" if head.find("1847") < head.find("1852") else "1852(asserted)"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()

    os.chdir(tempfile.mkdtemp(prefix="abtag-"))  # no repo to read from here

    # 3 samples per condition — more signal, and survives one bad run.
    N = 3
    outs = await asyncio.gather(*([_run(args.model, True) for _ in range(N)]
                                  + [_run(args.model, False) for _ in range(N)]))
    tagged = [o for o in outs[:N] if _valid(o)]
    stripped = [o for o in outs[N:] if _valid(o)]

    print(f"# A/B grounding-tag test v2 (model={args.model})")
    print(f"valid runs: tagged {len(tagged)}/{N}, stripped {len(stripped)}/{N}")
    for i, o in enumerate(outs):
        cond = "TAGGED" if i < N else "STRIPPED"
        print(f"  {cond}#{i%N}: {'OK' if _valid(o) else 'INVALID'} ({len(o)} chars) "
              f"endorses={_endorses(o) if _valid(o) else '-'}")
    if len(tagged) < 2 or len(stripped) < 2:
        raise SystemExit("\nNot enough valid runs to compute a baseline — re-run.")

    def _avg_pairs(xs, ys, same):
        vals = [_sim(a, b) for i, a in enumerate(xs) for j, b in enumerate(ys)
                if not (same and i >= j)]
        return sum(vals) / len(vals) if vals else float("nan")

    within = (_avg_pairs(tagged, tagged, True) + _avg_pairs(stripped, stripped, True)) / 2
    cross = _avg_pairs(tagged, stripped, False)
    print(f"\nwithin-condition similarity (noise floor): {within:.3f}")
    print(f"cross-condition  similarity (tag effect):  {cross:.3f}")
    verdict = ("tag has REAL effect (cross materially below noise floor)"
               if within - cross > 0.12 else
               "NO-OP: cross ≈ within, difference is just model noise")
    print(f"=> {verdict}\n")
    print("Directional — cited=1847, asserted=1852:")
    print(f"  TAGGED endorses:   {[_endorses(o) for o in tagged]}")
    print(f"  STRIPPED endorses: {[_endorses(o) for o in stripped]}\n")
    for label, txt in [("TAGGED#0", tagged[0]), ("STRIPPED#0", stripped[0])]:
        print("=" * 60, f"\n{label}:\n", "=" * 60, "\n", txt.strip(), "\n", sep="")


if __name__ == "__main__":
    asyncio.run(main())
