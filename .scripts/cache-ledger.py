#!/usr/bin/env python3
"""
Cache efficiency ledger for Claude Code sessions.

Usage:
  python3 cache-ledger.py [--session <jsonl-path>] [--last N]

Computes per-turn cache_read / total_input and flags turns where:
  - cache efficiency drops >20pp
  - a soul_context injection coincides with the drop

Formula (per Opus 4.8's correction):
  cache_efficiency = cache_read / (input + cache_write + cache_read)
"""
import argparse
import json
import os
import sys
from pathlib import Path


# Opus pricing ($/MTok)
RATES = {
    "input":       15.0,
    "output":      75.0,
    "cache_write": 18.75,   # 1.25× input
    "cache_read":   1.50,   # 0.10× input
}


def find_sessions(last_n: int) -> list[Path]:
    projects = Path.home() / ".claude" / "projects"
    files = sorted(projects.glob("*/*.jsonl"), key=os.path.getmtime, reverse=True)
    return files[:last_n]


def parse_session(path: Path) -> list[dict]:
    turns = []
    pending_soul = False

    for raw in open(path, errors="replace"):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue

        typ = d.get("type", "")

        # Detect soul_context injection in user turn (additionalContext)
        if typ == "user":
            content = d.get("message", {}).get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text += block.get("text", "") + " " + str(block.get("content", ""))
            if "[soul]" in text or "additionalContext" in text or "[correction]" in text:
                pending_soul = True

        # Extract usage from assistant turns
        if typ == "assistant":
            msg = d.get("message", {})
            usage = msg.get("usage", {})
            if not usage:
                continue

            inp  = usage.get("input_tokens", 0)
            out  = usage.get("output_tokens", 0)
            cw   = usage.get("cache_creation_input_tokens", 0)
            cr   = usage.get("cache_read_input_tokens", 0)
            total = inp + cw + cr

            efficiency = cr / total if total > 0 else 0.0

            cost = (
                inp  * RATES["input"]       / 1_000_000 +
                out  * RATES["output"]      / 1_000_000 +
                cw   * RATES["cache_write"] / 1_000_000 +
                cr   * RATES["cache_read"]  / 1_000_000
            )

            turns.append({
                "n":          len(turns) + 1,
                "input":      inp,
                "output":     out,
                "cache_write": cw,
                "cache_read": cr,
                "total_input": total,
                "efficiency": efficiency,
                "cost_usd":   cost,
                "soul_inject": pending_soul,
            })
            pending_soul = False

    return turns


def analyse(turns: list[dict]) -> None:
    if not turns:
        print("No turns with usage data found.")
        return

    # Per-turn table
    print(f"{'#':>3}  {'in':>7}  {'out':>6}  {'cw':>7}  {'cr':>7}  "
          f"{'eff%':>6}  {'cost$':>7}  soul  flags")
    print("-" * 72)

    prev_eff = None
    total_cost = 0.0
    flags_count = 0

    for t in turns:
        eff_pct = t["efficiency"] * 100
        drop_flag = ""
        if prev_eff is not None:
            drop = prev_eff - eff_pct
            if drop > 20:
                drop_flag = f"⚠ -{drop:.0f}pp"
                if t["soul_inject"]:
                    drop_flag += " [soul]"
                flags_count += 1

        soul_mark = "✓" if t["soul_inject"] else " "
        print(f"{t['n']:>3}  {t['input']:>7,}  {t['output']:>6,}  "
              f"{t['cache_write']:>7,}  {t['cache_read']:>7,}  "
              f"{eff_pct:>5.1f}%  ${t['cost_usd']:>6.4f}  {soul_mark:<4}  {drop_flag}")

        prev_eff = eff_pct
        total_cost += t["cost_usd"]

    # Summary
    total_input  = sum(t["input"]       for t in turns)
    total_output = sum(t["output"]      for t in turns)
    total_cw     = sum(t["cache_write"] for t in turns)
    total_cr     = sum(t["cache_read"]  for t in turns)
    total_all    = total_input + total_cw + total_cr
    avg_eff      = total_cr / total_all * 100 if total_all else 0

    print("-" * 72)
    print(f"{'TOT':>3}  {total_input:>7,}  {total_output:>6,}  "
          f"{total_cw:>7,}  {total_cr:>7,}  "
          f"{avg_eff:>5.1f}%  ${total_cost:>6.4f}")
    print()
    print(f"Turns: {len(turns)}  |  Avg cache efficiency: {avg_eff:.1f}%  "
          f"|  Total cost: ${total_cost:.4f}  |  Efficiency drops: {flags_count}")

    if avg_eff < 50:
        print("\n⚠  Cache efficiency below 50% — soul_context may be polluting history.")
    elif avg_eff > 80:
        print("\n✓  Cache efficiency healthy (>80%).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="Path to specific .jsonl file")
    ap.add_argument("--last",    type=int, default=1,
                    help="Analyse last N sessions (default: 1)")
    args = ap.parse_args()

    if args.session:
        paths = [Path(args.session)]
    else:
        paths = find_sessions(args.last)

    if not paths:
        sys.exit("No session files found.")

    for path in paths:
        print(f"\n{'='*72}")
        print(f"Session: {path.name}")
        print(f"{'='*72}")
        turns = parse_session(path)
        analyse(turns)


if __name__ == "__main__":
    main()
