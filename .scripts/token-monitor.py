#!/usr/bin/env python3
"""Live token usage monitor across all Claude Code sessions."""

import json, os, sys, time, glob
from pathlib import Path
from datetime import datetime, date

# Pricing (per million tokens)
IN_PRICE = 3.00
OUT_PRICE = 15.00
CR_PRICE = 0.30
CW_PRICE = 3.75


def parse_session(path):
    inp = out = cr = cw = turns = 0
    mtime = path.stat().st_mtime
    try:
        for line in open(path):
            d = json.loads(line)
            if d.get("type") in ("user", "assistant"):
                turns += 1
            u = d.get("message", {}).get("usage") or d.get("usage") or {}
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)
            cr  += u.get("cache_read_input_tokens", 0)
            cw  += u.get("cache_creation_input_tokens", 0)
    except Exception:
        pass
    return inp, out, cr, cw, turns, mtime


def cost(inp, out, cr, cw):
    return (inp * IN_PRICE + out * OUT_PRICE + cr * CR_PRICE + cw * CW_PRICE) / 1_000_000


def is_subagent(turns, cw):
    return turns <= 6 and cw > 5_000


def collect(since_ts):
    projects_dir = Path.home() / ".claude/projects"
    sessions = []
    for proj in projects_dir.iterdir():
        if not proj.is_dir():
            continue
        for f in proj.glob("*.jsonl"):
            try:
                mtime = f.stat().st_mtime
            except Exception:
                continue
            if mtime < since_ts:
                continue
            inp, out, cr, cw, turns, mtime = parse_session(f)
            if inp + out + cr + cw == 0:
                continue
            sessions.append({
                "id": f.stem[:16],
                "proj": proj.name[-30:],
                "mtime": mtime,
                "turns": turns,
                "inp": inp, "out": out, "cr": cr, "cw": cw,
                "cost": cost(inp, out, cr, cw),
                "subagent": is_subagent(turns, cw),
                "age": time.time() - mtime,
            })
    return sorted(sessions, key=lambda s: s["mtime"], reverse=True)


def render(sessions, window_label):
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")
    print(f"  Claude Token Monitor — {window_label}  [{now}]")
    print("  " + "─" * 90)

    main_s = [s for s in sessions if not s["subagent"]]
    agent_s = [s for s in sessions if s["subagent"]]

    def totals(ss):
        return (sum(s["inp"] for s in ss), sum(s["out"] for s in ss),
                sum(s["cr"] for s in ss), sum(s["cw"] for s in ss),
                sum(s["cost"] for s in ss))

    # Active sessions (touched in last 10 min)
    active = [s for s in main_s if s["age"] < 600]
    recent_main = main_s[:8]

    print(f"\n  MAIN SESSIONS (showing {len(recent_main)}, active={len(active)})")
    print(f"  {'ID':<18} {'Turns':>5} {'CW/t':>7} {'CacheR':>8} {'Out':>7} {'$':>7}  {'Project':<30}")
    print("  " + "─" * 88)
    for s in recent_main:
        age_str = f"{int(s['age']//60)}m" if s['age'] < 3600 else f"{int(s['age']//3600)}h"
        active_marker = "●" if s["age"] < 120 else " "
        cw_pt = s["cw"] // max(s["turns"], 1)
        proj = s["proj"].replace("-maps-projects-", "").replace("-home-kbd606-", "~")[:28]
        print(f"  {active_marker}{s['id']:<17} {s['turns']:>5} {cw_pt:>7,} {s['cr']//1000:>7,}k {s['out']//1000:>6,}k {s['cost']:>7.3f}  {proj} ({age_str})")

    ti, to, tcr, tcw, tc = totals(main_s)
    print(f"  {'TOTAL main':<24} {tcr//1_000_000:>7.1f}M CR  {to//1000:>6,}k out  ${tc:>7.3f}")

    print(f"\n  SUBAGENT SESSIONS ({len(agent_s)} sessions)")
    print(f"  {'ID':<18} {'Trns':>4} {'CW':>8} {'$':>7}  {'Project'}")
    print("  " + "─" * 70)
    for s in agent_s[:10]:
        proj = s["proj"].replace("-maps-projects-", "")[:30]
        print(f"  {s['id']:<18} {s['turns']:>4} {s['cw']//1000:>7,}k {s['cost']:>7.3f}  {proj}")
    if len(agent_s) > 10:
        print(f"  ... and {len(agent_s)-10} more")

    ai, ao, acr, acw, ac = totals(agent_s)
    print(f"  {'TOTAL agents':<24} {acw//1000:>7,}k CW               ${ac:>7.3f}")

    # Grand total
    print("\n  " + "─" * 90)
    gi, go, gcr, gcw, gc = totals(sessions)
    cache_hit = gcr / max(gi + gcr + gcw, 1) * 100
    without_cache = (gi * IN_PRICE + go * OUT_PRICE + (gcr + gcw) * IN_PRICE) / 1_000_000
    savings = without_cache - gc
    print(f"  GRAND TOTAL   cost=${gc:.3f}  cache-hit={cache_hit:.1f}%  savings=${savings:.2f}  sessions={len(sessions)}")
    print(f"  Prices: in=${IN_PRICE}/M  out=${OUT_PRICE}/M  cache-read=${CR_PRICE}/M  cache-write=${CW_PRICE}/M")
    print()


def main():
    # Default: today's sessions
    today_ts = datetime.combine(date.today(), datetime.min.time()).timestamp()
    since_ts = today_ts
    window_label = "today"

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "week":
            since_ts = time.time() - 7 * 86400
            window_label = "last 7 days"
        elif arg == "all":
            since_ts = 0
            window_label = "all time"
        elif arg.lstrip("-").isdigit():
            since_ts = time.time() - abs(int(arg)) * 3600
            window_label = f"last {abs(int(arg))}h"

    interval = 10  # seconds
    print(f"Monitoring (refresh every {interval}s, Ctrl-C to quit)...")
    try:
        while True:
            sessions = collect(since_ts)
            render(sessions, window_label)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
