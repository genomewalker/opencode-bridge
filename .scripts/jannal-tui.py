#!/usr/bin/env python3
"""
jannal-tui — TUI token/cost monitor hijacking /tmp/jannal.log.

Usage:
  python3 jannal-tui.py           # live (default)
  python3 jannal-tui.py --hours 6 # last 6h
  python3 jannal-tui.py --all     # all time
  python3 jannal-tui.py --groups N # show last N groups

Keybindings:
  r  refresh   t  toggle window   g  toggle group/request view   q  quit
"""

import re, sys, time, argparse, math, json
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict, deque

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static, Label, Input
from textual.containers import ScrollableContainer, Vertical, Horizontal
from textual.reactive import reactive
from textual.binding import Binding

LOG_PATH = Path("/tmp/jannal.log")
REFRESH  = 8   # seconds

# ── Caches ───────────────────────────────────────────────────────────────────
_sess_index_cache: list = []
_sess_index_ts:   float = 0.0
SESS_INDEX_TTL = 60.0   # rebuild session index every 60s

_log_groups:   dict  = {}
_log_file_pos: int   = 0  # byte offset — only parse new lines

IN_PRICE  = 3.00
OUT_PRICE = 15.00
CR_PRICE  = 0.30
CW_PRICE  = 3.75

# ── Log parser ──────────────────────────────────────────────────────────────

RE_REQ  = re.compile(
    r'\[R(\d+)\]\s+(\S+)\s+\|\s+(\d+)\s+segs\s+\|\s+~(\d+)\s+tokens\s+\|\s+\$([0-9.]+)'
)
RE_RESP = re.compile(
    r'→\s+\[R(\d+)\]\s+Response:\s+(\d+)\s+in\s*/\s*(\d+)\s+out\s+\[(\w+)\]\s+\|\s+\$([0-9.]+)'
    r'(?:\s+\(cache:\s+(\d+)\s+read,\s+(\d+)\s+created\))?'
    r'(?:\s+\|\s+tools:\s+(.+))?'
)
RE_GROUP = re.compile(
    r'\[group\]\s+(NEW|SAME)\s+group=(\d+)\s+reason=(\S+)\s+model=(\S+)\s+msgs=(\d+)'
)


def parse_log(since_ts: float = 0.0):
    """Parse jannal.log → dict of groups (incremental: only reads new bytes)."""
    global _log_groups, _log_file_pos
    if not LOG_PATH.exists():
        return _log_groups

    try:
        with open(LOG_PATH, "rb") as fh:
            fh.seek(_log_file_pos)
            new_bytes = fh.read()
            new_pos   = _log_file_pos + len(new_bytes)
    except Exception:
        return _log_groups

    if not new_bytes:
        return _log_groups

    lines = new_bytes.decode(errors="replace").splitlines()

    pending = {}
    groups  = dict(_log_groups)   # start from cached state

    for line in lines:
        line = line.strip()

        m = RE_REQ.search(line)
        if m:
            rid = int(m.group(1))
            pending[rid] = {
                "rid": rid, "model": m.group(2),
                "segs": int(m.group(3)), "est_tokens": int(m.group(4)),
                "est_cost": float(m.group(5)),
                "in_tokens": 0, "out_tokens": 0,
                "cr": 0, "cw": 0, "actual_cost": 0.0,
                "stop": "", "tools": [],
                "group_id": None, "reason": "",
            }
            continue

        m = RE_RESP.search(line)
        if m:
            rid = int(m.group(1))
            if rid in pending:
                pending[rid]["in_tokens"]    = int(m.group(2))
                pending[rid]["out_tokens"]   = int(m.group(3))
                pending[rid]["stop"]         = m.group(4)
                pending[rid]["actual_cost"]  = float(m.group(5))
                pending[rid]["cr"]           = int(m.group(6) or 0)
                pending[rid]["cw"]           = int(m.group(7) or 0)
                raw_tools = m.group(8) or ""
                pending[rid]["tools"]        = [t.strip() for t in raw_tools.split(",") if t.strip()]
            continue

        m = RE_GROUP.search(line)
        if m:
            is_new   = m.group(1) == "NEW"
            gid      = int(m.group(2))
            reason   = m.group(3)
            msgs     = int(m.group(5))

            # Find the most-recently-seen pending request not yet assigned
            # (jannal logs group line after the request line)
            latest_rid = max((r for r in pending if pending[r]["group_id"] is None), default=None)
            if latest_rid is None:
                continue

            req = pending[latest_rid]
            req["group_id"] = gid
            req["reason"]   = reason

            if gid not in groups:
                groups[gid] = {
                    "gid": gid,
                    "model": req["model"],
                    "requests": [],
                    "is_subagent": False,
                    "first_ts": time.time(),  # approximate
                    "max_msgs": 0,
                }

            g = groups[gid]
            g["requests"].append(req)
            g["max_msgs"] = max(g["max_msgs"], msgs)
            if "subagent" in reason:
                g["is_subagent"] = True
            del pending[latest_rid]

    _log_groups   = groups
    _log_file_pos = new_pos if _log_file_pos > 0 else LOG_PATH.stat().st_size
    return groups


def group_stats(g: dict) -> dict:
    reqs = g["requests"]
    if not reqs:
        return {}
    total_in  = sum(r["in_tokens"]   for r in reqs)
    total_out = sum(r["out_tokens"]  for r in reqs)
    total_cr  = sum(r["cr"]          for r in reqs)
    total_cw  = sum(r["cw"]          for r in reqs)
    total_cost = sum(r["actual_cost"] for r in reqs)
    max_segs  = max(r["segs"] for r in reqs)
    max_tok   = max(r["est_tokens"] for r in reqs)
    tools_all : list = []
    for r in reqs:
        tools_all.extend(r["tools"])
    tool_counts: dict = {}
    for t in tools_all:
        tool_counts[t] = tool_counts.get(t, 0) + 1
    top_tools = sorted(tool_counts, key=lambda k: -tool_counts[k])[:4]
    ctx_growth = [r["segs"] for r in reqs[-20:]]

    return {
        "n_reqs": len(reqs),
        "total_in": total_in, "total_out": total_out,
        "total_cr": total_cr, "total_cw": total_cw,
        "total_cost": total_cost,
        "max_segs": max_segs, "max_tok": max_tok,
        "top_tools": top_tools,
        "ctx_growth": ctx_growth,
        "last_stop": reqs[-1]["stop"],
        "last_reason": reqs[-1]["reason"],
    }


def sparkline(values: list, width: int = 10) -> str:
    """Mini sparkline using braille-style block chars."""
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    span = max(mx - mn, 1)
    chars = []
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]
    for v in sampled:
        idx = int((v - mn) / span * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars).ljust(width)


def ascii_lineplot(values: list, labels: list = None, title: str = "",
                   width: int = 72, height: int = 7, fmt=None) -> list[str]:
    """Return lines of an ASCII line chart (list of Rich-markup strings)."""
    if not values:
        return [f"  {title}  (no data)"]
    mn, mx = min(values), max(values)
    span = max(mx - mn, 1e-12)
    fmt = fmt or (lambda v: f"${v:.3f}")

    # downsample to width columns
    step = max(1, len(values) / width)
    pts: list[float] = [values[min(int(i * step), len(values) - 1)] for i in range(width)]

    # build grid (row 0 = top)
    grid = [[" "] * width for _ in range(height)]
    prev_row = None
    for col, v in enumerate(pts):
        row = height - 1 - int((v - mn) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "●"
        if prev_row is not None and abs(row - prev_row) > 1:
            lo, hi = sorted([row, prev_row])
            for r in range(lo + 1, hi):
                if grid[r][col] == " ":
                    grid[r][col] = "│"
        prev_row = row

    LABEL_W = 9
    lines = []
    if title:
        lines.append(f"  [bold]{title}[/]")
    for r, row in enumerate(grid):
        label_val = mx - span * r / max(height - 1, 1)
        label = fmt(label_val).rjust(LABEL_W)
        row_str = "".join(row)
        # colour the dots
        row_str = row_str.replace("●", "[cyan]●[/]").replace("│", "[dim]│[/]")
        lines.append(f"  [dim]{label}[/] [dim]│[/]{row_str}")
    lines.append(f"  {'':>{LABEL_W}} [dim]└{'─' * width}[/]")
    # x-axis tick labels (group ids or indices)
    if labels:
        tick_step = max(1, len(labels) // 8)
        tick_line = [" "] * width
        for i in range(0, len(labels), tick_step):
            col = min(int(i / max(len(labels) - 1, 1) * (width - 1)), width - 1)
            lbl = str(labels[min(int(i * step), len(labels) - 1)])[:5]
            for j, ch in enumerate(lbl):
                if col + j < width:
                    tick_line[col + j] = ch
        lines.append(f"  {'':>{LABEL_W}}  {''.join(tick_line)}")
    return lines


def build_session_index(force: bool = False) -> list:
    """Return list of {turns, proj} sorted by turns desc, for project matching."""
    global _sess_index_cache, _sess_index_ts
    now = time.time()
    if not force and _sess_index_cache and (now - _sess_index_ts) < SESS_INDEX_TTL:
        return _sess_index_cache

    projects_dir = Path.home() / ".claude/projects"
    sessions = []
    for proj in projects_dir.iterdir():
        if not proj.is_dir():
            continue
        for f in proj.glob("*.jsonl"):
            turns = 0
            try:
                for line in open(f, errors="replace"):
                    # fast path: avoid json.loads on every line
                    if '"type":"user"' in line or '"type": "user"' in line \
                            or '"type":"assistant"' in line or '"type": "assistant"' in line:
                        turns += 1
            except Exception:
                pass
            label = proj.name
            for prefix in ("-maps-projects-", "-home-kbd606-"):
                if label.startswith(prefix):
                    label = label[len(prefix):]
            parts = label.split("-")
            label = parts[-1] if len(parts) > 3 else label
            sessions.append({"turns": turns, "proj": label})
    sessions.sort(key=lambda s: s["turns"], reverse=True)

    _sess_index_cache = sessions
    _sess_index_ts    = now
    return sessions


def match_project(max_msgs: int, index: list) -> str:
    if not index:
        return "?"
    best = min(index, key=lambda s: abs(s["turns"] - max_msgs))
    diff = abs(best["turns"] - max_msgs)
    if diff > max(10, max_msgs * 0.15):
        return "?"
    return best["proj"][:20]


def fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n//1_000}k"
    return str(n)


# ── App ──────────────────────────────────────────────────────────────────────

class JannalTUI(App):
    TITLE = "jannal-tui"
    SUB_TITLE = "hijacking /tmp/jannal.log"

    CSS = """
    Screen { layout: vertical; }
    #summary {
        height: 3;
        background: $surface;
        border: tall $primary;
        padding: 0 2;
        content-align: center middle;
        text-style: bold;
        margin: 0 0 1 0;
    }
    #filter-bar {
        height: 1;
        layout: horizontal;
        margin: 0 0 1 0;
    }
    #filter-label { width: 10; content-align: left middle; color: $accent; }
    #filter-input { width: 1fr; }
    #sort-label   { width: 22; content-align: right middle; color: $text-muted; }
    .section-label {
        text-style: bold; color: $accent; padding: 0 1; height: 1;
    }
    DataTable { height: auto; max-height: 17; margin: 0 0 1 0; }
    #agents-table { max-height: 10; }
    """

    BINDINGS = [
        Binding("r",      "refresh",       "Refresh",      priority=True),
        Binding("t",      "toggle_window", "Window",       priority=True),
        Binding("s",      "cycle_sort",    "Sort",         priority=True),
        Binding("a",      "toggle_active", "Active only",  priority=True),
        Binding("p",      "toggle_proj",   "Page",         priority=True),
        Binding("slash",  "focus_filter",  "Filter"),
        Binding("escape", "clear_filter",  "Clear filter"),
        Binding("q",      "quit",          "Quit",         priority=True),
    ]

    # per-page sort keys — (label, key_fn)
    PAGE_SORTS = {
        0: [  # main groups
            ("newest",  lambda g, s: -g["gid"]),
            ("cost",    lambda g, s: -s.get("total_cost", 0)),
            ("segs",    lambda g, s: -s.get("max_segs", 0)),
            ("reqs",    lambda g, s: -s.get("n_reqs", 0)),
            ("cache$",  lambda g, s: -s.get("total_cr", 0)),
        ],
        1: [  # projects
            ("cost",    lambda p, v: -(v["cost"] + v["agent_cost"])),
            ("cache%",  lambda p, v: -(v["cr"] / max(v["cr"] + v["cw"], 1))),
            ("reqs",    lambda p, v: -v["reqs"]),
            ("groups",  lambda p, v: -v["groups"]),
            ("agents$", lambda p, v: -v["agent_cost"]),
        ],
        2: [  # trends
            ("oldest",  lambda x: x[0]),
            ("newest",  lambda x: -x[0]),
            ("cost",    lambda x: -x[1]),
            ("cw/req",  lambda x: -x[2]),
            ("segs",    lambda x: -x[3]),
        ],
    }

    _since_ts:    float = 0.0
    _window_label: str  = "all"
    _windows = [("all", 0), ("today", -1), ("6h", 6), ("1h", 1)]
    _widx        = 0
    _sort_idxs   = {0: 0, 1: 0, 2: 0}   # per-page sort index
    _active_only = False
    _filter_str  = ""
    _page        = 0   # 0=main 1=projects 2=trends
    _pages       = ["main", "projects", "trends"]
    _n_groups:  int = 50

    def __init__(self, since_ts: float, label: str, n_groups: int):
        super().__init__()
        self._since_ts    = since_ts
        self._window_label = label
        self._n_groups    = n_groups

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="summary")
        # filter bar (main view only)
        yield Horizontal(
            Label("Filter: ", id="filter-label"),
            Input(placeholder="project / tool name …", id="filter-input"),
            Label("", id="sort-label"),
            id="filter-bar",
        )
        # ── Page 0: main groups view ──
        yield ScrollableContainer(
            Label("● CONVERSATIONS (groups)", classes="section-label"),
            DataTable(id="main-table", zebra_stripes=True, cursor_type="row"),
            Label("◎ SUBAGENT GROUPS", classes="section-label"),
            DataTable(id="agents-table", zebra_stripes=True, cursor_type="row"),
            id="main-view",
        )
        # ── Page 1: project summary ──
        yield ScrollableContainer(
            Static("", id="proj-content"),
            id="proj-view",
        )
        # ── Page 2: trends ──
        yield ScrollableContainer(
            Static("", id="trend-content"),
            id="trend-view",
        )
        yield Footer()

    def on_mount(self):
        mt = self.query_one("#main-table", DataTable)
        mt.add_columns(
            "Grp", "Project", "Reqs", "Segs", "Ctx~", "CacheR",
            "Out", "Cost$", "Spark(ctx)", "Tools", "Stop"
        )
        at = self.query_one("#agents-table", DataTable)
        at.add_columns("Grp", "Project", "Reqs", "Reason", "Cost$", "Tools")
        # hide non-active views
        self.query_one("#proj-view").display  = False
        self.query_one("#trend-view").display = False
        self._update_sort_label()
        self._update_subtitle()
        self.action_refresh()
        self.set_interval(REFRESH, self.action_refresh)
        self.query_one("#main-table", DataTable).focus()

    def _update_subtitle(self):
        active_tag = "  [active]" if self._active_only else ""
        page_name  = self._pages[self._page]
        self.sub_title = (
            f"[{page_name}] p=next-page  |  "
            f"window: {self._window_label}{active_tag}  |  "
            f"refresh {REFRESH}s"
        )

    def _update_sort_label(self):
        sorts = self.PAGE_SORTS[self._page]
        name  = sorts[self._sort_idxs[self._page]][0]
        self.query_one("#sort-label", Label).update(f"sort: [{name}] s=cycle")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._filter_str = event.value.lower().strip()
            self.action_refresh()

    def action_focus_filter(self):
        self.query_one("#filter-input", Input).focus()

    def action_clear_filter(self):
        inp = self.query_one("#filter-input", Input)
        inp.value = ""
        self._filter_str = ""
        self.set_focus(None)
        self.action_refresh()

    def action_toggle_proj(self):
        self._page = (self._page + 1) % len(self._pages)
        self.query_one("#main-view").display  = (self._page == 0)
        self.query_one("#proj-view").display  = (self._page == 1)
        self.query_one("#trend-view").display = (self._page == 2)
        self.query_one("#filter-bar").display = (self._page == 0)
        if self._page == 0:
            self.query_one("#main-table", DataTable).focus()
        self._update_subtitle()
        self.action_refresh()

    def action_refresh(self):
        groups   = parse_log(self._since_ts)
        sess_idx = build_session_index()
        if self._page == 0:
            self._render(groups, sess_idx)
        elif self._page == 1:
            self._render_projects(groups, sess_idx)
        else:
            self._render_trends(groups, sess_idx)

    def action_cycle_sort(self):
        sorts = self.PAGE_SORTS[self._page]
        self._sort_idxs[self._page] = (self._sort_idxs[self._page] + 1) % len(sorts)
        self._update_sort_label()
        self.action_refresh()

    def action_toggle_active(self):
        self._active_only = not self._active_only
        self._update_subtitle()
        self.action_refresh()

    def action_toggle_window(self):
        self._widx = (self._widx + 1) % len(self._windows)
        label, hours = self._windows[self._widx]
        now = time.time()
        if hours == 0:
            self._since_ts = 0.0
            self._window_label = "all"
        elif hours == -1:
            self._since_ts = datetime.combine(date.today(), datetime.min.time()).timestamp()
            self._window_label = "today"
        else:
            self._since_ts = now - hours * 3600
            self._window_label = f"last {hours}h"
        self._update_subtitle()
        self.action_refresh()

    def _render(self, groups: dict, sess_index: list = None):
        # ── annotate groups with project and stats for sorting/filtering ──
        annotated = []
        for g in groups.values():
            s = group_stats(g)
            if not s:
                continue
            proj = match_project(g["max_msgs"], sess_index or [])
            g["_proj"] = proj
            g["_stats"] = s
            annotated.append(g)

        # ── filter ──
        if self._filter_str:
            f = self._filter_str
            annotated = [
                g for g in annotated
                if f in g["_proj"].lower()
                or any(f in t.lower() for t in g["_stats"]["top_tools"])
            ]

        # ── active-only filter ──
        if self._active_only:
            annotated = [g for g in annotated if g["_stats"]["last_stop"] == "tool_use"]

        # ── sort ──
        sort_fn = self.PAGE_SORTS[0][self._sort_idxs[0]][1]
        annotated.sort(key=lambda g: sort_fn(g, g["_stats"]))

        main_groups  = [g for g in annotated if not g["is_subagent"]]
        agent_groups = [g for g in annotated if g["is_subagent"]]

        # ── Summary bar ──
        def totals(gs):
            t_cost = t_cr = t_cw = t_out = 0
            for g in gs:
                s = group_stats(g)
                if not s:
                    continue
                t_cost += s["total_cost"]
                t_cr   += s["total_cr"]
                t_cw   += s["total_cw"]
                t_out  += s["total_out"]
            return t_cost, t_cr, t_cw, t_out

        mc, mcr, mcw, mo = totals(main_groups)
        ac, acr, acw, ao = totals(agent_groups)
        tc = mc + ac
        tcr = mcr + acr
        tcw = mcw + acw
        hit = tcr / max(tcr + tcw, 1) * 100
        woc = (tcr + tcw) * IN_PRICE / 1_000_000
        savings = woc - (tcr * CR_PRICE + tcw * CW_PRICE) / 1_000_000

        # ── Summary bar ──
        def sum_totals(gs):
            tc = tcr = tcw = tout = 0
            for g in gs:
                s = g["_stats"]
                tc  += s["total_cost"]; tcr += s["total_cr"]
                tcw += s["total_cw"];   tout += s["total_out"]
            return tc, tcr, tcw, tout

        mc, mcr, mcw, mo = sum_totals(main_groups)
        ac, acr, acw, ao = sum_totals(agent_groups)
        tc = mc + ac; tcr = mcr + acr; tcw = mcw + acw
        hit     = tcr / max(tcr + tcw, 1) * 100
        savings = (tcr + tcw) * IN_PRICE / 1_000_000 - (tcr * CR_PRICE + tcw * CW_PRICE) / 1_000_000
        n_shown = len(main_groups) + len(agent_groups)

        self.query_one("#summary", Static).update(
            f"[bold]Cost:[/] [green]${tc:.3f}[/]  "
            f"[dim]main ${mc:.3f}  agents ${ac:.3f}[/]   "
            f"[bold]Cache hit:[/] [cyan]{hit:.1f}%[/]   "
            f"[bold]Saved:[/] [yellow]${savings:.2f}[/]   "
            f"[bold]Shown:[/] {n_shown}  "
            f"[bold]Reqs:[/] {sum(len(g['requests']) for g in annotated)}"
        )

        # ── Main table ──
        mt = self.query_one("#main-table", DataTable)
        mt.clear()
        for g in main_groups[:self._n_groups]:
            s = g["_stats"]
            spark  = sparkline(s["ctx_growth"], 8)
            tools  = ",".join(s["top_tools"])[:20]
            active = "●" if s["last_stop"] == "tool_use" else " "
            mt.add_row(
                f"{active}{g['gid']}",
                g["_proj"],
                str(s["n_reqs"]),
                str(s["max_segs"]),
                fmt_k(s["max_tok"]),
                fmt_k(s["total_cr"]),
                fmt_k(s["total_out"]),
                f"${s['total_cost']:.3f}",
                spark,
                tools,
                s["last_stop"][:10],
                key=str(g["gid"]),
            )

        # ── Agents table ──
        at = self.query_one("#agents-table", DataTable)
        at.clear()
        for g in agent_groups[:30]:
            s = g["_stats"]
            tools = ",".join(s["top_tools"])[:22]
            at.add_row(
                str(g["gid"]),
                g["_proj"],
                str(s["n_reqs"]),
                s["last_reason"][:18],
                f"${s['total_cost']:.3f}",
                tools,
                key=str(g["gid"]),
            )


    def _render_projects(self, groups: dict, sess_index: list):
        """Per-project cost summary with line plot + horizontal bars."""
        proj_data: dict = defaultdict(lambda: {
            "cost": 0.0, "cr": 0, "cw": 0, "reqs": 0, "groups": 0, "agent_cost": 0.0,
        })

        # chronological cost series for the global line plot (by group id)
        sorted_groups = sorted(groups.values(), key=lambda g: g["gid"])
        cum_cost_vals: list[float] = []
        cum_cost_lbls: list[int]   = []
        running = 0.0
        for g in sorted_groups:
            s = group_stats(g)
            if not s:
                continue
            running += s["total_cost"]
            cum_cost_vals.append(running)
            cum_cost_lbls.append(g["gid"])
            proj = match_project(g["max_msgs"], sess_index or [])
            key  = proj if proj != "?" else "(unmatched)"
            if g["is_subagent"]:
                proj_data[key]["agent_cost"] += s["total_cost"]
            else:
                proj_data[key]["cost"]   += s["total_cost"]
                proj_data[key]["cr"]     += s["total_cr"]
                proj_data[key]["cw"]     += s["total_cw"]
                proj_data[key]["reqs"]   += s["n_reqs"]
                proj_data[key]["groups"] += 1

        # ── global line plot ──
        plot_lines = ascii_lineplot(
            cum_cost_vals, labels=cum_cost_lbls,
            title=f"Cumulative cost over time  (total ${running:.3f})",
            width=70, height=7,
        )

        # ── sort ──
        sort_fn = self.PAGE_SORTS[1][self._sort_idxs[1]][1]
        rows = sorted(proj_data.items(), key=lambda x: sort_fn(x[0], x[1]))
        max_cost = max((v["cost"] + v["agent_cost"] for _, v in rows), default=1)
        BAR = 38

        table_lines = [
            "",
            f"  {'Project':<22} {'Cost$':>8}  {'Bar':<{BAR}}  {'Cache%':>7}  {'Reqs':>5}  {'Agent$':>7}",
            "  " + "─" * 100,
        ]
        for proj, v in rows:
            tc      = v["cost"] + v["agent_cost"]
            bar_len = int(v["cost"] / max_cost * BAR)
            agt_len = int(v["agent_cost"] / max_cost * BAR)
            bar     = "[green]" + "█" * bar_len + "[/]" + "[yellow]" + "▒" * agt_len + "[/]"
            bar    += " " * max(0, BAR - bar_len - agt_len)
            hit     = v["cr"] / max(v["cr"] + v["cw"], 1) * 100
            table_lines.append(
                f"  {proj:<22} ${tc:>7.3f}  {bar}  {hit:>6.1f}%  "
                f"{v['reqs']:>5}  ${v['agent_cost']:>6.3f}"
            )
        table_lines += ["", "  [dim]█ = main cost   ▒ = agent cost   bar width ∝ total cost[/]"]

        self.query_one("#proj-content", Static).update(
            "\n".join(plot_lines + table_lines)
        )

    def _render_trends(self, groups: dict, sess_index: list):
        """CW/turn and cost per group over time — spot token efficiency trends."""
        annotated = []
        for g in groups.values():
            if g["is_subagent"]:
                continue
            s = group_stats(g)
            if not s or s["n_reqs"] < 2:
                continue
            cw_per_req = (s["total_cw"]) / max(s["n_reqs"], 1)
            annotated.append((g["gid"], s["total_cost"], cw_per_req, s["max_segs"], match_project(g["max_msgs"], sess_index or [])))

        sort_fn = self.PAGE_SORTS[2][self._sort_idxs[2]][1]
        annotated.sort(key=sort_fn)

        if not annotated:
            self.query_one("#trend-content", Static).update("  No data.")
            return

        # Rolling window averages (5 groups)
        W = 5
        costs  = [x[1] for x in annotated]
        cwpts  = [x[2] for x in annotated]

        def rolling(vals, w):
            out = []
            for i in range(len(vals)):
                sl = vals[max(0, i-w+1):i+1]
                out.append(sum(sl) / len(sl))
            return out

        avg_cost = rolling(costs, W)
        avg_cw   = rolling(cwpts, W)

        max_cost = max(costs, default=1)
        max_cw   = max(cwpts, default=1)
        BAR = 35

        lines = [
            f"  {'Grp':>5}  {'Project':<18}  {'Cost$':>7}  {'Cost trend':<{BAR}}  {'CW/req':>8}  {'CW trend':<{BAR}}",
            "  " + "─" * 115,
        ]

        # Trend arrow: compare last value to rolling avg
        def arrow(val, avg):
            if val > avg * 1.10: return "[red]↑[/]"
            if val < avg * 0.90: return "[green]↓[/]"
            return "[dim]→[/]"

        for i, (gid, cost, cw, segs, proj) in enumerate(annotated[-50:]):
            cb = int(cost / max_cost * BAR)
            wb = int(cw   / max_cw   * BAR)
            ca = arrow(cost, avg_cost[-(50-i)] if i < len(avg_cost) else cost)
            wa = arrow(cw,   avg_cw[-(50-i)]   if i < len(avg_cw)   else cw)
            cost_bar = "[cyan]" + "▪" * cb + "[/]" + " " * (BAR - cb)
            cw_bar   = "[magenta]" + "▪" * wb + "[/]" + " " * (BAR - wb)
            lines.append(
                f"  {gid:>5}  {proj:<18}  ${cost:>6.3f} {ca}  {cost_bar}  {fmt_k(int(cw)):>8} {wa}  {cw_bar}"
            )

        # Summary line
        first5_cost = sum(costs[:5]) / max(len(costs[:5]), 1)
        last5_cost  = sum(costs[-5:]) / max(len(costs[-5:]), 1)
        first5_cw   = sum(cwpts[:5]) / max(len(cwpts[:5]), 1)
        last5_cw    = sum(cwpts[-5:]) / max(len(cwpts[-5:]), 1)
        cost_delta  = (last5_cost - first5_cost) / max(first5_cost, 0.001) * 100
        cw_delta    = (last5_cw   - first5_cw)   / max(first5_cw,   1)    * 100
        sign_c = "+" if cost_delta > 0 else ""
        sign_w = "+" if cw_delta   > 0 else ""
        col_c  = "red" if cost_delta > 5 else "green" if cost_delta < -5 else "dim"
        col_w  = "red" if cw_delta   > 5 else "green" if cw_delta   < -5 else "dim"

        lines += [
            "",
            f"  [bold]Trend (first 5 vs last 5 groups):[/]  "
            f"cost [{col_c}]{sign_c}{cost_delta:.1f}%[/]   "
            f"CW/req [{col_w}]{sign_w}{cw_delta:.1f}%[/]",
            f"  [dim]▪ bars = absolute value   ↑/↓ = vs {W}-group rolling avg[/]",
        ]
        self.query_one("#trend-content", Static).update("\n".join(lines))


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="jannal-tui: Claude Code TUI monitor")
    p.add_argument("--hours",  type=float, help="Show last N hours")
    p.add_argument("--all",    action="store_true", help="All time (default)")
    p.add_argument("--today",  action="store_true", help="Today only")
    p.add_argument("--groups", type=int, default=50, help="Max groups to show")
    args = p.parse_args()

    now = time.time()
    if args.today:
        since = datetime.combine(date.today(), datetime.min.time()).timestamp()
        label = "today"
    elif args.hours:
        since = now - args.hours * 3600
        label = f"last {args.hours:.0f}h"
    else:
        since = 0.0
        label = "all"

    JannalTUI(since_ts=since, label=label, n_groups=args.groups).run()


if __name__ == "__main__":
    main()
