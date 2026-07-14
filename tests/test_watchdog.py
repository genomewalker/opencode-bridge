"""Progress deadline: STALL and CYCLE, on a fake clock.

The interesting cases are the ones that must NOT be killed -- a panel that only
reasons, an edit/test loop that is genuinely converging, a paginating poll that
drains. Every event payload here is shaped like the real thing (codex camelCase
tokenUsage, claude stream-json blocks), because keying on the wrong field name
is the failure mode that silently disables the detector.
"""

from chitta_bridge.watchdog import ProgressWatch


def _watch(t0=0.0):
    return ProgressWatch(t0, progress_timeout=900.0, cycle_window=600.0,
                         cycle_repeats=3)


def test_pure_reasoning_never_stalls():
    """A panelist that only thinks: no actions, but tokens climb."""
    w, t = _watch(), 0.0
    for i in range(1, 41):          # 40 * 60s = 40 min, well past 900s
        t += 60.0
        w.feed_codex("thread/tokenUsage/updated",
                     {"tokenUsage": {"total": {"outputTokens": 0,
                                               "reasoningOutputTokens": 100 * i}}}, t)
        assert w.verdict(t) is None, f"killed a live reasoning turn at t={t}"


def test_heartbeat_without_progress_stalls():
    """Events keep flowing (hooks), but nothing advances. This is the whole point."""
    w, t = _watch(), 0.0
    for _ in range(20):
        t += 60.0
        w.feed_claude({"type": "system", "subtype": "hook_started"}, t)
    assert "stalled" in (w.verdict(t) or ""), "a hook heartbeat kept a dead turn alive"


def test_claude_thinking_tokens_are_progress():
    """The claude analogue: silent reasoning emits only system/thinking_tokens."""
    w, t = _watch(), 0.0
    for i in range(1, 41):
        t += 60.0
        w.feed_claude({"type": "system", "subtype": "thinking_tokens",
                       "estimated_tokens": 50 * i}, t)
        assert w.verdict(t) is None


def test_repeated_command_cycles():
    """Same command, same tree, same output, 3x -> looping."""
    w, t = _watch(), 0.0
    for _ in range(3):
        t += 60.0
        w.feed_codex("item/completed",
                     {"item": {"type": "commandExecution", "command": "pytest",
                               "cwd": "/r", "exitCode": 1,
                               "aggregatedOutput": "1 failed"}}, t)
    assert "looping" in (w.verdict(t) or "")


def test_alternating_thrash_cycles():
    """A->B->A->B->A. The naive 'novel digest clears the counters' rule misses this."""
    w, t = _watch(), 0.0
    for cmd in ["A", "B", "A", "B", "A"]:
        t += 60.0
        w.feed_codex("item/completed",
                     {"item": {"type": "commandExecution", "command": cmd,
                               "cwd": "/r", "exitCode": 1, "aggregatedOutput": "x"}}, t)
    assert "looping" in (w.verdict(t) or "")


def test_edit_then_retest_is_not_a_cycle():
    """The false positive that matters: identical failing test, but the tree moved."""
    w, t = _watch(), 0.0
    for i in range(6):
        t += 30.0
        w.feed_codex("item/completed",
                     {"item": {"type": "fileChange",
                               "changes": [{"path": "/r/a.py", "content": f"v{i}"}],
                               "status": "completed"}}, t)
        t += 30.0
        w.feed_codex("item/completed",
                     {"item": {"type": "commandExecution", "command": "pytest",
                               "cwd": "/r", "exitCode": 1,
                               "aggregatedOutput": "1 failed"}}, t)
        assert w.verdict(t) is None, f"killed a converging edit/test loop at t={t}"


def test_noise_in_output_does_not_hide_a_cycle():
    """Identical reruns differ only in pid/duration. Unscrubbed, CYCLE never fires."""
    w, t = _watch(), 0.0
    for i in range(3):
        t += 60.0
        w.feed_codex("item/completed",
                     {"item": {"type": "commandExecution", "command": "poll",
                               "cwd": "/r", "exitCode": 1,
                               "aggregatedOutput": f"pid 91{i}2 took {i}.4s"}}, t)
    assert "looping" in (w.verdict(t) or "")


def test_draining_pagination_is_not_a_cycle():
    """Same command each time, but the result genuinely advances."""
    w, t = _watch(), 0.0
    for i in range(20):
        t += 30.0
        w.feed_codex("item/completed",
                     {"item": {"type": "commandExecution", "command": "fetch --next",
                               "cwd": "/r", "exitCode": 0,
                               "aggregatedOutput": f"page {i}: {'ab' * (i + 1)}"}}, t)
        assert w.verdict(t) is None, f"killed a draining poll at t={t}"


def test_rate_limit_backoff_is_not_a_stall():
    """An announced wait is the server's choice, not a wedged turn."""
    w, t = _watch(), 0.0
    w.feed_claude({"type": "rate_limit_event", "rate_limit_info": {}}, t)
    t += 800.0
    assert w.verdict(t) is None
    t += 200.0                      # past progress_timeout from t0, still suspended
    assert w.verdict(t) is None


def test_claude_tool_use_pairs_into_an_action():
    """tool_use (assistant) pairs with tool_result (user) by tool_use_id."""
    w, t = _watch(), 0.0
    for _ in range(3):
        t += 60.0
        w.feed_claude({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "tu1", "name": "Bash",
             "input": {"command": "ls"}}]}}, t)
        t += 1.0
        w.feed_claude({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "same"}]}}, t)
    assert "looping" in (w.verdict(t) or ""), "claude tool_use/tool_result never paired"


def test_claude_edit_moves_the_tree():
    """Same as the codex edit/retest case, through the claude event shapes."""
    w, t = _watch(), 0.0
    for i in range(6):
        t += 30.0
        w.feed_claude({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"e{i}", "name": "Edit",
             "input": {"file_path": "/r/a.py", "new_string": f"v{i}"}}]}}, t)
        w.feed_claude({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"e{i}", "content": "ok"}]}}, t)
        t += 30.0
        w.feed_claude({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t", "name": "Bash",
             "input": {"command": "pytest"}}]}}, t)
        w.feed_claude({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "1 failed"}]}}, t)
        assert w.verdict(t) is None, f"killed a converging claude edit/test loop at t={t}"


def test_left_shrinks_toward_the_deadline():
    w = _watch()
    assert w.left(0.0) == 900.0
    assert w.left(300.0) == 600.0
