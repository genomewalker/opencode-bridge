"""Token-cost estimation and room cost/audit ledger helpers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

__all__ = ["_estimate_cost_usd", "_append_room_cost", "_append_room_audit"]

_MODEL_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8":    (15.0, 75.0),
    "claude-opus-4-7":    (15.0, 75.0),
    "claude-sonnet-4-6":  (3.0,  15.0),
    "claude-haiku-4-5":   (0.8,   4.0),
}


def _estimate_cost_usd(model: str, in_tok: int, out_tok: int,
                        cache_write: int = 0, cache_read: int = 0) -> float:
    for prefix, (r_in, r_out) in _MODEL_RATES.items():
        if model.startswith(prefix):
            cost = (in_tok * r_in + out_tok * r_out) / 1_000_000
            cost += cache_write * r_in * 1.25 / 1_000_000  # cache write = 1.25× input rate
            cost += cache_read * r_in * 0.10 / 1_000_000   # cache read = 0.10× input rate
            return round(cost, 6)
    return 0.0


def _append_room_cost(rooms_dir: Path, room_id: str, participant_name: str,
                       backend: str, model: str, effort: Optional[str],
                       round_num: int, usage: dict) -> None:
    in_tok   = usage.get("input_tokens", 0)
    out_tok  = usage.get("output_tokens", 0)
    cw_tok   = usage.get("cache_creation_input_tokens", 0)
    cr_tok   = usage.get("cache_read_input_tokens", 0)
    est_usd  = _estimate_cost_usd(model, in_tok, out_tok, cw_tok, cr_tok)
    record = {
        "ts": datetime.now().isoformat(),
        "room_id": room_id, "participant": participant_name,
        "backend": backend, "model": model, "effort": effort, "round": round_num,
        "in_tok": in_tok, "out_tok": out_tok,
        "cache_write_tok": cw_tok, "cache_read_tok": cr_tok,
        "est_usd": est_usd,
        "estimated": bool(usage.get("estimated", False)),
    }
    cost_path = rooms_dir / f"{room_id}.costs.jsonl"
    try:
        with open(cost_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _append_room_audit(rooms_dir: Path, room_id: str, participant_name: str,
                       round_num: int, record: dict) -> None:
    """Append one audit record to {room_id}.audit.jsonl (provenance ledger).

    Parallel to _append_room_cost but captures epistemic provenance: prompt
    hashes, tool-call ids, memory-injection flag, unsourced flag. Best-effort.
    """
    audit_path = rooms_dir / f"{room_id}.audit.jsonl"
    try:
        with open(audit_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass
