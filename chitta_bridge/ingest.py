"""Ingest utilities: triplet extraction and document ingestion."""
from __future__ import annotations

import asyncio
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from chitta_bridge.soul import SoulClient

__all__ = ["chitta_ingest", "distill_event"]

# ---------------------------------------------------------------------------
# Regex patterns for triplet/decision extraction
# ---------------------------------------------------------------------------

_SSL_PATTERN = re.compile(r'\[[\w:]+\]\s+\S+→\S+→\S+(?:\s+@\S+)?')
_CORRECTION_PATTERN = re.compile(r'(?:wrong|incorrect|fix|correction):\s*(.+?)(?:\.|$)', re.I)
_DECISION_PATTERN = re.compile(r'(?:chose|use|prefer|adopt)\s+(\w+)\s+over\s+(\w+)', re.I)
_LOCUS_PATTERN = re.compile(r'@([\w/\.\-]+:\d+)')
_REVIEW_COMMENT_PATTERN = re.compile(r'(.+?)\s+at\s+([\w/\.\-]+\.[\w]+:\d+)')


# ---------------------------------------------------------------------------
# chitta_ingest
# ---------------------------------------------------------------------------

def chitta_ingest(text: str) -> int:
    """Extract SSL triplets and decisions from text, write each to soul memory.

    Pure regex — no LLM call. Returns count of memories written.
    """
    triplets: list[str] = []

    for m in _SSL_PATTERN.finditer(text):
        triplets.append(m.group(0).strip())

    for m in _CORRECTION_PATTERN.finditer(text):
        triplets.append(f"correction→is→{m.group(1).strip()}")

    for m in _DECISION_PATTERN.finditer(text):
        triplets.append(f"{m.group(1)}→preferred-over→{m.group(2)}")

    for m in _REVIEW_COMMENT_PATTERN.finditer(text):
        locus = m.group(2).strip()
        comment = m.group(1).strip()
        triplets.append(f"review-comment→at→{locus} {comment[:120]}")

    if not triplets:
        return 0

    written = 0
    for t in triplets:
        content = f"[source:opencode-bridge] {t}"
        r = SoulClient.remember(content, kind="episode", tags="bridge-ingest,episodic", confidence=0.6)
        if r is not None:
            written += 1

    return written


# ---------------------------------------------------------------------------
# _doc_ingest
# ---------------------------------------------------------------------------

async def _doc_ingest(
    source: str,
    realm: str = "research",
    tags: list | None = None,
    model: str = "gpt-5.5",
    dry_run: bool = True,
    max_memories: int = 50,
    *,
    codex_bridge: Any,
    rooms: Any,
    web_fetch_fn: Optional[Callable[[str, int], str]] = None,
) -> str:
    """Extract structured memory records from a document via frontier LLM.

    Parameters
    ----------
    codex_bridge:
        A ``CodexBridge`` instance (passed by the server to avoid circular import).
    rooms:
        A ``RoomManager`` instance (used for PDF reading).
    web_fetch_fn:
        Optional callable ``(url, max_chars) -> str`` for URL fetching.
        Defaults to a simple ``urllib``-based fallback when not provided.
    """
    import hashlib

    tags = tags or []
    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]

    # 1. Read source
    src_path = Path(source)
    if source.startswith("http://") or source.startswith("https://"):
        if web_fetch_fn is not None:
            raw_text = await asyncio.to_thread(web_fetch_fn, source, 80_000)
        else:
            import urllib.request
            def _fetch(url: str, max_chars: int) -> str:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return resp.read(max_chars).decode(errors="replace")
            raw_text = await asyncio.to_thread(_fetch, source, 80_000)
    elif src_path.exists():
        if source.lower().endswith(".pdf"):
            raw_text = await asyncio.to_thread(rooms._tool_pdf_read, {"path": source, "pages": "all"})
        else:
            raw_text = src_path.read_text(errors="replace")
    else:
        return f"Error: source not found: {source}"

    if not raw_text or (isinstance(raw_text, str) and raw_text.startswith("Error")):
        return f"Failed to read source: {str(raw_text)[:300]}"

    # Cap input — large context causes stall in codex
    text_sample = raw_text[:30_000]

    schema_example = json.dumps({
        "kind": "failure_mode",
        "title": "...",
        "claim": "...",
        "scope": {"doc": doc_id, "assay": "", "task": ""},
        "source": {"doc_id": doc_id, "page": 0, "section": "...", "span": "..."},
        "evidence": "stated",
        "tags": tags,
        "realm": realm,
        "type": "wisdom",
        "visibility": 1,
        "retrieval_text": "...",
    })

    extraction_prompt = (
        f"Extract all significant atomic knowledge records from the document below.\n\n"
        f"Output ONLY valid JSONL (one JSON object per line, no prose). Each record MUST follow:\n"
        f"{schema_example}\n\n"
        f"Rules:\n"
        f"- kind: failure_mode | constraint | grading_rule | procedure | definition | caveat\n"
        f"- title: 5-10 words, unique, search-optimized\n"
        f"- claim: 1-3 sentences, standalone, self-contained\n"
        f"- scope.assay: platform name (xenium/merfish/visium/curio/atlasxomics) or \"\"\n"
        f"- scope.task: task type (clustering/de/qc/spatial_analysis/cell_typing) or \"\"\n"
        f"- evidence: stated | inferred | cross_doc\n"
        f"- type: wisdom | procedural | insight | episode\n"
        f"- retrieval_text: 50-150 word self-contained summary, useful without surrounding doc\n"
        f"- tags: include {json.dumps(tags)} plus relevant domain tags\n"
        f"- Output at most {max_memories} records. Most important/actionable facts only.\n"
        f"- Output ONLY JSON lines — no markdown, no headers, no explanations.\n\n"
        f"DOCUMENT (id={doc_id}, source={source}):\n"
        f"{text_sample}"
    )

    # 2. Run extraction with generous timeouts (extraction of 50 records can take >3 min)
    with tempfile.TemporaryDirectory() as tmpdir:
        args = codex_bridge._build_exec_args(model, None, full_auto=False)
        output, code = await codex_bridge._run_codex_exec_stdin(
            args, extraction_prompt, tmpdir,
            timeout=600, stall_timeout=300,
        )
        if code != 0:
            reply = ""
        else:
            reply, _ = codex_bridge._parse_codex_jsonl(output)
            reply = reply or output

    if not reply:
        return f"Extraction failed (exit {code}): {output[:500] if output else '(no output)'}"

    # 3. Parse JSONL records
    records: list[dict] = []
    parse_errors: list[str] = []
    for line in reply.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict) and ("claim" in rec or "title" in rec):
                rec.setdefault("scope", {})["doc"] = doc_id
                records.append(rec)
        except json.JSONDecodeError as e:
            parse_errors.append(f"{e}: {line[:80]}")

    if not records:
        return json.dumps({
            "doc_id": doc_id, "source": source, "status": "no_records",
            "parse_errors": parse_errors[:5], "raw_sample": reply[:800],
        }, indent=2)

    records = records[:max_memories]

    if dry_run:
        return json.dumps({
            "doc_id": doc_id, "source": source, "status": "dry_run",
            "extracted": len(records), "records": records,
            "parse_errors": parse_errors[:5] if parse_errors else [],
        }, indent=2)

    # 4. Write to chitta
    written: list[str] = []
    skipped: list[dict] = []
    for rec in records:
        body = rec.get("retrieval_text") or rec.get("claim", "")
        if not body:
            skipped.append({"title": rec.get("title", "?"), "reason": "empty body"})
            continue
        full_content = (
            f"[{rec.get('kind', 'insight')}] {rec.get('title', '')}\n\n"
            f"{rec.get('claim', '')}\n\n{body}"
        )
        rec_tags = list(dict.fromkeys(
            (tags or []) + rec.get("tags", []) + [doc_id, rec.get("kind", "insight")]
        ))
        mem = await asyncio.to_thread(
            SoulClient.remember,
            content=full_content,
            kind=rec.get("type", "wisdom"),
            tags=",".join(rec_tags),
            confidence=0.85,
            realm=rec.get("realm", realm),
        )
        if mem is not None:
            written.append(rec.get("title", "?"))
        else:
            skipped.append({"title": rec.get("title", "?"), "reason": "write failed"})

    return json.dumps({
        "doc_id": doc_id, "source": source, "status": "applied",
        "written": len(written), "skipped": skipped,
        "parse_errors": parse_errors[:5] if parse_errors else [],
    }, indent=2)


# ---------------------------------------------------------------------------
# distill_event
# ---------------------------------------------------------------------------

def distill_event(event_type: str, content: str, context: dict) -> None:
    """Post-process a bridge event: ingest triplets + write a typed digest memory.

    event_type: one of room_synth | checkpoint | file_edit
    context keys:
      symbol  — for file_edit events, the symbol name (optional)
    """
    chitta_ingest(content)

    truncated = content[:500]

    if event_type == "room_synth":
        SoulClient.remember(
            f"[digest-node] {truncated}",
            kind="digest-node",
            tags="digest-node,room-synth",
            confidence=0.75,
        )
    elif event_type == "checkpoint":
        SoulClient.remember(
            f"[rollup] {truncated}",
            kind="rollup",
            tags="checkpoint,rollup",
            confidence=0.8,
        )
    elif event_type == "file_edit":
        symbol = context.get("symbol")
        if symbol:
            SoulClient.remember(
                f"[symbol-summary] {symbol}: {truncated}",
                kind="symbol-summary",
                tags="file-edit,symbol-summary",
                confidence=0.7,
            )
