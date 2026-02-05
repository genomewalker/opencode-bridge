"""Tests for chunked subagent processing of large files."""

import json
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from opencode_bridge.server import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNK_THRESHOLD,
    MAX_TOTAL_CHUNKS,
    OpenCodeBridge,
    build_chunk_prompt,
    build_synthesis_prompt,
    chunk_file,
    get_file_info,
    _file_info_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(lines: int, *, with_boundaries: bool = False) -> str:
    """Create a temp file with the given number of lines. Returns its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="test_chunk_"
    )
    for i in range(1, lines + 1):
        if with_boundaries and i % 200 == 0:
            tmp.write(f"\ndef function_{i}():\n")
        else:
            tmp.write(f"# line {i}\n")
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# chunk_file
# ---------------------------------------------------------------------------

class TestChunkFile:
    def test_small_file_single_chunk(self):
        path = _make_file(100)
        chunks = chunk_file(path, chunk_size=CHUNK_SIZE)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["total_chunks"] == 1
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 100

    def test_large_file_multiple_chunks(self):
        path = _make_file(2500)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        assert len(chunks) > 1
        # All chunks should reference the same file
        for c in chunks:
            assert c["filepath"] == path
            assert c["total_chunks"] == len(chunks)

    def test_chunks_cover_full_file(self):
        path = _make_file(3000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        # First chunk starts at line 1
        assert chunks[0]["start_line"] == 1
        # Last chunk ends at the file's last line
        assert chunks[-1]["end_line"] == 3000

    def test_chunk_overlap(self):
        path = _make_file(2000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1]["end_line"]
            cur_start = chunks[i]["start_line"]
            # The start of the next chunk should overlap with the end of the previous
            assert cur_start <= prev_end, (
                f"Chunk {i} starts at {cur_start} but previous ends at {prev_end}"
            )

    def test_boundary_snapping(self):
        """Chunks should prefer cutting near function definitions."""
        path = _make_file(2000, with_boundaries=True)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        # With boundaries every 200 lines, cuts should snap near those points
        assert len(chunks) >= 2

    def test_chunk_indices_sequential(self):
        path = _make_file(3000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i

    def test_empty_file(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="test_empty_"
        )
        tmp.close()
        chunks = chunk_file(tmp.name)
        assert chunks == []

    def test_nonexistent_file(self):
        chunks = chunk_file("/nonexistent/file.py")
        assert chunks == []

    def test_content_preserved(self):
        path = _make_file(100)
        chunks = chunk_file(path, chunk_size=800)
        original = Path(path).read_text()
        assert chunks[0]["content"] == original


# ---------------------------------------------------------------------------
# build_chunk_prompt
# ---------------------------------------------------------------------------

class TestBuildChunkPrompt:
    def test_includes_chunk_metadata(self):
        chunk_info = {
            "chunk_index": 2,
            "total_chunks": 5,
            "start_line": 1601,
            "end_line": 2400,
            "content": "...",
            "filepath": "/tmp/test.py",
        }
        file_info = {"name": "test.py", "language": "Python", "lines": 5000}
        prompt = build_chunk_prompt("Review this code", chunk_info, file_info)
        assert "chunk 3 of 5" in prompt
        assert "lines 1601" in prompt
        assert "2400" in prompt
        assert "test.py" in prompt
        assert "Python" in prompt
        assert "5000" in prompt

    def test_includes_user_prompt(self):
        chunk_info = {
            "chunk_index": 0, "total_chunks": 1,
            "start_line": 1, "end_line": 100,
            "content": "...", "filepath": "/tmp/x.py",
        }
        file_info = {"name": "x.py", "language": "Python", "lines": 100}
        prompt = build_chunk_prompt("Find security bugs", chunk_info, file_info)
        assert "Find security bugs" in prompt

    def test_review_mode_adds_categories(self):
        chunk_info = {
            "chunk_index": 0, "total_chunks": 1,
            "start_line": 1, "end_line": 100,
            "content": "...", "filepath": "/tmp/x.py",
        }
        file_info = {"name": "x.py", "language": "Python", "lines": 100}
        prompt = build_chunk_prompt("Review", chunk_info, file_info, mode="review")
        assert "bug" in prompt.lower()
        assert "security" in prompt.lower()

    def test_discuss_mode_no_categories(self):
        chunk_info = {
            "chunk_index": 0, "total_chunks": 1,
            "start_line": 1, "end_line": 100,
            "content": "...", "filepath": "/tmp/x.py",
        }
        file_info = {"name": "x.py", "language": "Python", "lines": 100}
        prompt = build_chunk_prompt("Explain this", chunk_info, file_info, mode="discuss")
        assert "Categorize findings" not in prompt


# ---------------------------------------------------------------------------
# build_synthesis_prompt
# ---------------------------------------------------------------------------

class TestBuildSynthesisPrompt:
    def test_includes_all_chunk_responses(self):
        results = [
            {"chunk_index": 0, "file": "/tmp/a.py", "response": "Found bug on line 10", "error": None},
            {"chunk_index": 1, "file": "/tmp/a.py", "response": "Performance issue at line 900", "error": None},
        ]
        file_infos = [{"name": "a.py", "lines": 2000}]
        prompt = build_synthesis_prompt("Review code", results, file_infos)
        assert "Found bug on line 10" in prompt
        assert "Performance issue at line 900" in prompt
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt

    def test_marks_failed_chunks(self):
        results = [
            {"chunk_index": 0, "file": "/tmp/a.py", "response": "OK", "error": None},
            {"chunk_index": 1, "file": "/tmp/a.py", "response": "", "error": "timeout"},
        ]
        file_infos = [{"name": "a.py", "lines": 2000}]
        prompt = build_synthesis_prompt("Review", results, file_infos)
        assert "analysis failed" in prompt
        assert "timeout" in prompt

    def test_includes_original_request(self):
        results = [{"chunk_index": 0, "file": "/tmp/a.py", "response": "OK", "error": None}]
        file_infos = [{"name": "a.py", "lines": 100}]
        prompt = build_synthesis_prompt("Find SQL injections", results, file_infos)
        assert "Find SQL injections" in prompt

    def test_review_mode_adds_grouping(self):
        results = [{"chunk_index": 0, "file": "/tmp/a.py", "response": "OK", "error": None}]
        file_infos = [{"name": "a.py", "lines": 100}]
        prompt = build_synthesis_prompt("Review", results, file_infos, mode="review")
        assert "Group findings by category" in prompt


# ---------------------------------------------------------------------------
# _parse_opencode_response
# ---------------------------------------------------------------------------

class TestParseOpenCodeResponse:
    def test_extracts_text_and_session_id(self):
        lines = [
            json.dumps({"sessionID": "sess-123"}),
            json.dumps({"type": "text", "part": {"text": "Hello "}}),
            json.dumps({"type": "text", "part": {"text": "World"}}),
        ]
        output = "\n".join(lines)
        text, sid = OpenCodeBridge._parse_opencode_response(output)
        assert text == "Hello World"
        assert sid == "sess-123"

    def test_no_session_id(self):
        lines = [
            json.dumps({"type": "text", "part": {"text": "Just text"}}),
        ]
        output = "\n".join(lines)
        text, sid = OpenCodeBridge._parse_opencode_response(output)
        assert text == "Just text"
        assert sid is None

    def test_skips_invalid_json(self):
        output = "not json\n" + json.dumps({"type": "text", "part": {"text": "OK"}})
        text, sid = OpenCodeBridge._parse_opencode_response(output)
        assert text == "OK"

    def test_empty_output(self):
        text, sid = OpenCodeBridge._parse_opencode_response("")
        assert text == ""
        assert sid is None


# ---------------------------------------------------------------------------
# Integration: chunking gate in send_message / review_code
# ---------------------------------------------------------------------------

def _mock_opencode_response(text: str, session_id: str = "mock-sess") -> str:
    """Build a mock JSON-lines response from opencode."""
    lines = [
        json.dumps({"sessionID": session_id}),
        json.dumps({"type": "text", "part": {"text": text}}),
    ]
    return "\n".join(lines)


@pytest.fixture
def bridge():
    b = OpenCodeBridge()
    return b


class TestChunkingGateIntegration:
    @pytest.mark.anyio
    async def test_small_file_bypasses_chunking(self, bridge, tmp_path):
        """Files under CHUNK_THRESHOLD should NOT trigger chunking."""
        small = tmp_path / "small.py"
        small.write_text("\n".join(f"# line {i}" for i in range(500)))

        await bridge.start_session("test-small")

        with patch.object(bridge, "_run_opencode", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (_mock_opencode_response("review done"), 0)
            with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
                result = await bridge.send_message(
                    "Review this", files=[str(small)]
                )
                mock_chunked.assert_not_called()
                assert "review done" in result

    @pytest.mark.anyio
    async def test_large_file_triggers_chunking(self, bridge, tmp_path):
        """Files over CHUNK_THRESHOLD should trigger _run_chunked."""
        large = tmp_path / "large.py"
        large.write_text("\n".join(f"# line {i}" for i in range(CHUNK_THRESHOLD + 100)))

        await bridge.start_session("test-large")

        with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
            mock_chunked.return_value = "chunked analysis done"
            result = await bridge.send_message(
                "Review this", files=[str(large)]
            )
            mock_chunked.assert_called_once()
            assert "chunked analysis done" in result

    @pytest.mark.anyio
    async def test_review_code_large_file_triggers_chunking(self, bridge, tmp_path):
        """review_code should also route large files through chunking."""
        large = tmp_path / "big.py"
        large.write_text("\n".join(f"# line {i}" for i in range(CHUNK_THRESHOLD + 100)))

        await bridge.start_session("test-review-large")

        with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
            mock_chunked.return_value = "chunked review done"
            result = await bridge.review_code(str(large))
            mock_chunked.assert_called_once()
            assert "chunked review done" in result

    @pytest.mark.anyio
    async def test_review_code_small_file_no_chunking(self, bridge, tmp_path):
        """Small files in review_code should not trigger chunking."""
        small = tmp_path / "tiny.py"
        small.write_text("\n".join(f"# line {i}" for i in range(200)))

        await bridge.start_session("test-review-small")

        with patch.object(bridge, "_run_opencode", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (_mock_opencode_response("looks good"), 0)
            with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
                result = await bridge.review_code(str(small))
                mock_chunked.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: _run_chunked map-reduce
# ---------------------------------------------------------------------------

class TestRunChunked:
    @pytest.mark.anyio
    async def test_successful_chunked_processing(self, bridge, tmp_path):
        """Full map-reduce pipeline with mocked _run_opencode."""
        large = tmp_path / "big.py"
        large.write_text("\n".join(f"# line {i}" for i in range(2500)))

        await bridge.start_session("test-chunked")
        session = bridge.sessions["test-chunked"]

        call_count = 0

        async def mock_run(*args, timeout=300):
            nonlocal call_count
            call_count += 1
            # Chunk calls return chunk analysis; synthesis call returns final
            return (_mock_opencode_response(f"analysis-{call_count}"), 0)

        with patch.object(bridge, "_run_opencode", side_effect=mock_run):
            result = await bridge._run_chunked(
                "Review this code", [str(large)], session, mode="review"
            )

        # Should have made multiple calls (chunks + synthesis)
        assert call_count > 2
        assert result  # non-empty

    @pytest.mark.anyio
    async def test_majority_failure_returns_error(self, bridge, tmp_path):
        """If >50% of chunks fail, return an error message."""
        large = tmp_path / "big.py"
        large.write_text("\n".join(f"# line {i}" for i in range(2500)))

        await bridge.start_session("test-fail")
        session = bridge.sessions["test-fail"]

        # All calls fail
        async def mock_fail(*args, timeout=300):
            return ("error: context_length_exceeded", 1)

        with patch.object(bridge, "_run_opencode", side_effect=mock_fail):
            result = await bridge._run_chunked(
                "Review", [str(large)], session
            )

        assert "failed" in result.lower()

    @pytest.mark.anyio
    async def test_synthesis_failure_falls_back(self, bridge, tmp_path):
        """If synthesis fails, fall back to concatenated chunk results."""
        large = tmp_path / "big.py"
        large.write_text("\n".join(f"# line {i}" for i in range(2500)))

        await bridge.start_session("test-fallback")
        session = bridge.sessions["test-fallback"]

        chunks = chunk_file(str(large), CHUNK_SIZE, CHUNK_OVERLAP)
        num_chunks = len(chunks)

        # Track calls to distinguish chunk calls from synthesis call.
        # Chunk calls have a --file arg pointing to a temp chunk file;
        # the synthesis call does NOT attach a chunk temp file.
        async def mock_run(*args, timeout=300):
            # The synthesis prompt includes "Synthesize" — detect it
            prompt_arg = args[1] if len(args) > 1 else ""
            if "Synthesize" in prompt_arg or "Chunk Analyses" in prompt_arg:
                return ("synthesis error", 1)
            return (_mock_opencode_response("chunk-ok"), 0)

        with patch.object(bridge, "_run_opencode", side_effect=mock_run):
            result = await bridge._run_chunked(
                "Review", [str(large)], session
            )

        # Should contain raw chunk results as fallback
        assert "Synthesis failed" in result
        assert "chunk-ok" in result

    @pytest.mark.anyio
    async def test_partial_chunk_failure_still_synthesizes(self, bridge, tmp_path):
        """If some chunks fail but not majority, synthesis should still run."""
        large = tmp_path / "big.py"
        large.write_text("\n".join(f"# line {i}" for i in range(2500)))

        await bridge.start_session("test-partial")
        session = bridge.sessions["test-partial"]

        first_call_done = False

        async def mock_run(*args, timeout=300):
            nonlocal first_call_done
            prompt_arg = args[1] if len(args) > 1 else ""
            # Synthesis call — always succeed
            if "Synthesize" in prompt_arg or "Chunk Analyses" in prompt_arg:
                return (_mock_opencode_response("synthesized"), 0)
            # Fail just the first chunk call
            if not first_call_done:
                first_call_done = True
                return ("error", 1)
            return (_mock_opencode_response("chunk-ok"), 0)

        with patch.object(bridge, "_run_opencode", side_effect=mock_run):
            result = await bridge._run_chunked(
                "Review", [str(large)], session
            )

        # Should have gotten a synthesis result (not a total failure)
        assert "synthesized" in result or "chunk-ok" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestChunkEdgeCases:
    def test_exactly_at_threshold(self):
        """A file with exactly CHUNK_THRESHOLD lines should produce one chunk."""
        path = _make_file(CHUNK_THRESHOLD)
        chunks = chunk_file(path, chunk_size=CHUNK_SIZE)
        # CHUNK_THRESHOLD (2000) > CHUNK_SIZE (800) → should produce multiple chunks
        assert len(chunks) > 1

    def test_one_line_over_threshold(self):
        """CHUNK_THRESHOLD+1 lines should chunk properly."""
        path = _make_file(CHUNK_THRESHOLD + 1)
        chunks = chunk_file(path, chunk_size=CHUNK_SIZE)
        assert len(chunks) > 1
        assert chunks[-1]["end_line"] == CHUNK_THRESHOLD + 1

    def test_single_line_file(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="test_one_"
        )
        tmp.write("x = 1\n")
        tmp.close()
        chunks = chunk_file(tmp.name)
        assert len(chunks) == 1
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 1

    def test_very_large_file_chunk_count(self):
        """A huge file should produce a reasonable number of chunks."""
        path = _make_file(10000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        # ~14 chunks (boundary snapping may reduce effective chunk size)
        assert 10 <= len(chunks) <= 25

    def test_binary_file_does_not_crash(self):
        """Binary file with replacement errors should not crash."""
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False, prefix="test_bin_"
        )
        tmp.write(b"\x00\xff" * 5000 + b"\n" * 100)
        tmp.close()
        chunks = chunk_file(tmp.name)
        # Should produce something without crashing
        assert isinstance(chunks, list)

    def test_file_info_cache_not_stale(self, tmp_path):
        """get_file_info cache shouldn't interfere with chunk decisions."""
        f = tmp_path / "grow.py"
        f.write_text("\n".join(f"# {i}" for i in range(100)))
        info1 = get_file_info(str(f))
        assert info1["lines"] == 100

        # Clear cache so re-read picks up new size
        _file_info_cache.pop(str(f.resolve()), None)

        f.write_text("\n".join(f"# {i}" for i in range(3000)))
        info2 = get_file_info(str(f))
        assert info2["lines"] == 3000

    def test_chunk_content_has_correct_lines(self):
        """Verify each chunk's content actually matches its line range."""
        path = _make_file(2000)
        all_lines = Path(path).read_text().splitlines(keepends=True)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        for c in chunks:
            start = c["start_line"] - 1  # 0-indexed
            end = c["end_line"]
            expected = "".join(all_lines[start:end])
            assert c["content"] == expected, (
                f"Chunk {c['chunk_index']} content mismatch: "
                f"lines {c['start_line']}-{c['end_line']}"
            )

    def test_no_content_loss_across_chunks(self):
        """Union of all chunk line ranges should cover every line in the file."""
        path = _make_file(3000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        covered = set()
        for c in chunks:
            for line in range(c["start_line"], c["end_line"] + 1):
                covered.add(line)
        assert covered == set(range(1, 3001))


class TestChunkGateBoundary:
    """Test the exact boundary where chunking kicks in."""

    @pytest.mark.anyio
    async def test_file_at_threshold_no_chunking_in_send(self, bridge, tmp_path):
        """File with exactly CHUNK_THRESHOLD lines should NOT trigger chunking
        in send_message (gate is strictly >)."""
        f = tmp_path / "exact.py"
        f.write_text("\n".join(f"# {i}" for i in range(CHUNK_THRESHOLD)))

        await bridge.start_session("test-exact")

        with patch.object(bridge, "_run_opencode", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (_mock_opencode_response("normal"), 0)
            with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
                result = await bridge.send_message("Review", files=[str(f)])
                mock_chunked.assert_not_called()
                assert "normal" in result

    @pytest.mark.anyio
    async def test_file_one_over_threshold_triggers_chunking(self, bridge, tmp_path):
        """File with CHUNK_THRESHOLD+1 lines SHOULD trigger chunking."""
        f = tmp_path / "over.py"
        f.write_text("\n".join(f"# {i}" for i in range(CHUNK_THRESHOLD + 1)))

        await bridge.start_session("test-over")

        with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
            mock_chunked.return_value = "chunked"
            result = await bridge.send_message("Review", files=[str(f)])
            mock_chunked.assert_called_once()

    @pytest.mark.anyio
    async def test_multiple_small_files_no_chunking(self, bridge, tmp_path):
        """Multiple small files (each under threshold) should NOT trigger chunking,
        even if total lines exceed threshold."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("\n".join(f"# {i}" for i in range(1500)))
        f2.write_text("\n".join(f"# {i}" for i in range(1500)))

        await bridge.start_session("test-multi-small")

        with patch.object(bridge, "_run_opencode", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (_mock_opencode_response("ok"), 0)
            with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
                result = await bridge.send_message(
                    "Review", files=[str(f1), str(f2)]
                )
                mock_chunked.assert_not_called()

    @pytest.mark.anyio
    async def test_mix_small_and_large_triggers_chunking(self, bridge, tmp_path):
        """If ANY file exceeds threshold, chunking should trigger."""
        small = tmp_path / "small.py"
        large = tmp_path / "large.py"
        small.write_text("\n".join(f"# {i}" for i in range(100)))
        large.write_text("\n".join(f"# {i}" for i in range(CHUNK_THRESHOLD + 1)))

        await bridge.start_session("test-mix")

        with patch.object(bridge, "_run_chunked", new_callable=AsyncMock) as mock_chunked:
            mock_chunked.return_value = "chunked"
            result = await bridge.send_message(
                "Review", files=[str(small), str(large)]
            )
            mock_chunked.assert_called_once()


class TestRunChunkedSmallFilesPassthrough:
    """Verify small files are passed to synthesis call as context."""

    @pytest.mark.anyio
    async def test_small_files_attached_to_synthesis(self, bridge, tmp_path):
        small = tmp_path / "helper.py"
        large = tmp_path / "main.py"
        small.write_text("\n".join(f"# {i}" for i in range(100)))
        large.write_text("\n".join(f"# {i}" for i in range(2500)))

        await bridge.start_session("test-passthrough")
        session = bridge.sessions["test-passthrough"]

        synthesis_args = []

        async def mock_run(*args, timeout=300):
            prompt_arg = args[1] if len(args) > 1 else ""
            if "Synthesize" in prompt_arg or "Chunk Analyses" in prompt_arg:
                synthesis_args.extend(args)
                return (_mock_opencode_response("final"), 0)
            return (_mock_opencode_response("chunk-ok"), 0)

        with patch.object(bridge, "_run_opencode", side_effect=mock_run):
            result = await bridge._run_chunked(
                "Review", [str(small), str(large)], session
            )

        # The synthesis call should have --file pointing to the small file
        assert str(small) in synthesis_args, (
            f"Small file not passed to synthesis. Args: {synthesis_args}"
        )
