"""Tests for file chunking utilities in chitta_bridge.prompts."""

import tempfile
from pathlib import Path

import pytest

from chitta_bridge.prompts import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_chunk_prompt,
    build_synthesis_prompt,
    chunk_file,
    get_file_info,
    _file_info_cache,
)

# Threshold used to decide when chunking kicks in (files > this many lines).
# Kept local since the bridge decides this internally; tests just exercise
# the chunking primitives directly.
_CHUNK_THRESHOLD = 2000


def _make_file(lines: int, *, with_boundaries: bool = False) -> str:
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
        for c in chunks:
            assert c["filepath"] == path
            assert c["total_chunks"] == len(chunks)

    def test_chunks_cover_full_file(self):
        path = _make_file(3000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        assert chunks[0]["start_line"] == 1
        assert chunks[-1]["end_line"] == 3000

    def test_chunk_overlap(self):
        path = _make_file(2000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1]["end_line"]
            cur_start = chunks[i]["start_line"]
            assert cur_start <= prev_end

    def test_boundary_snapping(self):
        path = _make_file(2000, with_boundaries=True)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
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


class TestBuildChunkPrompt:
    def test_includes_chunk_metadata(self):
        chunk_info = {
            "chunk_index": 2, "total_chunks": 5,
            "start_line": 1601, "end_line": 2400,
            "content": "...", "filepath": "/tmp/test.py",
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


class TestChunkEdgeCases:
    def test_exactly_at_chunk_size(self):
        path = _make_file(_CHUNK_THRESHOLD)
        chunks = chunk_file(path, chunk_size=CHUNK_SIZE)
        assert len(chunks) > 1

    def test_one_line_over_threshold(self):
        path = _make_file(_CHUNK_THRESHOLD + 1)
        chunks = chunk_file(path, chunk_size=CHUNK_SIZE)
        assert len(chunks) > 1
        assert chunks[-1]["end_line"] == _CHUNK_THRESHOLD + 1

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
        path = _make_file(10000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        assert 10 <= len(chunks) <= 25

    def test_binary_file_does_not_crash(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".bin", delete=False, prefix="test_bin_"
        )
        tmp.write(b"\x00\xff" * 5000 + b"\n" * 100)
        tmp.close()
        chunks = chunk_file(tmp.name)
        assert isinstance(chunks, list)

    def test_file_info_cache_not_stale(self, tmp_path):
        f = tmp_path / "grow.py"
        f.write_text("\n".join(f"# {i}" for i in range(100)))
        info1 = get_file_info(str(f))
        assert info1["lines"] == 100

        _file_info_cache.pop(str(f.resolve()), None)

        f.write_text("\n".join(f"# {i}" for i in range(3000)))
        info2 = get_file_info(str(f))
        assert info2["lines"] == 3000

    def test_chunk_content_has_correct_lines(self):
        path = _make_file(2000)
        all_lines = Path(path).read_text().splitlines(keepends=True)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        for c in chunks:
            start = c["start_line"] - 1
            end = c["end_line"]
            expected = "".join(all_lines[start:end])
            assert c["content"] == expected

    def test_no_content_loss_across_chunks(self):
        path = _make_file(3000)
        chunks = chunk_file(path, chunk_size=800, overlap=20)
        covered = set()
        for c in chunks:
            for line in range(c["start_line"], c["end_line"] + 1):
                covered.add(line)
        assert covered == set(range(1, 3001))
