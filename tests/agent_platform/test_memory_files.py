"""The file layer under long-term memory (ADR 0011).

Real files in a temporary directory: the behaviours under test — paths
that cannot escape, writes no reader sees half of, records that keep
their order, an index that rebuilds itself — are properties of the
filesystem, and a mocked one would only test the mock.
"""

from pathlib import Path

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import MemoryFilesSettings
from agent_platform.memory.files import (
    DerivedIndex,
    IndexedDoc,
    IndexEntry,
    MemoryDirectory,
    MemoryDocument,
    MemoryDocumentError,
    MemoryPathError,
    RecordLog,
    parse_document,
    render_document,
    render_index,
    safe_name,
)


@pytest.fixture
def files() -> MemoryFilesSettings:
    return get_settings().memory.files


@pytest.fixture
def directory(tmp_path: Path, files: MemoryFilesSettings) -> MemoryDirectory:
    return MemoryDirectory(tmp_path / "store", files.lock_file)


class TestPathsStayInside:
    @pytest.mark.parametrize("escape", ["../outside.md", "/etc/passwd", "a/../../b.md"])
    def test_a_path_that_would_leave_the_directory_is_refused(
        self, directory: MemoryDirectory, escape: str
    ) -> None:
        with pytest.raises(MemoryPathError):
            directory.path(escape)

    def test_a_symlink_pointing_out_is_refused(
        self, directory: MemoryDirectory, tmp_path: Path
    ) -> None:
        (tmp_path / "secret").mkdir()
        (directory.root / "link").symlink_to(tmp_path / "secret")
        with pytest.raises(MemoryPathError):
            directory.path("link/file.md")

    def test_a_name_that_is_already_safe_is_kept(self) -> None:
        assert safe_name("delivery_window", 8) == "delivery_window"

    def test_two_names_that_clean_to_the_same_text_get_different_files(self) -> None:
        assert safe_name("a b", 8) != safe_name("a_b", 8)
        assert "/" not in safe_name("a/../b", 8)
        assert safe_name("..", 8) not in ("", ".", "..")


class TestWrites:
    def test_a_write_replaces_the_file_and_leaves_no_temporary_behind(
        self, directory: MemoryDirectory
    ) -> None:
        directory.write("x/y.md", "one")
        directory.write("x/y.md", "two")
        assert directory.read("x/y.md") == "two"
        assert [p.name for p in (directory.root / "x").iterdir()] == ["y.md"]

    def test_the_sequence_counts_up_from_one(
        self, directory: MemoryDirectory, files: MemoryFilesSettings
    ) -> None:
        with directory.locked():
            numbers = [directory.next_sequence(files.sequence_file) for _ in range(3)]
        assert numbers == [1, 2, 3]

    def test_a_tree_is_removed_with_a_count(self, directory: MemoryDirectory) -> None:
        directory.write("g/a.md", "a")
        directory.write("g/b/c.md", "c")
        assert directory.remove_tree("g") == 2
        assert directory.remove_tree("g") == 0


class TestFrontmatter:
    def test_a_document_round_trips(self, files: MemoryFilesSettings) -> None:
        document = MemoryDocument(meta={"key": "k", "version": 2}, body="line one\nline two")
        text = render_document(document, files.frontmatter_delimiter)
        assert parse_document(text, files.frontmatter_delimiter) == document

    def test_text_without_frontmatter_is_all_body(self, files: MemoryFilesSettings) -> None:
        assert parse_document("just words", files.frontmatter_delimiter).body == "just words"

    def test_unclosed_frontmatter_is_an_error(self, files: MemoryFilesSettings) -> None:
        with pytest.raises(MemoryDocumentError):
            parse_document("---\nkey: k\nbody", files.frontmatter_delimiter)


class TestRecords:
    def test_records_come_back_in_write_order_across_groups(
        self, directory: MemoryDirectory, files: MemoryFilesSettings
    ) -> None:
        log = RecordLog(directory, files)
        with directory.locked():
            log.append(["s", "b"], MemoryDocument(body="first"))
            log.append(["s", "a"], MemoryDocument(body="second"))
        assert [r.document.body for r in log.records()] == ["first", "second"]
        assert [r.document.body for r in log.records(["s", "a"])] == ["second"]

    def test_an_unreadable_record_is_skipped_not_fatal(
        self, directory: MemoryDirectory, files: MemoryFilesSettings
    ) -> None:
        log = RecordLog(directory, files)
        with directory.locked():
            kept = log.append(["g"], MemoryDocument(body="kept"))
        directory.write(f"{log.group_path(['g'])}/broken{files.record_suffix}", "---\nnope")
        assert [r.seq for r in log.records()] == [kept.seq]

    def test_the_fingerprint_changes_when_a_record_does(
        self, directory: MemoryDirectory, files: MemoryFilesSettings
    ) -> None:
        log = RecordLog(directory, files)
        with directory.locked():
            record = log.append(["g"], MemoryDocument(body="x"))
        before = log.fingerprint()
        log.rewrite(record, {"deleted_at": "now"})
        assert log.fingerprint() != before


class TestDerivedIndex:
    def test_search_is_scoped_to_the_subject(self, tmp_path: Path) -> None:
        index = DerivedIndex(tmp_path / "idx.sqlite")
        index.replace(IndexedDoc(doc_id="1", subject="a", key="window", text="mornings"))
        index.replace(IndexedDoc(doc_id="2", subject="b", key="door", text="mornings"))
        assert [key for key, _ in index.search("mornings", "a", 5)] == ["window"]

    def test_a_rebuild_replaces_everything_and_records_its_source(self, tmp_path: Path) -> None:
        index = DerivedIndex(tmp_path / "idx.sqlite")
        index.replace(IndexedDoc(doc_id="old", subject="a", key="k", text="stale words"))
        index.rebuild([IndexedDoc(doc_id="new", subject="a", key="k", text="fresh")], "fp")
        assert index.search("stale", "a", 5) == []
        assert index.fingerprint() == "fp"

    def test_text_is_never_read_as_query_syntax(self, tmp_path: Path) -> None:
        index = DerivedIndex(tmp_path / "idx.sqlite")
        index.replace(IndexedDoc(doc_id="1", subject="", key="k", text="plain"))
        assert index.search('" OR * NEAR(', "", 5) == []


class TestMemoryIndexFile:
    def test_one_line_per_memory_with_a_link(self, files: MemoryFilesSettings) -> None:
        text = render_index(
            "Semantic",
            "agent",
            [IndexEntry(title="window", path="_agent/window.md", description="mornings")],
            files.index,
        )
        assert "[window](_agent/window.md)" in text

    def test_an_empty_store_says_so(self, files: MemoryFilesSettings) -> None:
        assert files.index.empty in render_index("Semantic", "agent", [], files.index)
