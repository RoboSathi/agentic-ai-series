"""Readable output and config: trees instead of JSON, relevant page windows, YAML merges."""

from pathlib import Path

import pytest

from agent_platform.config.yaml_config import _read_yaml
from agent_platform.context.excerpt import relevant_excerpt
from agent_platform.interface.presentation.text_tree import value_lines


class TestValueTree:
    def test_json_text_becomes_a_tree_with_no_braces(self) -> None:
        lines = value_lines('{"nodes": [{"node_id": "N1", "depends_on": []}], "why": "x"}')
        text = "\n".join(lines)
        assert "{" not in text and '"' not in text
        assert "node_id: N1" in text and lines[-1] == "└─ why: x"

    def test_fenced_json_is_read_too(self) -> None:
        assert "a: 1" in "\n".join(value_lines('```json\n{"a": 1}\n```'))

    def test_json_inside_a_mapping_becomes_a_subtree(self) -> None:
        text = "\n".join(value_lines({"response": '{"score": 0.9}'}))
        assert "response:" in text and "score: 0.9" in text

    def test_prose_is_left_alone(self) -> None:
        assert value_lines("It rained {a little} today.") == ["It rained {a little} today."]


class TestRelevantExcerpt:
    def test_a_short_text_is_kept_whole(self) -> None:
        assert relevant_excerpt("short", "q", 100, 10) == "short"

    def test_the_windows_on_the_question_are_kept_in_page_order(self) -> None:
        text = "\n".join(
            ["menu home login"] * 20 + ["iron-air cycle life is 10,000 cycles"] + ["footer"] * 20
        )
        excerpt = relevant_excerpt(text, "iron-air cycle life", 120, 40)
        assert "iron-air cycle life is 10,000 cycles" in excerpt and len(excerpt) <= 120


class TestYamlMerge:
    def test_a_merge_brings_the_anchor_and_a_written_key_overrides_it(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("base: &b {a: 1, b: 2}\nrole: {<<: *b, b: 3}\n")
        assert _read_yaml(path)["role"] == {"a": 1, "b": 3}

    def test_two_written_keys_are_still_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        path.write_text("x: 1\nx: 2\n")
        with pytest.raises(ValueError, match="Duplicate"):
            _read_yaml(path)
