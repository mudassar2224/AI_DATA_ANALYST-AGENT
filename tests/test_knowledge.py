"""Tests for dsagent.knowledge.base."""

from __future__ import annotations

from dsagent.knowledge.base import all_entries, retrieve


def test_retrieve_by_single_tag():
    entries = retrieve(["missing"])
    assert len(entries) > 0
    assert all("missing" in e.tags for e in entries)


def test_retrieve_deduplicates_across_tags():
    entries = retrieve(["missing", "quality"])
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))


def test_retrieve_unknown_tag_returns_empty():
    assert retrieve(["not_a_real_tag"]) == []


def test_all_entries_have_unique_ids():
    ids = [e.id for e in all_entries()]
    assert len(ids) == len(set(ids))


def test_all_entries_nonempty_text():
    assert all(len(e.text) > 20 for e in all_entries())
