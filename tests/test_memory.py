"""Tests for temporal knowledge graph memory."""

import pytest
from datetime import datetime, timezone, timedelta

from meta_optimizer.memory import Fact, TemporalMemory


class TestFact:
    def test_create_fact(self):
        f = Fact(content="Paris is the capital of France", tags=["geography"])
        assert f.content == "Paris is the capital of France"
        assert "geography" in f.tags
        assert f.invalidated is False
        assert f.fact_id != ""

    def test_is_valid_now(self):
        f = Fact(content="test")
        assert f.is_valid_at() is True

    def test_invalidated_fact(self):
        f = Fact(content="test", invalidated=True)
        assert f.is_valid_at() is False

    def test_expired_fact(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        f = Fact(content="test", valid_until=past)
        assert f.is_valid_at() is False

    def test_future_valid_until(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        f = Fact(content="test", valid_until=future)
        assert f.is_valid_at() is True

    def test_to_dict_roundtrip(self):
        f = Fact(content="hello", tags=["a", "b"], confidence=0.9)
        d = f.to_dict()
        f2 = Fact.from_dict(d)
        assert f2.content == f.content
        assert f2.tags == f.tags
        assert f2.confidence == f.confidence
        assert f2.fact_id == f.fact_id


class TestTemporalMemory:
    def setup_method(self):
        self.memory = TemporalMemory(session_id="test-session")

    def test_add_fact(self):
        fact = self.memory.add_fact("The sky is blue", tags=["nature"])
        assert fact.content == "The sky is blue"
        assert fact.session_id == "test-session"

    def test_query_by_keyword(self):
        self.memory.add_fact("Python is a programming language", tags=["tech"])
        self.memory.add_fact("Java is also a programming language", tags=["tech"])
        self.memory.add_fact("The sky is blue", tags=["nature"])

        results = self.memory.query(keyword="programming")
        assert len(results) == 2

    def test_query_by_tags(self):
        self.memory.add_fact("fact 1", tags=["a", "b"])
        self.memory.add_fact("fact 2", tags=["a"])
        self.memory.add_fact("fact 3", tags=["b"])

        results = self.memory.query(tags=["a", "b"])
        assert len(results) == 1
        assert results[0].content == "fact 1"

    def test_update_fact_invalidates_old(self):
        old = self.memory.add_fact("Old content")
        new = self.memory.update_fact(old.fact_id, "New content")

        assert self.memory.get_fact(old.fact_id).invalidated is True
        assert new.content == "New content"
        assert old.fact_id in new.related_facts

    def test_invalidate_fact(self):
        fact = self.memory.add_fact("temp fact")
        self.memory.invalidate_fact(fact.fact_id)
        assert self.memory.get_fact(fact.fact_id).invalidated is True

    def test_query_valid_only(self):
        valid = self.memory.add_fact("valid fact")
        invalid = self.memory.add_fact("invalid fact")
        self.memory.invalidate_fact(invalid.fact_id)

        results = self.memory.query(valid_only=True)
        ids = [f.fact_id for f in results]
        assert valid.fact_id in ids
        assert invalid.fact_id not in ids

    def test_relations(self):
        a = self.memory.add_fact("Fact A")
        b = self.memory.add_fact("Fact B")
        rel = self.memory.add_relation(a.fact_id, b.fact_id, "supports")
        assert rel.relation_type == "supports"
        rels = self.memory.get_relations(a.fact_id)
        assert len(rels) >= 1

    def test_cross_session_synthesis(self):
        mem = TemporalMemory(session_id="s1")
        mem.add_fact("fact from s1", tags=["x"])
        mem.session_id = "s2"
        mem.add_fact("fact from s2", tags=["y"])

        synthesis = mem.cross_session_synthesis()
        assert synthesis["sessions"] == 2
        assert synthesis["total_facts"] == 2

    def test_find_patterns(self):
        mem = TemporalMemory(session_id="s1")
        mem.add_fact("fact 1", tags=["pattern"])
        mem.add_fact("fact 2", tags=["pattern"])
        mem.add_fact("fact 3", tags=["pattern"])

        patterns = mem.find_patterns(min_occurrences=2)
        assert any(p["tag"] == "pattern" for p in patterns)

    def test_extract_knowledge(self):
        text = "Python is a language. It is widely used. Many developers love it."
        facts = self.memory.extract_knowledge(text)
        assert len(facts) >= 2

    def test_stats(self):
        self.memory.add_fact("f1")
        self.memory.add_fact("f2")
        f = self.memory.add_fact("f3")
        self.memory.invalidate_fact(f.fact_id)

        stats = self.memory.stats()
        assert stats["total_facts"] == 3
        assert stats["valid_facts"] == 2
        assert stats["invalidated_facts"] == 1

    def test_time_travel_query(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

        old_fact = self.memory.add_fact("old valid", valid_until=future)

        results = self.memory.query_at_time("old valid", future)
        assert len(results) >= 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "memory.json")
        mem = TemporalMemory(storage_path=path, session_id="persist-session")
        mem.add_fact("persistent fact", tags=["test"])
        mem.save()

        mem2 = TemporalMemory(storage_path=path, session_id="new-session")
        results = mem2.query(keyword="persistent")
        assert len(results) == 1
