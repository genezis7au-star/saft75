"""
Temporal knowledge graph memory system.

Implements:
- Temporal tracking (when facts were learned / updated / invalidated)
- Graph-style relationships between facts
- Cross-session synthesis
- Edge invalidation on fact updates
- Persistent JSON storage (with optional Neo4j upgrade path)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Fact:
    """A single piece of knowledge with temporal metadata."""

    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = ""
    session_id: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    valid_until: Optional[str] = None  # None = indefinitely valid
    invalidated: bool = False
    invalidated_by: Optional[str] = None  # fact_id that superseded this
    confidence: float = 1.0
    related_facts: List[str] = field(default_factory=list)

    def is_valid_at(self, timestamp_iso: Optional[str] = None) -> bool:
        """Check if this fact is valid at the given ISO timestamp (default: now)."""
        if self.invalidated:
            return False
        if self.valid_until is None:
            return True
        check = timestamp_iso or _utcnow_iso()
        return check <= self.valid_until

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        return cls(**d)


@dataclass
class Relation:
    """A directed relationship between two facts."""

    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_fact_id: str = ""
    to_fact_id: str = ""
    relation_type: str = "related_to"
    weight: float = 1.0
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Relation":
        return cls(**d)


# ---------------------------------------------------------------------------
# TemporalMemory
# ---------------------------------------------------------------------------


class TemporalMemory:
    """
    JSON-backed temporal knowledge graph.

    Supports:
    - Adding and updating facts with full temporal metadata
    - Graph-style relations between facts
    - Time-travel queries (facts valid at a given timestamp)
    - Edge invalidation: updating a fact creates a new version and
      invalidates the old one
    - Cross-session synthesis: find patterns across multiple sessions
    - Pattern recognition and keyword search

    Designed to be a drop-in foundation before a Neo4j/FalkorDB upgrade.
    The ``storage_path`` file persists all state across sessions.
    """

    def __init__(self, storage_path: Optional[str] = None, session_id: Optional[str] = None) -> None:
        self.storage_path = storage_path
        self.session_id = session_id or str(uuid.uuid4())
        self._facts: Dict[str, Fact] = {}
        self._relations: Dict[str, Relation] = {}

        if storage_path and os.path.exists(storage_path):
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        assert self.storage_path is not None
        with open(self.storage_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._facts = {k: Fact.from_dict(v) for k, v in data.get("facts", {}).items()}
        self._relations = {k: Relation.from_dict(v) for k, v in data.get("relations", {}).items()}

    def save(self) -> None:
        if not self.storage_path:
            return
        data = {
            "facts": {k: v.to_dict() for k, v in self._facts.items()},
            "relations": {k: v.to_dict() for k, v in self._relations.items()},
        }
        parent_dir = os.path.dirname(self.storage_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        source: str = "",
        confidence: float = 1.0,
        valid_until: Optional[str] = None,
        related_fact_ids: Optional[List[str]] = None,
    ) -> Fact:
        """Add a new fact and return it."""
        fact = Fact(
            content=content,
            tags=tags or [],
            source=source,
            session_id=self.session_id,
            confidence=confidence,
            valid_until=valid_until,
            related_facts=related_fact_ids or [],
        )
        self._facts[fact.fact_id] = fact
        return fact

    def update_fact(
        self,
        fact_id: str,
        new_content: str,
        confidence: float = 1.0,
        valid_until: Optional[str] = None,
    ) -> Fact:
        """
        Update an existing fact.

        Invalidates the old fact and creates a new version that references it.
        """
        if fact_id not in self._facts:
            raise KeyError(f"Fact {fact_id!r} not found")

        old_fact = self._facts[fact_id]
        old_fact.invalidated = True
        old_fact.updated_at = _utcnow_iso()

        new_fact = Fact(
            content=new_content,
            tags=list(old_fact.tags),
            source=old_fact.source,
            session_id=self.session_id,
            confidence=confidence,
            valid_until=valid_until,
            related_facts=[fact_id] + list(old_fact.related_facts),
        )
        old_fact.invalidated_by = new_fact.fact_id
        self._facts[new_fact.fact_id] = new_fact

        # Add a supersedes relation
        self.add_relation(new_fact.fact_id, fact_id, "supersedes")

        return new_fact

    def invalidate_fact(self, fact_id: str) -> None:
        if fact_id not in self._facts:
            raise KeyError(f"Fact {fact_id!r} not found")
        self._facts[fact_id].invalidated = True
        self._facts[fact_id].updated_at = _utcnow_iso()

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        return self._facts.get(fact_id)

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        from_fact_id: str,
        to_fact_id: str,
        relation_type: str = "related_to",
        weight: float = 1.0,
    ) -> Relation:
        rel = Relation(
            from_fact_id=from_fact_id,
            to_fact_id=to_fact_id,
            relation_type=relation_type,
            weight=weight,
        )
        self._relations[rel.relation_id] = rel
        if from_fact_id in self._facts and to_fact_id not in self._facts[from_fact_id].related_facts:
            self._facts[from_fact_id].related_facts.append(to_fact_id)
        return rel

    def get_relations(self, fact_id: str) -> List[Relation]:
        return [r for r in self._relations.values() if r.from_fact_id == fact_id or r.to_fact_id == fact_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query(
        self,
        keyword: Optional[str] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        valid_only: bool = True,
        timestamp_iso: Optional[str] = None,
    ) -> List[Fact]:
        """
        Query facts by keyword, tags, session, or validity at a point in time.

        Args:
            keyword: Case-insensitive substring match on content.
            tags: Require all listed tags to be present.
            session_id: Filter by originating session.
            valid_only: Only return currently valid (non-invalidated) facts.
            timestamp_iso: Check validity at this ISO timestamp (time travel).

        Returns:
            List of matching Facts.
        """
        results: List[Fact] = []
        for fact in self._facts.values():
            if valid_only and not fact.is_valid_at(timestamp_iso):
                continue
            if keyword and keyword.lower() not in fact.content.lower():
                continue
            if tags and not all(t in fact.tags for t in tags):
                continue
            if session_id and fact.session_id != session_id:
                continue
            results.append(fact)
        return results

    def query_at_time(self, keyword: str, timestamp_iso: str) -> List[Fact]:
        """Time-travel query: return facts matching keyword that were valid at timestamp."""
        return self.query(keyword=keyword, valid_only=True, timestamp_iso=timestamp_iso)

    # ------------------------------------------------------------------
    # Cross-session synthesis
    # ------------------------------------------------------------------

    def get_all_sessions(self) -> List[str]:
        return list({f.session_id for f in self._facts.values()})

    def cross_session_synthesis(self, keyword: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize knowledge across all sessions.

        Returns statistics and grouped facts by session.
        """
        sessions = self.get_all_sessions()
        synthesis: Dict[str, Any] = {
            "total_facts": len(self._facts),
            "valid_facts": len([f for f in self._facts.values() if f.is_valid_at()]),
            "sessions": len(sessions),
            "by_session": {},
            "common_tags": {},
        }

        for sid in sessions:
            session_facts = self.query(keyword=keyword, session_id=sid, valid_only=False)
            synthesis["by_session"][sid] = {
                "count": len(session_facts),
                "valid": sum(1 for f in session_facts if f.is_valid_at()),
            }

        # Count tags across all valid facts
        for fact in self._facts.values():
            if fact.is_valid_at():
                for tag in fact.tags:
                    synthesis["common_tags"][tag] = synthesis["common_tags"].get(tag, 0) + 1

        return synthesis

    # ------------------------------------------------------------------
    # Pattern recognition
    # ------------------------------------------------------------------

    def find_patterns(self, min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """
        Identify recurring themes/patterns in stored facts.

        Groups facts by tags and returns groups that appear in multiple sessions.
        """
        tag_groups: Dict[str, List[str]] = {}
        for fact in self._facts.values():
            if not fact.is_valid_at():
                continue
            for tag in fact.tags:
                tag_groups.setdefault(tag, []).append(fact.session_id)

        patterns = []
        for tag, sessions in tag_groups.items():
            unique_sessions = set(sessions)
            if len(sessions) >= min_occurrences:
                patterns.append(
                    {
                        "tag": tag,
                        "occurrences": len(sessions),
                        "sessions": list(unique_sessions),
                    }
                )

        return sorted(patterns, key=lambda p: p["occurrences"], reverse=True)

    # ------------------------------------------------------------------
    # Knowledge extraction
    # ------------------------------------------------------------------

    def extract_knowledge(self, text: str, tags: Optional[List[str]] = None) -> List[Fact]:
        """
        Extract simple atomic facts from free-form text.

        Splits by sentence boundaries and stores each sentence as a fact.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        extracted: List[Fact] = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:
                fact = self.add_fact(content=sent, tags=tags or ["extracted"], source="auto_extract")
                extracted.append(fact)
        return extracted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        valid = [f for f in self._facts.values() if f.is_valid_at()]
        return {
            "total_facts": len(self._facts),
            "valid_facts": len(valid),
            "invalidated_facts": len(self._facts) - len(valid),
            "total_relations": len(self._relations),
            "sessions": len(self.get_all_sessions()),
            "avg_confidence": sum(f.confidence for f in valid) / max(len(valid), 1),
        }
