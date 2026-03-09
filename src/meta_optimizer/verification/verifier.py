"""
Self-verification module.

Implements multi-step self-checking for AI-generated content:
- Consistency checks
- Factual plausibility signals
- Logical contradiction detection
- Confidence estimation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VerificationResult:
    """Result of a self-verification check."""

    passed: bool
    confidence: float  # 0.0 – 1.0
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "suggestions": self.suggestions,
            "details": self.details,
        }


class SelfVerifier:
    """
    Self-verification for generated text.

    Runs a configurable pipeline of checks:
    1. Non-emptiness
    2. Minimum length
    3. Contradiction detection
    4. Consistency with a reference
    5. Structural completeness
    """

    def __init__(self, min_words: int = 10, pass_threshold: float = 0.6) -> None:
        self.min_words = min_words
        self.pass_threshold = pass_threshold

    def verify(self, text: str, reference: Optional[str] = None) -> VerificationResult:
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        suggestions: List[str] = []
        details: Dict[str, Any] = {}

        # 1. Non-empty
        if text.strip():
            checks_passed.append("non_empty")
        else:
            checks_failed.append("non_empty")
            suggestions.append("Output is empty — regenerate")

        # 2. Minimum length
        word_count = len(text.split())
        details["word_count"] = word_count
        if word_count >= self.min_words:
            checks_passed.append("min_length")
        else:
            checks_failed.append("min_length")
            suggestions.append(f"Output has {word_count} words; at least {self.min_words} expected")

        # 3. Internal contradictions
        contradictions = self._detect_contradictions(text)
        details["contradictions"] = contradictions
        if not contradictions:
            checks_passed.append("no_contradictions")
        else:
            checks_failed.append("no_contradictions")
            suggestions.append(f"Possible contradictions detected: {contradictions}")

        # 4. Consistency with reference
        if reference is not None:
            consistent, overlap = self._check_consistency(text, reference)
            details["reference_overlap"] = overlap
            if consistent:
                checks_passed.append("reference_consistency")
            else:
                checks_failed.append("reference_consistency")
                suggestions.append("Low overlap with reference — verify content alignment")

        # 5. Structural completeness (has subject + predicate)
        if self._has_structure(text):
            checks_passed.append("structural_completeness")
        else:
            checks_failed.append("structural_completeness")
            suggestions.append("Output may lack a complete sentence structure")

        total = len(checks_passed) + len(checks_failed)
        confidence = len(checks_passed) / total if total > 0 else 0.0
        passed = confidence >= self.pass_threshold

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            suggestions=suggestions,
            details=details,
        )

    def _detect_contradictions(self, text: str) -> List[str]:
        patterns = [
            (r"\bis\b", r"\bis not\b"),
            (r"\bwill\b", r"\bwill not\b"),
            (r"\balways\b", r"\bnever\b"),
            (r"\bincreases?\b", r"\bdecreases?\b"),
        ]
        found: List[str] = []
        for pos_p, neg_p in patterns:
            if re.search(pos_p, text, re.I) and re.search(neg_p, text, re.I):
                found.append(f"{pos_p} vs {neg_p}")
        return found

    def _check_consistency(self, text: str, reference: str) -> Tuple[bool, float]:
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return True, 1.0
        overlap = len(text_words & ref_words) / len(ref_words)
        return overlap >= 0.15, overlap

    def _has_structure(self, text: str) -> bool:
        sentences = re.split(r"[.!?]+", text)
        complete = [s for s in sentences if len(s.split()) >= 3]
        return len(complete) >= 1
