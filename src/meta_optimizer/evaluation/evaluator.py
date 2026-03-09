"""
Evaluation framework: 6D quality scoring, multi-level coherence checking,
LLM-as-judge evaluation, and A/B testing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 6D Quality Scorer
# ---------------------------------------------------------------------------


@dataclass
class QualityDimension:
    name: str
    score: float  # 0.0 – 1.0
    explanation: str = ""


@dataclass
class QualityReport:
    """6-dimensional quality report for a prompt or response."""

    coherence: float = 0.0
    completeness: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    conciseness: float = 0.0
    dimensions: List[QualityDimension] = field(default_factory=list)
    overall: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        default_weights = {
            "coherence": 0.20,
            "completeness": 0.20,
            "specificity": 0.15,
            "accuracy": 0.20,
            "relevance": 0.15,
            "conciseness": 0.10,
        }
        w = weights or default_weights
        total_w = sum(w.values())
        self.overall = (
            self.coherence * w.get("coherence", 0)
            + self.completeness * w.get("completeness", 0)
            + self.specificity * w.get("specificity", 0)
            + self.accuracy * w.get("accuracy", 0)
            + self.relevance * w.get("relevance", 0)
            + self.conciseness * w.get("conciseness", 0)
        ) / total_w
        return self.overall

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence": self.coherence,
            "completeness": self.completeness,
            "specificity": self.specificity,
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "conciseness": self.conciseness,
            "overall": self.overall,
            "warnings": self.warnings,
        }


class QualityScorer:
    """
    6-dimensional quality scorer for prompts and responses.

    Dimensions:
    1. Coherence     - logical flow and consistency
    2. Completeness  - coverage of required aspects
    3. Specificity   - level of detail and precision
    4. Accuracy      - factual correctness signals
    5. Relevance     - alignment with the question/task
    6. Conciseness   - absence of unnecessary repetition
    """

    def score(self, text: str, reference: Optional[str] = None) -> QualityReport:
        """
        Score ``text`` across all 6 dimensions.

        Args:
            text: The text to evaluate.
            reference: Optional reference text for accuracy scoring.

        Returns:
            QualityReport with individual and overall scores.
        """
        report = QualityReport(
            coherence=self._score_coherence(text),
            completeness=self._score_completeness(text),
            specificity=self._score_specificity(text),
            accuracy=self._score_accuracy(text, reference),
            relevance=self._score_relevance(text, reference),
            conciseness=self._score_conciseness(text),
        )
        report.compute_overall()

        if report.coherence < 0.5:
            report.warnings.append("Low coherence: check logical flow")
        if report.completeness < 0.5:
            report.warnings.append("Low completeness: possibly missing key aspects")
        if report.conciseness < 0.4:
            report.warnings.append("Low conciseness: possible repetition")

        return report

    # ------------------------------------------------------------------
    # Heuristic dimension scorers
    # ------------------------------------------------------------------

    def _score_coherence(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return 0.8
        # Penalise very short sentences mixed with very long ones (incoherence signal)
        lengths = [len(s.split()) for s in sentences]
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        coherence = max(0.0, 1.0 - variance / (avg ** 2 + 1) * 0.1)
        return min(1.0, coherence)

    def _score_completeness(self, text: str) -> float:
        words = text.split()
        if len(words) < 20:
            return 0.3
        if len(words) < 50:
            return 0.6
        if len(words) < 150:
            return 0.8
        return 1.0

    def _score_specificity(self, text: str) -> float:
        # Numbers, proper nouns (capitalized mid-sentence), technical terms
        has_numbers = bool(re.search(r"\d+", text))
        has_technical = len(re.findall(r"\b[A-Z][a-z]{2,}\b", text)) > 2
        has_examples = bool(re.search(r"e\.g\.|for example|such as|:|\(", text, re.I))
        score = 0.4 + 0.2 * has_numbers + 0.2 * has_technical + 0.2 * has_examples
        return score

    def _score_accuracy(self, text: str, reference: Optional[str] = None) -> float:
        if reference is None:
            return 0.7  # neutral when no reference
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.7
        overlap = len(text_words & ref_words) / len(ref_words)
        return min(1.0, overlap * 1.5)

    def _score_relevance(self, text: str, reference: Optional[str] = None) -> float:
        if reference is None:
            return 0.7
        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.7
        overlap = len(text_words & ref_words) / len(ref_words | text_words)
        return min(1.0, overlap * 2.0)

    def _score_conciseness(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        unique = len(set(w.lower() for w in words))
        ratio = unique / len(words)
        return min(1.0, ratio * 1.2)


# ---------------------------------------------------------------------------
# Multi-level Coherence Checker
# ---------------------------------------------------------------------------


@dataclass
class CoherenceReport:
    """Multi-level coherence analysis result."""

    lexical: float = 0.0      # word-level consistency
    syntactic: float = 0.0    # sentence structure
    semantic: float = 0.0     # meaning-level consistency
    discourse: float = 0.0    # paragraph / overall flow
    overall: float = 0.0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lexical": self.lexical,
            "syntactic": self.syntactic,
            "semantic": self.semantic,
            "discourse": self.discourse,
            "overall": self.overall,
            "issues": self.issues,
        }


class CoherenceChecker:
    """
    4-level coherence analysis.

    Levels: Lexical → Syntactic → Semantic → Discourse
    """

    def check(self, text: str) -> CoherenceReport:
        report = CoherenceReport(
            lexical=self._lexical(text),
            syntactic=self._syntactic(text),
            semantic=self._semantic(text),
            discourse=self._discourse(text),
        )
        report.overall = (report.lexical + report.syntactic + report.semantic + report.discourse) / 4.0

        if report.lexical < 0.5:
            report.issues.append("Lexical: inconsistent vocabulary or excessive jargon")
        if report.syntactic < 0.5:
            report.issues.append("Syntactic: irregular sentence lengths or broken structure")
        if report.semantic < 0.5:
            report.issues.append("Semantic: possible contradictions or topic drift")
        if report.discourse < 0.5:
            report.issues.append("Discourse: poor paragraph transitions or flow")

        return report

    def _lexical(self, text: str) -> float:
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        if not words:
            return 0.5
        unique_ratio = len(set(words)) / len(words)
        return min(1.0, unique_ratio + 0.2)

    def _syntactic(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            return 0.8
        lengths = [len(s.split()) for s in sentences]
        avg = sum(lengths) / len(lengths)
        std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5
        cv = std / (avg + 1e-9)
        return max(0.0, 1.0 - cv * 0.5)

    def _semantic(self, text: str) -> float:
        negations = len(re.findall(r"\bnot?\b|\bnever\b|\bno\b", text, re.I))
        words = len(text.split())
        if words == 0:
            return 0.5
        negation_density = negations / words
        return max(0.0, 1.0 - negation_density * 5)

    def _discourse(self, text: str) -> float:
        connectors = re.findall(
            r"\b(however|therefore|furthermore|additionally|in addition|"
            r"consequently|as a result|for example|in contrast|on the other hand)\b",
            text,
            re.I,
        )
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return 0.7
        connector_density = len(connectors) / len(paragraphs)
        return min(1.0, 0.5 + connector_density * 0.1)


# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Result from an LLM-as-judge evaluation."""

    score: float  # 0.0 – 1.0
    reasoning: str
    passed: bool
    criteria: str
    model: str = "heuristic"


class LLMJudge:
    """
    LLM-as-judge evaluation framework.

    By default uses heuristic scoring when no LLM client is provided.
    Plug in any callable ``judge_fn(text, criteria) -> JudgeResult``
    for real LLM-backed evaluation.
    """

    def __init__(self, judge_fn: Optional[Callable[[str, str], JudgeResult]] = None, pass_threshold: float = 0.6) -> None:
        self._judge_fn = judge_fn
        self.pass_threshold = pass_threshold

    def evaluate(self, text: str, criteria: str) -> JudgeResult:
        if self._judge_fn is not None:
            return self._judge_fn(text, criteria)
        return self._heuristic_judge(text, criteria)

    def _heuristic_judge(self, text: str, criteria: str) -> JudgeResult:
        scorer = QualityScorer()
        report = scorer.score(text, reference=criteria)
        score = report.overall
        return JudgeResult(
            score=score,
            reasoning=f"Heuristic evaluation: overall={score:.2f}, "
            f"coherence={report.coherence:.2f}, completeness={report.completeness:.2f}",
            passed=score >= self.pass_threshold,
            criteria=criteria,
            model="heuristic",
        )

    def batch_evaluate(self, items: List[Tuple[str, str]]) -> List[JudgeResult]:
        """Evaluate a list of (text, criteria) pairs."""
        return [self.evaluate(text, criteria) for text, criteria in items]


# ---------------------------------------------------------------------------
# A/B Testing Framework
# ---------------------------------------------------------------------------


@dataclass
class ABVariant:
    name: str
    texts: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / max(len(self.scores), 1)

    @property
    def n(self) -> int:
        return len(self.scores)


@dataclass
class ABTestResult:
    variant_a: ABVariant
    variant_b: ABVariant
    winner: Optional[str]
    delta: float
    significant: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ABTestFramework:
    """
    Simple A/B testing for comparing two prompt/response variants.

    Uses the 6D quality scorer to evaluate each item and computes
    mean scores with a basic significance check (t-test approximation).
    """

    def __init__(self, scorer: Optional[QualityScorer] = None) -> None:
        self.scorer = scorer or QualityScorer()

    def run(
        self,
        variant_a_texts: List[str],
        variant_b_texts: List[str],
        reference: Optional[str] = None,
        significance_threshold: float = 0.05,
    ) -> ABTestResult:
        a = ABVariant(name="A", texts=variant_a_texts)
        b = ABVariant(name="B", texts=variant_b_texts)

        for text in variant_a_texts:
            a.scores.append(self.scorer.score(text, reference).overall)
        for text in variant_b_texts:
            b.scores.append(self.scorer.score(text, reference).overall)

        delta = a.mean_score - b.mean_score
        significant = self._is_significant(a.scores, b.scores, significance_threshold)
        winner = None
        if significant:
            winner = "A" if delta > 0 else "B"

        return ABTestResult(
            variant_a=a,
            variant_b=b,
            winner=winner,
            delta=delta,
            significant=significant,
            details={
                "a_mean": a.mean_score,
                "b_mean": b.mean_score,
                "a_n": a.n,
                "b_n": b.n,
            },
        )

    def _is_significant(self, a_scores: List[float], b_scores: List[float], alpha: float) -> bool:
        """Welch's t-test approximation."""
        import math

        na, nb = len(a_scores), len(b_scores)
        if na < 2 or nb < 2:
            return False

        mean_a = sum(a_scores) / na
        mean_b = sum(b_scores) / nb
        var_a = sum((x - mean_a) ** 2 for x in a_scores) / (na - 1)
        var_b = sum((x - mean_b) ** 2 for x in b_scores) / (nb - 1)

        se = math.sqrt(var_a / na + var_b / nb)
        if se == 0:
            return False

        t_stat = abs(mean_a - mean_b) / se

        # Approximation: |t| > 2.0 ≈ p < 0.05 for moderate sample sizes
        return t_stat > 2.0
