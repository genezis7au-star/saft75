"""Evaluation package: quality scoring, coherence checking, LLM-as-judge, A/B testing."""

from .evaluator import (
    ABTestFramework,
    ABTestResult,
    ABVariant,
    CoherenceChecker,
    CoherenceReport,
    JudgeResult,
    LLMJudge,
    QualityReport,
    QualityScorer,
)

__all__ = [
    "ABTestFramework",
    "ABTestResult",
    "ABVariant",
    "CoherenceChecker",
    "CoherenceReport",
    "JudgeResult",
    "LLMJudge",
    "QualityReport",
    "QualityScorer",
]
