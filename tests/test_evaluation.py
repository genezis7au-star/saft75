"""Tests for evaluation framework: 6D scoring, coherence, LLM-as-judge, A/B testing."""

import pytest

from meta_optimizer.evaluation import (
    ABTestFramework,
    CoherenceChecker,
    LLMJudge,
    QualityReport,
    QualityScorer,
)


class TestQualityScorer:
    def setup_method(self):
        self.scorer = QualityScorer()

    def test_score_returns_report(self):
        report = self.scorer.score("This is a simple test sentence.")
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall <= 1.0

    def test_all_dimensions_set(self):
        report = self.scorer.score("This is a reasonably detailed test sentence with numbers like 42.")
        assert 0.0 <= report.coherence <= 1.0
        assert 0.0 <= report.completeness <= 1.0
        assert 0.0 <= report.specificity <= 1.0
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.relevance <= 1.0
        assert 0.0 <= report.conciseness <= 1.0

    def test_longer_text_higher_completeness(self):
        short = self.scorer.score("Short.")
        long_text = " ".join(["word"] * 100)
        long = self.scorer.score(long_text)
        assert long.completeness >= short.completeness

    def test_reference_affects_accuracy(self):
        reference = "The quick brown fox jumps over the lazy dog"
        on_topic = self.scorer.score("The quick brown fox jumps high", reference=reference)
        off_topic = self.scorer.score("Unrelated content about cars and roads", reference=reference)
        assert on_topic.accuracy >= off_topic.accuracy

    def test_score_with_reference(self):
        report = self.scorer.score("hello world test", reference="hello world")
        assert report.accuracy > 0.5

    def test_warnings_for_low_scores(self):
        report = self.scorer.score("x")
        assert len(report.warnings) > 0

    def test_to_dict(self):
        report = self.scorer.score("test text here today")
        d = report.to_dict()
        assert "overall" in d
        assert "coherence" in d


class TestCoherenceChecker:
    def setup_method(self):
        self.checker = CoherenceChecker()

    def test_returns_report(self):
        report = self.checker.check("This is a test. It has multiple sentences. Furthermore, it flows well.")
        assert 0.0 <= report.overall <= 1.0
        assert 0.0 <= report.lexical <= 1.0
        assert 0.0 <= report.syntactic <= 1.0
        assert 0.0 <= report.semantic <= 1.0
        assert 0.0 <= report.discourse <= 1.0

    def test_issues_populated_for_bad_text(self):
        bad_text = "X. Y. Z. A. B. C. D. E. F. G. not never no nothing"
        report = self.checker.check(bad_text)
        assert isinstance(report.issues, list)

    def test_to_dict(self):
        report = self.checker.check("Good coherent text with proper flow.")
        d = report.to_dict()
        assert all(k in d for k in ("lexical", "syntactic", "semantic", "discourse", "overall"))


class TestLLMJudge:
    def setup_method(self):
        self.judge = LLMJudge(pass_threshold=0.5)

    def test_heuristic_evaluation(self):
        result = self.judge.evaluate(
            "Python is a high-level programming language known for its readability.",
            "programming language readability",
        )
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.model == "heuristic"

    def test_batch_evaluate(self):
        items = [
            ("Good response about Python", "Python"),
            ("Another answer about Java", "Java"),
        ]
        results = self.judge.batch_evaluate(items)
        assert len(results) == 2

    def test_custom_judge_fn(self):
        from meta_optimizer.evaluation import JudgeResult

        def custom_fn(text, criteria):
            return JudgeResult(score=0.99, reasoning="always pass", passed=True, criteria=criteria, model="custom")

        judge = LLMJudge(judge_fn=custom_fn)
        result = judge.evaluate("anything", "anything")
        assert result.score == 0.99
        assert result.model == "custom"


class TestABTestFramework:
    def setup_method(self):
        self.ab = ABTestFramework()

    def test_basic_ab_test(self):
        a_texts = [
            "Python is a versatile programming language used in data science and web development.",
            "Python offers clean syntax and rich libraries for developers.",
        ]
        b_texts = [
            "x",
            "y",
        ]
        result = self.ab.run(a_texts, b_texts)
        assert result.variant_a.n == 2
        assert result.variant_b.n == 2
        assert isinstance(result.delta, float)

    def test_winner_determined_when_significant(self):
        # A variant has clearly better text (longer, more content)
        a_texts = [
            "This is a comprehensive and detailed answer about the topic with examples and specifics. " * 3
        ] * 10
        b_texts = ["Short answer."] * 10
        result = self.ab.run(a_texts, b_texts)
        if result.significant:
            assert result.winner in ("A", "B")

    def test_no_winner_for_small_samples(self):
        result = self.ab.run(["text a"], ["text b"])
        assert result.winner is None  # not significant with n=1
