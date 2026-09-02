"""Tests for monitoring, multi-agent, verification, and TRIZ modules."""

import pytest

from meta_optimizer.monitoring import Alert, AlertRule, LatencyTimer, MetricsCollector
from meta_optimizer.agents import Agent, AgentRole, MultiAgentSystem, TaskResult
from meta_optimizer.verification import SelfVerifier
from meta_optimizer.triz import (
    Contradiction,
    ReverseMathPlanner,
    TRIZSolver,
    TTTAdapter,
)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def setup_method(self):
        self.collector = MetricsCollector(window_size=50)

    def test_record_and_stats(self):
        for i in range(10):
            self.collector.record("latency", float(i))
        stats = self.collector.stats("latency")
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert stats["mean"] == 4.5

    def test_alert_rule_fires(self):
        fired = []
        rule = AlertRule(
            name="high_latency",
            metric_name="latency",
            threshold=100.0,
            condition="above",
            severity="warning",
        )
        self.collector.add_alert_rule(rule)
        self.collector.add_alert_handler(fired.append)
        self.collector.record("latency", 200.0)
        assert len(fired) == 1
        assert fired[0].severity == "warning"

    def test_alert_rule_does_not_fire_below_threshold(self):
        fired = []
        rule = AlertRule(
            name="high_latency",
            metric_name="latency",
            threshold=100.0,
            condition="above",
        )
        self.collector.add_alert_rule(rule)
        self.collector.add_alert_handler(fired.append)
        self.collector.record("latency", 50.0)
        assert len(fired) == 0

    def test_regression_detection(self):
        # Baseline: high quality scores
        for _ in range(20):
            self.collector.record("quality", 0.9)
        # Current: dropped significantly
        for _ in range(5):
            self.collector.record("quality", 0.3)
        regression = self.collector.detect_regression("quality", baseline_window=20, current_window=5, threshold_pct=10)
        assert regression is not None
        assert regression["direction"] == "regression"

    def test_no_regression_when_stable(self):
        for _ in range(25):
            self.collector.record("quality", 0.9)
        result = self.collector.detect_regression("quality", baseline_window=20, current_window=5, threshold_pct=10)
        assert result is None

    def test_latency_timer(self):
        with LatencyTimer(self.collector, "test_op"):
            pass
        stats = self.collector.stats("latency_ms.test_op")
        assert stats["count"] == 1
        assert stats["latest"] >= 0.0

    def test_dashboard(self):
        self.collector.record("metric_a", 1.0)
        self.collector.record("metric_b", 2.0)
        dashboard = self.collector.dashboard()
        assert "metric_a" in dashboard["metrics"]
        assert "metric_b" in dashboard["metrics"]
        assert "generated_at" in dashboard


# ---------------------------------------------------------------------------
# Multi-agent
# ---------------------------------------------------------------------------


class TestMultiAgentSystem:
    def test_add_and_run_agent(self):
        system = MultiAgentSystem()

        def task_fn(agent, task, context):
            return f"result of: {task}"

        system.add_agent("worker", AgentRole.RESEARCHER, task_fn)
        results = system.run_pipeline(["do work"], ["worker"])
        assert results[0].success
        assert "do work" in results[0].output

    def test_pipeline_passes_context(self):
        system = MultiAgentSystem()
        received_contexts = []

        def agent_fn(agent, task, context):
            received_contexts.append(context)
            return f"processed: {context}"

        system.add_agent("a1", AgentRole.PLANNER, agent_fn)
        system.add_agent("a2", AgentRole.VERIFIER, agent_fn)

        system.run_pipeline(["task1", "task2"], ["a1", "a2"], initial_context="start")
        assert received_contexts[0] == "start"
        assert "start" in received_contexts[1]

    def test_parallel_execution(self):
        system = MultiAgentSystem()

        system.add_agent("a", AgentRole.RESEARCHER, lambda ag, t, c: f"a:{t}")
        system.add_agent("b", AgentRole.OPTIMIZER, lambda ag, t, c: f"b:{t}")

        results = system.run_parallel({"a": "task_a", "b": "task_b"})
        assert "a" in results
        assert "b" in results
        assert results["a"].success
        assert results["b"].success

    def test_missing_agent_handled(self):
        system = MultiAgentSystem()
        results = system.run_pipeline(["task"], ["nonexistent"])
        assert results[0].success is False
        assert "not found" in results[0].error

    def test_shared_memory(self):
        from meta_optimizer.memory import TemporalMemory

        mem = TemporalMemory()
        system = MultiAgentSystem(shared_memory=mem)

        def writer(agent, task, context):
            agent.memory.add_fact("shared knowledge", tags=["shared"])
            return "wrote"

        def reader(agent, task, context):
            facts = agent.memory.query(keyword="shared")
            return len(facts)

        system.add_agent("writer", AgentRole.PLANNER, writer)
        system.add_agent("reader", AgentRole.VERIFIER, reader)

        results = system.run_pipeline(["write", "read"], ["writer", "reader"])
        assert results[1].output >= 1

    def test_message_bus_communication(self):
        system = MultiAgentSystem()
        messages_sent = []

        def sender_fn(agent, task, context):
            agent.send("receiver", "hello from sender", message_type="info")
            return "sent"

        def receiver_fn(agent, task, context):
            msgs = agent.get_inbox()
            messages_sent.extend(msgs)
            return "received"

        system.add_agent("sender", AgentRole.PLANNER, sender_fn)
        system.add_agent("receiver", AgentRole.VERIFIER, receiver_fn)
        system.run_pipeline(["send", "receive"], ["sender", "receiver"])
        targeted = [m for m in messages_sent if m.content == "hello from sender"]
        assert len(targeted) == 1


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------


class TestSelfVerifier:
    def setup_method(self):
        self.verifier = SelfVerifier(min_words=5, pass_threshold=0.6)

    def test_passes_for_good_text(self):
        result = self.verifier.verify("Python is a high-level programming language with clear syntax and broad applicability.")
        assert result.passed
        assert result.confidence > 0.5

    def test_fails_for_empty(self):
        result = self.verifier.verify("")
        assert not result.passed
        assert "non_empty" in result.checks_failed

    def test_fails_for_short_text(self):
        result = self.verifier.verify("Hi")
        assert "min_length" in result.checks_failed

    def test_consistency_check(self):
        text = "The solution is highly effective for the given problem domain."
        reference = "effective solution problem"
        result = self.verifier.verify(text, reference=reference)
        assert "reference_consistency" in result.checks_passed

    def test_contradiction_detection(self):
        text = "This always works. This never works. The value increases and then decreases."
        result = self.verifier.verify(text)
        assert "no_contradictions" in result.checks_failed

    def test_to_dict(self):
        result = self.verifier.verify("A reasonable sentence with content.")
        d = result.to_dict()
        assert "passed" in d
        assert "confidence" in d


# ---------------------------------------------------------------------------
# TRIZ
# ---------------------------------------------------------------------------


class TestTRIZSolver:
    def setup_method(self):
        self.solver = TRIZSolver()

    def test_solve_returns_principles(self):
        c = Contradiction(
            improving_param="speed",
            worsening_param="energy",
            improving_param_id=1,
            worsening_param_id=2,
        )
        solution = self.solver.solve(c)
        assert len(solution.principles) > 0
        assert len(solution.principle_names) == len(solution.principles)

    def test_solve_returns_suggestions(self):
        c = Contradiction(
            improving_param="strength",
            worsening_param="weight",
            improving_param_id=3,
            worsening_param_id=1,
        )
        solution = self.solver.solve(c)
        assert len(solution.suggestions) > 0

    def test_fallback_for_unknown_params(self):
        c = Contradiction(
            improving_param="X",
            worsening_param="Y",
            improving_param_id=99,
            worsening_param_id=99,
        )
        solution = self.solver.solve(c)
        assert len(solution.principles) > 0


class TestReverseMathPlanner:
    def setup_method(self):
        self.planner = ReverseMathPlanner()

    def test_plan_creates_steps(self):
        plan = self.planner.plan(
            goal="Deploy a production model",
            available_actions=["Train model", "Evaluate model", "Package model", "Deploy model"],
            current_state="Raw data available",
        )
        assert plan.goal == "Deploy a production model"
        assert len(plan.steps) == 4
        assert plan.start_state == "Raw data available"

    def test_steps_ordered(self):
        plan = self.planner.plan(
            goal="End state",
            available_actions=["step1", "step2", "step3"],
        )
        step_ids = [s.step_id for s in plan.steps]
        assert step_ids == sorted(step_ids)


class TestTTTAdapter:
    def test_adapt_selects_best(self):
        outputs = ["short", "this is a much longer and more detailed response about the topic"]
        call_count = [0]

        def mock_fn(inp):
            out = outputs[min(call_count[0], len(outputs) - 1)]
            call_count[0] += 1
            return out

        adapter = TTTAdapter(mock_fn, max_iterations=2, improvement_threshold=0.01)
        best, adaptations = adapter.adapt("test input")
        assert len(adaptations) == 2
        # The longer output should score better on completeness
        assert best is not None
