"""Tests for graph-based orchestration (StateGraph)."""

import pytest

from meta_optimizer.orchestration import (
    GraphState,
    NodeResult,
    NodeStatus,
    StateGraph,
)


def make_simple_graph() -> StateGraph:
    """Build a simple 3-node graph: start -> middle -> end."""
    graph = StateGraph(initial_node="start")

    def start_fn(state: GraphState) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS, data={"visited_start": True})

    def middle_fn(state: GraphState) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS, data={"visited_middle": True})

    def end_fn(state: GraphState) -> NodeResult:
        return NodeResult(status=NodeStatus.SUCCESS, data={"visited_end": True})

    graph.add_node("start", start_fn)
    graph.add_node("middle", middle_fn)
    graph.add_node("end", end_fn)
    graph.add_edge("start", "middle")
    graph.add_edge("middle", "end")
    graph.set_end_node("end")
    return graph


class TestStateGraph:
    def test_basic_execution(self):
        graph = make_simple_graph()
        state = graph.run()
        assert state.get("visited_start") is True
        assert state.get("visited_middle") is True
        assert state.get("visited_end") is True
        assert not state.has_errors()

    def test_history_recorded(self):
        graph = make_simple_graph()
        state = graph.run()
        assert state.history == ["start", "middle", "end"]

    def test_conditional_routing(self):
        graph = StateGraph(initial_node="check")

        def check_fn(state: GraphState) -> NodeResult:
            return NodeResult(status=NodeStatus.SUCCESS, data={"score": 0.8})

        def high_fn(state: GraphState) -> NodeResult:
            return NodeResult(status=NodeStatus.SUCCESS, data={"branch": "high"})

        def low_fn(state: GraphState) -> NodeResult:
            return NodeResult(status=NodeStatus.SUCCESS, data={"branch": "low"})

        graph.add_node("check", check_fn)
        graph.add_node("high", high_fn)
        graph.add_node("low", low_fn)
        graph.add_conditional_edge(
            "check",
            lambda s: "high" if s.get("score", 0) > 0.5 else "low",
        )
        graph.set_end_node("high")
        graph.set_end_node("low")

        state = graph.run()
        assert state.get("branch") == "high"
        assert not state.has_errors()

    def test_node_failure_stops_execution(self):
        graph = StateGraph(initial_node="fail_node")

        def fail_fn(state: GraphState) -> NodeResult:
            raise RuntimeError("intentional failure")

        graph.add_node("fail_node", fail_fn)
        state = graph.run()
        assert state.has_errors()
        assert "fail_node" in state.errors[0]

    def test_retry_on_failure(self):
        call_count = [0]

        def flaky_fn(state: GraphState) -> NodeResult:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("transient error")
            return NodeResult(status=NodeStatus.SUCCESS, data={"retried": True})

        graph = StateGraph(initial_node="flaky")
        graph.add_node("flaky", flaky_fn, max_retries=3)
        graph.set_end_node("flaky")
        state = graph.run()
        assert state.get("retried") is True
        assert call_count[0] == 3

    def test_execution_summary(self):
        graph = make_simple_graph()
        graph.run()
        summary = graph.get_execution_summary()
        assert summary["steps"] == 3
        assert summary["failed_nodes"] == []

    def test_max_steps_safety(self):
        graph = StateGraph(initial_node="loop")

        def loop_fn(state: GraphState) -> NodeResult:
            return NodeResult(status=NodeStatus.SUCCESS)

        graph.add_node("loop", loop_fn)
        graph.add_edge("loop", "loop")
        state = graph.run(max_steps=5)
        assert state.has_errors()

    def test_initial_state_preserved(self):
        graph = StateGraph(initial_node="read")

        def read_fn(state: GraphState) -> NodeResult:
            val = state.get("initial_val")
            return NodeResult(status=NodeStatus.SUCCESS, data={"read_val": val})

        graph.add_node("read", read_fn)
        graph.set_end_node("read")
        initial = GraphState(data={"initial_val": 42})
        state = graph.run(initial)
        assert state.get("read_val") == 42

    def test_unknown_node_error(self):
        graph = StateGraph(initial_node="missing")
        state = graph.run()
        assert state.has_errors()
        assert "missing" in state.errors[0]
