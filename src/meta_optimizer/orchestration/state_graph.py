"""
Graph-based orchestration: LangGraph-style state machines.

Provides explicit state machines with branching logic, error handling, and retries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GraphState:
    """Mutable state passed between graph nodes."""

    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    current_node: str = ""
    history: List[str] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass
class NodeResult:
    """Result produced by a graph node."""

    status: NodeStatus
    next_node: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: float = 0.0


class GraphNode:
    """
    A single node in the state graph.

    Each node runs a function, handles errors, and decides the next node.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[GraphState], NodeResult],
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> None:
        self.name = name
        self.func = func
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

    def execute(self, state: GraphState) -> NodeResult:
        """Execute this node with retry logic."""
        attempts = 0
        last_error: Optional[str] = None
        start = time.monotonic()

        while attempts <= self.max_retries:
            start = time.monotonic()
            try:
                result = self.func(state)
                result.duration_ms = (time.monotonic() - start) * 1000
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                attempts += 1
                if attempts <= self.max_retries and self.retry_delay_s > 0:
                    time.sleep(self.retry_delay_s)

        duration_ms = (time.monotonic() - start) * 1000
        return NodeResult(
            status=NodeStatus.FAILED,
            error=last_error,
            duration_ms=duration_ms,
        )


class StateGraph:
    """
    Explicit state-machine graph for AI workflow orchestration.

    Usage::

        graph = StateGraph(initial_node="verify")
        graph.add_node("verify", verify_fn)
        graph.add_node("enhance", enhance_fn)
        graph.add_node("optimize", optimize_fn)
        graph.add_node("validate", validate_fn)

        graph.add_edge("verify", "enhance")
        graph.add_conditional_edge(
            "enhance",
            condition=lambda s: "optimize" if s.get("score", 0) > 0.5 else "verify",
        )
        graph.add_edge("optimize", "validate")
        graph.set_end_node("validate")

        state = GraphState(data={"input": "..."})
        final_state = graph.run(state)
    """

    END = "__END__"

    def __init__(self, initial_node: str = "") -> None:
        self._initial_node = initial_node
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, Callable[[GraphState], str]] = {}
        self._end_nodes: set[str] = set()
        self.execution_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        func: Callable[[GraphState], NodeResult],
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> "StateGraph":
        self._nodes[name] = GraphNode(name, func, max_retries, retry_delay_s)
        return self

    def add_edge(self, from_node: str, to_node: str) -> "StateGraph":
        self._edges[from_node] = to_node
        return self

    def add_conditional_edge(
        self,
        from_node: str,
        condition: Callable[[GraphState], str],
    ) -> "StateGraph":
        self._conditional_edges[from_node] = condition
        return self

    def set_end_node(self, node: str) -> "StateGraph":
        self._end_nodes.add(node)
        return self

    def set_initial_node(self, node: str) -> "StateGraph":
        self._initial_node = node
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _next_node(self, current: str, state: GraphState) -> Optional[str]:
        if current in self._end_nodes:
            return None
        if current in self._conditional_edges:
            return self._conditional_edges[current](state)
        if current in self._edges:
            target = self._edges[current]
            return None if target == self.END else target
        return None

    def run(self, state: Optional[GraphState] = None, max_steps: int = 100) -> GraphState:
        """
        Execute the graph starting from the initial node.

        Args:
            state: Initial state (creates empty state if None).
            max_steps: Safety limit to prevent infinite loops.

        Returns:
            Final GraphState after execution completes.
        """
        if state is None:
            state = GraphState()

        if not self._initial_node:
            raise ValueError("initial_node must be set before calling run()")

        self.execution_log = []
        current = self._initial_node
        steps = 0

        while current and steps < max_steps:
            if current not in self._nodes:
                state.add_error(f"Unknown node: {current!r}")
                break

            state.current_node = current
            state.history.append(current)

            node = self._nodes[current]
            result = node.execute(state)

            log_entry: Dict[str, Any] = {
                "node": current,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
            }
            if result.error:
                log_entry["error"] = result.error

            self.execution_log.append(log_entry)

            if result.data:
                state.update(**result.data)

            if result.status == NodeStatus.FAILED:
                if result.error:
                    state.add_error(f"Node {current!r} failed: {result.error}")
                break

            if result.next_node is not None:
                current = result.next_node
            else:
                current = self._next_node(current, state)  # type: ignore[assignment]

            steps += 1

        if steps >= max_steps:
            state.add_error(f"Graph exceeded max_steps={max_steps}")

        return state

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def get_execution_summary(self) -> Dict[str, Any]:
        total_ms = sum(e.get("duration_ms", 0) for e in self.execution_log)
        failed = [e["node"] for e in self.execution_log if e["status"] == "failed"]
        return {
            "steps": len(self.execution_log),
            "total_duration_ms": total_ms,
            "failed_nodes": failed,
            "log": self.execution_log,
        }
