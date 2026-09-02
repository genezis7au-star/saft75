"""
Multi-agent collaboration framework.

Implements CrewAI-style role-based agents with:
- Agent-to-agent communication via shared message bus
- Role assignments and task delegation
- Shared memory layer (TemporalMemory)
- Orchestrated multi-agent execution
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from meta_optimizer.memory.temporal_memory import TemporalMemory


class AgentRole(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    VERIFIER = "verifier"
    OPTIMIZER = "optimizer"
    COORDINATOR = "coordinator"
    CUSTOM = "custom"


@dataclass
class Message:
    """A message passed between agents."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""  # empty = broadcast
    content: Any = None
    message_type: str = "task"  # "task" | "result" | "error" | "info"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result produced by an agent for a task."""

    agent_name: str
    task: str
    output: Any
    success: bool
    error: Optional[str] = None


class Agent:
    """
    A role-based agent that executes tasks and communicates via a message bus.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        task_fn: Callable[["Agent", str, Any], Any],
        memory: Optional[TemporalMemory] = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.role = role
        self._task_fn = task_fn
        self.memory = memory or TemporalMemory()
        self.description = description
        self._inbox: List[Message] = []
        self._bus: Optional["MessageBus"] = None

    def receive(self, message: Message) -> None:
        self._inbox.append(message)

    def send(self, recipient: str, content: Any, message_type: str = "task") -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Message(
                sender=self.name,
                recipient=recipient,
                content=content,
                message_type=message_type,
            )
        )

    def broadcast(self, content: Any, message_type: str = "info") -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Message(
                sender=self.name,
                recipient="",
                content=content,
                message_type=message_type,
            )
        )

    def execute(self, task: str, context: Any = None) -> TaskResult:
        try:
            output = self._task_fn(self, task, context)
            return TaskResult(agent_name=self.name, task=task, output=output, success=True)
        except Exception as exc:  # noqa: BLE001
            return TaskResult(agent_name=self.name, task=task, output=None, success=False, error=str(exc))

    def get_inbox(self) -> List[Message]:
        messages = list(self._inbox)
        self._inbox.clear()
        return messages


class MessageBus:
    """Simple in-process message bus for agent communication."""

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}
        self._history: List[Message] = []

    def register(self, agent: Agent) -> None:
        self._agents[agent.name] = agent
        agent._bus = self

    def publish(self, message: Message) -> None:
        self._history.append(message)
        if message.recipient:
            if message.recipient in self._agents:
                self._agents[message.recipient].receive(message)
        else:
            for name, agent in self._agents.items():
                if name != message.sender:
                    agent.receive(message)

    def get_history(self) -> List[Message]:
        return list(self._history)


class MultiAgentSystem:
    """
    Orchestrates a team of agents to collaboratively solve tasks.

    Usage::

        system = MultiAgentSystem(shared_memory=TemporalMemory())
        system.add_agent("planner", AgentRole.PLANNER, plan_fn)
        system.add_agent("researcher", AgentRole.RESEARCHER, research_fn)
        system.add_agent("verifier", AgentRole.VERIFIER, verify_fn)

        results = system.run_pipeline(
            tasks=["Research the topic", "Verify findings"],
            agent_sequence=["researcher", "verifier"],
        )
    """

    def __init__(self, shared_memory: Optional[TemporalMemory] = None) -> None:
        self.shared_memory = shared_memory or TemporalMemory()
        self.bus = MessageBus()
        self._agents: Dict[str, Agent] = {}

    def add_agent(
        self,
        name: str,
        role: AgentRole,
        task_fn: Callable[[Agent, str, Any], Any],
        description: str = "",
    ) -> "MultiAgentSystem":
        agent = Agent(name=name, role=role, task_fn=task_fn, memory=self.shared_memory, description=description)
        self.bus.register(agent)
        self._agents[name] = agent
        return self

    def get_agent(self, name: str) -> Optional[Agent]:
        return self._agents.get(name)

    def run_pipeline(
        self,
        tasks: List[str],
        agent_sequence: List[str],
        initial_context: Any = None,
    ) -> List[TaskResult]:
        """
        Execute tasks sequentially, passing each result as context to the next agent.

        Args:
            tasks: List of task strings (same length as agent_sequence, or one task per agent).
            agent_sequence: Ordered list of agent names.
            initial_context: Initial context passed to the first agent.

        Returns:
            List of TaskResult objects.
        """
        results: List[TaskResult] = []
        context = initial_context

        for i, agent_name in enumerate(agent_sequence):
            task = tasks[i] if i < len(tasks) else tasks[-1]
            agent = self._agents.get(agent_name)
            if agent is None:
                results.append(TaskResult(agent_name=agent_name, task=task, output=None, success=False, error=f"Agent {agent_name!r} not found"))
                continue

            result = agent.execute(task, context)
            results.append(result)

            if result.success:
                context = result.output
                agent.broadcast(f"Completed task: {task[:50]}", message_type="info")

        return results

    def run_parallel(
        self,
        tasks: Dict[str, str],
    ) -> Dict[str, TaskResult]:
        """
        Execute tasks in parallel across named agents (sequential simulation).

        Args:
            tasks: Dict mapping agent_name -> task_string.

        Returns:
            Dict mapping agent_name -> TaskResult.
        """
        results: Dict[str, TaskResult] = {}
        for agent_name, task in tasks.items():
            agent = self._agents.get(agent_name)
            if agent is None:
                results[agent_name] = TaskResult(
                    agent_name=agent_name, task=task, output=None, success=False, error=f"Agent {agent_name!r} not found"
                )
            else:
                results[agent_name] = agent.execute(task)
        return results

    def get_message_history(self) -> List[Message]:
        return self.bus.get_history()
