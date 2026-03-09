"""
TRIZ integration: contradiction-oriented problem solving.

Implements core TRIZ concepts:
- Contradiction matrix
- 40 Inventive Principles
- Reverse mathematics (backward planning from goals)
- Test-Time Training (TTT) adaptation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# TRIZ Inventive Principles (subset of 40)
# ---------------------------------------------------------------------------

INVENTIVE_PRINCIPLES: Dict[int, str] = {
    1: "Segmentation",
    2: "Taking out / Extraction",
    3: "Local quality",
    4: "Asymmetry",
    5: "Merging",
    6: "Universality",
    7: "Nested doll",
    8: "Anti-weight / Counterweight",
    9: "Preliminary anti-action",
    10: "Preliminary action",
    11: "Beforehand cushioning",
    12: "Equipotentiality",
    13: "The other way round",
    14: "Spheroidality – Curvature",
    15: "Dynamics",
    16: "Partial or excessive actions",
    17: "Another dimension",
    18: "Mechanical vibration",
    19: "Periodic action",
    20: "Continuity of useful action",
    21: "Skipping",
    22: "Blessing in disguise",
    23: "Feedback",
    24: "Intermediary",
    25: "Self-service",
    26: "Copying",
    27: "Cheap short-living",
    28: "Mechanics substitution",
    29: "Pneumatics and hydraulics",
    30: "Flexible shells and thin films",
    31: "Porous materials",
    32: "Color changes",
    33: "Homogeneity",
    34: "Discarding and recovering",
    35: "Parameter changes",
    36: "Phase transitions",
    37: "Thermal expansion",
    38: "Strong oxidants",
    39: "Inert atmosphere",
    40: "Composite materials",
}

# Simplified contradiction matrix: maps (improving_param, worsening_param) -> [principle_ids]
# Only a representative subset is included.
CONTRADICTION_MATRIX: Dict[Tuple[int, int], List[int]] = {
    (1, 2): [15, 8, 29, 34],
    (1, 3): [29, 17, 38, 34],
    (2, 1): [10, 1, 29, 35],
    (2, 3): [1, 7, 4, 17],
    (3, 1): [1, 8, 15, 34],
    (3, 2): [17, 10, 4, 1],
    (4, 5): [35, 28, 6, 37],
    (5, 4): [2, 35, 16, 10],
    (6, 7): [10, 2, 13, 28],
    (7, 6): [1, 6, 15, 8],
}


@dataclass
class Contradiction:
    """A TRIZ-style technical contradiction."""

    improving_param: str
    worsening_param: str
    context: str = ""
    improving_param_id: int = 0
    worsening_param_id: int = 0


@dataclass
class TRIZSolution:
    """A proposed solution based on TRIZ principles."""

    principles: List[int]
    principle_names: List[str]
    contradiction: Contradiction
    suggestions: List[str] = field(default_factory=list)


class TRIZSolver:
    """
    TRIZ-based problem solver.

    Resolves technical contradictions by mapping parameter pairs to
    inventive principles from the TRIZ contradiction matrix.
    """

    def solve(self, contradiction: Contradiction) -> TRIZSolution:
        """
        Find inventive principles for the given contradiction.

        Args:
            contradiction: The technical contradiction to resolve.

        Returns:
            TRIZSolution with applicable principles and suggestions.
        """
        key = (contradiction.improving_param_id, contradiction.worsening_param_id)
        principles = CONTRADICTION_MATRIX.get(key, [1, 2, 10, 35])  # default fallback

        names = [INVENTIVE_PRINCIPLES.get(p, f"Principle {p}") for p in principles]
        suggestions = self._generate_suggestions(principles, contradiction)

        return TRIZSolution(
            principles=principles,
            principle_names=names,
            contradiction=contradiction,
            suggestions=suggestions,
        )

    def _generate_suggestions(self, principles: List[int], contradiction: Contradiction) -> List[str]:
        templates = {
            1: f"Segment '{contradiction.improving_param}' into smaller independent parts",
            2: f"Extract the essential component of '{contradiction.improving_param}'",
            10: f"Apply a preliminary action to address '{contradiction.worsening_param}' before it occurs",
            13: f"Try the opposite approach to '{contradiction.improving_param}'",
            15: f"Make '{contradiction.improving_param}' dynamic and adaptable",
            35: f"Change the parameter settings related to '{contradiction.worsening_param}'",
        }
        return [templates.get(p, f"Apply Principle {p}: {INVENTIVE_PRINCIPLES.get(p, '')} to resolve the contradiction") for p in principles]


# ---------------------------------------------------------------------------
# Reverse Mathematics (backward planning)
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step in a backward-planned solution."""

    step_id: int
    description: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class ReversePlan:
    """A plan constructed via backward reasoning from a goal."""

    goal: str
    steps: List[PlanStep]
    start_state: str


class ReverseMathPlanner:
    """
    Backward planning: starts from the goal and works backward to find
    the initial state and required steps.

    This is the 'reverse mathematics' concept: define what you want to
    achieve, then deduce what is required.
    """

    def plan(
        self,
        goal: str,
        available_actions: List[str],
        current_state: str = "",
    ) -> ReversePlan:
        """
        Generate a backward plan from goal to initial state.

        Args:
            goal: The desired end state.
            available_actions: List of action descriptions.
            current_state: Description of the current state.

        Returns:
            ReversePlan with ordered steps from start to goal.
        """
        # Simulate backward chaining
        steps: List[PlanStep] = []
        remaining = list(reversed(available_actions))

        for idx, action in enumerate(remaining):
            step = PlanStep(
                step_id=idx,
                description=action,
                preconditions=[f"State before: {action}"],
                postconditions=[f"Enables: {remaining[idx - 1] if idx > 0 else goal}"],
            )
            steps.append(step)

        # Reverse to get forward order
        steps = list(reversed(steps))
        for i, step in enumerate(steps):
            step.step_id = i + 1

        return ReversePlan(
            goal=goal,
            steps=steps,
            start_state=current_state or "Initial state",
        )


# ---------------------------------------------------------------------------
# Test-Time Training (TTT) Adaptation
# ---------------------------------------------------------------------------


@dataclass
class TTTAdaptation:
    """Records an adaptation made during inference."""

    input_pattern: str
    original_output: str
    adapted_output: str
    confidence_delta: float
    iteration: int


class TTTAdapter:
    """
    Test-Time Training (TTT) adaptation.

    Adapts the output of a callable function (e.g. an LLM call) at
    inference time by iteratively refining the output based on a
    self-consistency objective.
    """

    def __init__(
        self,
        fn: Any,
        max_iterations: int = 3,
        improvement_threshold: float = 0.05,
    ) -> None:
        self._fn = fn
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.adaptations: List[TTTAdaptation] = []

    def adapt(self, input_data: Any, score_fn: Any = None) -> Tuple[Any, List[TTTAdaptation]]:
        """
        Adapt the output iteratively.

        Args:
            input_data: Input to the wrapped function.
            score_fn: Optional callable(output) -> float for quality scoring.

        Returns:
            Tuple of (best_output, list_of_adaptations).
        """
        from meta_optimizer.evaluation.evaluator import QualityScorer

        if score_fn is None:
            scorer = QualityScorer()

            def score_fn(out: Any) -> float:
                return scorer.score(str(out)).overall

        best_output = self._fn(input_data)
        best_score = score_fn(best_output)
        adaptations: List[TTTAdaptation] = []

        for iteration in range(1, self.max_iterations + 1):
            candidate = self._fn(input_data)
            candidate_score = score_fn(candidate)
            delta = candidate_score - best_score

            adaptation = TTTAdaptation(
                input_pattern=str(input_data)[:100],
                original_output=str(best_output)[:200],
                adapted_output=str(candidate)[:200],
                confidence_delta=delta,
                iteration=iteration,
            )
            adaptations.append(adaptation)
            self.adaptations.append(adaptation)

            if delta > self.improvement_threshold:
                best_output = candidate
                best_score = candidate_score

        return best_output, adaptations
