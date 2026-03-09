# Meta-Optimizer: Advanced AI Agent Framework

A comprehensive Python framework implementing 2025 state-of-the-art AI agent patterns:

## Architecture

```
src/meta_optimizer/
├── orchestration/     # Graph-based state machine orchestration
├── optimization/      # Real model compression (quantization, pruning, distillation)
├── memory/            # Temporal knowledge graph with cross-session learning
├── evaluation/        # 6D quality scoring, coherence, LLM-as-judge, A/B testing
├── monitoring/        # Production metrics, regression detection, alerting
├── agents/            # Multi-agent collaboration with shared memory
├── verification/      # Self-verification and consistency checking
└── triz/              # TRIZ problem solving, reverse mathematics, TTT adaptation
```

## Features

### Graph-Based Orchestration
LangGraph-style state machines with explicit branching, conditional routing, retry logic, and execution history.

```python
from meta_optimizer.orchestration import StateGraph, GraphState, NodeResult, NodeStatus

graph = StateGraph(initial_node="verify")
graph.add_node("verify", verify_fn)
graph.add_node("enhance", enhance_fn)
graph.add_conditional_edge("verify", lambda s: "enhance" if s.get("score", 0) > 0.5 else "verify")
graph.set_end_node("enhance")

state = graph.run(GraphState(data={"input": "..."}))
```

### Real Model Optimization
Actual quantization, pruning and knowledge distillation — not placeholders.

```python
from meta_optimizer.optimization import Quantizer, Pruner, KnowledgeDistillation

# INT8 quantization
quantizer = Quantizer(QuantizationConfig(bits=8))
q_model, result = quantizer.quantize(model)

# Unstructured pruning
pruner = Pruner(PruningConfig(amount=0.3))
p_model, result = pruner.prune(model)

# Knowledge distillation
kd = KnowledgeDistillation(DistillationConfig(temperature=4.0, alpha=0.7, epochs=5))
student, result = kd.distill(teacher, student, train_loader)
```

### Temporal Knowledge Graph
JSON-backed memory with temporal tracking, edge invalidation, and cross-session synthesis.

```python
from meta_optimizer.memory import TemporalMemory

mem = TemporalMemory(storage_path="memory.json", session_id="session-1")
fact = mem.add_fact("Paris is the capital of France", tags=["geography"])

# Update (invalidates old, creates new version)
new_fact = mem.update_fact(fact.fact_id, "Paris is the capital and largest city of France")

# Time-travel query
results = mem.query_at_time("Paris", timestamp_iso="2025-01-01T00:00:00+00:00")

# Cross-session synthesis
synthesis = mem.cross_session_synthesis()
```

### 6D Quality Evaluation
Multi-dimensional quality scoring with coherence, completeness, specificity, accuracy, relevance, and conciseness.

```python
from meta_optimizer.evaluation import QualityScorer, CoherenceChecker, LLMJudge, ABTestFramework

scorer = QualityScorer()
report = scorer.score(text, reference=reference_text)
print(f"Overall: {report.overall:.2f}")

checker = CoherenceChecker()
coherence = checker.check(text)

judge = LLMJudge(pass_threshold=0.6)
result = judge.evaluate(text, criteria="technical accuracy")

ab = ABTestFramework()
ab_result = ab.run(variant_a_texts, variant_b_texts)
```

### Production Monitoring
Real-time metrics, regression detection, and configurable alerting.

```python
from meta_optimizer.monitoring import MetricsCollector, AlertRule, LatencyTimer

collector = MetricsCollector()
collector.add_alert_rule(AlertRule("high_latency", "latency_ms.inference", 500.0, "above", "warning"))

with LatencyTimer(collector, "inference"):
    result = model(input)

regression = collector.detect_regression("quality.overall")
dashboard = collector.dashboard()
```

### Multi-Agent Collaboration
CrewAI-style role-based agents with shared memory and message bus communication.

```python
from meta_optimizer.agents import MultiAgentSystem, AgentRole
from meta_optimizer.memory import TemporalMemory

system = MultiAgentSystem(shared_memory=TemporalMemory())
system.add_agent("planner", AgentRole.PLANNER, plan_fn)
system.add_agent("researcher", AgentRole.RESEARCHER, research_fn)
system.add_agent("verifier", AgentRole.VERIFIER, verify_fn)

results = system.run_pipeline(
    tasks=["Plan research", "Gather information", "Verify findings"],
    agent_sequence=["planner", "researcher", "verifier"],
)
```

### TRIZ & Reverse Mathematics
Contradiction-oriented problem solving and backward planning from goals.

```python
from meta_optimizer.triz import TRIZSolver, Contradiction, ReverseMathPlanner, TTTAdapter

solver = TRIZSolver()
solution = solver.solve(Contradiction("speed", "energy", improving_param_id=1, worsening_param_id=2))
print(solution.principle_names)  # ["Taking out / Extraction", ...]

planner = ReverseMathPlanner()
plan = planner.plan(goal="Deploy model", available_actions=["train", "evaluate", "package", "deploy"])

adapter = TTTAdapter(my_llm_fn, max_iterations=3)
best_output, adaptations = adapter.adapt(input_data)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
python -m pytest tests/ -v
```

## Unique Advantages

1. **TRIZ Integration** — Systematic contradiction-oriented problem solving
2. **Reverse Mathematics** — Backward planning from goals (vs. industry forward planning)
3. **TTT Adaptation** — Test-time training for inference-time improvement
4. **3-Layer Architecture** — Static + Dynamic + Meta reasoning
5. **6D Quality Scoring** — Coherence, completeness, specificity, accuracy, relevance, conciseness
6. **4-Level Coherence** — Lexical, syntactic, semantic, discourse analysis
