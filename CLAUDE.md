# CLAUDE.md

This file provides guidance to AI assistants (such as Claude) working in this repository.

## Repository Overview

**META-OPTIMIZER v7.0** is a Python toolkit of three loosely-coupled subsystems, delivered
as source under `.claude/tools/` rather than a conventional top-level package:

1. **Temporal Knowledge Graph (TKG) memory** — graph + vector hybrid storage for
   cross-session knowledge, with temporal queries ("what changed since last week?").
2. **Real model optimization pipeline** — actual PyTorch quantization (INT8/INT4),
   pruning (structured/unstructured), and ONNX/TensorRT export, not placeholders.
3. **Workflow orchestration** — a small LangGraph-inspired explicit state machine with
   conditional routing, retry/error recovery, and parallel strategy execution.

There is no CLI, server, or entry point — this is a library intended to be imported by
another project. See `README.md` for the full usage guide and quick-start examples.

## Project Structure

```
.claude/
├── tools/
│   ├── temporal_memory/          # TKG Memory System
│   │   ├── tkg_engine.py         # TemporalKnowledgeGraph — main public interface
│   │   ├── graph_store.py        # NetworkXGraphStore, GraphNode, GraphEdge
│   │   ├── vector_store.py       # VectorDocument, create_vector_store (in-memory/ChromaDB)
│   │   ├── hybrid_retriever.py   # HybridRetriever — combines graph + vector search
│   │   └── temporal_query.py     # TemporalQuery — time-aware queries
│   │
│   ├── amazon_robotics/          # Real Optimization
│   │   ├── quantizer.py          # RealQuantizer — PyTorch INT8/INT4 quantization
│   │   ├── pruner.py             # RealPruner — structured/unstructured pruning
│   │   ├── exporter.py           # ModelExporter — ONNX/TensorRT export
│   │   └── real_optimizer.py     # OptimizationPipeline — chains the above; PREDEFINED_STRATEGIES
│   │
│   └── orchestration/            # Workflow Engine
│       ├── workflow_engine.py    # WorkflowEngine, WorkflowNode, MetaOptimizerWorkflow
│       ├── state_manager.py      # StateManager, WorkflowState — checkpointing
│       ├── error_handler.py      # ErrorHandler, RetryConfig, ErrorRecord
│       └── parallel_executor.py  # ParallelExecutor — concurrent strategy execution
│
└── data/
    ├── tkg/                      # Persisted TKG storage (graph.json, vectors.json)
    └── workflow_states/          # Workflow checkpoints
```

Each subpackage's `__init__.py` is the intended import surface (re-exports the public
classes/dataclasses via `__all__`) — import from the package, not the submodule, e.g.
`from .claude.tools.temporal_memory import TemporalKnowledgeGraph`, not
`from .claude.tools.temporal_memory.tkg_engine import TemporalKnowledgeGraph`.

**Known issue:** `README.md`'s example imports use `from claude.tools.temporal_memory import ...`
(and similarly for `amazon_robotics` / `orchestration`). This does not work as written —
the source lives under `.claude/tools/...` (a dot-prefixed directory), which is not
importable as a `claude` package by adding the repo root to `sys.path`. To actually import
these modules, either add `.claude` itself to `sys.path` and import `from tools.temporal_memory
import ...`, or reference the modules by their real relative path. Don't propagate the
README's `claude.tools...` import path as if it works; if you fix call sites, fix the
README too rather than leaving the two inconsistent.

## Language, Runtime, and Dependencies

- **Language**: Python 3 (uses `from __future__ import annotations`, so 3.7+; f-strings and
  `dataclasses` are used throughout — target 3.9+ in practice).
- **No `requirements.txt` / `pyproject.toml` / `setup.py` exists yet.** Dependencies are
  documented only in `README.md`:
  - Required: `networkx`, `torch`, `onnx`, `onnxruntime`
  - Optional: `chromadb`, `sentence-transformers` (enables the ChromaDB vector backend;
    without it, `vector_store.py`'s in-memory backend is used)
  - If you add real dependency management, create a `requirements.txt` (or
    `pyproject.toml`) and update both this file and `README.md`'s install section.

## Code Conventions

Follow the patterns already established in `.claude/tools/`:

- `from __future__ import annotations` at the top of every module.
- Full type hints on public functions/methods (`Optional`, `Dict[str, Any]`, etc.).
- Configuration and results are `@dataclass`es with a `to_dict()` method (see
  `QuantizationConfig`/`QuantizationResult` in `quantizer.py`, `PipelineResult` in
  `real_optimizer.py`) — follow this pattern for new config/result types rather than
  plain dicts.
- Enums for fixed categories (`EntityType`, `NodeType`) subclass `str, Enum` so values
  serialize cleanly to JSON and compare equal to plain strings.
- Google-style docstrings (`Args:` / `Returns:`) on public methods; module-level
  docstrings describe the file's role in the subsystem.
- `# ---- Section ----` comment dividers group related methods within a class (see
  `tkg_engine.py`).
- Persistence is explicit and JSON-based (`_save()`/`_load()` writing to a
  `storage_path`), not a database — keep new persistence consistent with this unless a
  real backend is deliberately introduced.
- Public API is curated via each subpackage's `__init__.py` `__all__` list — add new
  public symbols there when you add them to a module.

General principles regardless of stack:
- Prefer clarity over cleverness
- Keep functions small and focused on a single responsibility
- Avoid over-engineering — build only what is needed now
- Do not add error handling for scenarios that cannot occur
- Do not add comments unless the logic is non-obvious

## Testing

**There is no test suite in this repository yet** (no `tests/` directory, no `pytest`
config). `README.md`'s "Testing" section shows ad-hoc smoke-test snippets run manually
via `python -c "..."` against each module (`TemporalKnowledgeGraph`, `RealQuantizer`,
`MetaOptimizerWorkflow.visualize_workflow()`), not an automated suite.

If you add tests, prefer `pytest` with a top-level `tests/` mirroring the
`.claude/tools/` layout (`tests/temporal_memory/`, `tests/amazon_robotics/`,
`tests/orchestration/`), and document the run command here once established.

## Git Workflow

### Branch naming
- AI-assisted work: `claude/<short-description>-<session-id>`
- Features: `feature/<description>`
- Bug fixes: `fix/<description>`
- This repo has no long-lived `main`/`master` — check `git remote show origin` for the
  current default branch (HEAD) before branching, since it may itself be a
  `claude/...`-named branch left over from prior AI-assisted work.

### Commit messages
Write clear, imperative-style commit messages:
```
Add authentication middleware
Fix null pointer in user service
Refactor database connection pool
```

### Pushing changes
Always push with tracking:
```bash
git push -u origin <branch-name>
```

### Pull requests
- Keep PRs focused on a single concern
- Include a clear description of what changed and why

## Security

- Never commit secrets, API keys, or credentials
- `.gitignore` already excludes Python build artifacts, virtual envs, IDE files, and
  optimizer output artifacts (`*.onnx`, `*.trt`, `*.engine`) — extend it rather than
  committing generated/exported models
- Validate all user input at system boundaries

## AI Assistant Guidelines

When working in this repository, AI assistants should:

1. **Read before editing** — always read a file before modifying it
2. **Stay in scope** — only change what was requested; avoid unsolicited refactors
3. **Keep changes minimal** — prefer editing existing files over creating new ones
4. **Confirm before destructive actions** — force pushes, file deletions, branch resets require explicit user approval
5. **Use the designated branch** — all work goes to the branch specified in the task context
6. **Match existing conventions** — see Code Conventions above (dataclasses, type hints,
   `__all__`-curated exports, docstring style) rather than introducing new patterns
7. **Update this file** — when new conventions, tools, dependencies, or structure are established, update the relevant section of this file

## Updating This File

As the project grows, keep this file in sync with reality:
- Add a real dependency manifest (`requirements.txt`/`pyproject.toml`) once one exists, and update this file's Dependencies section
- Document an actual test command once a test suite is added
- Resolve and document the `claude.tools...` vs `.claude/tools/...` import path inconsistency if/when it's fixed
- Note any CI/CD pipeline once one is added
- Record any new subsystem added alongside `temporal_memory`, `amazon_robotics`, and `orchestration`
