# System Convergence Audit

## Product Spine Statement

Spectral Packet Engine is a Python-first bounded-domain spectral computation library for packet simulation, modal analysis, profile compression, and inverse reconstruction, with file, SQL, CLI, MCP, and API surfaces over the same engine.

## Runtime Spine Statement

The product runs as one shared Python engine that validates environment assumptions explicitly, executes bounded in-process workflows, records task and artifact state, and leaves restart supervision to the host process manager.

## Fragmentation Audit

### Where the product already feels unified

- The spectral core remains the real center of gravity.
- The hero profile-table report workflow already connects validation, modal analysis, compression, and artifacts through one shared Python workflow.
- SQL-backed report workflows already reuse the same materialization contract and artifact layer.
- CLI, MCP, and API mostly wrap shared workflows instead of reimplementing math.

### Where the product still felt split before this pass

- Product identity was stated in docs, but not surfaced through the package or machine-facing interfaces as one shared report.
- CLI vocabulary still mixed short command names like `env` and `inspect-table` with more explicit names used by MCP and API.
- MCP, API, and CLI all exposed the same operations, but users still had to mentally map command names, tool names, and routes by hand.
- Runtime trust signals were split across environment inspection, service status, and transport/runtime inspection without one explicit product-level explanation.
- `workflows.py` is still a large orchestration hub, which makes internal ownership clearer than external package shape.
- Generated `__pycache__` directories and similar local residue make the working tree feel less intentional during audit passes.

## Biggest Convergence Failures

1. No code-owned product identity object existed for Python, CLI, MCP, and API to expose consistently.
2. Human-facing command naming still made the same workflow read differently across surfaces.
3. Runtime explanation existed, but mostly as scattered docs and endpoint descriptions rather than one shared contract.
4. The repository still leans on a monolithic workflow hub for too many responsibilities, even when the external product story is now narrower and clearer.

## Phase 1 Convergence Plan

1. Add one shared product identity and workflow catalog layer in the Python package.
2. Expose that same identity through Python, CLI, MCP, and API.
3. Normalize the CLI vocabulary around explicit `inspect-*`, `analyze-*`, `compress-*`, and `fit-*` names while preserving compatibility aliases.
4. Update public docs so README, quickstart, MCP docs, API docs, and architecture all point to the same spine statements and the same hero workflow.
5. Keep deeper runtime, workflow-module decomposition, and surface-level compatibility cleanup as later convergence work rather than forcing a broad refactor into this phase.

## Explicit Deferrals

- Do not split `workflows.py` in this pass unless a concrete bug or incoherent public contract forces it.
- Do not add a generic job model or scheduler just to make service surfaces look symmetrical.
- Do not rename MCP tool names that are already public unless a compatibility path exists and the gain clearly outweighs the churn.
