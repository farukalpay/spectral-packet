# Final Convergence Audit

Date: 2026-04-04

This is now the canonical internal audit for the final convergence pass. Older internal audits remain historical context, not the active release gate.

## Session Evidence

Validated directly in this session:

- repository-wide source audit across engine, workflows, ML, SQL/data, CLI, API, MCP, docs, and tests
- `pytest -q`: `73 passed, 3 skipped`
- `python3 -m build --sdist --wheel`
- `python3 -m twine check dist/*`
- in-process CLI smoke path for `forward` writing a real artifact bundle
- direct numerical probes for projection quality, norm preservation, and inverse fitting behavior

Phase 1 defects found and fixed immediately:

- `modal_energy()` and `modal_tail()` treated complex coefficients incorrectly and could fail on valid spectral inputs
- `SpectralPropagator.propagate()` silently accepted time vectors and flattened multi-time output into a single state

## Current Verdict

The repository is stronger than a prototype, but it is not yet ready for an honest public release.

The release gate is currently too willing to certify a codebase whose wrappers are broader than its core contracts.

## Deepest Remaining Blockers

### 1. Wrapper breadth still outpaces engine depth

The core engine files are materially smaller than the extension layers around them.

Current line-count snapshot:

- engine core: about 1500 lines
- workflow and runtime glue: about 2773 lines
- interfaces and public façades: about 3083 lines
- data and SQL layer: about 2084 lines
- ML layer: about 2084 lines

That does not prove weakness by itself, but in this repository it matches the architectural smell: the product idea is heavier than the current engine contract.

### 2. `workflows.py` is carrying too many responsibilities

`src/spectral_packet_engine/workflows.py` currently mixes:

- environment inspection
- install validation
- engine construction
- forward and inverse spectral workflows
- profile-table workflows
- SQL-backed workflows
- dataset benchmarking
- ML training and evaluation

This makes the library harder to reason about, harder to stabilize, and too easy to keep extending without deepening the engine.

### 3. Engine contracts are still too thin for the stated vision

The engine still lacks a strong shared contract for:

- engine configuration objects instead of parameter bundles
- projection and truncation error budgets
- propagation-mode and basis-mode validation
- engine-level diagnostics beyond summary tensors
- explicit sanity checks around weak or invalid numerical states
- deterministic reproducibility rules where they matter

The numerics are promising, but the engine still feels like a compact capability set with large wrappers around it.

### 4. Runtime and orchestration are still status-only

`src/spectral_packet_engine/service_status.py` tracks tasks, but the repository still does not have a real runtime backbone for:

- startup validation across subsystems
- capacity or overload policy
- queueing and backpressure
- coherent task lifecycle management
- artifact tracking as a runtime concern
- job-oriented behavior for long-running surfaces

The current status registry is useful observability, not yet a runtime.

### 5. The SQL layer still violates the intended architecture

`src/spectral_packet_engine/database.py` still relies heavily on handwritten SQL strings and raw `where` fragments.

That conflicts with the repository rule to prefer explicit shared SQL policy and SQLAlchemy Core for non-trivial SQL behavior. It is workable for SQLite utilities, but too loose for a trustworthy extension layer.

### 6. Public Python surface is too wide and too facaded

`src/spectral_packet_engine/__init__.py`, `core.py`, `data.py`, `interfaces.py`, and `surrogates.py` expose a very large surface while the stable engine boundaries are still implicit.

This makes the library look rich, but it weakens discoverability and makes it harder to distinguish:

- stable engine contracts
- workflow conveniences
- beta interfaces
- experimental compatibility paths

### 7. Validation is still too happy-path and too local

The current suite is good at consistency and representative success paths. It is not yet strong enough on:

- failure-mode coverage
- cross-platform evidence
- clean-install evidence
- backend-combination evidence
- concurrency and overload behavior
- engine-parameter sweeps that could expose numerical fragility

### 8. Public docs are still ahead of validated truth

The public release-facing docs still speak more confidently than this session's evidence supports, especially around install-path validation and overall release readiness.

That is a release blocker until implementation truth and docs truth converge.

## Scope Freeze

### Must Be Fixed Now

- strengthen the engine contract before expanding any surface
- split the monolithic workflow hub into clearer engine, data, ML, and service workflow boundaries
- introduce stronger engine diagnostics and inspectable configuration objects
- harden runtime startup, subsystem inspection, and overload behavior
- centralize backend and platform safety so Python, CLI, API, and MCP report the same truth
- narrow and clarify the stable Python surface
- expand validation around failure paths, clean installs, and backend/platform combinations

### Must Be Deferred

- new benchmark or dataset integrations
- broader feature-pipeline work
- richer remote SQL ambitions beyond an explicitly beta path
- model-resume or checkpoint productization beyond what is needed for trust

### Must Not Be Added

- generic AI-platform behavior
- generic MLOps or model-serving abstractions
- generic SQL-platform features
- generic backend-framework capabilities
- document-ingestion claims beyond structured table extraction

### Must Remain Clearly Experimental Or Beta

- TensorFlow compatibility workflow: experimental
- transport dataset benchmarking: experimental
- remote SQL backends: beta
- API and MCP: beta until runtime/backpressure behavior is stronger
- JAX modal surrogate: beta, platform-conditional

## Phase Plan

### Phase 2: Engine, Library, Backend, Runtime

- deepen engine state, propagation, diagnostics, and validation contracts
- introduce build-worthy engine and workflow configuration objects
- harden backend inspection and routing around platform and Python-version truth
- build a real runtime backbone instead of status-only tracking

### Phase 3: MCP/API Convergence, Validation, Cleanup

- make MCP and API reflect one runtime and one job model
- expand failure-path and concurrency-sensitive validation
- remove duplicated wrapper logic and leftover shallow abstractions
- clean dead or misleading code paths that survived earlier expansion

### Phase 4: Docs And Release Verdict

- update README, help output, and docs only after code truth is stronger
- publish an honest release verdict with validated, partially validated, and experimental surfaces separated sharply

## Release-Gate Rule

The repository is not release-ready because tests are green and the package builds.

It becomes release-ready only when:

- the engine carries more of the product weight than the wrappers around it
- runtime and backend behavior are explicit and trustworthy
- the validation matrix covers real failure and platform risk
- public docs describe only what has actually been validated
