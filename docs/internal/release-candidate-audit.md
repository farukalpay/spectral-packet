# Release Candidate Audit

This note captures the Phase 1 release-gate audit for the final public-readiness pass.

Its purpose is to separate true release blockers from deferred product ideas, freeze scope around the real engine, and define the minimum conditions the repository must satisfy before it can honestly be presented as a public-ready library product.

## Internal Product Statement

Spectral Packet Engine is a bounded-domain Gaussian and spectral compute engine for simulation, modal decomposition, inverse reconstruction, and backend-aware surrogate workflows. SQL, tabular data, MCP, and API layers exist to route real data into that same engine and expose the same workflows through different interfaces. Public readiness depends on making that one engine trustworthy, installable, and validated, not on widening the feature surface.

## Deepest Remaining Blockers

### Technical blockers

1. Package metadata is still thin for a public distribution:
   - no license metadata,
   - no author or maintainer metadata,
   - no project URLs.
   These do not block local use, but they still weaken public release readiness.

2. The core build path now works, but release validation was previously implicit rather than enforced:
   - version drift between `pyproject.toml` and `version.py` could have slipped in,
   - wheel and sdist checks were manual rather than part of the project discipline,
   - repo-facing path leaks were only being checked ad hoc.

3. The API surface remains beta in a meaningful sense:
   - the product now reports API stack incompatibility clearly,
   - a clean `.[api]` install can produce a compatible stack,
   - but mixed environments can still end up with conflicting Starlette versions.
   This is not a core blocker if the API remains explicitly beta and clean-environment guidance stays honest.

4. The ML backend surface is coherent, but not equally strong across every backend:
   - PyTorch is the primary supported path,
   - JAX is beta and platform-conditional,
   - TensorFlow is still an experimental compatibility path.
   That means release messaging must not blur the distinction between “available” and “primary”.

### Architectural blockers

1. Raw-record normalization and feature-pipeline work are still missing shared library implementations.
   These were identified in earlier audits and remain important, but they should not be rushed into the release-candidate pass. The correct release move is to freeze the stable surface around clean tabular, profile-table, SQLite, and modal-surrogate workflows rather than widening the public promise.

2. Runtime orchestration remains intentionally simple:
   - no queue manager,
   - no background-job registry,
   - no backpressure layer.
   For the current product shape, that is acceptable if CLI, Python, MCP, and API continue to expose synchronous, bounded jobs honestly. It would be a mistake to bolt on a generic job framework during the release pass.

### Trust blockers

1. The repo still needs an explicit release gate in writing, otherwise the public story can drift toward “many surfaces” rather than “one validated engine”.
2. Public-facing docs are mostly aligned, but they must continue to avoid implying:
   - raw messy-record ingestion,
   - first-class production API hosting,
   - uniform backend support across all Python and OS combinations.

## Release Gate

### Stable now and must pass release quality

- core bounded-domain spectral engine,
- Python import surface,
- packaged CLI,
- clean file-backed profile-table workflows,
- tabular and local SQLite workflows,
- artifact generation,
- PyTorch-backed modal-surrogate path,
- wheel and editable install behavior,
- version and entrypoint consistency.

### Beta and must not destabilize the core

- MCP server,
- HTTP API,
- remote SQL through SQLAlchemy,
- JAX backend path.

These may ship publicly, but only as clearly labeled beta surfaces.

### Experimental and allowed to remain limited

- TensorFlow surrogate compatibility path,
- transport dataset wrapper.

These should remain clearly marked experimental and should not drive release decisions for the stable core.

### Deferred from this release pass

- raw-record normalization for mixed-schema and nested inputs,
- analytical-table and feature-pipeline abstractions,
- feature-table-native model workflows,
- checkpoint-resume and model reload deepening,
- generic queueing and long-running job orchestration.

### Must be removed from the release promise

- any wording that implies the product is already a generic data platform,
- any wording that implies the API is first-class production hosting everywhere,
- any wording that implies TensorFlow and JAX are equal to the PyTorch path in maturity.

## Minimum Conditions For Public Readiness

Before calling the project public-ready, the following conditions must hold:

1. `pytest -q` passes.
2. `python -m build --sdist --wheel` succeeds.
3. `python -m twine check dist/*` succeeds.
4. At least one isolated standard install and one isolated editable install succeed.
5. The installed CLI entrypoint runs `--version` and `validate-install`.
6. The core Python import path works from an installed environment.
7. Stable versus beta versus experimental surfaces remain truthful.
8. Repository-facing markdown does not leak local machine paths.

## Validation Findings From This Phase

- wheel build: passed,
- sdist build: passed,
- `twine check`: passed,
- isolated standard install: passed in a virtual environment with core dependencies visible,
- isolated editable install: passed in a virtual environment with core dependencies visible,
- installed CLI `--version` and `validate-install`: passed,
- API extra: compatible in a clean-enough environment, but shared environments can still report dependency conflicts and should be treated carefully,
- public path leak scan: passed.

## Convergence Plan For The Remaining Phases

1. Harden the engine and public library boundaries first.
2. Keep packaging, metadata, and install guidance explicit.
3. Tighten API, MCP, and backend messaging around validated truth instead of equalizing surfaces that are not equally mature.
4. Avoid adding any new feature family unless it closes a concrete release blocker.
