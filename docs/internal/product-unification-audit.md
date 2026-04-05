# Product Unification Audit

This note records the Phase 1 product-fragmentation audit for the unification pass.

Its purpose is to name where the repository already behaves like one product, where it still feels split across subsystems, and what architectural guardrails must hold before deeper SQL, feature, model, MCP, and API work lands.

## Internal Product Statement

Spectral Packet Engine is a bounded-domain Gaussian and spectral compute engine. SQL, tabular data, feature engineering, backend-aware ML, MCP, and API layers exist only to move real user data into, through, and back out of that same spectral and inverse-reconstruction core. If a capability cannot be explained as a better path into spectral analysis, packet reconstruction, bounded-domain simulation, or modal-surrogate workflows, it does not belong in the primary product surface.

## Where The Product Spine Is Already Strong

- The bounded-domain simulation, projection, profile compression, spectral analysis, and inverse-fitting layers are real and already form a coherent scientific core.
- The shared workflow layer keeps CLI, MCP, and API thinner than a typical wrapper-heavy repo.
- SQLite-backed workflows and the tabular bridge already let clean file-backed and SQL-backed profile workflows enter the same engine.
- Backend-aware surrogate workflows already exist behind one public ML surface instead of three disconnected training stacks.

## Main Fragmentation Risks

### 1. Backend policy was under-expressed

The repo already had multiple ML backends, but the product-level routing logic was not visible enough. Users could see available backends, but the project did not expose a clear distinction between:

- runtime availability,
- project-supported install surfaces,
- routed default choice,
- backend warnings tied to Python version or operating system.

That makes the product look more fragmented than it actually is.

### 2. API compatibility was collapsed into a boolean

The serving surface could only say “available” or “unavailable.” That hid the real state:

- whether the `api` extra was missing,
- whether FastAPI and Uvicorn were installed,
- whether the local FastAPI/Starlette/Pydantic combination was incompatible,
- what a user should do next.

That weakens both human-user trust and AI-client routing.

### 3. SQL and ML still connect to the core more clearly than raw data does

The current product can ingest clean tables, query SQL, and train or evaluate modal surrogates, but users still fall out of the library when the upstream data is messy:

- nested JSON,
- mixed-schema records,
- analytical-table construction over multiple tables,
- reusable feature-table preparation.

If those are added carelessly, the repo will drift into a generic data platform.

### 4. MCP still has more tools than workflow guidance

MCP discovery is now cleaner than before, but the server still mostly assumes the caller already knows which tool chain to assemble. Without shared normalization, analytical-table, and feature-pipeline layers in the library, interface-side guidance would become product drift.

## Compatibility Risks That Need Explicit Handling

- JAX works in the project, but Windows should remain a conditional path rather than being implied as a first-class default.
- TensorFlow remains a compatibility backend, not the primary ML route; Python-version and platform limits must stay explicit.
- FastAPI compatibility depends on the local FastAPI/Starlette/Pydantic stack, not just on import presence.
- Backend inspection must respect the requested device consistently across Python, CLI, MCP, and API surfaces.

## Implemented In This Phase

- `ml.py` now reports the requested device explicitly, reports runtime-available backends separately from project-supported backends, and threads the requested device into JAX backend inspection instead of silently inspecting JAX with a different policy than PyTorch.
- `service_runtime.py` now owns explicit API stack inspection so the product can report installed versions, compatibility, and remediation guidance for the serving layer.
- `workflows.inspect_environment()` now aggregates ML routing state and API stack state into one machine report for Python, CLI, MCP, and API callers.
- `docs/architecture.md` and `AGENTS.md` now codify backend-routing and service-compatibility policy as shared library responsibilities, not wrapper responsibilities.

## Guardrails For The Next Phases

- Do not add raw-record, SQL, or feature logic directly to CLI, MCP, or API wrappers.
- Do not add backend-specific fallback rules outside `ml.py`.
- Do not add service-stack heuristics outside shared runtime inspection.
- Raw-record normalization, analytical-table construction, and feature-pipeline definitions must remain library-first layers that terminate in the same spectral or modal workflows.
- SQL, feature, model, MCP, and API work must deepen one engine, not create parallel product identities.
