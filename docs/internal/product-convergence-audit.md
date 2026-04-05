# Product Convergence Audit

This note records the Phase 1 multi-persona audit for the final “make it one real product” pass.

Its purpose is to identify where the repository already behaves like one coherent engine, where users still experience it as several adjacent products, and what architectural and validation rules must govern the remaining convergence work.

## One-Sentence Internal Product Spine

Spectral Packet Engine is a bounded-domain Gaussian and spectral compute engine whose SQL, tabular-data, model, MCP, CLI, and API layers exist only to move real user data into, through, and back out of that same spectral and inverse-reconstruction core.

## Product Spine Rule

- The spectral and bounded-domain engine is the mathematical and architectural center.
- SQL, data, feature, model, MCP, CLI, and API layers are extension layers around that same center.
- No extension layer may replace the core identity, bypass the shared workflow layer, or create a separate product story.
- Broader usefulness must come from one coherent engine, not from disconnected subsystems.

## Multi-Persona Audit

### A. Security and automation user

What works:

- packaged CLI,
- environment inspection,
- explicit ML backend inspection,
- database bootstrap and query workflows,
- artifact bundle generation,
- bounded MCP tool surface instead of shell-first automation.

What feels weak:

- no structured runtime log surface,
- no explicit overload or concurrency model,
- no job registry or task status layer,
- no retry or backpressure semantics.

What creates distrust:

- MCP is powerful, but still tool-heavy instead of workflow-guided,
- API stack health depends on environment compatibility rather than a product-owned runtime sandbox,
- server behavior under pressure is still implicit.

### B. Database and data-management expert

What works:

- SQLite-first workflow layer,
- parameterized queries,
- table writes and query materialization,
- tabular and profile-table conversion boundary,
- consistent artifact output for query-driven workflows.

What feels weak:

- analytical-table construction still depends on user-written SQL,
- no library-owned multi-table denormalization layer,
- no first-class feature-table abstraction,
- remote SQL remains beta and lightly validated compared with SQLite.

What creates friction:

- the product can ingest clean relational outputs, but it does not yet reduce enough SQL glue work upstream of the spectral engine.

### C. Spectral and mathematically oriented user

What works:

- bounded-domain basis and packet projection,
- modal propagation,
- profile compression,
- inverse reconstruction,
- diagnostics and truncation summaries,
- domain-aware comparison metrics.

What feels weak:

- the core is strong, but still under-explained by surrounding surfaces,
- richer state and basis configuration depth is narrower than the product’s long-term ambition,
- the packaging and ML/server layers can visually overshadow the actual engine.

What creates misunderstanding:

- outsiders may still see the spectral core as one feature among many instead of the spine that everything else serves.

### D. Capstone and end-to-end workflow builder

What works:

- file-backed profile workflows,
- SQLite-backed workflows,
- SQL-backed spectral analysis,
- backend-aware modal-surrogate training and evaluation,
- stable artifact bundles across interfaces.

What feels weak:

- raw messy JSON and log-like inputs still fall out of the library,
- normalization, reconciliation, and feature-table preparation still require external glue code,
- there is no one library-owned path from messy records to analytical tables to feature tables to spectral or model workflows.

What creates friction:

- the product is strongest once the user already has a clean table.

### E. MCP-driven AI client

What works:

- clean tool names and descriptions,
- environment and backend inspection,
- job-shaped simulation, SQL, spectral, and model tools,
- artifact listing after compute jobs.

What feels weak:

- no prompts or resources,
- limited workflow scaffolding,
- too many tools at one flat level,
- no higher-order job guidance for choosing the next step.

What creates prompt burden:

- the AI still has to infer the product’s intended tool chains from tool names rather than from explicit workflow grouping.

### F. Python library user

What works:

- stable core imports,
- rich workflow return objects,
- package entrypoint and module entrypoint,
- wheel and editable installs now validated,
- explicit workflow layer that most surfaces already share.

What feels weak:

- the top-level public surface is very broad,
- package import pulls in many subsystems eagerly,
- boundaries between core, storage bridge, interfaces, and optional surfaces are not obvious enough from import ergonomics alone.

What creates distrust:

- the library can feel more like a “bring everything in” façade than a sharply layered Python package.

### G. Server and API user

What works:

- API parity with much of the workflow layer,
- explicit compatibility inspection for the FastAPI stack,
- artifact retrieval,
- shared error mapping for common request failures.

What feels weak:

- synchronous-only request model,
- no explicit background task or job status semantics,
- no concurrency or overload policy,
- API remains environment-sensitive even when installed,
- no structured runtime logs or request-scoped execution diagnostics.

What creates distrust:

- the API is honest, but still feels beta because the runtime model is thin compared with the rest of the product.

## Biggest Fragmentation Risks

1. The core engine is real, but the repo still presents it through too many equal-looking surfaces.
2. SQL and model workflows are useful, but the upstream normalization and analytical-table gap makes them feel detached from the core.
3. MCP discovery is better than before, but still flatter and less guided than the compute surface deserves.
4. The API shares workflows, but not yet a strong runtime model for load, task tracking, or observability.
5. Backend support is coherent in policy but unequal in maturity; if the public story blurs that, trust drops quickly.
6. The package is installable, but the public metadata and layered import ergonomics still look less mature than the engine itself.

## Biggest Compatibility Risks

1. PyTorch is the primary path and should stay that way in product messaging.
2. JAX is beta and still platform-conditional.
3. TensorFlow is a compatibility path and should remain explicitly experimental.
4. The API surface is viable in clean environments, but mixed environments can still produce incompatible FastAPI and Starlette combinations.
5. The current runtime model is intentionally synchronous and bounded. That is acceptable only if overload and concurrency limits are stated clearly and not implied away.

## Biggest Misunderstanding Risks

1. Users can still mistake the product for a generic data or ML toolkit because the extension surfaces are easier to notice than the mathematical spine.
2. Spectral-domain users can still think the SQL and ML layers drifted too far, because the bridge back into the core is not explicit enough in end-to-end workflows.
3. Data and automation users can still think the product is too spectral to be practical, because the raw-to-structured-to-spectral bridge is still incomplete.
4. AI clients can still misuse MCP because the server exposes many tools but little guided composition.

## Convergence Plan

### Phase 2

- tighten package layering so the core engine, storage bridge, workflow layer, and interfaces are more obvious,
- reconnect SQL and model workflows to the core with explicit workflow bridges,
- simplify MCP into clearer job families without adding interface-only logic,
- strengthen the spectral core where its current abstractions are too thin relative to the public ambition.

### Phase 3

- harden backend compatibility and runtime routing further,
- add structured logging and clearer runtime diagnostics,
- define honest synchronous and overload behavior for API and MCP surfaces,
- run the validation matrix across installs, interfaces, backends, and failure paths.

### Phase 4

- update README, help text, docs, and examples to describe the validated product shape,
- keep stable, beta, and experimental surfaces sharply separated,
- publish an honest final statement of what the product is and what remains limited.

## Validation Matrix Plan

The next pass should validate the product as one engine through all of these views:

- core Python imports and workflow chaining,
- packaged CLI help, version, and golden-path commands,
- isolated standard install and isolated editable install,
- wheel and sdist builds,
- PyTorch, JAX, and TensorFlow capability reporting,
- clean API-extra install behavior versus mixed-environment API failure reporting,
- MCP discovery and one golden-path tool chain,
- file-backed workflows,
- SQLite-backed workflows,
- SQL-backed spectral workflows,
- backend-aware modal-surrogate workflows,
- bad-input and partial-data paths,
- cross-platform notes against validated truth.

## Scope Freeze For This Pass

Do not add any new product family in the remaining convergence work.

Specifically defer:

- raw-record normalization,
- feature-pipeline abstractions,
- generic job queues,
- broad server-side concurrency frameworks,
- any new backend family,
- any generic dashboard or orchestration layer.

The correct move is to converge the current product into one strong public-ready engine, not to widen its idea surface again.
