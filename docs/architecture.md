# Architecture

## Internal Product Statement

Spectral Packet Engine is a cross-platform Gaussian and spectral compute engine for bounded-domain simulation, modal decomposition, inverse reconstruction, and backend-aware surrogate modeling. Raw data ingestion, SQL workflows, feature engineering, and MCP or API interfaces exist only to move real user data into that same bounded-domain engine, persist its outputs, and let machine-side tooling run the same library workflows without external glue code.

## Convergence Rule

The repository must read as one engine with multiple interfaces, not as multiple mini-products:

- the spectral core is the center,
- the workflow layer is the shared conductor,
- SQL, tabular, model, MCP, CLI, and API layers are product extensions around that conductor,
- if a new abstraction makes the product harder to explain as one engine, it is architectural drift.

## Product Spine

Core mathematical engine:

- [`../src/spectral_packet_engine/domain.py`](../src/spectral_packet_engine/domain.py)
- [`../src/spectral_packet_engine/basis.py`](../src/spectral_packet_engine/basis.py)
- [`../src/spectral_packet_engine/state.py`](../src/spectral_packet_engine/state.py)
- [`../src/spectral_packet_engine/projector.py`](../src/spectral_packet_engine/projector.py)
- [`../src/spectral_packet_engine/dynamics.py`](../src/spectral_packet_engine/dynamics.py)
- [`../src/spectral_packet_engine/observables.py`](../src/spectral_packet_engine/observables.py)
- [`../src/spectral_packet_engine/simulation.py`](../src/spectral_packet_engine/simulation.py)
- [`../src/spectral_packet_engine/inference.py`](../src/spectral_packet_engine/inference.py)

Workflow and diagnostics layer:

- [`../src/spectral_packet_engine/workflows.py`](../src/spectral_packet_engine/workflows.py)
- [`../src/spectral_packet_engine/diagnostics.py`](../src/spectral_packet_engine/diagnostics.py)
- [`../src/spectral_packet_engine/artifacts.py`](../src/spectral_packet_engine/artifacts.py)

Data and storage bridge:

- [`../src/spectral_packet_engine/table_io.py`](../src/spectral_packet_engine/table_io.py)
- [`../src/spectral_packet_engine/tabular.py`](../src/spectral_packet_engine/tabular.py)
- [`../src/spectral_packet_engine/database.py`](../src/spectral_packet_engine/database.py)

Planned raw-record and feature boundary:

- raw records, nested JSON, heterogeneous logs, and document-extracted tables should be normalized into a shared tabular layer before any profile-table or ML workflow begins,
- relational joins and analytical table construction should terminate in the same tabular boundary,
- feature tables should remain product-aligned artifacts that feed spectral comparison, inverse fitting, or modal surrogate workflows instead of becoming a generic data platform.

Backend-aware ML layer:

- [`../src/spectral_packet_engine/ml.py`](../src/spectral_packet_engine/ml.py)
- [`../src/spectral_packet_engine/tf_surrogate.py`](../src/spectral_packet_engine/tf_surrogate.py)

Service compatibility layer:

- [`../src/spectral_packet_engine/service_runtime.py`](../src/spectral_packet_engine/service_runtime.py)

Interface layer:

- [`../src/spectral_packet_engine/cli.py`](../src/spectral_packet_engine/cli.py)
- [`../src/spectral_packet_engine/mcp.py`](../src/spectral_packet_engine/mcp.py)
- [`../src/spectral_packet_engine/api.py`](../src/spectral_packet_engine/api.py)

Optional secondary layers:

- [`../src/spectral_packet_engine/datasets.py`](../src/spectral_packet_engine/datasets.py)
- [`../src/spectral_packet_engine/plotting.py`](../src/spectral_packet_engine/plotting.py)

## Architectural Rule

All user-facing surfaces should call shared workflow functions instead of reimplementing the math, SQL handling, ML orchestration, or artifact logic.

That keeps the package coherent:

- scientific logic stays in the library,
- tabular and SQL conversion stays reusable,
- backend-aware surrogate logic stays in one ML layer,
- CLI, MCP, and API stay thin,
- tests can target one compute spine instead of fragmented wrappers.

## Harmonic Extension Rule

The Gaussian and bounded-domain spectral engine is the product spine.

Everything else must strengthen that spine:

- raw-record ingestion should reduce user glue code before spectral or surrogate workflows,
- SQL support should make relational data usable by the same library workflows,
- feature engineering should produce reproducible analytical tables that still terminate in spectral analysis, inverse reconstruction, comparison, or surrogate training,
- model engineering should stay attached to modal or profile-derived tasks instead of expanding into a generic model zoo.

If a new capability cannot be explained as “this helps users get their data into, through, or back out of the spectral engine,” it does not belong in the primary product surface.

## Current Stability Model

Stable:

- core bounded-domain modal engine,
- file-backed and SQLite-backed profile compression workflows,
- inverse reconstruction workflows,
- shared tabular bridge,
- local SQLite workflow layer,
- packaged CLI,
- Python package API,
- PyTorch-backed modal surrogate workflow surface.

Beta:

- MCP server,
- HTTP API,
- remote SQL backends through SQLAlchemy,
- JAX backend-aware modal surrogate path.

Experimental:

- TensorFlow compatibility surrogate,
- published transport dataset wrapper.

## Current Production Shape

### 1. Scientific compute core

The bounded-domain basis, packet/state handling, projection, propagation, observables, and inverse fitting logic remain the product center. Everything else exists to move real data into and out of those workflows without forking the compute logic.

### 2. Shared data boundary

`tabular.py`, `table_io.py`, and `database.py` define one boundary between files or SQL results and library-native profile-table workflows. File-backed and SQL-backed inputs should converge here before entering spectral or ML workflows.

The missing deepening target is upstream of this boundary, not outside it: users need a raw-record normalization layer that can reconcile nested or messy inputs into `TabularDataset` without bypassing the library.

Target layering for the next implementation pass:

- raw records and document-extracted tables,
- normalization and reconciliation into `TabularDataset`,
- relational and analytical-table construction,
- feature-pipeline execution,
- explicit conversion into profile or modal workflows,
- spectral analysis, inverse reconstruction, comparison, or surrogate training.

### 3. Shared backend-aware surrogate layer

`ml.py` owns backend inspection, backend resolution, preprocessing, scaling, training, evaluation, and export behavior for modal-regression surrogates. TensorFlow remains supported through a compatibility adapter, but the public ML story is no longer TensorFlow-only.

The next phase should deepen this layer through reproducible feature-table workflows and stronger model persistence, not by adding disconnected backend-specific products.

### 4. Backend routing and service compatibility

`ml.py` is the backend conductor for modal-surrogate work:

- inspect the runtime state of PyTorch, JAX, and TensorFlow,
- distinguish runtime availability from the project's supported install surface,
- expose one route decision for Python, CLI, MCP, and API callers,
- keep backend policy in one place instead of leaking fallback rules into wrappers.

`service_runtime.py` is the compatibility surface for optional serving layers:

- inspect FastAPI, Starlette, Pydantic, and Uvicorn as one API stack,
- report compatibility explicitly instead of collapsing to a bare boolean,
- feed the same machine-state report into Python, CLI, MCP, and API diagnostics.

`workflows.inspect_environment()` is the shared machine report for the product. New interface work should extend that shared report instead of inventing interface-local capability checks.

### 5. Thin interfaces over shared workflows

CLI, MCP, and API expose the same workflow functions with surface-specific validation and serialization only. New product capabilities should land in the workflow and library layers before they appear in interface code.

## Recent Productionization Work

Completed in the current consolidation pass:

- added the generic tabular bridge,
- added the SQLite-first database subsystem and SQL-backed workflow helpers,
- added backend-aware ML inspection, training, and evaluation over PyTorch, JAX, and TensorFlow compatibility paths,
- aligned CLI, MCP, and API surfaces around the same SQL and ML workflows,
- expanded artifact bundles for SQL-backed and backend-aware ML runs,
- expanded tests across the shared library and interface layers.

## Next Deepening Targets

Appropriate next product work stays inside the existing spine:

- a raw-record normalization layer for nested JSON, heterogeneous logs, and partially structured records that materializes cleanly into `TabularDataset`,
- relational analytical-table workflows over multiple source tables without moving users into custom glue scripts,
- reproducible feature-pipeline definitions that can feed both profile conversion and surrogate workflows,
- model reload and checkpoint-resume flows for backend-aware surrogates,
- deeper inverse fitting beyond single-Gaussian reconstruction,
- richer plotting and benchmark bundles over existing workflows,
- broader remote-database validation once SQLAlchemy-backed backends are exercised in compatible environments.

## Implementation Shape

The next product-aligned deepening should look like this:

- a raw-record normalization module that owns flattening, reconciliation, coercion, and profiling,
- a relational workflow layer that owns analytical-table construction on top of the current database subsystem,
- a feature-pipeline layer that transforms normalized or relational tables into reproducible feature tables,
- workflow functions that route those feature tables into spectral, comparison, inverse, or backend-aware surrogate paths,
- thin CLI, MCP, and API entrypoints over those same shared workflows.

The product should not add interface-only data transformations or backend-specific side pipelines.
