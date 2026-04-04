# Product Spine Audit

## Phase 1 Scope

This audit is not a release-facing marketing document.

Its purpose is to identify where the repository already delivers on its Gaussian and spectral product spine, where users still fall out of the library into custom glue code, and what architectural guardrails must hold before deeper data, SQL, feature-engineering, and model-engineering work lands.

## Where The Spine Is Already Strong

- The bounded-domain spectral core is real and product-grade: domain construction, modal basis generation, packet-state projection, exact propagation, observables, compression, and inverse fitting all exist as reusable library code.
- `workflows.py` already provides a shared job layer for forward simulation, projection, compression, comparison, inverse fitting, SQL-backed spectral analysis, and modal-surrogate training or evaluation.
- CLI, MCP, and API are thin enough that the product mostly behaves like one engine instead of three wrappers.
- SQLite-first workflows and the generic tabular bridge already make clean file-backed and SQL-backed profile workflows usable without notebooks.
- Backend-aware modal surrogates are real implementations, not just “backend selection” theater.

## Exact Drift Risks

- The product story can drift if raw-data, SQL, feature, and model work is added as generic tabular tooling instead of as a bridge into spectral or modal workflows.
- `TabularDataset` is currently a clean intermediate format, but it is still too low-level to absorb messy nested records, schema drift, or heterogeneous log-style data. If wrappers start solving that ad hoc, the repo will split into inconsistent mini-pipelines.
- `database.py` is already more than a toy query wrapper, but it still centers on query, CRUD, and table materialization. If feature-table construction gets bolted directly into CLI or MCP commands instead of shared library code, the data layer will become interface-driven instead of product-driven.
- The ML layer is coherent for modal surrogates, but it currently assumes users already have profile tables or clean SQL queries. If feature-table model work is added without a shared preprocessing and feature boundary, the repo will fork into incompatible “profile ML” and “tabular ML” stories.
- MCP is useful, but it still mostly exposes clean-table workflows. If raw ingestion and feature work arrive as isolated tools, MCP will become a tool dump instead of a coherent machine-side compute interface.

## Unmet User Needs

### Raw data and ingestion

- Users with nested JSON, mixed-schema records, or log-like structured data still need external Python to flatten, reconcile, and type-coerce records before the library can help.
- There is no shared schema-inference or schema-reconciliation layer for merging multiple raw files with partially overlapping fields.
- Missing-value handling, deduplication, timestamp parsing, and explicit normalization policies are still too weak for real ingestion workloads.
- There is no honest PDF or DOCX extraction boundary yet; if those formats are added carelessly, they will invite overclaiming and drift.

### SQL and relational workflows

- The current SQL subsystem is strongest once a clean table already exists.
- Users with multiple relational tables still need custom code to build analytical tables, resolve joins, and persist cleaned feature datasets.
- The library does not yet own a high-level “ingest files -> infer schema -> create tables -> build analytical table -> run spectral or ML workflow” path.
- Data-quality checks around relational joins and analytical-table construction are still too shallow.

### Feature engineering

- There is no first-class feature pipeline layer.
- Users cannot yet define reusable transformations for missing-value policies, numerical transforms, categorical encoding, time-window aggregation, or uncertainty-aware smoothing inside the library.
- Feature artifacts are not yet modeled as a first-class output that can be reused across Python, CLI, MCP, and API surfaces.

### Model engineering

- The current model layer is strong for modal-regression surrogates, but weaker for users who need to start from feature tables rather than already-clean profile tables.
- Backend-aware training and evaluation exist, but model persistence, reload, experiment comparison, and reproducible feature-pipeline attachment are still narrower than they should be.
- If broader model support is added naively, it will drift toward a generic model zoo. The safe extension is feature-table-to-modal or feature-table-to-profile-adjacent workflows, not arbitrary ML breadth.

### MCP

- MCP can already run core spectral, SQL-backed spectral, and surrogate jobs, but it cannot yet own the raw-data cleanup path users actually need before those jobs.
- Schema inference, normalization, analytical-table construction, feature-pipeline execution, and feature-table model workflows are not yet first-class MCP jobs.
- Without shared library abstractions underneath, expanding MCP now would increase drift instead of solving user pain.

## Structural Weaknesses To Fix Before Deeper Feature Growth

- `tabular.py` is intentionally small, but it has no reusable notion of record normalization, schema reconciliation, or configurable coercion policy.
- `table_io.py` cleanly converts already-structured profile tables, but that conversion is strict enough that users still need external cleanup before it can help.
- `database.py` owns durable storage and safe query execution, but not yet higher-level analytical-table construction or data-quality checks.
- `workflows.py` has grown into the right orchestration layer, but it needs a new shared normalization and feature layer beneath it rather than more direct file and SQL helpers.
- Tests mostly cover clean-table paths. Coverage is still thin for heterogeneous raw records, join mismatches, schema reconciliation, and feature-pipeline reproducibility.

## Current Code-Level Blockers

These are the exact places where the current implementation is still forcing user-side glue code:

- `TabularDataset.from_rows()` currently requires every row to contain the same keys. That is correct for a finalized table, but too strict for raw-record ingestion where partial or mixed schemas are the norm.
- `load_tabular_dataset_json()` currently expects a non-empty list of flat mappings or an object with a `rows` list. That means nested JSON and log-like record collections still need external flattening before the library can help.
- `TabularDataset.join()` is useful for small clean joins, but there is no shared relational workflow for multi-table analytical-table construction, grouped aggregation, or quality validation.
- `database.py` can create, inspect, query, and materialize tables, but it does not yet own a reusable “build analytical table from multiple source tables” layer.
- `workflows.py` currently orchestrates clean tables very well, but it has no first-class normalization pipeline or feature-pipeline abstraction beneath the spectral and surrogate workflows.
- `ml.py` is strong for modal surrogates over profile tables, but there is no feature-table-native model workflow for users who begin with structured records or relational analytical tables.
- `mcp.py` exposes the existing engine cleanly, but it mostly assumes the caller already has a usable table or query result. It does not yet own raw-record normalization, feature construction, or analytical-table jobs.

## Where Users Still Need External Python

These are the concrete cases where the product is still making users leave the library:

- flattening nested JSON into rows,
- reconciling mixed-schema records across many files,
- coercing timestamps and numeric-like strings consistently,
- deciding how to handle missing or ambiguous fields before loading into `TabularDataset`,
- building denormalized analytical tables from multiple SQL tables,
- materializing reusable feature tables for downstream spectral or model workflows,
- attaching reproducible preprocessing to model training or evaluation.

## Internal Product Statement

Spectral Packet Engine is a Gaussian and bounded-domain spectral compute engine for users who need to turn raw observational, simulated, or relational data into validated profile tables, analytical feature tables, modal decompositions, inverse reconstructions, and backend-aware surrogate workflows without rewriting the scientific core. Data ingestion, SQL, feature engineering, and model engineering are supportive layers around that same engine, not independent product identities.

## Architecture Guardrails

- Raw files, nested records, or extracted document tables must flow into a shared normalization layer before they become `TabularDataset`.
- `TabularDataset` remains the common intermediate boundary for relational, file-backed, and normalized raw data.
- `ProfileTable` remains the explicit spectral boundary. Converting generic tables into profile tables must stay deliberate and validated.
- Feature pipelines should produce explicit analytical tables or feature tables that remain traceable to source data and reusable across Python, CLI, MCP, and API.
- Model-engineering work must remain attached to modal, profile-derived, or feature-table workflows that support the spectral engine.
- CLI, MCP, and API should only expose workflows after the shared library layer owns them.

## Harmonized Architecture Plan

### 1. Raw-record normalization layer

Add a shared library layer responsible for:

- nested-object flattening,
- schema inference and reconciliation across many raw files,
- explicit coercion policies,
- timestamp normalization,
- missing-value policies,
- duplicate detection,
- normalization artifacts and validation reports.

This layer should terminate in `TabularDataset`, not bypass it.

### 2. Relational and analytical-table layer

Extend the current SQL subsystem with shared analytical-table workflows for:

- joining source tables,
- selecting and renaming columns,
- grouped aggregation,
- table-level quality checks,
- feature-table materialization,
- writing normalized or engineered outputs back to storage when appropriate.

This layer should terminate in either `TabularDataset` or an explicit feature-table abstraction, not in interface-specific JSON payloads.

### 3. Feature-pipeline layer

Add a reusable feature-engineering abstraction over normalized and relational tables for:

- derived numeric features,
- time-window aggregation,
- categorical encoding when justified,
- uncertainty-aware smoothing or denoising,
- explicit export of feature tables plus diagnostics and artifacts.

This layer should feed both spectral workflows and model workflows without forking the product story.

### 4. Model-engineering extension

Extend the current backend-aware surrogate layer so the library can own:

- feature-table-backed training and evaluation,
- reproducible preprocessing attachment,
- model persistence and comparison,
- file-backed and SQL-backed model workflows over the same shared abstractions.

The target is not generic ML breadth. The target is stronger model workflows that still terminate in modal, profile-derived, or bounded-domain prediction tasks.

## Concrete Next Implementation Targets

### 1. Upgrade the tabular boundary

- add mixed-schema record ingestion instead of requiring identical row keys up front,
- add schema-reconciliation and coercion-policy objects,
- add normalization reports and deterministic raw-to-tabular artifacts.

### 2. Add analytical-table workflows

- add multi-table join and denormalization helpers in shared library code,
- add grouped aggregation and quality checks,
- add SQL-backed analytical-table materialization that Python, CLI, MCP, and API can all share.

### 3. Add first-class feature pipelines

- define reusable feature-pipeline configuration and execution,
- materialize feature tables as reusable outputs,
- connect them to both spectral and model workflows.

### 4. Deepen model workflows without drift

- add feature-table-backed training and evaluation,
- preserve backend-aware modal surrogates as the main model story,
- add reproducible preprocessing attachment, persistence, and comparison around that same workflow surface.

## Non-Goals For The Next Phase

- No generic dashboarding or BI layer.
- No generic database administration feature set.
- No unconstrained document-understanding claims.
- No generic AutoML or model-zoo expansion.
- No MCP tool expansion without shared library ownership underneath.
