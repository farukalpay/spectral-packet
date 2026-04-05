# Product Validation Audit

## Scope

This audit validates the repository as a product used by real humans and AI clients, not only as a codebase with passing tests.

The goal is to identify:

- where users still misunderstand the product,
- where AI clients get access but not good workflow design,
- where library semantics are not chainable enough,
- where raw-data, SQL, feature, and model workflows still force external glue code,
- where the product risks drifting away from the Gaussian and spectral core.

## Internal Product Statement

Spectral Packet Engine is a Gaussian and bounded-domain spectral compute library. Data ingestion, SQL, feature engineering, model workflows, MCP, and API paths exist to feed the same spectral, inverse-reconstruction, comparison, and modal-surrogate engine, not to create separate generic data or AI products.

## Human-User Misunderstanding Risks

- The top-level CLI is real and broad, but the command list is now long enough that it can feel like a tool collection unless the user already understands the product spine.
- JSON support is easy to misunderstand. The product currently supports clean tabular JSON and profile-table JSON, but not general mixed-schema nested JSON ingestion yet.
- The README and architecture docs correctly describe the core, but the extension layers are still stronger for clean tables than for messy raw-record workflows.
- The SQL path looks deeper than it is for analytical-table construction. Users can query, materialize, and analyze tables, but they still need manual SQL or external code for many multi-table feature workflows.
- The model story is strong for modal surrogates, but users who start from feature tables rather than profile tables still face a gap.

## AI-Client Misunderstanding Risks

- MCP previously exposed internal function-shaped tool names such as `inspect_environment_tool`; that made discovery feel wrapper-driven instead of product-shaped. This phase fixes the tool naming and descriptions.
- MCP currently exposes tools only. The official MCP model expects discoverable tools, and the broader protocol also supports resources and prompts for richer workflow guidance. The current server still has zero prompts and zero resources.
- MCP can execute core spectral, SQL-backed spectral, and surrogate workflows, but it still cannot own the raw-record cleanup or feature-table construction path users actually need before those workflows.
- Tool coverage is broad, but workflow scaffolding is still thinner than it should be for AI clients. The server exposes capability, but not yet enough guided entry points for messy-data and analytical-table jobs.

## Persona Validation

### 1. Direct Python spectral user

Validated path:

- load a bundled profile table,
- compress it into modal coefficients,
- analyze its spectral structure.

Result:

- works cleanly today,
- return objects are reusable and library-shaped,
- this is the strongest current product path.

### 2. Feature-engineering user with messy JSON

Validated path:

- write a nested JSON file with partially overlapping keys,
- call `load_tabular_dataset()`.

Observed result:

- fails with `ValueError: all rows must contain the same columns`.

Interpretation:

- the current tabular boundary is still a finalized-table abstraction,
- the raw-record normalization layer is still missing,
- this is a concrete reason users still need external Python.

### 3. SQL and data-pipeline user with scattered relational tables

Validated path:

- write separate `profiles` and `runs` tables into SQLite,
- join them with SQL and materialize the result into a tabular dataset.

Observed result:

- works when the user already writes the join query,
- the library can consume the result once the analytical table exists.

Interpretation:

- the SQL subsystem is useful,
- but the product still lacks a shared analytical-table workflow layer,
- users still need manual SQL for common relational feature work.

### 4. MCP-driven AI user

Validated path:

- inspect the server with `list_tools()`, `list_prompts()`, and `list_resources()`.

Observed result:

- 30 tools are available,
- tool names and descriptions are now product-shaped in this phase,
- prompt count is `0`,
- resource count is `0`,
- resource template count is `0`.

Interpretation:

- the tool layer is useful,
- but AI-user discoverability and workflow scaffolding are still thinner than the MCP ecosystem makes possible.

### 5. API and server user

Validated path:

- construct the FastAPI app in the current environment.

Observed result:

- the API fails fast with a clear runtime error when the local FastAPI and Starlette stack is incompatible.

Interpretation:

- failure behavior is honest and product-safe,
- but the practical API path still lags the library and CLI because the stack can be unavailable even when installed,
- this remains a beta surface in real use.

## Strongest Signs Of Drift Or Fragmentation

- JSON is implemented, but the product still cannot honestly claim strong messy-record ingestion.
- SQL is implemented, but analytical-table and feature-table workflows are still weaker than the query surface suggests.
- MCP is implemented, but the workflow design is still shallower than the compute surface.
- Model workflows are implemented, but the feature-table entry path is still missing.
- The repo remains coherent at the spectral core, but the upstream data boundary is still the main place where drift could occur if future work lands ad hoc.

## Most Dangerous Untested Assumptions

- assuming raw JSON means normalized row-shaped JSON,
- assuming identical row keys are acceptable at ingestion time,
- assuming users are comfortable writing custom SQL joins for analytical tables,
- assuming tool exposure alone is enough for a good MCP experience,
- assuming backend-aware modal surrogates cover broader model-engineering needs without a feature-table pipeline,
- assuming Python’s permissive JSON handling is representative of stricter downstream consumers when `NaN` appears in serialized outputs.

## Validation-Driven Hardening Priorities

### 1. Raw-record normalization

Build the missing normalization layer so messy JSON and heterogeneous records stop failing at the `TabularDataset` boundary.

### 2. Analytical-table workflows

Add shared relational workflows for joins, denormalization, grouped aggregation, and quality checks so users stop writing manual bridge SQL for common product use cases.

### 3. Feature-pipeline layer

Add a reusable feature-engineering layer that feeds both spectral and model workflows instead of leaving preprocessing outside the product.

### 4. AI-client workflow design

Keep improving MCP as a product surface:

- product-shaped tool names and descriptions are now fixed,
- next steps should add better guidance through prompts, resources, or higher-level job entry points where they genuinely improve discovery.

### 5. Library chainability

Keep the library first:

- normalized data should flow into reusable tables,
- reusable tables should flow into feature pipelines,
- feature or profile tables should flow into spectral, inverse, comparison, or surrogate workflows,
- CLI, MCP, and API should remain thin over those same shared abstractions.
