# Adoption And Replaceability Audit

## Why The Product Was Still Replaceable

- The engine was strong, but users still had to choose the workflow graph themselves.
- File-backed and SQL-backed report paths were coherent, but the product still made the user discover them instead of recommending them.
- MCP exposed real power, but it still looked more like a tool shelf than an operator brain.
- Opinionated defaults existed in practice, but they were scattered across Python, CLI, MCP, API, examples, and docs instead of being declared once.
- The product story still leaned on explanation more than on shared workflow guidance embedded in code.

## Defensible Product Loop

The product is hardest to replace when it does this well:

1. Take one file-backed or SQL-backed profile-table input.
2. Validate and materialize it through one explicit bounded-domain contract.
3. Run one inspect-analyze-compress report workflow with stable defaults.
4. Emit one artifact-backed result chain that the user or machine client can inspect and continue from.

That is better than a stitched stack because the validation boundary, modal basis logic, compression semantics, artifact schema, and runtime/status model all belong to one engine.

## Canonical Killer Workflows

### 1. File Profile Report Loop

- Target user: Python or CLI user with one profile-table file and a need for a fast, trustworthy spectral answer.
- Inputs: CSV, TSV, JSON, or optionally XLSX profile table.
- Steps: validate table -> inspect modal structure -> compress -> write artifacts -> inspect artifacts.
- Outputs: profile report overview, table summary, spectral analysis bundle, compression bundle, root artifact manifest.
- Why better here: avoids custom glue across loaders, projection code, reconstruction logic, and reporting.

### 2. SQL Profile Report Loop

- Target user: user with profile-table-shaped data already in SQLite or SQLAlchemy-backed storage.
- Inputs: database reference plus one profile-table-shaped SQL query.
- Steps: explicit query materialization -> inspect/analyze/compress report -> provenance-aware artifacts.
- Outputs: same report bundle as the file workflow plus SQL provenance in artifact metadata.
- Why better here: removes ad hoc CSV export and makes the relational-to-spectral boundary explicit and reproducible.

### 3. MCP Operator Loop

- Target user: machine-side tool client that needs structured workflows instead of prompt glue.
- Inputs: file-backed or SQL-backed profile-table source plus an MCP client over stdio.
- Steps: inspect product -> inspect environment -> run one report tool -> inspect artifacts -> inspect service status.
- Outputs: structured workflow guidance, bounded compute calls, artifact inspection, canonical runtime/status records.
- Why better here: the product now recommends the high-value path instead of forcing the client to infer a tool chain.

## Biggest Sources Of Decision Burden

- Choosing between file, SQL, and MCP paths without a shared recommendation layer.
- Remembering the default mode budgets, artifact locations, and runtime expectations by reading docs instead of code.
- Knowing when to use the report workflow instead of lower-level analyze/compress commands.
- Knowing what MCP tool order is intended for a first successful run.

## Biggest Sources Of Glue Burden

- Translating “I have a file” or “I have a SQL query” into the right product entrypoint.
- Building the recommended next steps manually after a successful run.
- Teaching an AI client the intended MCP workflow instead of asking the product for it.

## Adoption-Focused Plan

1. Centralize the killer workflows and defaults in the shared product layer.
2. Expose a surface-aware workflow guide through Python, CLI, MCP, and API.
3. Reuse the same report defaults across the workflow layer and wrappers.
4. Update docs and examples so the product tells one adoption story: report-first, artifact-backed, and continuation-friendly.
