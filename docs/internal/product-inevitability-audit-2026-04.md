# Product Inevitability Audit

Date: 2026-04-04

This note defines the Phase 1 adoption pass: why the product is still replaceable, which workflows should become the obvious reasons to adopt it, and what the implementation order should be.

## Why The Product Is Still Replaceable

The repository is already coherent and technically serious, but it can still be respected more than adopted.

The main replaceability risks are:

1. The product language has still leaned too hard on transport boundaries.
   File versus SQL versus MCP tells the user how data arrives, not why the product is the best place to run the job.

2. The workflow guidance has been too input-centric.
   The old guidance mostly asked the user to choose between file-backed and SQL-backed paths instead of choosing between report, inverse reconstruction, and feature-model intents.

3. Some high-value workflows existed but were not visible in the shared product map.
   A concrete example was SQL inverse fitting: implemented in code, but not fully represented in the shared workflow catalog.

4. MCP has exposed tools more clearly than it has exposed operational intent.
   That makes an AI client more likely to compose prompt-side tool chains than to trust the product’s preferred loop.

5. The strongest originality has been present but still partially implicit.
   The actual differentiator is not “spectral math plus some interfaces.” It is:
   bounded-domain spectral diagnostics + compression + inverse reconstruction + file/SQL materialization + artifact continuity + machine-side operation in one product.

6. The feature-model story has improved, but it still risks feeling optional.
   Without a stronger report-first loop, predictive workflows can still look like one more bridge the user assembled on top of generic libraries.

## Canonical Killer Workflows

These are the three workflows that should increasingly define the product.

### 1. Spectral Evidence Loop

Target user:
- scientific or engineering user with one observed or simulated 1D density/profile table

Input types:
- file-backed profile table
- profile-table-shaped SQL query

Steps:
1. validate the table shape and numeric quality
2. inspect moments and spectral budgets
3. compress into modal coefficients and reconstruction
4. write one inspectable artifact bundle

Outputs:
- profile report
- spectral analysis bundle
- compression bundle
- artifact manifest

Why better here than a stitched stack:
- one engine owns validation, modal projection, diagnostics, compression, and artifacts
- the user does not need custom glue between loaders, spectral code, diagnostics, and reporting

### 2. Inverse Reconstruction Loop

Target user:
- user who needs interpretable Gaussian packet parameters from observed densities

Input types:
- file-backed profile table
- profile-table-shaped SQL query

Steps:
1. run the spectral evidence loop first
2. fit Gaussian packet parameters through the bounded inverse workflow
3. inspect reconstruction quality and optimization diagnostics
4. persist inverse artifacts with the same provenance chain

Outputs:
- spectral report bundle
- inverse-fit summary
- inverse reconstruction artifacts

Why better here than a stitched stack:
- the same engine that explains the data spectrally performs the inverse reconstruction
- this is stronger than generic optimization over CSVs because the inverse path remains tied to the bounded-domain model

### 3. Spectral Feature Model Loop

Target user:
- data or ML user who wants predictive modeling over profile-table-shaped results

Input types:
- file-backed profile table
- profile-table-shaped SQL query
- one supervised target column joined explicitly after feature export

Steps:
1. run the spectral evidence loop first
2. export the explicit spectral feature table and schema
3. join or append the target column explicitly
4. train or tune the model through the shared workflow layer
5. inspect model artifacts and provenance

Outputs:
- spectral report bundle
- feature table + schema
- training or tuning artifacts
- predictions and model export

Why better here than a stitched stack:
- spectral compression becomes a stable, inspectable feature contract rather than an ad hoc notebook step
- provenance, schema, and artifact continuity remain part of the product instead of being invented per project

## Biggest Sources Of Decision Burden

1. Users still need help selecting the right outcome, not just the right transport.
2. New users still need clearer product-driven guidance about when to stay report-first.
3. AI clients still need stronger next-step recommendations after a report completes.
4. Users still have to infer which follow-up workflows are evidence-driven versus optional.

## Biggest Sources Of Glue Burden

1. Supervised targets still need to be joined explicitly after feature export.
2. Continuation between report -> inverse -> model still relies on the user following artifact guidance instead of a single chained workflow.
3. Some adoption-critical workflows had previously been underrepresented in the shared catalog.

## Adoption / Intelligence / Defaults Plan

### Phase 1

- make the killer workflows goal-based rather than interface-based
- teach `guide_workflow(...)` to route by user intent:
  - `report`
  - `inverse-fit`
  - `feature-model`
- keep report-first as the default routing policy
- expose the same intent-aware guidance through CLI, MCP, and API
- fill product-catalog gaps that weaken the one-engine story

### Phase 2

- add stronger chained no-glue workflows over the same shared workflow layer
- reduce explicit handoff steps around feature-model work where possible
- push MCP further toward continuation and next-step intelligence

### Phase 3

- strengthen summary outputs, artifacts, and adoption-facing reports
- make the product’s originality visible in actual outputs, not just in descriptions
- continue hardening runtime and optional-backend behavior

### Phase 4

- rewrite README, examples, quickstarts, and interface docs around the killer workflows
- make the shortest path to value obvious
- sharpen the product language so users describe it as a workflow engine, not a toolkit bundle

## Phase 1 Implementation Completed In This Pass

The codebase now reflects the start of this adoption pass:

- the shared product layer now models three goal-centric killer workflows:
  - spectral evidence
  - inverse reconstruction
  - spectral feature modeling
- `guide_workflow(...)` now accepts a `goal` argument and stays report-first across Python, CLI, MCP, and API
- SQL inverse fitting is now represented in the shared workflow catalog instead of remaining a partially hidden capability
- `inspect_product` now reports replaceability risks, decision burdens, glue burdens, and adoption priorities directly from code-owned product metadata

This does not complete the adoption pass, but it removes one important source of replaceability:
the product is now clearer about the workflows it wants users and AI clients to follow.
