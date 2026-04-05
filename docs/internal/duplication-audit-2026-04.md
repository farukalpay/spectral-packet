# Duplication Audit

Date: 2026-04-04

This audit groups duplication by behavior rather than by filename similarity. The goal is to consolidate true duplicate implementations into the existing shared workflow and artifact layers without flattening domain-specific logic.

## 1. SQL-backed profile-table workflow wrappers

- Duplicate area:
  SQL-backed workflow entrypoints repeat the same `materialize profile table -> delegate to shared spectral workflow` behavior.
- Files involved:
  - `src/spectral_packet_engine/workflows.py`
- Why this is a true duplicate:
  `analyze_profile_table_from_database_query`, `compress_profile_table_from_database_query`, `build_profile_table_report_from_database_query`, `fit_gaussian_packet_to_profile_table_from_database_query`, `export_feature_table_from_database_query`, `train_modal_surrogate_from_database_query`, and `evaluate_modal_surrogate_from_database_query` all thread the same database/materialization arguments into `materialize_profile_table_from_database_query(...)` and then immediately delegate.
- Stronger implementation:
  `materialize_profile_table_from_database_query(...)` is already the canonical shared implementation for SQL-to-profile-table behavior.
- Risk level of consolidation:
  Medium-low. The behavioral center is already shared; only the repeated orchestration needs consolidation.
- Proposed refactor strategy:
  Add one internal helper in `workflows.py` that materializes once and hands the resulting `DatabaseProfileTableMaterialization` into a callback. Migrate the SQL-backed wrappers to that helper.
- Backward compatibility needed:
  No public API removal. Existing public functions remain as stable entrypoints.

## 2. SQL artifact metadata merge across CLI / MCP / API

- Duplicate area:
  SQL-backed surface handlers repeatedly build:
  `{"workflow": "...", **database_profile_query_artifact_metadata(...)}`
- Files involved:
  - `src/spectral_packet_engine/cli.py`
  - `src/spectral_packet_engine/mcp.py`
  - `src/spectral_packet_engine/api.py`
  - `src/spectral_packet_engine/workflows.py`
- Why this is a true duplicate:
  The behavior is identical across surfaces: take canonical SQL provenance metadata and attach the workflow id for artifact writing.
- Stronger implementation:
  `database_profile_query_artifact_metadata(...)` in `workflows.py` is already the canonical provenance builder. The missing piece is one shared wrapper that also records the workflow id.
- Risk level of consolidation:
  Low.
- Proposed refactor strategy:
  Add `database_profile_query_workflow_artifact_metadata(...)` in `workflows.py` and switch the surface wrappers to it.
- Backward compatibility needed:
  No external behavior change. Artifact shape should remain identical.

## 3. Generic database-query artifact metadata merge across CLI / MCP / API

- Duplicate area:
  Surface handlers repeatedly build:
  `{"workflow": "...", **database_query_artifact_metadata(...)}`
- Files involved:
  - `src/spectral_packet_engine/cli.py`
  - `src/spectral_packet_engine/mcp.py`
  - `src/spectral_packet_engine/api.py`
  - `src/spectral_packet_engine/workflows.py`
- Why this is a true duplicate:
  The behavior is the same as the SQL profile-query case, but for generic tabular query artifacts.
- Stronger implementation:
  `database_query_artifact_metadata(...)` is already the canonical query-provenance builder.
- Risk level of consolidation:
  Low.
- Proposed refactor strategy:
  Add `database_query_workflow_artifact_metadata(...)` in `workflows.py` and migrate wrappers.
- Backward compatibility needed:
  No.

## 4. File-backed convenience wrappers

- Duplicate area:
  File-backed helper functions such as `load_profile_table_report(...)` and `load_and_compress_profile_table(...)` look similar to the base workflow functions.
- Files involved:
  - `src/spectral_packet_engine/workflows.py`
- Why this is not a true duplicate:
  These are stable convenience adapters that convert one public input boundary into the canonical shared workflow. They do not reimplement the spectral work or artifact semantics.
- Stronger implementation:
  The canonical implementations remain `build_profile_table_report(...)` and `compress_profile_table(...)`; the file-backed adapters are acceptable specialization.
- Risk level of consolidation:
  Medium if removed, with limited payoff.
- Proposed refactor strategy:
  Keep them. Do not collapse them unless the public API surface is deliberately narrowed later.
- Backward compatibility needed:
  Yes, which is another reason to leave them intact.

## 5. Artifact writers in `artifacts.py`

- Duplicate area:
  Multiple artifact writers share similar structure.
- Files involved:
  - `src/spectral_packet_engine/artifacts.py`
- Why this is not a first-wave true duplicate:
  They already rely on shared index/finalization helpers, but the bundles themselves are meaningfully different: report bundles, inverse bundles, feature bundles, training bundles, and comparison bundles have different stable contracts.
- Stronger implementation:
  Current specialized writers are the right architectural home because artifact semantics are product-facing.
- Risk level of consolidation:
  Medium-high if overdone, because it would blur bundle identity.
- Proposed refactor strategy:
  Leave specialized writers separate. Only deduplicate metadata/index plumbing, not bundle semantics.
- Backward compatibility needed:
  Yes.

## 6. Tree-model dataset wrappers

- Duplicate area:
  `train_tree_model(...)` / `tune_tree_model(...)` look similar to `train_tree_model_on_dataset(...)` / `tune_tree_model_on_dataset(...)`.
- Files involved:
  - `src/spectral_packet_engine/workflows.py`
  - `src/spectral_packet_engine/tree_models.py`
- Why this is not a true duplicate:
  The workflow-layer functions are intentionally surface-oriented adapters from file-or-dataset input into the dataset-native implementation in `tree_models.py`.
- Stronger implementation:
  `*_on_dataset(...)` in `tree_models.py` is the canonical implementation; workflow-layer wrappers are acceptable specialization.
- Risk level of consolidation:
  Low, but payoff is also low because the wrappers are already thin.
- Proposed refactor strategy:
  Keep as-is.
- Backward compatibility needed:
  Yes.

## Safest Consolidation Sequence

1. Consolidate SQL artifact metadata merge helpers.
2. Consolidate generic database-query artifact metadata merge helpers.
3. Consolidate SQL-backed profile-table workflow wrappers around one internal materialize-and-delegate helper.
4. Re-run wrapper and artifact tests across CLI / MCP / API.
5. Reassess whether any further consolidation is needed in artifact plumbing, but avoid flattening semantically distinct bundle writers.
