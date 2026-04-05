# PR Draft: Spectral Feature Export And Tree-Model Workflows

## Why

The repository already exposed spectral compression and backend-aware surrogate workflows, but it did not yet provide one explicit downstream path for:

- exporting a reusable spectral feature table,
- preserving a stable feature contract with schema and provenance,
- training or tuning a bounded tree-model workflow through the same shared engine surfaces.

This PR keeps that path product-aligned instead of growing a separate generic tabular ML subsystem.

## What

- added shared `export-features`, `tree-train`, and `tree-tune` workflows across Python, CLI, MCP, and API,
- made feature export write `features_schema.json` with column order, dtype, nullability, semantic role, and semantic meaning,
- made feature-export metadata record input kind, normalization settings, mode count, ordering policy, and library versions where available,
- made SQL-backed feature export record whether time or position columns were reordered during materialization,
- made tree training and tuning artifacts record backend library metadata and persisted-model provenance,
- classified tree-model workflows as beta in shared installation validation, not only in docs,
- documented the new workflows in the README, workflow guide, MCP guide, and release-readiness notes.

## How Tested

### Repository test suite

- `pytest -q`
  Result: `160 passed, 3 skipped`

### CLI smoke

- `PYTHONPATH=src python3 -m spectral_packet_engine.cli export-features examples/data/synthetic_profiles.csv --modes 8 --device cpu --output-dir <tmp>/features`
- joined a synthetic `target` column into the exported feature table
- `PYTHONPATH=src python3 -m spectral_packet_engine.cli tree-train <tmp>/features_with_target.csv --target-column target --library sklearn --params '{"n_estimators": 64, "max_depth": 4}' --output-dir <tmp>/tree_train`

Observed:

- feature export wrote `features.csv`, `feature_table_export.json`, `features_schema.json`, and `artifacts.json`,
- tree training wrote `tree_training.json`, `predictions.csv`, `feature_importance.csv`, the persisted sklearn model file, and `artifacts.json`.

### MCP tool discovery

Validated from the real server path via `create_mcp_server().list_tools()`:

- `inspect_tree_backends`
- `export_feature_table`
- `export_feature_table_from_sql`
- `train_tree_model`
- `tune_tree_model`

### Artifact completeness

Feature-export bundle:

```text
features/
  artifacts.json
  feature_table_export.json
  features.csv
  features_schema.json
```

Observed artifact state:

- `inspect-artifacts` reported the bundle as complete, which is driven by `artifacts.json`,
- metadata included `input_kind`, `normalize_each_profile`, `num_modes`, `ordering`, and `library_versions`.

Tree-training bundle:

```text
tree_train/
  artifacts.json
  feature_importance.csv
  predictions.csv
  sklearn_randomforestregressor.pkl
  tree_training.json
```

Observed artifact state:

- `inspect-artifacts` reported the bundle as complete, which is driven by `artifacts.json`,
- metadata included `library`, `library_version`, `source_kind`, `target_column`, and `task`.

### Packaging and release checks

- `python3 -m build --sdist --wheel`
- `python3 -m twine check dist/*`

Both commands passed in this session.

## Risks

- the workflow family is beta and should not yet be presented as part of the stable core,
- the scikit-learn baseline is the most directly validated backend in this session,
- XGBoost, LightGBM, and CatBoost remain optional runtime-gated paths with much narrower validation,
- tiny datasets can still produce weak metrics such as undefined `R^2`; this is expected and already explicit in outputs rather than hidden.

## Follow-Up TODOs

1. Run clean-environment validation for `ml-xgboost`, `ml-lightgbm`, and `ml-catboost` extras on Linux and Windows.
2. Add backend-specific serialization and reload coverage for LightGBM and CatBoost.
3. Expand artifact and tuning coverage for non-sklearn backends so feature importance and model-export behavior are validated consistently.
4. Add broader overload and queue-behavior validation for long-running MCP tree-model jobs.
5. Reassess whether any tree backend can graduate from beta once clean-environment install and reload evidence is stronger.
