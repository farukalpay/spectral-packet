# Day-1 Workflows

These are the workflows that explain the product quickly without broadening what it is.

The repository now has four major workflow families:

- report-first spectral evidence loops,
- inverse and uncertainty-aware inference,
- controlled reduced models,
- differentiable and domain-specific vertical workflows.

If you are unsure which path to use first, ask the product directly:

```bash
spectral-packet-engine guide-workflow
```

Unless stated otherwise, file-based examples assume you are running from a repository checkout so `examples/data/` is available.

## 1. Hero Workflow: Python Profile Report

Input:

- one validated profile table

Code:

```python
from spectral_packet_engine import load_profile_table_report

report = load_profile_table_report(
    "examples/data/synthetic_profiles.csv",
    analyze_num_modes=16,
    compress_num_modes=8,
    device="cpu",
)
artifacts = report.write_artifacts("artifacts/profile_report_python")
```

Expected outputs:

- one inspection summary of the input table,
- one spectral analysis result,
- one compression result,
- one top-level report overview with dominant modes, capture thresholds, and compression error.
- one artifact inspection result describing completeness and visible files.

Artifact locations:

- `artifacts/profile_report_python/profile_table_report.json`
- `artifacts/profile_report_python/profile_table_summary.json`
- `artifacts/profile_report_python/analysis/spectral_analysis.json`
- `artifacts/profile_report_python/compression/compression_summary.json`
- `artifacts/profile_report_python/artifacts.json`

What the user learns:

- the real center of gravity is profile-table analysis and compression,
- Python is the cleanest surface for structured result objects,
- the artifact layer stays shared and inspectable rather than hidden in wrapper code.

## 2. Hero Workflow: CLI Profile Report

Input:

- a profile table in CSV, TSV, JSON, or optionally XLSX

Command:

```bash
spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report
spectral-packet-engine inspect-artifacts artifacts/profile_report
```

Expected outputs:

- one reference-grade profile report in JSON,
- one validated table summary,
- nested analysis and compression artifact bundles,
- one artifact inspection summary that confirms completion and provenance.

Artifact locations:

- `artifacts/profile_report/profile_table_report.json`
- `artifacts/profile_report/profile_table_summary.json`
- `artifacts/profile_report/analysis/artifacts.json`
- `artifacts/profile_report/compression/artifacts.json`
- `artifacts/profile_report/artifacts.json`

What the user learns:

- the CLI is a thin operational wrapper over the same library workflow,
- artifact bundles are stable enough to inspect after the run,
- the workflow is useful without needing the optional service layers.

## 3. SQL Ingress Into The Same Report Workflow

Input:

- a local SQLite path,
- a profile-table-shaped SQL query

Command:

```bash
spectral-packet-engine bootstrap-database artifacts/example.sqlite
spectral-packet-engine write-database-table artifacts/example.sqlite profiles examples/data/synthetic_profiles.csv --if-exists replace
spectral-packet-engine sql-profile-report artifacts/example.sqlite 'SELECT * FROM "profiles" ORDER BY time' \
  --time-column time \
  --sort-by-time \
  --device cpu \
  --output-dir artifacts/sql_profile_report
```

Expected outputs:

- one profile report built from a SQL result set,
- explicit SQL provenance in the root artifact index,
- the same analysis and compression structure used by the file-backed workflow.

Artifact locations:

- `artifacts/sql_profile_report/profile_table_report.json`
- `artifacts/sql_profile_report/analysis/spectral_analysis.json`
- `artifacts/sql_profile_report/compression/compression_summary.json`
- `artifacts/sql_profile_report/artifacts.json`

What the user learns:

- SQL is a first-class ingress path into the same spectral engine,
- the SQL-to-profile-table boundary stays explicit through `time_column`, `position_columns`, and `sort_by_time`,
- the report semantics stay aligned between file-backed and SQL-backed runs.

## 4. Uncertainty-Aware Inverse Fit

Input:

- one observed profile table,
- one physically plausible initial guess for packet center, width, and wavenumber.

Command:

```bash
spectral-packet-engine fit-profile-table examples/data/synthetic_profiles.csv \
  --center 0.36 \
  --width 0.11 \
  --wavenumber 22.0 \
  --device cpu \
  --output-dir artifacts/inverse_fit
```

Expected outputs:

- one inverse-fit summary with fitted packet parameters and optimization history,
- one local posterior summary over the inferred physical parameters,
- one modal-coefficient posterior summary,
- one sensitivity map bundle that shows which parts of the profile constrain which parameters,
- one posterior-predictive interval summary over the observed density,
- one observation-information map that shows where local Fisher information concentrates,
- one predicted density table and artifact index.

Artifact locations:

- `artifacts/inverse_fit/inverse_fit.json`
- `artifacts/inverse_fit/predicted_density.csv`
- `artifacts/inverse_fit/uncertainty_summary.json`
- `artifacts/inverse_fit/parameter_posterior.csv`
- `artifacts/inverse_fit/modal_posterior.csv`
- `artifacts/inverse_fit/sensitivity_map.json`
- `artifacts/inverse_fit/observation_posterior.json`
- `artifacts/inverse_fit/observation_information.json`
- `artifacts/inverse_fit/artifacts.json`

What the user learns:

- inverse reconstruction is no longer only a point estimate; it now exposes confidence intervals, identifiability, and parameter-to-observation sensitivity explicitly,
- the uncertainty story lives in the shared workflow layer, so Python, CLI, MCP, and API stay aligned,
- modal coefficients remain part of the answer, but only as one consequence of a broader physical inference result.

## 5. Explicit SQLite Setup Before SQL Query Workflows

Input:

- a local SQLite path,
- an explicit SQL setup script for schema and seed rows,
- a read-only query that consumes the resulting tables.

Command:

```bash
spectral-packet-engine execute-database-script artifacts/war.sqlite '
CREATE TABLE IF NOT EXISTS war_commodities (
  month_idx INTEGER PRIMARY KEY,
  month_label TEXT,
  brent_usd REAL,
  gold_usd REAL
);
INSERT OR REPLACE INTO war_commodities VALUES
  (1, "2026-01", 65.0, 5400.0),
  (2, "2026-02", 69.4, 5070.0);
'
spectral-packet-engine query-database artifacts/war.sqlite 'SELECT * FROM "war_commodities" ORDER BY month_idx'
```

Expected outputs:

- one explicit execution summary for the setup script,
- one read-only query summary over the seeded table,
- no ambiguity about whether the SQL surface is mutating or read-only.

Artifact locations:

- optional: `query-database --output-dir ...` writes `db_query_summary.json`, `query_result.csv`, and `artifacts.json`

What the user learns:

- `query-database` is intentionally read-only,
- schema creation and seed inserts belong to `execute-database-statement` or `execute-database-script`,
- later SQL-backed spectral workflows can depend on an explicit, inspectable database-setup step.

## 6. Spectroscopy-Style Family Inference

Input:

- an observed low-lying spectrum,
- one or more explicit candidate potential families.

Command:

```bash
spectral-packet-engine infer-potential-spectrum 5.22 15.83 26.41 \
  --family harmonic \
  --family double-well \
  --device cpu \
  --output-dir artifacts/spectroscopy
```

Expected outputs:

- one ranking over candidate potential families,
- one best-fit calibration summary,
- local posterior, sensitivity, and observation-information outputs for the best family,
- one vertical artifact bundle with family-comparison provenance.

Artifact locations:

- `artifacts/spectroscopy/vertical_workflow_summary.json`
- `artifacts/spectroscopy/family_inference/potential_family_inference.json`
- `artifacts/spectroscopy/family_inference/candidate_ranking.csv`
- `artifacts/spectroscopy/family_inference/best_family_calibration.json`
- `artifacts/spectroscopy/family_inference/best_family_observation_posterior.json`
- `artifacts/spectroscopy/family_inference/best_family_observation_information.json`

What the user learns:

- the inverse question can now be “which family explains this spectrum?” instead of only “what is one fitted parameter vector?”,
- family comparison is explicit and inspectable rather than hidden in generic regression logic,
- the uncertainty story remains local and artifact-backed.

## 7. Controlled Reduced Models

Input:

- a separable bounded 2D structure, a reduced coupled-channel structure, or a radial effective coordinate.

Code:

```python
from spectral_packet_engine import build_separable_2d_report

report = build_separable_2d_report(
    num_modes_x=4,
    num_modes_y=4,
    num_combined_states=6,
    device="cpu",
)
artifacts = report.write_artifacts("artifacts/separable_2d_report")
```

Expected outputs:

- one structured separable 2D report over a bounded box-plus-box problem,
- one retained tensor-basis summary with explicit x-major/y-minor indexing,
- one mode-budget and truncation summary,
- one Kronecker-sum operator summary,
- one eigenvalue table compared against the closed-form additive box-plus-box reference.

Artifact locations:

- `artifacts/separable_2d_report/reduced_model_summary.json`
- `artifacts/separable_2d_report/separable_2d_report.json`
- `artifacts/separable_2d_report/separable_2d_summary.json`
- `artifacts/separable_2d_report/eigenvalues.csv`
- `artifacts/separable_2d_report/mode_budget.json`
- `artifacts/separable_2d_report/structured_operator.json`
- `artifacts/separable_2d_report/artifacts.json`

What the user learns:

- structured dimensional lift means separable tensor bases plus Kronecker-sum operators, not arbitrary 2D grids or general 3D infrastructure,
- mode budgets and truncation cutoffs are first-class outputs rather than hidden heuristics,
- the artifact-backed report is a narrow beta path centered on one analytically interpretable separable 2D example.

Other reduced-model entry points remain available when the structure is explicit:

```bash
spectral-packet-engine analyze-separable-spectrum \
  --family-x harmonic --params-x '{"omega": 8.0}' \
  --family-y harmonic --params-y '{"omega": 6.0}' \
  --device cpu

spectral-packet-engine analyze-coupled-surfaces --device cpu
spectral-packet-engine solve-radial-reduction --family morse --params '{"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7}' --device cpu
```

## 8. Differentiable Design And Control

Input:

- a target transition energy,
- or a target observable for packet steering.

Commands:

```bash
spectral-packet-engine design-transition \
  --family harmonic \
  --target-transition 12.0 \
  --initial-guess '{"omega": 5.0}' \
  --device cpu
```

```bash
spectral-packet-engine optimize-packet-control \
  --objective target_position \
  --target-value 0.55 \
  --final-time 0.004 \
  --device cpu
```

What the user learns:

- gradients are now first-class when the physics map is differentiable,
- the control story stays tied to explicit packet-preparation parameters,
- differentiable workflows remain subordinate to the spectral core rather than forming a generic training subsystem.

## 9. Vertical Scientific Workflow

Input:

- one profile table,
- one plausible initial packet guess.

Command:

```bash
spectral-packet-engine profile-inference-workflow examples/data/synthetic_profiles.csv \
  --device cpu \
  --output-dir artifacts/profile_vertical
```

Expected outputs:

- one report-first profile summary,
- one uncertainty-aware inverse fit,
- one spectral feature export,
- one nested artifact bundle with `report/`, `inverse/`, and `features/`.

What the user learns:

- report-first remains the default even when the end goal is inverse inference or downstream features,
- the tabular vertical is one coherent workflow rather than three disconnected commands,
- Python, CLI, and MCP all follow the same nested artifact contract.

## 10. CLI Batch Packet Sweep

Input:

- several packet parameter tuples

Command:

```bash
spectral-packet-engine packet-sweep \
  --centers 0.25 0.35 \
  --widths 0.07 0.08 \
  --wavenumbers 22 24 \
  --device cpu \
  --output-dir artifacts/packet_sweep
```

Expected outputs:

- one JSON summary for the sweep,
- one CSV table with final metrics for each run,
- artifact index.

Artifact locations:

- `artifacts/packet_sweep/packet_sweep.json`
- `artifacts/packet_sweep/packet_sweep.csv`
- `artifacts/packet_sweep/artifacts.json`

What the user learns:

- the backend can absorb parameter-sweep work,
- the project is usable for repeated compute jobs, not only single demos.

## 11. Backend-Aware ML Workflow

Input:

- a profile table,
- an explicit backend choice such as `torch` or `jax`

Command:

```bash
spectral-packet-engine ml-backends --device cpu
spectral-packet-engine ml-evaluate-table examples/data/synthetic_profiles.csv --backend torch --modes 8 --epochs 6 --batch-size 2 --device cpu --output-dir artifacts/ml_eval
```

Expected outputs:

- backend capability report,
- ML evaluation summary in JSON,
- reconstructed profile table,
- predicted coefficient table,
- predicted moment table,
- artifact index.

Artifact locations:

- `artifacts/ml_eval/ml_evaluation.json`
- `artifacts/ml_eval/ml_reconstruction.csv`
- `artifacts/ml_eval/ml_coefficients.csv`
- `artifacts/ml_eval/ml_predicted_moments.csv`
- `artifacts/ml_eval/artifacts.json`

What the user learns:

- ML backend choice is explicit and discoverable,
- PyTorch and JAX share one workflow surface,
- surrogate workloads stay subordinate to the spectral engine instead of becoming a separate product.

## 12. Spectral Feature Export And Tree Models

Status: beta. These workflows are integrated and artifact-backed, but they should still be treated as downstream spectral workflows rather than as a fully generalized tabular ML subsystem.

Input:

- one validated profile table for feature export,
- one feature table that already includes the supervised target column for tree training or tuning.

Command:

```bash
spectral-packet-engine tree-backends
spectral-packet-engine export-features examples/data/synthetic_profiles.csv --modes 16 --device cpu --output-dir artifacts/features
spectral-packet-engine tree-train artifacts/features_with_target.csv --target-column target --library sklearn --params '{"n_estimators": 128, "max_depth": 6}' --output-dir artifacts/tree_train
spectral-packet-engine tree-tune artifacts/features_with_target.csv --target-column target --library sklearn --search-kind grid --search-space '{"n_estimators": [64, 128], "max_depth": [4, 6]}' --cv 3 --output-dir artifacts/tree_tune
```

Expected outputs:

- one tree-backend capability report,
- one explicit spectral feature table with schema, provenance, ordering policy, and library-version metadata,
- one training summary with predictions, feature importance, and a persisted model artifact,
- one tuning summary plus a nested best-model artifact bundle.

Artifact locations:

- `artifacts/features/features.csv`
- `artifacts/features/feature_table_export.json`
- `artifacts/features/features_schema.json`
- `artifacts/tree_train/tree_training.json`
- `artifacts/tree_train/predictions.csv`
- `artifacts/tree_train/feature_importance.csv`
- `artifacts/tree_tune/tree_tuning.json`
- `artifacts/tree_tune/best_model/artifacts.json`

What the user learns:

- feature-table construction is owned by the shared workflow layer instead of wrappers,
- the exported feature contract is inspectable: ordered columns, dtypes, semantic meaning, normalization, input kind, and ordering policy are all recorded,
- tree workflows stay tied to spectral inputs and provenance instead of becoming generic tabular tooling,
- backend availability and optional dependency boundaries are explicit before training starts.

## 13. MCP Workflow

Input:

- an MCP client connected to the local machine

Command:

```bash
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-level warning
```

Expected operations through the client:

- inspect the shared product and runtime contract,
- ask `guide_workflow` for the default operator path before composing a tool chain,
- inspect the MCP runtime and environment,
- validate the installation,
- run `profile_table_report` or `report_database_profile_query`,
- inspect profile tables,
- run compression or inverse reconstruction,
- launch packet sweeps,
- retrieve produced artifact bundles.

Artifact locations:

- wherever `output_dir` is directed by the client

What the user learns:

- MCP is useful because the machine-side backend can do real numerical work through structured domain tools.
- MCP runtime limits, artifact completion, and logging behavior are explicit instead of hidden in wrapper code.

## 14. Optional API Workflow

Input:

- a local or self-hosted HTTP client

Command:

```bash
spectral-packet-engine serve-api --host localhost --port 8000
```

Example:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

Expected outputs:

- health inspection with product name and version,
- status inspection with canonical workflow ids plus raw API route bindings,
- JSON responses for profile reports, forward runs, spectral analysis, profile compression, inverse fitting, comparison, sweeps, feature export, tree-model training or tuning, backend-aware ML training or evaluation, and TensorFlow compatibility workflows when installed.

Artifact locations:

- inline JSON by default,
- optional artifact bundles when an endpoint request includes `output_dir`,
- artifact listing through `GET /artifacts`

What the user learns:

- the HTTP surface is a thin wrapper over the same compute engine, not a separate product.

## 15. CLI Table Comparison Workflow

Input:

- a reference profile table,
- a candidate profile table on the same grid and sample times

Command:

```bash
spectral-packet-engine compare-tables examples/data/synthetic_profiles.csv examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/comparison
```

Expected outputs:

- comparison summary in JSON,
- residual profile table,
- per-sample error and drift metrics,
- artifact index.

Artifact locations:

- `artifacts/comparison/table_comparison.json`
- `artifacts/comparison/residual_profiles.csv`
- `artifacts/comparison/sample_metrics.csv`
- `artifacts/comparison/artifacts.json`

What the user learns:

- the product can compare real runs instead of only producing them,
- domain metrics like relative error, mass drift, and width drift are part of the compute backend.
