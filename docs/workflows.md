# Day-1 Workflows

These are the workflows that explain the product quickly without broadening what it is.

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

## 4. CLI Batch Packet Sweep

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

## 5. Backend-Aware ML Workflow

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

## 6. MCP Workflow

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

## 7. Optional API Workflow

Input:

- a local or self-hosted HTTP client

Command:

```bash
spectral-packet-engine serve-api --host 127.0.0.1 --port 8000
```

Example:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

Expected outputs:

- health inspection with product name and version,
- status inspection with canonical workflow ids plus raw API route bindings,
- JSON responses for profile reports, forward runs, spectral analysis, profile compression, inverse fitting, comparison, sweeps, backend-aware ML training or evaluation, and TensorFlow compatibility workflows when installed.

Artifact locations:

- inline JSON by default,
- optional artifact bundles when an endpoint request includes `output_dir`,
- artifact listing through `GET /artifacts`

What the user learns:

- the HTTP surface is a thin wrapper over the same compute engine, not a separate product.

## 8. CLI Table Comparison Workflow

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
