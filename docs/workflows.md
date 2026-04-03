# Day-1 Workflows

These are the canonical workflows that explain the product quickly.

Unless stated otherwise, file-based examples assume you are running from a repository checkout so `examples/data/` is available.

## 1. Direct Python Workflow

Input:

- packet parameters in your own Python code

Command or code:

```python
from spectral_packet_engine import simulate_gaussian_packet

summary = simulate_gaussian_packet(
    center=0.30,
    width=0.07,
    wavenumber=25.0,
    times=[0.0, 1e-3, 5e-3],
    device="cpu",
)
```

Expected outputs:

- packet densities through time,
- expectation-position trajectory,
- interval probabilities,
- truncation summary.

Artifact locations:

- none unless you write them yourself or use the packaged example script

What the user learns:

- the project is a real importable library,
- forward bounded-domain simulation is a first-class workflow,
- the result objects are structured enough for downstream code.

## 2. CLI Spectral Analysis Workflow

Input:

- a profile table in CSV, TSV, JSON, or optionally XLSX

Command:

```bash
spectral-packet-engine analyze-table examples/data/synthetic_profiles.csv --modes 16 --device cpu --output-dir artifacts/analysis
```

Expected outputs:

- spectral analysis summary in JSON,
- coefficient table,
- per-sample metrics,
- aggregated modal-weight table,
- artifact index.

Artifact locations:

- `artifacts/analysis/spectral_analysis.json`
- `artifacts/analysis/coefficients.csv`
- `artifacts/analysis/sample_metrics.csv`
- `artifacts/analysis/mean_modal_weights.csv`
- `artifacts/analysis/artifacts.json`

What the user learns:

- the project accepts practical file inputs,
- spectral decomposition and mode-budget analysis are day-1 user jobs,
- the CLI is meant for reproducible compute runs.

## 3. CLI SQL Bootstrap And Analysis Workflow

Input:

- a local SQLite path,
- a profile-table-shaped SQL workflow

Command:

```bash
spectral-packet-engine db-bootstrap artifacts/example.sqlite
spectral-packet-engine db-write-table artifacts/example.sqlite profiles examples/data/synthetic_profiles.csv --if-exists replace
spectral-packet-engine sql-analyze-table artifacts/example.sqlite 'SELECT * FROM "profiles" ORDER BY time' --modes 8 --device cpu
```

Expected outputs:

- database capability report,
- persisted profile table,
- spectral analysis summary from a SQL-backed data source.

Artifact locations:

- optional when `--output-dir` is provided to the analysis step

What the user learns:

- the product can bootstrap a usable local database path,
- SQL is a real data ingress path into the bounded-domain engine,
- the SQL layer supports structured compute jobs rather than only raw queries.
- read-oriented database commands expect an existing database; bootstrap or write one first.

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
spectral-packet-engine serve-mcp
```

Expected operations through the client:

- validate the installation,
- inspect profile tables,
- run compression or inverse reconstruction,
- launch packet sweeps,
- retrieve produced artifact bundles.

Artifact locations:

- wherever `output_dir` is directed by the client

What the user learns:

- MCP is useful because the machine-side backend can do real numerical work through structured domain tools.

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
```

Expected outputs:

- health and capability inspection,
- JSON responses for forward runs, spectral analysis, profile compression, inverse fitting, comparison, sweeps, backend-aware ML training or evaluation, and TensorFlow compatibility workflows when installed.

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
