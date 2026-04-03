# Spectral Packet Engine

Spectral Packet Engine is a cross-platform scientific compute library for bounded-domain modal simulation, spectral decomposition, and inverse reconstruction.

It is designed as a real product surface, not a manuscript archive:

- use it directly as a Python package,
- run packaged workflows through a CLI,
- expose the same compute engine through MCP,
- optionally serve it as a local or self-hosted HTTP API.

## Day-1 Path

From a fresh source checkout:

1. install the package,
2. validate the local environment,
3. run one bundled table workflow,
4. choose Python, CLI, MCP, or API based on how you want to integrate the engine.

Minimal path:

```bash
python3 -m pip install -e .
spectral-packet-engine --version
spectral-packet-engine validate-install --device cpu
spectral-packet-engine compress-table examples/data/synthetic_profiles.csv --modes 8 --device cpu --output-dir artifacts/compression
```

If you want the shortest clone-to-first-run guide, start with [docs/getting-started.md](docs/getting-started.md).

## What It Is For

Install this project if you need one or more of these workflows:

- project localized packet states into a bounded modal basis,
- analyze the spectral structure of observed profile tables,
- ingest tabular data from files or SQL queries before routing it into spectral workflows,
- propagate those modal states exactly in time,
- reconstruct compact packet parameters from observed densities,
- compress bounded profile data into modal coefficients,
- compare reference and candidate profile tables with domain metrics,
- bootstrap a local SQLite workflow or query a remote database when SQLAlchemy is installed,
- run those workflows through Python, CLI, MCP, or an HTTP service without rewriting the compute logic.

## What It Is Not

- Not a generic AI wrapper.
- Not a paper-first repository.
- Not a generic physics-everything engine.
- Not a chat platform.

The product identity is narrow on purpose:

`bounded-domain spectral packet engine -> decomposition / propagation / inverse reconstruction -> reusable compute interfaces`

This project is for researchers, scientific programmers, and engineering teams who need a reusable bounded-domain compute engine instead of a one-off notebook or a manuscript companion repo.

## Product Surfaces

| Surface | Purpose | Status |
| --- | --- | --- |
| Python library | In-process scientific compute and integration into your own code | Stable |
| CLI | Local workflows, artifacts, scripting, CI, reproducible runs | Stable |
| Tabular + SQLite workflows | File-backed and local-database data ingress into the spectral engine | Stable |
| MCP server | Structured machine-side compute for LLM clients | Beta |
| HTTP API | Optional self-hosted service layer over the same workflows | Beta |
| Remote SQL backends | SQLAlchemy-backed access to non-SQLite databases | Beta |
| Backend-aware modal surrogate | PyTorch-first modal-regression training and evaluation over profile tables | Beta |
| JAX modal surrogate | Optional JAX/XLA path over the same backend-aware workflow surface | Beta |
| TensorFlow surrogate | Compatibility path for TensorFlow-backed modal regression when installed | Experimental |
| Published transport dataset wrapper | Optional real-data benchmark path | Experimental |

Run this first on any machine:

```bash
spectral-packet-engine validate-install --device cpu
```

For the final release-readiness summary and remaining known limits, see [docs/release-readiness-audit.md](docs/release-readiness-audit.md).

## Why Install It

This package closes a common gap in scientific repos:

- the math is not detached from the implementation,
- the implementation is not trapped in notebooks,
- the interface layers are thin wrappers over the real engine,
- artifacts and outputs are stable enough to script and test.

## Direct Python Vs MCP Vs API

Use direct Python when:

- you are writing your own code,
- you want in-process tensors and result objects,
- you want the lowest integration overhead.

Use MCP when:

- you want an LLM client to delegate structured compute to a machine-side backend,
- you want bounded tools instead of arbitrary shell access,
- you want reusable simulation, compression, inverse-fit, and training operations exposed as domain tools.
- you want the backend to inspect tables, compare runs, and write artifact bundles after compute jobs.

Use the API when:

- you want a local or self-hosted network service,
- you need HTTP integration from another process or machine,
- you want the same workflows behind a web boundary.

## Installation

### Core package

After cloning the repository:

Linux and macOS:

```bash
python3 -m pip install .
```

Windows:

```powershell
py -m pip install .
```

You can also use the module entrypoint directly after installation:

Linux and macOS:

```bash
python3 -m spectral_packet_engine --help
```

Windows:

```powershell
py -m spectral_packet_engine --help
```

Check the installed release surface:

```bash
spectral-packet-engine --version
```

### Editable install for development

Linux and macOS:

```bash
python3 -m pip install -e ".[dev]"
```

Windows:

```powershell
py -m pip install -e ".[dev]"
```

### Optional extras

- `data`: published dataset download and MATLAB loading
- `files`: optional XLSX and Parquet file support through `openpyxl` and `pyarrow`
- `sql`: remote SQL backends through SQLAlchemy
- `examples`: plotting dependencies for the example scripts
- `mcp`: MCP server runtime
- `api`: FastAPI and Uvicorn
- `ml`: TensorFlow surrogate on supported Python versions
- `ml-jax`: JAX backend on supported non-Windows environments
- `ml-cuda`: Linux CUDA TensorFlow path

Examples:

```bash
python3 -m pip install -e ".[data,examples]"
python3 -m pip install -e ".[files]"
python3 -m pip install -e ".[sql]"
python3 -m pip install -e ".[mcp,api]"
python3 -m pip install -e ".[data,examples,ml]"
python3 -m pip install -e ".[ml-jax]"
```

## Platform Support

The core library and CLI are designed for Linux, Windows, and macOS.

Platform notes:

- Linux: best overall path for CPU and CUDA-oriented workflows.
- macOS: core library is fully supported; Torch MPS can be used when available; TensorFlow depends on a compatible Python and local TensorFlow stack.
- Windows: core library, CLI, MCP, and API are supported; TensorFlow GPU is not the primary target, and WSL2 is the recommended path for GPU-oriented TensorFlow work.

Run this to inspect the active machine:

```bash
spectral-packet-engine env
```

Detailed platform notes live in [docs/platforms.md](docs/platforms.md).
Canonical day-1 workflows live in [docs/workflows.md](docs/workflows.md).

## First Commands To Run

These commands assume you are working from a repository checkout so the bundled example data is present under `examples/data/`.

Inspect the machine:

```bash
spectral-packet-engine validate-install --device cpu
```

Run a forward packet simulation and write artifacts:

```bash
spectral-packet-engine forward --device cpu --output-dir artifacts/forward
```

Inspect the bundled table example:

```bash
spectral-packet-engine inspect-table examples/data/synthetic_profiles.csv --device cpu
```

Inspect the same file through the generic tabular layer:

```bash
spectral-packet-engine tabular-inspect examples/data/synthetic_profiles.csv
```

Bootstrap a local SQLite workflow from the bundled table data:

```bash
spectral-packet-engine db-bootstrap artifacts/example.sqlite
spectral-packet-engine db-write-table artifacts/example.sqlite profiles examples/data/synthetic_profiles.csv --if-exists replace
spectral-packet-engine db-query artifacts/example.sqlite 'SELECT * FROM "profiles" ORDER BY time'
spectral-packet-engine db-materialize-query artifacts/example.sqlite profiles_copy 'SELECT * FROM "profiles"' --replace
spectral-packet-engine sql-analyze-table artifacts/example.sqlite 'SELECT * FROM "profiles" ORDER BY time' --modes 8 --device cpu
```

Read-oriented database commands such as `db-inspect`, `db-list-tables`, `db-describe-table`, `db-query`, and SQL-backed analysis expect an existing database. Use `db-bootstrap` to create a local SQLite file when you need one.

Inspect ML backend support and run the backend-aware surrogate path:

```bash
spectral-packet-engine ml-backends --device cpu
spectral-packet-engine ml-evaluate-table examples/data/synthetic_profiles.csv --backend torch --modes 8 --epochs 6 --batch-size 2 --device cpu --output-dir artifacts/ml_eval
```

Compress the bundled table example:

```bash
spectral-packet-engine compress-table examples/data/synthetic_profiles.csv --modes 8 --device cpu --output-dir artifacts/compression
```

Fit a Gaussian packet to the same CSV table:

```bash
spectral-packet-engine fit-table examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/inverse
```

Run a batch sweep:

```bash
spectral-packet-engine packet-sweep --centers 0.25 0.35 --widths 0.07 0.08 --wavenumbers 22 24 --device cpu --output-dir artifacts/packet_sweep
```

## Python Quickstart

```python
from spectral_packet_engine import simulate_gaussian_packet

summary = simulate_gaussian_packet(
    center=0.30,
    width=0.07,
    wavenumber=25.0,
    times=[0.0, 1e-3, 5e-3],
    num_modes=128,
    grid_points=512,
    device="cpu",
)

print(summary.expectation_position)
print(summary.total_probability)
print(summary.truncation.dominant_modes)
```

## CLI Workflows

Main packaged commands:

- `spectral-packet-engine env`
- `spectral-packet-engine validate-install`
- `spectral-packet-engine forward`
- `spectral-packet-engine project`
- `spectral-packet-engine inspect-table`
- `spectral-packet-engine analyze-table`
- `spectral-packet-engine compress-table`
- `spectral-packet-engine compression-sweep`
- `spectral-packet-engine fit-table`
- `spectral-packet-engine compare-tables`
- `spectral-packet-engine packet-sweep`
- `spectral-packet-engine transport-benchmark`
- `spectral-packet-engine tf-train-table`
- `spectral-packet-engine tf-evaluate-table`
- `spectral-packet-engine serve-mcp`
- `spectral-packet-engine serve-api`

CLI runs write stable artifacts when `--output-dir` is provided:

- `forward_summary.json`
- `forward_densities.csv`
- `compression_summary.json`
- `spectral_analysis.json`
- `reconstruction.csv`
- `coefficients.csv`
- `inverse_fit.json`
- `predicted_density.csv`
- `table_comparison.json`
- `residual_profiles.csv`
- `sample_metrics.csv`
- `transport_benchmark.json`
- `ml_training.json`
- `ml_evaluation.json`
- `ml_reconstruction.csv`
- `ml_coefficients.csv`
- `ml_predicted_moments.csv`
- `tf_training.json`
- `tf_evaluation.json`
- `tf_reconstruction.csv`
- `tf_predicted_moments.csv`
- `artifacts.json`

## File Support

Supported table formats:

- CSV
- TSV
- JSON
- XLSX with the optional `files` extra

The packaged table workflow expects:

- one `time` column,
- one column per spatial position,
- strictly increasing position columns,
- rows shaped as one profile sample per time.

Example file:

- [examples/data/synthetic_profiles.csv](examples/data/synthetic_profiles.csv)
- [examples/data/synthetic_profiles.tsv](examples/data/synthetic_profiles.tsv)
- [examples/data/synthetic_profiles.json](examples/data/synthetic_profiles.json)

This makes the local file path workflows portable across Linux, Windows, and macOS without hidden notebook assumptions. If you install the package outside a source checkout, bring your own table files instead of relying on `examples/data/`.

## MCP Usage

Install the MCP extra:

```bash
python3 -m pip install -e ".[mcp]"
```

Run the server over stdio:

```bash
spectral-packet-engine serve-mcp
```

The MCP layer exposes domain-aligned tools for:

- environment inspection,
- installation validation,
- ML backend inspection,
- database bootstrap, inspection, and query materialization,
- spectral analysis of profile tables,
- packet simulation,
- modal projection,
- packet sweeps,
- table inspection and compression,
- table comparison,
- inverse fitting from table inputs,
- transport benchmark runs,
- backend-aware modal surrogate training and evaluation.

More detail: [docs/mcp.md](docs/mcp.md)

## API Usage

Install the API extra:

```bash
python3 -m pip install -e ".[api]"
```

Start the service:

```bash
spectral-packet-engine serve-api --host 127.0.0.1 --port 8000
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

The API can also write artifact bundles when an endpoint request includes an `output_dir`, and artifact listings can be retrieved through `GET /artifacts`.

More detail: [docs/api.md](docs/api.md)

## TensorFlow And ML

The product now has one backend-aware modal-surrogate surface with multiple implementations:

- PyTorch is the primary installed-by-default training backend because `torch` is already part of the core package.
- JAX is available through the optional `ml-jax` extra on supported environments.
- TensorFlow remains supported as an optional compatibility path through the `ml` extra.

It is useful when:

- you have repeated profile-to-mode inference workloads,
- you want a trainable surrogate over modal coefficients and profile moments,
- you want to evaluate reconstructed profiles and predicted modal coefficients on machine-side tables,
- you want the same workflow surface from Python, CLI, MCP, or API while choosing the backend explicitly,
- you understand that this path is secondary to the core spectral engine.

If an ML backend is missing or incompatible, the core spectral library, SQL/data workflows, CLI, MCP, and API remain usable.

## Examples To Run First

- [examples/python_workflow.py](examples/python_workflow.py): direct Python workflow with artifact output
- [examples/csv_profile_workflow.py](examples/csv_profile_workflow.py): table-driven spectral analysis, compression, inverse fitting, and comparison
- [examples/ml_backend_workflow.py](examples/ml_backend_workflow.py): backend-aware modal-surrogate evaluation over a profile table
- [examples/api_client_demo.py](examples/api_client_demo.py): minimal HTTP client against the optional API

Additional compatibility and experimental examples:

- [examples/tensorflow_evaluation_workflow.py](examples/tensorflow_evaluation_workflow.py): TensorFlow compatibility-path evaluation
- [examples/forward_reference.py](examples/forward_reference.py): older reference simulation example
- [examples/experimental_transport_analysis.py](examples/experimental_transport_analysis.py): published dataset benchmark

## Documentation

- [docs/architecture.md](docs/architecture.md): product spine, architecture, and release-direction summary
- [docs/release-readiness-audit.md](docs/release-readiness-audit.md): final release-readiness audit, trust-breaking issues, and remaining limits
- [docs/platforms.md](docs/platforms.md): cross-platform support notes
- [docs/mcp.md](docs/mcp.md): MCP setup and tool surface
- [docs/api.md](docs/api.md): API setup and endpoint surface
- [docs/getting-started.md](docs/getting-started.md): clone-to-first-run setup
- [docs/workflows.md](docs/workflows.md): canonical day-1 workflows

Legacy reference material is retained under [`docs/legacy/`](docs/legacy/README.md), but the files above are the current release-facing docs.

## Repository Layout

- [`src/spectral_packet_engine/`](src/spectral_packet_engine): library, workflows, CLI, MCP, API
- [`examples/`](examples): runnable demos
- [`tests/`](tests): regression and interface coverage
- [`docs/`](docs): product and interface documentation

## Testing

```bash
pytest -q
```

The test suite covers:

- core modal math,
- inverse fitting,
- CSV profile I/O,
- SQL and tabular workflows,
- backend-aware PyTorch and JAX modal surrogates,
- packaged workflows,
- CLI behavior,
- optional API and MCP surfaces when available.

## Troubleshooting

- If `validate-install` reports that the API is unavailable, reinstall the `api` extra in a clean environment. The package checks for FastAPI stack compatibility, not just import presence.
- If XLSX loading fails, install the `files` extra: `python3 -m pip install -e ".[files]"`.
- If TensorFlow workflows are unavailable, use `spectral-packet-engine ml-backends --device cpu` to inspect the available PyTorch and JAX paths. The TensorFlow path remains optional.
- If JAX is unavailable, install the `ml-jax` extra on a supported environment. Native Windows JAX support is not the primary target; prefer Linux, macOS, or WSL2.
- If you are on Windows and want a GPU-oriented TensorFlow workflow, prefer WSL2 over native Windows TensorFlow GPU expectations.
