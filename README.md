# Spectral Packet Engine

Spectral Packet Engine is a Python-first scientific compute library for bounded-domain spectral packet simulation, modal decomposition, inverse reconstruction, and data workflows built around that same engine.

It is not a generic AI platform or a paper companion repo. The product spine is narrow on purpose:

`bounded-domain spectral engine -> profile-table analysis / compression / inverse reconstruction -> Python, CLI, MCP, and API interfaces over the same workflows`

The runtime spine is equally narrow:

`one shared Python engine -> explicit environment validation -> bounded in-process workflows -> shared task/artifact state -> external supervision when needed`

The clearest day-1 workflow is:

`validated profile table -> inspect -> spectral analysis -> modal compression -> artifact-backed report`

The product’s three killer workflows are now explicit in code and runtime inspection:

- file profile report loop,
- SQL profile report loop,
- MCP operator loop.

## Why Use It

Use this project when you need a real reusable engine for one or more of these jobs:

- simulate a localized packet in a bounded 1D domain,
- project wavefunctions or observed profiles into a modal basis,
- quantify truncation error and observable behavior,
- fit Gaussian packet parameters back from observed densities,
- compress profile tables into modal coefficients,
- move file-backed or SQL-backed data into the same spectral workflows,
- expose those workflows through Python, CLI, MCP, or an HTTP boundary without rewriting the math.
- keep SQL-to-profile-table conversion explicit instead of hiding schema assumptions in wrappers.

## What It Is Not

- Not a generic backend framework.
- Not a generic data platform.
- Not a notebook dump.
- Not a vague “AI for science” wrapper.

## 5-Minute Path

From a fresh checkout:

```bash
python3 -m pip install -e .
spectral-packet-engine inspect-product
spectral-packet-engine guide-workflow
spectral-packet-engine inspect-environment --device cpu
spectral-packet-engine validate-install --device cpu
spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report
spectral-packet-engine inspect-artifacts artifacts/profile_report
```

That gives you:

- an environment and compatibility report,
- one reference-grade profile-table report,
- one artifact inspection report that tells you exactly what was written.

Typical artifact bundle:

```text
artifacts/profile_report/
  artifacts.json
  profile_table_report.json
  profile_table_summary.json
  analysis/
    artifacts.json
    spectral_analysis.json
    coefficients.csv
    sample_metrics.csv
    mean_modal_weights.csv
  compression/
    artifacts.json
    compression_summary.json
    coefficients.csv
    reconstruction.csv
```

If you are not running from a source checkout, use your own profile-table files instead of `examples/data/`.

## Python Example

```python
from spectral_packet_engine import guide_workflow, load_profile_table_report

print(guide_workflow(surface="python", input_kind="profile-table-file").defaults)

report = load_profile_table_report(
    "examples/data/synthetic_profiles.csv",
    analyze_num_modes=16,
    compress_num_modes=8,
    device="cpu",
)
report.write_artifacts("artifacts/profile_report_python")

print(report.overview.dominant_modes)
print(report.overview.capture_mode_budgets)
print(report.overview.mean_relative_l2_error)
```

Direct Python is the primary surface when you want in-process tensors, structured result objects, and composition into your own code.

`report.write_artifacts(...)` writes the same stable bundle structure used by the CLI, MCP, and API surfaces, then returns an artifact inspection report for immediate verification.

SQL-backed Python workflows use the same engine through an explicit table-materialization contract and the same report surface:

```python
from spectral_packet_engine import (
    build_profile_table_report_from_database_query,
)

report = build_profile_table_report_from_database_query(
    "sqlite:///artifacts/example.sqlite",
    'SELECT time, "x=0.0", "x=0.5", "x=1.0" FROM "profiles" ORDER BY time',
    time_column="time",
    position_columns=("x=0.0", "x=0.5", "x=1.0"),
    sort_by_time=True,
    analyze_num_modes=16,
    compress_num_modes=8,
    device="cpu",
)
```

## Product Surfaces

| Surface | Purpose | Status |
| --- | --- | --- |
| Python library | In-process engine access and composition into your own code | Stable |
| CLI | Reproducible local workflows and artifact bundles | Stable |
| File + SQLite workflows | Practical data ingress into the same spectral engine | Stable |
| MCP server | Structured machine-side access for external tool clients over stdio with bounded in-process execution | Beta |
| HTTP API | Optional local or self-hosted JSON service | Beta |
| Remote SQL backends | SQLAlchemy-backed non-SQLite access | Beta |
| Backend-aware modal surrogate | Shared PyTorch-first modal-regression workflow | Beta |
| JAX backend | Optional JAX path through the same surrogate workflow | Beta |
| TensorFlow compatibility path | Optional TensorFlow-specific workflow when installed | Experimental |
| Published transport dataset wrapper | Optional benchmark path | Experimental |

## Install

Core package:

```bash
python3 -m pip install .
```

Editable install:

```bash
python3 -m pip install -e ".[dev]"
```

Windows users can replace `python3` with `py`.

### Optional extras

- `files`: XLSX and Parquet support
- `sql`: remote SQL backends through SQLAlchemy
- `mcp`: MCP server runtime
- `api`: FastAPI and Uvicorn
- `ml-jax`: JAX backend on supported environments
- `ml`: TensorFlow compatibility path on supported Python versions
- `data`: published dataset download helpers
- `examples`: plotting dependencies for example scripts

Example installs:

```bash
python3 -m pip install -e ".[mcp]"
python3 -m pip install -e ".[mcp,api]"
python3 -m pip install -e ".[files,sql]"
python3 -m pip install -e ".[examples]"
python3 -m pip install -e ".[ml-jax]"
```

## MCP Runtime Quickstart

Install the optional MCP runtime, inspect the environment, then start the stdio server:

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine inspect-product
spectral-packet-engine inspect-environment --device cpu
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-level warning
```

Important operational rules:

- stdout is reserved for MCP protocol traffic,
- repository-managed logs go to stderr by default or to `--log-file` if you choose one,
- heavy MCP tools run under a bounded in-process execution limit and fail clearly when saturated,
- artifact bundles are written atomically and are only marked complete once `artifacts.json` is committed,
- automatic restarts belong to `systemd`, `launchd`, Docker restart policy, or an equivalent external supervisor.

## First Things To Try

- `spectral-packet-engine validate-install --device cpu`
- `spectral-packet-engine guide-workflow`
- `spectral-packet-engine inspect-product`
- `spectral-packet-engine inspect-environment --device cpu`
- `spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report`
- `spectral-packet-engine inspect-artifacts artifacts/profile_report`
- `spectral-packet-engine release-gate --device cpu`
- `spectral-packet-engine ml-backends --device cpu`

## Examples

Start with the curated examples index in [examples/README.md](examples/README.md).

Stable examples:

- [examples/core_engine_workflow.py](examples/core_engine_workflow.py)
- [examples/profile_table_workflow.py](examples/profile_table_workflow.py) for the hero profile-table report workflow
- [examples/modal_surrogate_workflow.py](examples/modal_surrogate_workflow.py)
- [examples/api_workflow.py](examples/api_workflow.py)

Reference and experimental material is separated under:

- [examples/reference/README.md](examples/reference/README.md)
- [examples/experimental/README.md](examples/experimental/README.md)

## Interface Choice

Use Python when you are writing your own code and want result objects directly.

Use the CLI when you want a reproducible local run that writes artifact bundles.

Use MCP when you want an external tool client to call bounded numerical tools instead of raw shell commands.

Use the API when another process needs a stable HTTP boundary over the same workflows.

Use the profile-table report workflow when you want the clearest first proof that the engine is useful: it inspects the input table, explains the modal structure, compresses it, and writes a bundle you can verify afterward.

Use `guide_workflow(...)`, `spectral-packet-engine guide-workflow`, MCP `guide_workflow`, or `GET /workflow/guide` when you want the product to recommend the default path instead of choosing among surfaces and workflow fragments yourself.

For SQL-backed spectral jobs, the Python library, CLI, MCP, and API all expose the same three controls:

- `time_column`
- optional explicit `position_columns`
- optional `sort_by_time`

Python users can inspect artifact bundles directly with `inspect_artifact_directory(...)`, while MCP and API expose the same report through `list_artifacts` and `GET /artifacts`.

## Validation And Trust Signals

The repository now includes:

- engine-level tests,
- end-to-end golden-path tests across Python, CLI, API, and MCP,
- a local executable release gate,
- packaging checks with `build` and `twine`,
- machine capability inspection for PyTorch, JAX, TensorFlow, API, and MCP support,
- service status reporting for API and MCP execution.

Release evidence and remaining limits are documented in [docs/release-readiness-audit.md](docs/release-readiness-audit.md).

## Current Honest Limits

- remote SQL backends are still beta,
- JAX is supported but not the primary Windows target,
- TensorFlow remains a compatibility path, not the primary ML story,
- Linux and macOS are the first-class operational targets for supervised MCP deployments,
- Windows remains a best-effort local stdio MCP target with clear degradation instead of a supervised runtime promise,
- fully clean cross-platform validation still needs broader Linux and Windows evidence.

## Documentation

- [docs/getting-started.md](docs/getting-started.md): shortest validated path from install to artifact output
- [docs/architecture.md](docs/architecture.md): how the engine, workflow, data, and service layers fit together
- [docs/workflows.md](docs/workflows.md): canonical workflows and outputs
- [docs/mcp.md](docs/mcp.md): MCP setup and runtime model
- [docs/api.md](docs/api.md): API setup, health/status, and service usage
- [docs/platforms.md](docs/platforms.md): platform and backend notes
- [docs/release-readiness-audit.md](docs/release-readiness-audit.md): validated truth and remaining limits

Legacy manuscript material is retained under [docs/legacy/README.md](docs/legacy/README.md) and is intentionally outside the main product path.
