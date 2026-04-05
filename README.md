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

The physics core goes deep: eigensolver for arbitrary potentials, split-operator propagation, Wigner phase-space distributions, density matrix formalism, Green’s functions, perturbation theory, WKB semiclassical analysis, operator algebra, symplectic integrators, spectral zeta functions, quantum scattering, Berry geometric phases, and quantum information measures — all built on PyTorch tensors and exposed through every surface.

## Why Use It

Use this project when you need a real reusable engine for one or more of these jobs:

- simulate a localized packet in a bounded 1D domain,
- solve the Schrödinger eigenvalue problem for arbitrary potentials (harmonic, double-well, Morse, Pöschl-Teller, or custom),
- propagate wavefunctions through potentials with split-operator time evolution,
- compute Wigner quasi-probability distributions for phase-space analysis,
- analyze density matrices, von Neumann entropy, fidelity, and purity,
- compute Green's functions, local density of states, and spectral functions,
- apply perturbation theory (1st/2nd order, degenerate) to modified potentials,
- estimate tunneling probabilities and Bohr-Sommerfeld quantization via WKB,
- evaluate commutators, generalized uncertainty relations, and operator algebra,
- run symplectic integrators (Verlet, Forest-Ruth, Yoshida) with energy conservation guarantees,
- compute spectral zeta functions, heat kernels, partition functions, and Casimir energies,
- analyze quantum scattering (transfer matrix, S-matrix, resonance detection),
- compute Berry geometric phases, Berry curvature, and Chern numbers,
- evaluate quantum Fisher information, entanglement entropy, concurrence, and quantum channels,
- project wavefunctions or observed profiles into a modal basis,
- quantify truncation error and observable behavior,
- fit Gaussian packet parameters back from observed densities,
- compress profile tables into modal coefficients,
- export traceable spectral feature tables for downstream supervised learning,
- train or tune tree models on those spectral features with explicit backend/runtime checks,
- move file-backed or SQL-backed data into the same spectral workflows,
- expose those workflows through Python, CLI, MCP, or an HTTP boundary without rewriting the math,
- keep SQL-to-profile-table conversion explicit instead of hiding schema assumptions in wrappers,
- model server request patterns using spectral decomposition for intelligent load analysis,
- compute adaptive throttling parameters (cooldown, interval, concurrency) from the spectral structure of traffic,
- detect anomalous load by comparing spectral signatures against a known-good baseline.

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
| Advanced physics modules | Eigensolver, split-operator, Wigner, density matrix, Green's function, perturbation theory, WKB, operator algebra, symplectic, spectral zeta, scattering, Berry phase, quantum info | Stable |
| MCP server (60+ tools) | Structured machine-side access for external tool clients over stdio with bounded in-process execution | Beta |
| HTTP API | Optional local or self-hosted JSON service | Beta |
| Remote SQL backends | SQLAlchemy-backed non-SQLite access | Beta |
| Spectral feature export | Reusable feature tables with explicit schema and provenance | Beta |
| Tree-model workflows | Shared sklearn-first train/tune workflows over spectral features | Beta |
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
- `ml-tree-core`: sklearn-based tree-model workflows
- `ml-xgboost`: optional XGBoost backend for tree-model workflows
- `ml-lightgbm`: optional LightGBM backend for tree-model workflows
- `ml-catboost`: optional CatBoost backend for tree-model workflows
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
python3 -m pip install -e ".[ml-tree-core]"
python3 -m pip install -e ".[ml-tree-core,ml-xgboost]"
python3 -m pip install -e ".[ml-jax]"
```

## MCP Runtime Quickstart

Install the optional MCP runtime, inspect the environment, then start the stdio server:

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine inspect-product
spectral-packet-engine inspect-environment --device cpu
spectral-packet-engine serve-mcp --max-concurrent-tasks 2 --log-level warning
```

For a restartable long-running MCP endpoint, expose streamable HTTP instead of stdio:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log
```

### Connecting an AI Agent

The repository includes `.mcp.json` at the root for local source-checkout use. Clients that support repo-local MCP discovery can use it directly, and other MCP clients can import the same command block manually.

For the hosted shared deployment, the default user endpoint is `https://lightcap.ai/mcp` over HTTPS.

Local stdio MCP needs no IP address:

- the client launches `python -m spectral_packet_engine serve-mcp`,
- the spectral engine runs in-process on that machine,
- the AI client calls tools instead of writing Python.

Important distinction:

- `https://lightcap.ai/mcp` is the published public MCP URL for users and AI clients.
- The streamable-HTTP command examples below describe a self-hosted private origin listener, not the public client URL.
- If you publish your own deployment, keep that private origin behind a tunnel or reverse proxy unless you intentionally want a wider attack surface.

For Claude Desktop, VS Code, Cursor, or another client that wants an MCP config block, generate it instead of hand-writing it:

```bash
spectral-packet-engine generate-mcp-config --transport stdio
```

Paste that JSON into the client's MCP config file such as `claude_desktop_config.json` or `.vscode/mcp.json`.

For remote connections (e.g., deployed server):

```json
{
  "mcpServers": {
    "spectral-packet-engine": {
      "command": "ssh",
      "args": ["user@example-host", "cd /srv/spectral-packet-engine && .venv/bin/python -m spectral_packet_engine serve-mcp --max-concurrent-tasks 2 --log-level warning"]
    }
  }
}
```

Generate that block with:

```bash
spectral-packet-engine generate-mcp-config --transport ssh --host user@example-host --remote-cwd /srv/spectral-packet-engine
```

For a restartable remote deployment over streamable HTTP, keep the service bound on the remote machine and tunnel to it:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log

spectral-packet-engine plan-mcp-tunnel \
  --host user@example-host \
  --local-port 8765 \
  --remote-port 8765 \
  --streamable-http-path /mcp
```

Use the reported `local_endpoint_url` with an HTTP-capable MCP client. For the shared hosted deployment, connect directly to `https://lightcap.ai/mcp`.

Public deployment note:

- the published public MCP endpoint is `https://lightcap.ai/mcp`,
- users should connect over HTTPS only; plain HTTP is not part of the public contract,
- external verification on April 5, 2026 confirmed that `https://lightcap.ai/mcp` is reachable and initializes as a real MCP endpoint,
- a raw request without the MCP session flow now returns a protocol error instead of a site `404`, which confirms the request is reaching the MCP server,
- a real MCP client initialize succeeds against `https://lightcap.ai/mcp` and currently discovers `79` tools,
- a public `probe_mcp_runtime(profile="audit")` run passed `15` probes with `0` failures and `0` bugs found.

After connecting, the agent should call `self_test`, then `server_info` to inspect the authoritative connection facts:

- `transport`
- `bind_host`
- `bind_port`
- `streamable_http_path`
- `endpoint_url`
- `allowed_hosts`
- `allowed_origins`

`best_effort_ipv4` is observational only. `endpoint_url` is the internal listener URL on the server host. The published public MCP route is `https://lightcap.ai/mcp`.

### What the MCP Server Offers

| Category | Highlights |
| --- | --- |
| Environment & Status | `self_test`, `server_info`, `probe_mcp_runtime`, `inspect_product` |
| Profile & Spectral | `profile_table_report`, `compress_profile_table`, `export_feature_table` |
| Analysis Pipelines | `analyze_quantum_state_pipeline`, `analyze_potential_pipeline` |
| Advanced Physics | `solve_eigenproblem`, `split_operator_propagate`, `scattering_analysis` |
| Load Modeling | `analyze_server_load`, `compute_adaptive_throttle`, `detect_load_anomaly` |
| Code Execution | `execute_python` (trusted opt-in only), `create_scratch_database`, `demo_spectral_pipeline` |
| SQL Workflows | `bootstrap_database`, `query_database`, `execute_database_script`, `report_database_profile_query` |
| Tree Models | `train_tree_model`, `tune_tree_model`, `inspect_tree_backends` |

### The Showcase Prompt

Give this prompt to an AI agent connected to the MCP server to demonstrate the full engine:

> 1. Run `self_test` to verify the server.
> 2. Solve the quantum harmonic oscillator for 20 eigenvalues and compare with exact E_n = (n+1/2).
> 3. Propagate a wavepacket through a double-well potential with split-operator time evolution. Report norm conservation.
> 4. Compute the Wigner function of the ground state and check for non-classical negativity.
> 5. Generate 100 synthetic profiles, analyze convergence, compress them, and report quality.
> 6. Export spectral features to a scratch SQL database, train a tree model, and show feature importances.
> 7. Analyze mock server traffic (1000 request timestamps with burst patterns) using spectral load modeling. Report the regime and recommended throttling.
> 8. Write all findings to an audit log.

This exercises eigenvalue solver, time propagation, phase-space analysis, spectral compression, SQL, tree models, load modeling, and audit logging in one session — all through MCP tool calls, no external code needed.

For a one-call deeper physics workflow, use `tunneling_experiment`. It now keeps the orchestration server-side:

- packet energy windows come from the packet's own spectral decomposition,
- exact and WKB transmission are compared at both `V0/2` and the packet mean energy,
- propagation reports transmitted and reflected probability,
- the AI client does not have to assemble low-level physics calls manually.

### Quick One-Liners for AI Agents

- **"Generate synthetic profiles and analyze them"** — agent calls `generate_synthetic_profiles` then `analyze_spectral_profile_pipeline`
- **"What's the optimal throttling for 80 req/s traffic?"** — agent calls `analyze_server_load`
- **"Solve Schrodinger for a Morse potential"** — agent calls `solve_eigenproblem`, or `execute_python` only on a trusted runtime started with `--allow-unsafe-python`
- **"Create a scratch database, populate it, and run a SQL query"** — agent calls `create_scratch_database`, then `execute_database_script` or `execute_database_statement`, then `query_database`
- **"Run any Python code with the library"** — only on a trusted local runtime intentionally started with `--allow-unsafe-python`

See [docs/mcp-usage-guide.md](docs/mcp-usage-guide.md) for detailed session examples.

### Operational Rules

- stdout is reserved for MCP protocol traffic,
- repository-managed logs go to stderr by default or to `--log-file`,
- heavy tools run under bounded in-process execution and fail clearly when saturated,
- artifact bundles are written atomically and marked complete only when `artifacts.json` is committed,
- `execute_python` is disabled by default and must be explicitly enabled for trusted local sessions,
- restartable MCP deployment should use `--transport streamable-http` together with `install-mcp-service`, `systemd`, `launchd`, Docker restart policy, or an equivalent external supervisor.

Important architectural boundary:

- the Python library remains an explicit in-process library,
- MCP is the machine-facing layer for external AI clients,
- `pip install` does not silently reroute Python API calls to a remote default server.

### Probe And Service Ops

Run the built-in MCP probe suite against the real protocol surface:

```bash
spectral-packet-engine probe-mcp --profile smoke --output-dir artifacts/mcp_probe
spectral-packet-engine probe-mcp --profile audit --output-dir artifacts/mcp_probe_audit
python3 scripts/mcp_stress_test.py --profile stress --output-dir artifacts/mcp_stress
spectral-packet-engine inspect-artifacts artifacts/mcp_probe
```

That writes:

- `mcp_probe_report.json`
- `mcp_probe_results.jsonl`
- `mcp_tool_calls.jsonl`
- `mcp_probe_summary.md`
- `server.log`
- `artifacts.json`

Install an auto-restarting user service for streamable HTTP MCP:

```bash
spectral-packet-engine install-mcp-service --dry-run
spectral-packet-engine install-mcp-service --yes --enable
```

Render the SSH tunnel plan for a remote streamable-HTTP deployment:

```bash
spectral-packet-engine plan-mcp-tunnel --host user@remote-host --remote-port 8765 --streamable-http-path /mcp
```

Generate a client config block instead of editing Claude or VS Code MCP JSON by hand:

```bash
spectral-packet-engine generate-mcp-config --transport stdio
spectral-packet-engine generate-mcp-config --transport ssh --host user@remote-host --remote-cwd /srv/spectral-packet-engine
```

## First Things To Try

- `spectral-packet-engine validate-install --device cpu`
- `spectral-packet-engine guide-workflow`
- `spectral-packet-engine inspect-product`
- `spectral-packet-engine inspect-environment --device cpu`
- `spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report`
- `spectral-packet-engine inspect-artifacts artifacts/profile_report`
- `spectral-packet-engine release-gate --device cpu`
- `spectral-packet-engine ml-backends --device cpu`
- `spectral-packet-engine tree-backends`
- `spectral-packet-engine export-features examples/data/synthetic_profiles.csv --modes 16 --device cpu --output-dir artifacts/features`

## Tree Models Over Spectral Features

The tree-model path stays subordinate to the spectral engine:

`CSV profile table -> modal coefficients -> feature table with schema/provenance -> supervised tree model`

Status: beta. The shared feature export, training, and tuning workflows are integrated across Python, CLI, MCP, and API, but the scikit-learn baseline is the most directly validated path in this release.

Minimal flow:

```bash
python3 -m pip install -e ".[ml-tree-core,ml-xgboost]"

spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report
spectral-packet-engine export-features examples/data/synthetic_profiles.csv --modes 16 --device cpu --output-dir artifacts/features

# Add or join your supervised label column into artifacts/features/features.csv first.
spectral-packet-engine tree-train artifacts/features_with_target.csv \
  --target-column target \
  --library xgboost \
  --params '{"n_estimators": 256, "max_depth": 6, "learning_rate": 0.05}' \
  --output-dir artifacts/tree_train
```

What each stage gives you:

- `artifacts/profile_report/compression/coefficients.csv`: raw modal coefficients from the report-first workflow
- `artifacts/features/features.csv`: reusable feature table for downstream models
- `artifacts/features/features_schema.json`: ordered columns, dtypes, semantic meaning, ordering policy, and library versions
- `artifacts/tree_train/tree_training.json`: train/test metrics, config, and persisted model provenance

Use `export-features` instead of treating `coefficients.csv` as the final model input contract. The dedicated feature workflow records normalization, mode count, input kind, and ordering policy explicitly.

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
- tree-model feature export, training, and tuning remain beta; the scikit-learn path is the most directly validated backend in this release,
- XGBoost, LightGBM, and CatBoost remain optional backend-specific beta paths and still need broader clean-environment validation,
- JAX is supported but not the primary Windows target,
- TensorFlow remains a compatibility path, not the primary ML story,
- Linux and macOS are the first-class operational targets for supervised MCP deployments,
- Windows remains a best-effort local stdio MCP target with clear degradation instead of a supervised runtime promise,
- fully clean cross-platform validation still needs broader Linux and Windows evidence.

## Analysis Pipelines

The library provides high-level analysis pipelines that chain multiple functions automatically. No parameter guessing — the library uses its own convergence diagnostics, truncation recommendations, and spectral analysis to determine everything from the input data.

| Pipeline | Input | What It Does |
| --- | --- | --- |
| `analyze_quantum_state` | Spectral coefficients + basis | Energy, momentum, uncertainty, entropy, convergence, Wigner — one call |
| `analyze_potential_landscape` | V(x) function + domain | Eigenvalues, WKB comparison, thermodynamics, zeta function, Weyl law |
| `analyze_scattering_system` | Potential segments | T(E), resonances, S-matrix, WKB tunneling — energy range auto-determined |
| `analyze_spectral_profile` | Position grid + profiles | Auto mode count, convergence, energy budget, compression quality, Gibbs |
| `compare_quantum_states` | Two coefficient vectors | Fidelity, trace distance, energy/momentum differences |

Example — one-call complete state analysis:

```python
from spectral_packet_engine import (
    analyze_quantum_state, InfiniteWell1D, InfiniteWellBasis,
    make_truncated_gaussian_packet, StateProjector,
)
import torch

domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
basis = InfiniteWellBasis(domain, 64)
projector = StateProjector(basis)
packet = make_truncated_gaussian_packet(domain, center=0.3, width=0.07, wavenumber=25.0)
state = projector.project_packet(packet)

report = analyze_quantum_state(state.coefficients, basis)
print(f"Recommended modes: {report.recommended_truncation}")
print(f"Uncertainty product: {report.uncertainty_product:.4f}")
print(f"Spectral decay: {report.spectral_decay_type}")
```

Example — analyze a profile CSV without knowing any parameters:

```python
from spectral_packet_engine import analyze_spectral_profile, load_profile_table
import torch

table = load_profile_table("profiles.csv")
report = analyze_spectral_profile(table.grid, table.profiles)
print(f"Auto-determined modes: {report.auto_num_modes}")
print(f"Mean L2 error: {report.mean_relative_l2_error:.6f}")
print(f"99% capture: {report.recommended_modes_for_99pct} modes")
```

The same pipelines are available as MCP tools (`analyze_quantum_state_pipeline`, `analyze_potential_pipeline`, `analyze_scattering_pipeline`, `analyze_spectral_profile_pipeline`, `compare_quantum_states_pipeline`) so an AI agent can run complete analyses without knowing the internal function chain.

## Spectral Load Modeling

The engine uses its own spectral decomposition machinery for infrastructure optimization. Server request rate r(t) over a time window [0, T] is a signal on a bounded domain — exactly what the engine was built to decompose. No hardcoded thresholds — the engine's convergence diagnostics and spectral entropy determine everything.

| Function | What It Does |
| --- | --- |
| `analyze_request_load` | End-to-end: timestamps → decomposition → classification → throttle → capacity |
| `decompose_load_signal` | Project request rate onto spectral basis, get per-mode energy fractions |
| `analyze_load_spectrum` | Classify traffic via spectral decay (smooth/bursty/anomalous) |
| `compute_adaptive_throttle` | Derive cooldown, interval, concurrency from spectral structure |
| `detect_load_anomaly` | Compare current spectrum to baseline via spectral distance and JSD |
| `estimate_capacity` | Separate sustained load (low modes) from burst spikes (high modes) |

**CLI:**
```bash
spectral-packet-engine load-analyze timestamps.txt --capacity 100
spectral-packet-engine load-throttle rates.txt --capacity 100 --modes 64
spectral-packet-engine load-capacity rates.txt --modes 64
```

**MCP tools:** `analyze_server_load`, `decompose_load_signal`, `compute_adaptive_throttle`, `detect_load_anomaly`, `estimate_server_capacity`

Example — spectral-derived adaptive throttling:

```python
from spectral_packet_engine import analyze_request_load

report = analyze_request_load(timestamps, capacity_rps=100.0)
print(f"Regime: {report.throttle.regime}")  # smooth, bursty, saturated, anomalous
print(f"Cooldown: {report.throttle.recommended_cooldown_seconds:.1f}s")
print(f"Max concurrent: {report.throttle.recommended_max_concurrent}")
print(f"Sustained RPS: {report.capacity.sustained_rps:.1f}")
```

## Advanced Physics Modules

The engine includes deep mathematical physics beyond basic spectral decomposition. These modules are available through Python, MCP tools, and the CLI:

| Module | Domain | Key Capabilities |
| --- | --- | --- |
| `eigensolver` | Quantum eigenvalue problems | Chebyshev collocation for arbitrary V(x); harmonic, double-well, Morse, Pöschl-Teller potentials |
| `split_operator` | Time-dependent Schrödinger | 2nd/4th-order Trotter-Suzuki propagation with norm/energy conservation |
| `wigner` | Phase-space quantum mechanics | Wigner quasi-probability W(x,p), marginals, negativity witness |
| `density_matrix` | Mixed quantum states | Pure/mixed/thermal states, von Neumann entropy, fidelity, trace distance, partial trace |
| `greens_function` | Spectral propagators | Retarded Green's function, LDOS, spectral function, DOS, free propagator |
| `perturbation` | Perturbation theory | 1st/2nd order energy/state corrections, degenerate perturbation theory |
| `semiclassical` | WKB methods | Bohr-Sommerfeld quantization, tunneling probability, connection formulas |
| `operator_algebra` | Quantum operator calculus | Commutators, generalized uncertainty, BCH expansion, ladder operators |
| `symplectic` | Hamiltonian dynamics | Störmer-Verlet, Forest-Ruth, Yoshida 4th/6th order integrators |
| `spectral_zeta` | Mathematical physics | Spectral zeta, heat kernel, partition function, Weyl law, Casimir energy |
| `scattering` | Quantum scattering | Transfer matrix, T/R coefficients, S-matrix, resonance detection |
| `berry_phase` | Geometric phases | Berry phase, Berry curvature, Chern numbers, adiabatic evolution |
| `quantum_info` | Quantum information | Fisher information, entanglement entropy, concurrence, quantum channels |

Example — solve a double-well potential and analyze the energy spectrum:

```python
from spectral_packet_engine import (
    solve_eigenproblem, double_well_potential, InfiniteWell1D,
    compute_wigner, analyze_density_matrix, pure_state_density_matrix,
)
import torch

domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
result = solve_eigenproblem(
    lambda x: double_well_potential(x, a_param=2000.0, b_param=100.0, domain=domain),
    domain, num_points=256, num_states=20,
)
print(result.eigenvalues[:10])  # First 10 energy levels
```

Example — complete tunneling experiment in one call:

```python
from spectral_packet_engine import analyze_tunneling

report = analyze_tunneling(barrier_height=50.0, barrier_width_sigma=0.03, device="cpu")
print(f"Packet mean energy: {report.packet_mean_energy:.6f}")
print(f"T(E_packet) exact:  {report.transmission_at_packet_energy:.6e}")
print(f"T(E_packet) WKB:    {report.wkb_transmission_at_packet_energy:.6e}")
print(f"Transmitted mass:   {report.transmitted_probability:.6f}")
print(f"Reflected mass:     {report.reflected_probability:.6f}")
```

Or from the CLI:

```bash
python examples/tunneling_experiment.py --barrier-height 50 --grid-points 512
```

**Example CPU output:**

```
Barrier FWHM window: [0.4646, 0.5354]
Packet mean energy: 878.125006
T(E_packet) exact:  9.999940e-01
T(E_packet) WKB:    1.000000e+00
Transmitted mass:   0.000000
Reflected mass:     0.999995
```

The same experiment via MCP: call `tunneling_experiment` with the same parameters. Via `execute_python`: only use the two-line Python snippet above on a trusted runtime that was started with `--allow-unsafe-python`.

## License

This project is licensed under the **Spectral Packet Engine Reciprocal Public License v1.0** — see the [LICENSE](LICENSE) file for the full text.

Key terms:
- **Open source with copyleft**: you may use, modify, and distribute the software, but derivative works and network-service deployments must be released under this same license.
- **Mandatory attribution**: all uses must credit the original project and author.
- **Contribution-back**: modifications must be contributed back to the upstream repository within 90 days.
- **Patent grant with retaliation**: contributors grant a patent license that terminates if you initiate patent litigation against the project.
- **Anti-misappropriation**: creating substantially similar works without attribution or license compliance is prohibited.
- **Donation encouragement**: commercial users are encouraged to support the project's sustainability.

The license is designed to be enforceable under United States (Title 17 USC), European Union (Directives 2001/29/EC, 2004/48/EC), and Republic of Turkey (FSEK, Berne Convention) law.

## Documentation

- [docs/getting-started.md](docs/getting-started.md): shortest validated path from install to artifact output
- [docs/architecture.md](docs/architecture.md): how the engine, workflow, data, and service layers fit together
- [docs/workflows.md](docs/workflows.md): canonical workflows and outputs
- [docs/mcp.md](docs/mcp.md): MCP setup and runtime model
- [docs/mcp-usage-guide.md](docs/mcp-usage-guide.md): detailed MCP usage sessions, example prompts, connection guide
- [docs/mcp-operations.md](docs/mcp-operations.md): restart, tunnel, self-audit, and stress-test operator playbook
- [docs/api.md](docs/api.md): API setup, health/status, and service usage
- [docs/platforms.md](docs/platforms.md): platform and backend notes
- [docs/release-readiness-audit.md](docs/release-readiness-audit.md): validated truth and remaining limits
- [docs/internal/advanced-physics-review-2026-04.md](docs/internal/advanced-physics-review-2026-04.md): TUM×RWTH physics depth review

Legacy manuscript material is retained under [docs/legacy/README.md](docs/legacy/README.md) and is intentionally outside the main product path.
