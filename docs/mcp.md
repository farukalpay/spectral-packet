# MCP Usage

## Purpose

The MCP surface exists so an external tool client can call the same bounded-domain engine through explicit tools instead of shelling into the machine.

It is useful when you want:

- structured numerical tools,
- stable artifact bundles,
- explicit environment and health inspection,
- the same workflows the Python library and CLI already use.

It is not a separate product. MCP is a machine-facing wrapper over the shared workflow layer.

## Install

```bash
python3 -m pip install -e ".[mcp]"
```

Validate the local environment before wiring a client to the server:

```bash
spectral-packet-engine inspect-product
spectral-packet-engine inspect-environment --device cpu
spectral-packet-engine validate-install --device cpu
```

## Start

```bash
spectral-packet-engine serve-mcp
```

The default server runs over stdio, which is the normal local-machine MCP transport.

Useful runtime controls:

```bash
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --slot-timeout-seconds 60 --log-level warning
spectral-packet-engine serve-mcp --log-file logs/mcp.log
spectral-packet-engine serve-mcp --transport streamable-http --port 8765 --streamable-http-path /mcp
```

Important rule: stdout is reserved for MCP protocol messages. Repository-managed logs go to stderr by default or to the log file you configure.

Minimal `.mcp.json` example:

```json
{
  "mcpServers": {
    "spectral-packet-engine": {
      "command": "spectral-packet-engine",
      "args": ["serve-mcp", "--max-concurrent-tasks", "1", "--log-level", "warning"]
    }
  }
}
```

Local stdio MCP is the simplest AI-client path:

- no bind IP is needed,
- the client launches the server process on demand,
- the engine runs in-process on the same machine.

For a remote always-on deployment, prefer streamable HTTP plus an external supervisor, then tunnel or otherwise expose the endpoint explicitly.

For the shared hosted deployment, the public MCP endpoint is [https://lightcap.ai/mcp](https://lightcap.ai/mcp).

## Core Tool Families

- environment and install inspection
- service status inspection
- file-format and tabular capability inspection
- profile-table report, analysis, compression, comparison, and inverse fitting
- spectral feature export plus tree-model training and tuning
- bounded-domain packet simulation and modal projection
- SQLite bootstrap, query, materialization, and SQL-backed report/analyze/compress/fit workflows
- backend-aware modal-surrogate training and evaluation
- spectral load modeling, anomaly detection, adaptive throttling, and capacity estimation

### Analysis Pipelines (auto-parameterized, one-call)

These are the highest-value MCP tools for data engineers and AI agents who don't want to specify every parameter. Each pipeline uses the library's own diagnostics to determine mode counts, energy ranges, and convergence thresholds automatically.

| Tool | Input | What It Does |
| --- | --- | --- |
| `analyze_quantum_state_pipeline` | Packet params | Complete state report: energy, momentum, uncertainty, entropy, convergence, Wigner |
| `analyze_potential_pipeline` | Potential name | Eigenvalues, WKB comparison, thermodynamics, spectral zeta, Weyl law |
| `analyze_scattering_pipeline` | Barrier params | T(E), resonances, S-matrix, WKB tunneling — auto energy range |
| `analyze_spectral_profile_pipeline` | CSV path | Auto mode count, convergence, compression quality, Gibbs detection |
| `compare_quantum_states_pipeline` | Two packet params | Fidelity, trace distance, energy/momentum differences |

An AI client can say "analyze this profile CSV" and receive a complete structured report without knowing any internal function names or parameter conventions.

### Advanced Physics Tools (60+ tools total)

These tools expose the deep physics modules directly to AI clients:

| Tool | Physics | Key Output |
| --- | --- | --- |
| `solve_eigenproblem` | Schrödinger eigenvalue for arbitrary V(x) | Eigenvalues, eigenstates, orthonormality check |
| `split_operator_propagate` | Time-dependent wavepacket propagation | Density evolution, norm/energy conservation |
| `compute_wigner_function` | Phase-space Wigner distribution | Negativity, marginals, non-classicality |
| `analyze_density_matrix` | Quantum state characterization | Purity, von Neumann entropy, rank |
| `compute_greens_function` | Spectral Green's function | LDOS, DOS, spectral function |
| `perturbation_analysis` | Quantum perturbation theory | Energy corrections, convergence parameter |
| `wkb_analysis` | Semiclassical WKB | Bohr-Sommerfeld energies, tunneling T |
| `operator_commutator` | Operator algebra | [X,P], BCH expansion |
| `symplectic_propagation` | Hamiltonian dynamics | Phase-space trajectory, energy drift |
| `spectral_zeta_analysis` | Mathematical physics | ζ_H(s), partition function, Weyl law, Casimir |
| `scattering_analysis` | Quantum scattering | T(E), R(E), resonances, S-matrix |
| `berry_phase_analysis` | Geometric phases | Berry phase, solid angle validation |
| `quantum_info_analysis` | Quantum information | Fisher info, entanglement, concurrence, channels |
| `write_audit_log` | Development audit | Structured docs/internal/ entries |

### Spectral Load Modeling Tools

These tools use the spectral engine itself for infrastructure optimization — request rate r(t) is a bounded-domain signal, decomposed and classified using the same convergence diagnostics and spectral entropy the physics tools use.

| Tool | Input | What It Does |
| --- | --- | --- |
| `analyze_server_load` | Timestamps + capacity | Full pipeline: decomposition, classification, throttle, capacity, anomaly |
| `decompose_load_signal` | Rate values | Spectral coefficients and per-mode energy fractions |
| `compute_adaptive_throttle` | Rate values + capacity | Cooldown, interval, concurrency — derived from spectral structure |
| `detect_load_anomaly` | Current + baseline rates | Spectral distance, JSD, entropy shift — no hardcoded thresholds |
| `estimate_server_capacity` | Rate values | Sustained vs peak RPS via low/high frequency mode separation |

An AI agent can call `analyze_server_load` with raw timestamps and receive a complete traffic classification, throttle recommendation, and capacity estimate — the engine determines everything from the spectral structure.

### Code Execution, Scratch, And Runtime Audit

These tools let AI agents manage scratch data, run trusted-only local Python when explicitly enabled, and ask the server to probe its own MCP surface.

| Tool | What It Does |
| --- | --- |
| `execute_python` | Trusted opt-in only. When enabled at startup, run a Python snippet with `spe`, `torch`, and `np` pre-imported. Assign to `result` for structured output. |
| `create_scratch_database` | Create a temporary SQLite DB, optionally pre-populated with synthetic profiles. |
| `generate_synthetic_profiles` | Generate Gaussian density profiles as CSV or SQLite. |
| `demo_spectral_pipeline` | One-call end-to-end demo: generate → project → analyze → compress → report. |
| `probe_mcp_runtime` | Launch a child MCP server and run the built-in self-probe suite through the real MCP client. Supports `profile=smoke|stress|audit`. |
| `self_test` | Validate all engine subsystems (core, physics, load, SQL, scratch directory). |
| `server_info` | Return hostname plus authoritative bind/endpoint facts, version, scratch directory, and runtime config. |

The `execute_python` tool is intentionally disabled by default. It becomes available only when the operator starts the server with `--allow-unsafe-python` for a trusted local session. The `scratch_dir` variable is pre-set to the managed MCP scratch directory reported by `server_info`.

For detailed usage sessions and example prompts, see [mcp-usage-guide.md](mcp-usage-guide.md).

## Runtime Trust Signals

The MCP server exposes the same runtime model as the rest of the product:

- `inspect_product` for the shared product spine, runtime spine, and workflow map
- `guide_workflow` for the recommended high-value loop and defaults for file-backed or SQL-backed work
- `inspect_environment` for machine capability inspection
- `inspect_tree_backends` for sklearn-first and optional boosted-tree backend availability
- `inspect_mcp_runtime` for transport, platform, bounded-execution, and logging policy inspection
- `validate_installation` for surface readiness
- `inspect_service_status` for uptime, task counters, and recent runs with canonical `workflow_id` plus raw `surface_action`

That lets an MCP client ask:

- is this machine ready,
- which workflow should I run instead of inventing a tool chain,
- how is the MCP runtime configured,
- what backends are installed,
- what just ran,
- where did artifacts go.

## Runtime Model

The repository guarantees a narrow in-process runtime model for MCP:

- stdio transport for local client-launched sessions, or streamable HTTP for supervised network service mode,
- repository-managed logging on stderr or an explicit file,
- bounded in-process execution slots for compute-heavy tools,
- shared workflow execution with the same spectral engine used by Python, CLI, and API,
- atomic artifact writes and output-directory locking for artifact-producing tools.

The repository does not promise:

- forced in-process cancellation of arbitrary running numerical work,
- automatic restarts inside the repository process itself,
- distributed scheduling,
- a generic job-queue platform.

If a heavy tool cannot acquire an execution slot before the configured timeout, it fails clearly so the client can retry or reduce concurrency expectations.

## Artifacts

Most compute tools can write artifact bundles when `output_dir` is provided.

Typical outputs include:

- JSON summaries
- top-level profile report overviews
- reconstructed profile tables
- coefficient tables
- feature tables with explicit schema
- predicted moments
- tree-model summaries, predictions, and persisted best-model bundles
- artifact indexes

Artifact-writing behavior is shared with the CLI and API instead of being reimplemented inside the MCP layer.

Artifact directories are restart-safe at the repository level:

- data files are staged and atomically moved into place,
- artifact directories are locked during writes to avoid cross-request collisions,
- temporary runtime files are cleaned up on the next managed write,
- a bundle is only considered complete when `artifacts.json` reports `"status": "complete"`.

Use `list_artifacts` to inspect:

- whether the directory exists,
- whether the bundle is complete,
- which files are present,
- what metadata was recorded.

For SQL-backed profile-table tools, MCP exposes the same explicit controls as Python and the CLI:

- `time_column`
- optional `position_columns`
- optional `sort_by_time`

Artifact indexes record the database URL, query text, bound parameters, and the profile-table materialization policy when those SQL-backed tools write outputs.

For the strongest bounded day-1 workflow through MCP, prefer:

- `inspect_product` before first use if the client needs the product and workflow map
- `profile_table_report` for a file-backed table
- `report_database_profile_query` for a SQL-backed table
- `inspect_tree_backends` before feature-model work
- `export_feature_table` before `train_tree_model` or `tune_tree_model`
- `list_artifacts` after the run if you asked the tool to write outputs

## Feature And Tree Tools

The feature-model tool chain is intentionally small and explicit:

Status: beta. The bounded MCP wrappers are real and validated, but the scikit-learn baseline remains the most directly exercised backend path in this release.

- `inspect_tree_backends`
  Arguments:
  `library` with values such as `auto`, `sklearn`, `xgboost`, `lightgbm`, or `catboost`
- `export_feature_table`
  Arguments:
  `table_path`, `num_modes`, `device`, `normalize_each_profile`, `include`, `format`, `output_dir`
- `export_feature_table_from_sql`
  Arguments:
  `database`, `query`, optional `parameters`, `time_column`, optional `position_columns`, `sort_by_time`, `num_modes`, `device`, `normalize_each_profile`, `include`, `format`, `output_dir`
- `train_tree_model`
  Arguments:
  `features_path`, `target_column`, optional `feature_columns`, `task`, `library`, optional `model`, optional `params`, `test_fraction`, `random_state`, optional `export_dir`, optional `output_dir`
- `tune_tree_model`
  Arguments:
  `features_path`, `target_column`, `search_space`, optional `feature_columns`, `task`, `library`, optional `model`, `search_kind`, `n_iter`, `cv`, optional `scoring`, `test_fraction`, `random_state`, optional `export_dir`, optional `output_dir`

Example tool sequence:

1. `inspect_tree_backends` to see which optional backends are actually available.
2. `export_feature_table` or `export_feature_table_from_sql` to write `features.csv` or `features.parquet` plus `features_schema.json`.
3. Add or join the supervised target column outside the MCP server if your labels live elsewhere.
4. `train_tree_model` for one explicit run or `tune_tree_model` for search-driven selection.

The feature export artifacts record input kind, normalization, mode count, ordering policy, and library versions so the training input contract is inspectable after the run.

## Deployment Guidance

For local use, stdio MCP is enough:

```bash
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-file logs/mcp.log
```

For restartable long-running use, switch to streamable HTTP first:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log
```

Then prefer an external service manager:

- Linux: `systemd` or a container restart policy
- macOS: `launchd`
- Windows: local stdio use is supported, but supervised long-running deployment is best-effort rather than a first-class promise

The process itself is restart-safe around artifact state, but restart policy belongs outside the repository process.

The repository includes an install helper that renders and optionally enables a user-level service manifest:

```bash
spectral-packet-engine install-mcp-service --dry-run
spectral-packet-engine install-mcp-service --yes --enable
```

Use `install-mcp-service --dry-run` as the source of truth for the actual manifest contents. The generated manifest is the supported contract; hand-written examples drift too easily.

For remote HTTP access through SSH tunneling, render the exact tunnel plan instead of hand-typing it:

```bash
spectral-packet-engine plan-mcp-tunnel \
  --host user@example-host \
  --local-port 8765 \
  --remote-port 8765 \
  --streamable-http-path /mcp
```

That prints:

- the exact `ssh -L ...` command,
- the local endpoint URL to hand to an HTTP-capable MCP client,
- the remote endpoint URL the tunnel targets.

Use `server_info` after connecting. Treat these fields as authoritative:

- `transport`
- `bind_host`
- `bind_port`
- `streamable_http_path`
- `endpoint_url`

Treat `best_effort_ipv4` as observational only.

## Troubleshooting

- If `serve-mcp` reports a missing module, install the `mcp` extra in the active environment.
- If a request fails with a saturation error, reduce concurrent heavy requests or increase `--max-concurrent-tasks` conservatively.
- If artifact creation reports that an output directory is busy, avoid sharing one `output_dir` across overlapping heavy jobs.
- If you need a reproducible health and hardening check, run `spectral-packet-engine probe-mcp --profile smoke --output-dir artifacts/mcp_probe`.
- If you need a deeper repeated-load audit, run `spectral-packet-engine probe-mcp --profile audit --output-dir artifacts/mcp_probe_audit` or `python3 scripts/mcp_stress_test.py --profile stress --output-dir artifacts/mcp_stress`.
- If you need logs, use stderr or `--log-file`; do not emit anything to stdout from wrapper code or shell glue around the MCP process.

## Status

MCP is currently a beta surface. Linux and macOS are the first-class operational targets for supervised MCP use. Windows local stdio use degrades gracefully, but it is not yet claimed as a first-class supervised deployment target. The Python library and CLI remain the most directly validated surfaces.
