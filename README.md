# Spectral Packet Engine

Spectral Packet Engine is a Python-first spectral inverse physics engine.

The repository is intentionally narrow:

`bounded-domain spectral core -> uncertainty-aware inference and controlled reductions -> coherent Python / CLI / MCP / API surfaces`

It is not a generic ML framework, not a grab-bag of physics demos, and not a notebook dump. The spectral engine remains the center of gravity. Inference, reduced models, differentiable optimization, and vertical workflows all sit on top of that same core.

## What It Does

The stable core already covers:

- bounded-domain packet construction, projection, propagation, and observables,
  including Gaussian and bounded plane-wave state preparation with explicit support diagnostics,
- modal decomposition, compression, and artifact-backed profile-table workflows,
- inverse Gaussian-packet fitting with local uncertainty summaries,
- arbitrary 1D potentials through the eigensolver and related physics modules.

This upgrade extends that core in four focused directions:

1. Inverse problems + uncertainty quantification
   - local posterior summaries for inverse packet fitting,
   - parameter confidence intervals,
   - identifiability, sensitivity, and observation-information maps,
   - posterior-predictive intervals over fitted densities or calibrated spectra,
   - explicit potential-family inference from observed spectra.
2. Controlled reduced models beyond plain 1D
   - separable tensor-product spectra,
   - phase-1 structured dimensional lift for separable 2D bounded problems,
   - near-separable, block-coupled, and low-rank coupling diagnostics over retained tensor bases,
   - reduced coupled-channel surfaces,
   - radial effective-coordinate reductions,
   - structured low-rank summaries.
3. Differentiable physics
   - gradient-based potential calibration,
   - transition-oriented inverse design,
   - differentiable packet-steering workflows.
4. Domain-specific vertical workflows
   - spectroscopy / family inference,
   - transport / barrier / resonance,
   - control / packet steering,
   - scientific tabular report-first inference workflows.

Two structural contracts keep this direction product-aligned rather than generic:

- `open-system measurement` for Lindblad evolution, finite-resolution instrument response, and inspectable measurement-noise summaries over explicit matrix or grid bases,
- `SpectralDataset` for grid metadata, units, uncertainty, regime splits, content hashes, and artifact lineage before data enter inverse, reduced, benchmark, or surrogate workflows.

## Product Shape

The intended product language is:

- `physics contracts` for shared PotentialFamily, HamiltonianOperator, BasisSpec, BoundaryCondition, ObservableSet, and MeasurementModel objects,
- `profile-table report` for evidence-first diagnostics,
- `inverse inference` for physical parameters plus uncertainty,
- `reduced models` when the physics has explicit structure,
- `open-system measurement` when real data includes decoherence, finite resolution, or measurement noise,
- `SpectralDataset` when data need grid metadata, units, uncertainty, splits, and artifact lineage,
- `differentiable design` when gradients are the right tool,
- `benchmark registry` for evidence-backed release and performance claims,
- `vertical workflows` when the user cares about spectroscopy, transport, control, or scientific tabular analysis end-to-end.

That same language now appears in:

- Python result objects,
- CLI commands,
- MCP tools/resources/prompts,
- artifact bundle names,
- product metadata and docs.

## Quickstart

```bash
python3 -m pip install -e .
spectral-packet-engine inspect-product
spectral-packet-engine validate-install --device cpu
spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report
spectral-packet-engine infer-potential-spectrum 5.22 15.83 26.41 --family harmonic --family double-well --device cpu
```

### Non-Canonical Profile Tables

Real-world tables do not always use numeric position headers. If your columns are semantic labels such as `h00`, `h01`, `h02`, keep that source schema explicit and provide the physical coordinate mapping at load time:

```bash
spectral-packet-engine inspect-profile-table data/semantic_profiles.csv \
  --time-column sample \
  --position-column h02 --position-value 1.0 \
  --position-column h00 --position-value 0.0 \
  --position-column h01 --position-value 0.5 \
  --sort-by-time
```

The same explicit `time_column`, repeated `position_column`, repeated `position_value`, and `sort_by_time` controls are available in the Python loader and across file-backed and SQL-backed CLI and MCP profile workflows.

### Direct Meteorology Ingest

If your upstream source is hourly weather data, fetch it directly into the shared tabular contract first and only materialize a profile table when you have an explicit coordinate mapping for the selected variables:

```bash
spectral-packet-engine ingest-open-meteo \
  --latitude 41.01 \
  --longitude 28.97 \
  --start-date 2026-04-08 \
  --end-date 2026-04-08 \
  --hourly temperature_2m \
  --hourly relative_humidity_2m \
  --output-path artifacts/weather.csv
```

If those hourly variables should enter the spectral workflow, keep the coordinate mapping explicit instead of renaming columns into a temporary schema:

```bash
spectral-packet-engine ingest-open-meteo \
  --latitude 41.01 \
  --longitude 28.97 \
  --start-date 2026-04-08 \
  --end-date 2026-04-08 \
  --hourly temperature_2m \
  --hourly relative_humidity_2m \
  --position-column relative_humidity_2m --position-value 1.0 \
  --position-column temperature_2m --position-value 0.0 \
  --profile-output artifacts/weather_profile.csv
```

### Urban Microclimate And Human Thermal Load

Hourly weather ingest stays separate from the city-scale closure layer. Once the tabular weather dataset exists, derive neutral boundary-layer exchange, radiative temperature, operative temperature, and net human thermal storage through an explicit mapping contract:

```bash
spectral-packet-engine derive-urban-microclimate artifacts/weather.csv \
  --mapping-profile open-meteo \
  --output-path artifacts/weather_microclimate.csv
```

This workflow does not guess column semantics from loose keywords. You either select an inspectable named profile such as `open-meteo` or pass the source columns explicitly.
If shortwave and longwave fluxes are missing, the radiative closure falls back transparently to air-temperature equivalence and reports that fallback in the summary instead of silently inventing extra weather priors.

## Python Examples

### Report-first profile workflow

```python
from spectral_packet_engine import load_profile_table_report

report = load_profile_table_report(
    "examples/data/synthetic_profiles.csv",
    analyze_num_modes=16,
    compress_num_modes=8,
    device="cpu",
)
report.write_artifacts("artifacts/profile_report")
```

### Spectroscopy-style family inference

```python
import torch

from spectral_packet_engine import (
    InfiniteWell1D,
    GradientOptimizationConfig,
    harmonic_potential,
    infer_potential_family_from_spectrum,
    solve_eigenproblem,
)

domain = InfiniteWell1D.from_length(1.0, dtype=torch.float64)
target = solve_eigenproblem(
    lambda x: harmonic_potential(x, omega=8.0, domain=domain),
    domain,
    num_points=128,
    num_states=3,
).eigenvalues

summary = infer_potential_family_from_spectrum(
    target_eigenvalues=target,
    families=("harmonic", "double-well"),
    initial_guesses={
        "harmonic": {"omega": 5.0},
        "double-well": {"a_param": 1.5, "b_param": 1.0},
    },
    optimization_config=GradientOptimizationConfig(steps=180, learning_rate=0.04),
    device="cpu",
)

print(summary.best_family)
print(summary.family_weights)
```

### Reduced-model analysis

```python
from spectral_packet_engine import analyze_separable_tensor_product_spectrum, build_separable_2d_report

summary = analyze_separable_tensor_product_spectrum(
    family_x="harmonic",
    parameters_x={"omega": 8.0},
    family_y="harmonic",
    parameters_y={"omega": 6.0},
    device="cpu",
)

print(summary.combined_eigenvalues[:4])
print(summary.ground_density_low_rank.energy_capture)

report = build_separable_2d_report(
    num_modes_x=4,
    num_modes_y=4,
    num_combined_states=6,
    device="cpu",
)
report.write_artifacts("artifacts/separable_2d_report")

print(report.overview.example_name)
print(report.overview.max_absolute_reference_error)
```

### Packet-family projection and propagation

```python
from spectral_packet_engine import (
    InfiniteWell1D,
    compare_state_trajectories,
    make_box_mode_spectral_state,
    make_plane_wave_packet,
    make_windowed_plane_wave_packet,
    project_packet_state,
    simulate_packet_state,
    simulate_spectral_state,
)

domain = InfiniteWell1D.from_length(1.0)
packet = make_plane_wave_packet(domain, wavenumber=9.0)
windowed = make_windowed_plane_wave_packet(domain, center=0.35, window_width=0.24, wavenumber=20.0)
mode = make_box_mode_spectral_state(domain, mode_index=8, num_modes=96)

projection = project_packet_state(packet, num_modes=96, device="cpu")
forward = simulate_packet_state(packet, times=[0.0, 1e-3, 2e-3], num_modes=96, device="cpu")
spectral_forward = simulate_spectral_state(mode, times=[0.0, 1e-3], interval=[0.55, 0.9], num_modes=96, device="cpu")
comparison = compare_state_trajectories(
    [("windowed", windowed), ("mode", mode)],
    times=[0.0, 1e-3],
    interval=[0.55, 0.9],
    num_modes=96,
    device="cpu",
)

print(projection.initial_support.boundary_density_mismatch)
print(forward.position_standard_deviation)
print(forward.phase_space.negativity)
print(spectral_forward.tracked_interval_probability)
print(comparison.pairs[0].comparison.fidelity)
```

Input is an explicit bounded-domain packet or spectral state. Outputs expose projection quality, support diagnostics, probability-preserving propagation, interval traces, position-space uncertainty summaries, density-matrix diagnostics, explicit Wigner phase-space diagnostics, and pairwise state-comparison summaries without baking Gaussian-only assumptions into the shared workflow layer. Python remains the most general surface; MCP now also accepts explicit bounded-state specifications when `execute_python` is unavailable.

### Official benchmark registry

```python
from spectral_packet_engine import run_benchmark_registry

report = run_benchmark_registry(
    case_ids=("harmonic-oscillator", "double-well", "barrier-scattering"),
    device="cpu",
)

for result in report.case_results:
    print(result.definition.case_id, result.status, result.metrics.score)

report.write_artifacts("artifacts/benchmark_registry")
```

Input is an explicit case list or the full official suite. Outputs include per-case error metrics, elapsed time, Python peak memory, mode or energy-grid budget, local identifiability evidence, backend metadata, and requested backend comparisons. The artifact bundle writes `benchmark_registry.json`, `benchmark_cases.csv`, and `artifacts.json`; it tells a user whether the engine’s claims are backed by reproducible measurement rather than a feature list.

### Physics-aware datasets and measurement response

```python
import torch

from spectral_packet_engine import (
    MeasurementNoiseModel,
    finite_resolution_response_matrix,
    apply_instrument_response,
    load_profile_table,
    spectral_dataset_from_profile_table,
    write_spectral_dataset_artifacts,
)

profile_table = load_profile_table("examples/data/synthetic_profiles.csv")
dataset = spectral_dataset_from_profile_table(profile_table)
write_spectral_dataset_artifacts("artifacts/spectral_dataset", dataset)

response = finite_resolution_response_matrix(dataset.grids[-1].coordinates, sigma=0.04)
summary = apply_instrument_response(
    dataset.values[0],
    response,
    noise_model=MeasurementNoiseModel(model="independent-gaussian", scale=0.01),
)

print(dataset.content_hash)
print(torch.max(summary.normalization_error))
```

Input is an already-validated `ProfileTable` or explicit spectral-grid tensor. Outputs include grid metadata, units, uncertainty, train/test regime splits, artifact lineage, and a finite-resolution measurement response summary. This is a physics-aware data contract, not a generic dataframe or backend abstraction.

### Differentiable inverse design

```python
from spectral_packet_engine import (
    GradientOptimizationConfig,
    design_potential_for_target_transition,
)

summary = design_potential_for_target_transition(
    family="harmonic",
    target_transition=12.0,
    initial_guess={"omega": 5.0},
    optimization_config=GradientOptimizationConfig(steps=200, learning_rate=0.05),
    device="cpu",
)

print(summary.optimized_parameters)
print(summary.achieved_transition)
```

## CLI Highlights

### Inverse and UQ

```bash
spectral-packet-engine fit-profile-table examples/data/synthetic_profiles.csv \
  --center 0.36 \
  --width 0.11 \
  --wavenumber 22.0 \
  --device cpu \
  --output-dir artifacts/inverse_fit
```

### Spectroscopy / family inference

```bash
spectral-packet-engine infer-potential-spectrum 5.22 15.83 26.41 \
  --family harmonic \
  --family double-well \
  --device cpu \
  --output-dir artifacts/spectroscopy
```

### Reduced models

```bash
spectral-packet-engine analyze-separable-spectrum \
  --family-x harmonic \
  --params-x '{"omega": 8.0}' \
  --family-y harmonic \
  --params-y '{"omega": 6.0}' \
  --device cpu \
  --output-dir artifacts/separable
```

```bash
spectral-packet-engine solve-radial-reduction \
  --family morse \
  --params '{"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7}' \
  --angular-momentum 1 \
  --device cpu
```

### Differentiable design and control

```bash
spectral-packet-engine design-transition \
  --family harmonic \
  --target-transition 12.0 \
  --initial-guess '{"omega": 5.0}' \
  --device cpu
```

```bash
spectral-packet-engine optimize-packet-control \
  --objective position \
  --target-value 0.55 \
  --final-time 0.004 \
  --device cpu
```

### Vertical workflows

```bash
spectral-packet-engine transport-workflow --device cpu --output-dir artifacts/transport
spectral-packet-engine profile-inference-workflow examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_vertical
```

## Artifact Bundles

Artifacts remain shared and inspectable across surfaces.

New bundle patterns include:

- inverse/UQ:
  - `uncertainty_summary.json`
  - `parameter_posterior.csv`
  - `modal_posterior.csv`
  - `observation_posterior.json`
  - `observation_information.json`
  - `sensitivity_map.json`
- potential-family inference:
  - `potential_family_inference.json`
  - `candidate_ranking.csv`
  - `best_family_calibration.json`
- reduced models:
  - `reduced_model_summary.json`
  - `combined_spectrum.csv`
  - `separable_2d_report.json`
  - `separable_2d_summary.json`
  - `eigenvalues.csv`
  - `mode_budget.json`
  - `structured_operator.json`
  - `adiabatic_surfaces.csv`
  - `effective_potential.csv`
- differentiable design:
  - `differentiable_summary.json`
  - `transition_design_spectrum.csv`
  - `transition_gradient.csv`
  - `optimization_history.csv`
- benchmark registry:
  - `benchmark_registry.json`
  - `benchmark_cases.csv`
- spectral dataset:
  - `spectral_dataset.json`
  - `spectral_dataset_values.json`
- vertical workflows:
  - nested bundles such as `family_inference/`, `report/`, `inverse/`, `features/`

## MCP

### Hosted server

A public MCP endpoint is available at:

```
https://lightcap.ai/mcp
```

Connect from Claude Desktop, Cursor, Windsurf, or any MCP-compatible client by adding the server URL to your MCP configuration. No installation or API key required.

For streamable-HTTP compatibility routes, prefer the path-scoped bridge under the same mount:

- `https://lightcap.ai/mcp/tool_registry`
- `https://lightcap.ai/mcp/server_info`
- `https://lightcap.ai/mcp/<tool_name>`

### Tools

The MCP server exposes first-class tools for:

- planning and routing: `plan_experiment`, `guide_workflow`,
- inverse/UQ: `infer_potential_spectrum`, `fit_packet_to_profile_table`,
- reduced models: `analyze_separable_spectrum`, `analyze_coupled_surfaces`, `solve_radial_reduction`,
- differentiable physics: `design_transition`, `optimize_packet_control`,
- vertical workflows: `transport_workflow`, `profile_inference_workflow`,
- data loading and microclimate: `write_scratch_file`, `upload_csv_to_database`, `upload_csv_for_analysis`, `create_scratch_database`, `ingest_open_meteo_hourly_dataset`, `derive_urban_microclimate`,
- SQL workflows: `query_database`, `execute_database_script`, `materialize_query_table`.

It also exposes structured capability resources:

- `spectral://capabilities/inverse-uq`
- `spectral://capabilities/reduced-models`
- `spectral://capabilities/differentiable-physics`
- `spectral://capabilities/vertical-workflows`

Tool descriptions are intent-oriented, and every MCP tool response includes `related_tools` so an agent can inspect adjacent next steps without hardcoded routing.

### Self-hosting

To run your own instance:

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine serve-mcp --transport streamable-http --port 8765 --streamable-http-path /mcp
```

All security limits (rate limiting, query timeouts, database size caps, audit logging) are enabled by default. See `spectral-packet-engine serve-mcp --help` for tunable parameters.

Managed SQLite databases now sit behind a persistent storage economy and snapshot guard:

- existing scratch databases are sealed on first startup and become protected mass,
- write and delete cost is measured from the actual SQLite page diff, not only from request count,
- the spendable mutation budget refills over `--storage-protection-window-seconds` and scales with protected database size,
- `write_scratch_file` cannot overwrite `.db` / `.sqlite` files and `delete_scratch_file` cannot remove managed databases,
- missing managed databases are restored automatically from guarded snapshots on server startup,
- `inspect_storage_economy` and `server_info` expose the live budget, refill rate, and registered database set.

The default protection window is one day. In other words, mutating one current-database worth of SQLite pages again takes roughly one day of accumulated budget unless you lower the protection settings explicitly.

If you publish MCP behind a reverse proxy, expose both the exact mount (`/mcp`) and the scoped prefix (`/mcp/`). Publishing only the exact mount can make the MCP session itself work while path-based bridge routes such as `/mcp/tool_registry` and `/mcp/inspect_product` fail.

For a restartable containerized deployment:

```bash
docker compose up -d --build
```

That starts the MCP server on `http://127.0.0.1:8765/mcp`, mounts persistent scratch/log volumes under `docker-data/`, and health-checks the real callable MCP bridge route at `/mcp/server_info`.
The compose file binds to `127.0.0.1` by default so reverse-proxy deployments do not accidentally expose the MCP listener directly; override `SPE_PUBLISHED_HOST` only when you intentionally want a wider bind.
The repository Docker image installs the CPU PyTorch wheel explicitly so Linux hosts do not silently pull a multi-gigabyte CUDA runtime just to serve the default MCP surface.
The same container keeps the storage-economy ledger and guarded SQLite snapshots under the persisted scratch volume, so a container restart can restore managed databases instead of starting from an empty scratch directory.

If you keep `.env` populated with your remote host settings, you can push the Docker deployment with:

```bash
./scripts/deploy_mcp_docker.sh
```

The deploy script uses a stable compose project name, tears down old compose containers for the same product, and can stop legacy `spectral_packet_engine serve-mcp` systemd units that still own the target port before starting Docker. If some unrelated process still owns the port, deployment fails loudly instead of silently publishing a broken surface.
For public hosting, generate one aligned ingress contract before editing nginx by hand:

```bash
spectral-packet-engine plan-mcp-ingress --public-host lightcap.ai --public-host www.lightcap.ai
```

That prints the canonical public endpoint, the `allowed_hosts` / `allowed_origins` values the MCP runtime should trust, the Docker env block, and the nginx site snippet that keeps `/`, `/mcp`, and `/mcp/` consistent.
If the hostname is already owned by an existing site server block, use the emitted `nginx_location_snippet` instead of adding a second competing `server_name` block.

See [docs/mcp.md](docs/mcp.md) and [docs/mcp-capabilities.md](docs/mcp-capabilities.md).

## Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/workflows.md](docs/workflows.md)
- [docs/inverse-and-uq.md](docs/inverse-and-uq.md)
- [docs/reduced-models.md](docs/reduced-models.md)
- [docs/differentiable-physics.md](docs/differentiable-physics.md)
- [docs/vertical-workflows.md](docs/vertical-workflows.md)
- [docs/mcp.md](docs/mcp.md)
- [docs/mcp-capabilities.md](docs/mcp-capabilities.md)

## Install

Core package:

```bash
python3 -m pip install .
```

Editable install with dev tools:

```bash
python3 -m pip install -e ".[dev]"
```

Useful extras:

- `mcp` for the MCP runtime,
- `api` for FastAPI/Uvicorn,
- `sql` for SQLAlchemy-backed database workflows,
- `files` for optional Parquet/XLSX file support,
- `ml-tree-core` and friends for tree-model workflows,
- `examples` for plotting/example dependencies when needed.

## Limits

The repository is explicit about what it does not claim:

- local posterior approximations are not global Bayesian inference,
- reduced models are structured reductions, not arbitrary 2D/3D solvers,
- phase-1 structured dimensional lift is a separable 2D tensor-product path with explicit budgets and truncation, not a generic multidimensional PDE stack,
- open-system evolution is an explicit matrix-basis Lindblad contract, not a generic environment simulator,
- `SpectralDataset` is a physics-aware data contract over spectral grids, not a replacement for clean tabular ingestion,
- differentiable workflows optimize explicit parameterizations, not unrestricted control spaces,
- vertical workflows are productized orchestrations over the same spectral core, not separate subsystems.
