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
    make_plane_wave_packet,
    project_packet_state,
    simulate_packet_state,
)

domain = InfiniteWell1D.from_length(1.0)
packet = make_plane_wave_packet(domain, wavenumber=9.0)

projection = project_packet_state(packet, num_modes=96, device="cpu")
forward = simulate_packet_state(packet, times=[0.0, 1e-3, 2e-3], num_modes=96, device="cpu")

print(projection.initial_support.boundary_density_mismatch)
print(forward.position_standard_deviation)
```

Input is an explicit bounded-domain packet state. Outputs expose projection quality, support diagnostics, probability-preserving propagation, and position-space uncertainty summaries without baking Gaussian-only assumptions into the shared workflow layer.

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

### Tools

The MCP server exposes first-class tools for:

- planning and routing: `plan_experiment`, `guide_workflow`,
- inverse/UQ: `infer_potential_spectrum`, `fit_packet_to_profile_table`,
- reduced models: `analyze_separable_spectrum`, `analyze_coupled_surfaces`, `solve_radial_reduction`,
- differentiable physics: `design_transition`, `optimize_packet_control`,
- vertical workflows: `transport_workflow`, `profile_inference_workflow`,
- data loading: `write_scratch_file`, `upload_csv_to_database`, `upload_csv_for_analysis`, `create_scratch_database`,
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
spectral-packet-engine mcp --transport streamable-http --port 8765
```

All security limits (rate limiting, query timeouts, database size caps, audit logging) are enabled by default. See `spectral-packet-engine mcp --help` for tunable parameters.

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
