# Reduced Models

## Scope

The repository does not claim a general high-dimensional solver.

It exposes controlled reduced models that preserve explicit physical structure:

- separable tensor-product spectra,
- near-separable and block-coupled tensor-basis diagnostics,
- phase-1 structured dimensional lift for separable 2D bounded problems,
- reduced coupled-channel adiabatic surfaces,
- radial effective-coordinate reductions,
- low-rank summaries of structured coefficient objects.

## Python

```python
from spectral_packet_engine import analyze_separable_tensor_product_spectrum, build_separable_2d_report

summary = analyze_separable_tensor_product_spectrum(
    family_x="harmonic",
    parameters_x={"omega": 8.0},
    family_y="harmonic",
    parameters_y={"omega": 6.0},
    device="cpu",
)

report = build_separable_2d_report(
    num_modes_x=4,
    num_modes_y=4,
    num_combined_states=6,
    device="cpu",
)
report.write_artifacts("artifacts/separable_2d_report")
```

```python
import torch

from spectral_packet_engine import analyze_structured_coupling

coupling = torch.eye(6)
structure = analyze_structured_coupling(
    coupling,
    tensor_shape=(2, 3),
    block_partitions=((0, 1, 2), (3, 4, 5)),
)

print(structure.low_rank_energy_capture)
print(structure.within_block_energy_fraction)
```

```python
from spectral_packet_engine import analyze_coupled_channel_surfaces, solve_radial_reduction

surfaces = analyze_coupled_channel_surfaces(device="cpu")
radial = solve_radial_reduction(
    family="morse",
    parameters={"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7},
    angular_momentum=1,
    device="cpu",
)
```

## CLI

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
spectral-packet-engine analyze-coupled-surfaces --device cpu --output-dir artifacts/coupled_surfaces
spectral-packet-engine solve-radial-reduction --family morse --params '{"D_e": 8.0, "alpha": 2.0, "x_eq": 0.7}' --device cpu
```

## MCP

- `analyze_separable_spectrum`
- `analyze_coupled_surfaces`
- `solve_radial_reduction`
- resource: `spectral://capabilities/reduced-models`

## Artifact Contract

Reduced-model bundles are typed by the summary they contain.

Depending on the workflow they may include:

- `reduced_model_summary.json`
- `combined_spectrum.csv`
- `separable_2d_report.json`
- `separable_2d_summary.json`
- `eigenvalues.csv`
- `mode_budget.json`
- `structured_operator.json`
- `adiabatic_surfaces.csv`
- `effective_potential.csv`
- `singular_values.csv`

## Honesty Rule

Every reduced-model summary includes explicit assumptions.

Examples:

- separable workflows state that the total Hamiltonian is assumed to split into independent 1D components and that the structured operator is a retained-basis Kronecker sum,
- structured coupling diagnostics analyze an explicit retained tensor-product basis and do not create a generic multidimensional solver,
- the structured-dimensional-lift report states that it is one separable 2D bounded-domain path rather than a general 2D/3D solver,
- coupled-channel workflows state that they model a reduced avoided crossing rather than a full electronic-structure problem,
- radial workflows state that they solve a 1D effective-coordinate problem with a centrifugal term on a finite interval.
