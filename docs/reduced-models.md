# Reduced Models

## Scope

The repository does not claim a general high-dimensional solver.

It exposes controlled reduced models that preserve explicit physical structure:

- separable tensor-product spectra,
- reduced coupled-channel adiabatic surfaces,
- radial effective-coordinate reductions,
- low-rank summaries of structured coefficient objects.

## Python

```python
from spectral_packet_engine import analyze_separable_tensor_product_spectrum

summary = analyze_separable_tensor_product_spectrum(
    family_x="harmonic",
    parameters_x={"omega": 8.0},
    family_y="harmonic",
    parameters_y={"omega": 6.0},
    device="cpu",
)
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
- `adiabatic_surfaces.csv`
- `effective_potential.csv`
- `singular_values.csv`

## Honesty Rule

Every reduced-model summary includes explicit assumptions.

Examples:

- separable workflows state that the total Hamiltonian is assumed to split into independent 1D components,
- coupled-channel workflows state that they model a reduced avoided crossing rather than a full electronic-structure problem,
- radial workflows state that they solve a 1D effective-coordinate problem with a centrifugal term on a finite interval.
