# Differentiable Physics

## Positioning

PyTorch is used here to differentiate through the spectral core, not to turn the repository into a generic training framework.

Supported workflows include:

- `potential -> spectrum` calibration,
- `potential -> target transition` inverse design,
- `state preparation -> target observable` optimization.

## Python

```python
from spectral_packet_engine import (
    GradientOptimizationConfig,
    calibrate_potential_from_spectrum,
    design_potential_for_target_transition,
    optimize_packet_control,
)

calibration = calibrate_potential_from_spectrum(
    family="harmonic",
    target_eigenvalues=[5.22, 15.83, 26.41],
    initial_guess={"omega": 5.0},
    optimization_config=GradientOptimizationConfig(steps=180, learning_rate=0.04),
    device="cpu",
)

design = design_potential_for_target_transition(
    family="harmonic",
    target_transition=12.0,
    initial_guess={"omega": 5.0},
    device="cpu",
)

control = optimize_packet_control(
    initial_guess={"center": 0.30, "width": 0.07, "wavenumber": 25.0, "phase": 0.0},
    objective="target_position",
    target_value=0.55,
    final_time=0.004,
    device="cpu",
)
```

## CLI

```bash
spectral-packet-engine design-transition \
  --family harmonic \
  --target-transition 12.0 \
  --initial-guess '{"omega": 5.0}' \
  --device cpu
```

```bash
spectral-packet-engine optimize-packet-control \
  --objective target_position \
  --target-value 0.55 \
  --final-time 0.004 \
  --device cpu
```

## MCP

- `design_transition`
- `optimize_packet_control`
- resource: `spectral://capabilities/differentiable-physics`

## Artifact Contract

Differentiable bundles include:

- `differentiable_summary.json`
- `predicted_eigenvalues.csv` for calibration,
- `transition_design_spectrum.csv` and `transition_gradient.csv` for transition design,
- `optimization_history.csv`, `final_density.csv`, and `objective_gradient.csv` for packet control.

## Limits

These gradients are only as smooth as the underlying physics map.

Be careful near:

- eigenvalue crossings and degeneracies,
- non-smooth parameterizations,
- highly constrained bounded-domain fits where multiple parameters become weakly identifiable.

The repository keeps those limitations explicit instead of advertising generic optimizer magic.
