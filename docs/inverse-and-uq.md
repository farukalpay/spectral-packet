# Inverse And UQ

## Scope

The inverse layer is intentionally lightweight and physics-centered.

It supports:

- Gaussian packet inverse fitting against observed density profiles,
- local posterior summaries over inferred packet parameters,
- coefficient posterior summaries,
- sensitivity and identifiability outputs,
- explicit potential-family inference from observed low-lying spectra.

It does not add a heavyweight probabilistic framework or a generic Bayesian platform.

## Main Surfaces

### Python

```python
from spectral_packet_engine import fit_gaussian_packet_to_profile_table

summary = fit_gaussian_packet_to_profile_table(
    table,
    initial_guess={"center": 0.36, "width": 0.11, "wavenumber": 22.0, "phase": 0.0},
    num_modes=96,
    quadrature_points=1024,
    device="cpu",
)

print(summary.physical_inference.parameter_posterior.identifiability_score)
```

```python
from spectral_packet_engine import infer_potential_family_from_spectrum

summary = infer_potential_family_from_spectrum(
    target_eigenvalues=[5.22, 15.83, 26.41],
    families=("harmonic", "double-well"),
    initial_guesses={
        "harmonic": {"omega": 5.0},
        "double-well": {"a_param": 1.5, "b_param": 1.0},
    },
    device="cpu",
)
```

### CLI

```bash
spectral-packet-engine fit-profile-table examples/data/synthetic_profiles.csv \
  --center 0.36 --width 0.11 --wavenumber 22.0 --device cpu \
  --output-dir artifacts/inverse_fit
```

```bash
spectral-packet-engine infer-potential-spectrum 5.22 15.83 26.41 \
  --family harmonic --family double-well --device cpu \
  --output-dir artifacts/spectroscopy
```

### MCP

- `fit_packet_to_profile_table`
- `fit_packet_to_database_profile_query`
- `infer_potential_spectrum`
- resource: `spectral://capabilities/inverse-uq`

## Artifact Contract

Packet inverse fit bundles may include:

- `inverse_fit.json`
- `predicted_density.csv`
- `uncertainty_summary.json`
- `parameter_posterior.csv`
- `modal_posterior.csv`
- `sensitivity_map.json`

Potential-family inference bundles may include:

- `potential_family_inference.json`
- `candidate_ranking.csv`
- `best_family_calibration.json`
- `best_family_parameter_posterior.csv`
- `best_family_sensitivity_map.json`

## Interpretation

These uncertainty summaries are local.

- Packet inverse fit uses a local linearization around the optimized parameter vector.
- Potential-family ranking uses an explicit BIC-style evidence score over a small family set.
- Confidence intervals and identifiability scores describe local observability, not global uniqueness.

Those limitations are surfaced in result objects and artifacts instead of being hidden behind probabilistic language.
