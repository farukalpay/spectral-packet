# Vertical Workflows

Vertical workflows are not separate products. They are opinionated orchestrations over the same spectral core.

## 1. Spectroscopy / Family Inference

Goal:

- determine which explicit potential family best explains an observed low-lying spectrum,
- return best-fit parameters plus local uncertainty and sensitivity.

Surfaces:

- Python: `run_spectroscopy_workflow(...)`
- CLI: `infer-potential-spectrum`
- MCP: `infer_potential_spectrum`

Typical outputs:

- family ranking,
- best family,
- transition energies from the best family,
- posterior and sensitivity artifacts.

## 2. Transport / Barrier / Resonance

Goal:

- analyze tunneling through a barrier experiment with one structured vertical bundle.

Surfaces:

- Python: `run_transport_resonance_workflow(...)`
- CLI: `transport-workflow`
- MCP: `transport_workflow`

Typical outputs:

- resonance energies and widths,
- WKB vs exact transmission comparison,
- propagation and Wigner diagnostics,
- one transport-focused artifact bundle.

## 3. Control / Packet Steering

Goal:

- optimize initial packet preparation parameters toward a target observable.

Surfaces:

- Python: `run_control_workflow(...)` or `optimize_packet_control(...)`
- CLI: `optimize-packet-control`
- MCP: `optimize_packet_control`

Typical outputs:

- optimized packet parameters,
- optimization history,
- final density,
- observable gradient summary.

## 4. Scientific Tabular Workflow

Goal:

- keep scientific table analysis report-first while still connecting it to inverse fitting and downstream features.

Surfaces:

- Python: `run_profile_inference_workflow(...)`
- CLI: `profile-inference-workflow`
- MCP: `profile_inference_workflow`

Typical outputs:

- profile report,
- inverse fit with uncertainty,
- feature table export,
- one nested vertical artifact bundle with `report/`, `inverse/`, and `features/`.

## Design Rule

Each vertical is honest about scope.

- Spectroscopy fits explicit bounded-domain model families.
- Transport uses the repository’s tunneling experiment pipeline, not arbitrary open-system modeling.
- Control optimizes packet preparation parameters, not unrestricted pulse design.
- Scientific tabular workflows stay grounded in inspectable spectral summaries before any inverse or feature-model step.
