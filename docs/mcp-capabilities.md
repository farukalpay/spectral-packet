# MCP Capabilities

The MCP server is not a freeform text wrapper. It exposes structured spectral operations with inspectable assumptions.

## Capability Resources

The server now publishes machine-readable capability resources:

- `spectral://capabilities/inverse-uq`
- `spectral://capabilities/reduced-models`
- `spectral://capabilities/differentiable-physics`
- `spectral://capabilities/vertical-workflows`

Use these when an MCP client needs to decide which tool family to call before running compute.

## Prompt Templates

The server also publishes prompt templates for workflow selection:

- `select_inverse_physics_workflow`
- `select_reduced_model_strategy`
- `select_vertical_workflow`

These are intended for MCP clients that want a reusable planning layer without inventing tool-routing policy on their own.

## New Tool Families

### Inverse / UQ

- `infer_potential_spectrum`
- `fit_packet_to_profile_table`
- `fit_packet_to_database_profile_query`

### Reduced Models

- `analyze_separable_spectrum`
- `analyze_coupled_surfaces`
- `solve_radial_reduction`

### Differentiable Physics

- `design_transition`
- `optimize_packet_control`

### Vertical Workflows

- `transport_workflow`
- `profile_inference_workflow`

## Tool Selection Heuristics

Use:

- `infer_potential_spectrum` when the observation is a low-lying spectrum and the question is “which family explains this data?”
- `analyze_separable_spectrum` when the Hamiltonian is explicitly separable into independent 1D components.
- `analyze_coupled_surfaces` when the problem is a reduced avoided crossing or channel-coupling analysis.
- `solve_radial_reduction` when the system is best represented as a bounded radial effective coordinate.
- `design_transition` when the user wants gradient-based inverse design of a spectral gap or transition.
- `optimize_packet_control` when the target is an observable reached by steering initial packet preparation parameters.
- `transport_workflow` when the user wants a barrier/resonance answer rather than separate scattering/WKB/propagation calls.
- `profile_inference_workflow` when the input is a scientific profile table and the user wants a report-first end-to-end result.

## Artifact Alignment

All of these tools align with the shared artifact layer. When `output_dir` is provided, MCP writes the same bundle structures that Python and CLI workflows use.
