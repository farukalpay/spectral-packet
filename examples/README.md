# Examples

Install the package first, then choose the example surface that matches what you want to verify.

## Stable First-Run Examples

- `core_engine_workflow.py`: direct Python engine workflow with forward artifacts and modal projection output
- `profile_table_workflow.py`: the hero Python-first profile-table report workflow with inspectable artifacts
- `inverse_physics_workflow.py`: spectroscopy-style potential-family inference with uncertainty-aware artifact output
- `reduced_model_workflow.py`: artifact-backed separable 2D report plus one coupled-surface reduced-model summary
- `modal_surrogate_workflow.py`: backend-aware modal-surrogate evaluation over a profile table
- `api_workflow.py`: minimal HTTP client for the same profile-table report workflow exposed through the optional API

These are the examples a new user should try first.

## Reference Examples

`reference/` contains deeper reference-case scripts for inspecting the core engine numerics and generating figures from a fixed bounded-domain packet setup.

They are useful for understanding the engine more deeply, but they are not the shortest path into the product.

## Experimental Examples

`experimental/` contains compatibility-path and benchmark scripts that are intentionally outside the core first-run experience.

They are kept in the repository because they are useful, but they should not be confused with the stable surface.
