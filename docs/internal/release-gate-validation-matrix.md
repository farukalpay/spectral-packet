# Final Convergence Validation Matrix

Date: 2026-04-04

This matrix tracks what was validated in this session, what is covered only by code or policy, and what still blocks a strong public release decision.

## Validated In This Session

| Area | Evidence | Status |
| --- | --- | --- |
| Repository test suite | `pytest -q` -> `73 passed, 3 skipped` | Validated locally |
| Packaging metadata | `python3 -m build --sdist --wheel` and `python3 -m twine check dist/*` | Validated locally |
| Python import path | direct imports from the local source tree | Validated locally |
| CLI artifact flow | in-process `forward` command wrote a real artifact bundle | Validated locally |
| Engine numerics | direct probes for norm preservation, projection accuracy, and inverse-fit recovery | Spot-validated locally |
| API availability | `api_is_available()` returned `True` and API tests passed | Validated locally |
| MCP availability | `mcp_is_available()` returned `True` and MCP tests passed | Validated locally |
| PyTorch backend | runtime inspection plus ML tests | Validated locally |
| JAX backend | runtime inspection plus ML tests | Validated locally |
| TensorFlow absence handling | backend inspection reports unavailable in this environment | Validated locally |

## Covered By Tests But Still Too Narrow

| Area | Current coverage | Remaining gap |
| --- | --- | --- |
| Engine workflows | representative forward, projection, compression, and inverse-fit paths | insufficient parameter sweeps, truncation budgets, and invalid-state coverage |
| CLI | help plus representative file, SQL, ML, and release-gate paths | too few chained failure paths and no strong external process validation evidence |
| API | representative health, status, DB, profile, and ML paths | no sustained concurrency or overload characterization |
| MCP | tool discovery and representative execution paths | no long-running or multi-step job validation |
| SQLite workflows | bootstrap, write, query, describe, materialize | remote-backend parity still unproven |
| Artifact bundles | representative JSON and CSV outputs | lineage and trace metadata still light |

## Not Validated In This Session

| Area | Gap | Current status |
| --- | --- | --- |
| Linux | no direct run in this session | Unvalidated here |
| Windows | no direct run in this session | Unvalidated here |
| TensorFlow-supported Python versions | no direct run in this session | Unvalidated here |
| Clean standard install without inherited site packages | not executed in this session | Still unvalidated |
| Clean editable install without inherited site packages | not executed in this session | Still unvalidated |
| Remote SQL backends | no direct backend-specific run in this session | Beta and under-validated |
| Overload and backpressure behavior | no sustained pressure test | Under-validated |
| Restart and crash recovery | not exercised | Under-validated |

## Core Versus Extension Validation Priority

Core and must be strong before release:

1. engine numerics and diagnostics
2. Python library contracts
3. CLI and file-backed workflows
4. SQLite-backed spectral workflows
5. backend inspection and safe routing

Extension surfaces that must not destabilize the core:

1. API
2. MCP
3. remote SQL
4. JAX surrogate backend
5. TensorFlow compatibility path
6. transport dataset benchmark

## Canonical Golden Paths

These are the workflows the final release gate should rely on.

1. direct Python engine path:
   packet state -> spectral projection -> time propagation -> diagnostics
2. file-backed CLI path:
   profile table -> compression -> artifact bundle
3. SQLite-backed path:
   profile table -> SQLite -> spectral or ML workflow -> artifact bundle
4. API path:
   JSON profile table -> shared workflow -> status and artifact visibility
5. MCP path:
   tool execution -> shared workflow -> status and artifact visibility

## Canonical Failure Paths

These must remain part of the release decision.

1. unsupported or unavailable backend selection
2. missing input file through CLI, API, and MCP
3. invalid profile-table shape or schema
4. invalid numerical contract such as non-finite or non-physical inputs
5. runtime startup with missing optional service stacks

## Release Interpretation

- Validated locally: executed in this session or in tests that ran here
- Spot-validated locally: exercised directly but not yet across a strong matrix
- Under-validated: code exists, but current evidence is not strong enough for public release claims

The current matrix is enough to guide hardening work. It is not yet enough to justify a high-confidence public release.
