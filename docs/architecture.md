# Architecture

## Product Statement

Spectral Packet Engine is one scientific compute product with multiple interfaces.

The core is a bounded-domain spectral engine for:

- packet construction,
- modal projection,
- forward propagation,
- observables,
- truncation diagnostics,
- inverse reconstruction,
- local uncertainty-aware physical inference,
- controlled reduced models,
- differentiable design workflows.

The engine also includes deep mathematical physics modules:

- eigenvalue problems for arbitrary potentials (Chebyshev collocation),
- split-operator time-dependent propagation (2nd/4th order Trotter-Suzuki),
- Wigner quasi-probability phase-space distributions,
- density matrix formalism (pure/mixed/thermal states, entropy, fidelity),
- spectral Green's functions and local density of states,
- quantum perturbation theory (1st/2nd order, degenerate),
- WKB semiclassical methods (Bohr-Sommerfeld, tunneling),
- operator algebra (commutators, generalized uncertainty, BCH, ladder operators),
- symplectic integrators (Störmer-Verlet, Forest-Ruth, Yoshida 4th/6th order),
- spectral zeta functions, heat kernels, partition functions, Casimir energy,
- quantum scattering (transfer matrix, S-matrix, resonances),
- Berry geometric phases (curvature, Chern numbers, adiabatic evolution),
- quantum information measures (Fisher information, entanglement, concurrence, quantum channels).

Everything else exists to move real data into that engine, run it through the same workflows, and expose the results through different integration surfaces.

## Runtime Statement

The runtime is one shared Python engine that validates environment assumptions explicitly, executes bounded in-process workflows, records task and artifact state, and leaves restart supervision to the host process manager.

## Product Spine

### 1. Core engine

The core mathematical engine lives in:

- `src/spectral_packet_engine/domain.py`
- `src/spectral_packet_engine/basis.py`
- `src/spectral_packet_engine/physics_contracts.py`
- `src/spectral_packet_engine/state.py`
- `src/spectral_packet_engine/projector.py`
- `src/spectral_packet_engine/dynamics.py`
- `src/spectral_packet_engine/observables.py`
- `src/spectral_packet_engine/simulation.py`
- `src/spectral_packet_engine/inference.py`

`physics_contracts.py` owns the shared `PotentialFamily`, `HamiltonianOperator`, `BasisSpec`, `BoundaryCondition`, `ObservableSet`, and `MeasurementModel` contract so forward, inverse, reduced-model, and surrogate workflows can refer to the same mathematical problem object.

`state.py` owns packet-family preparation and bounded-support diagnostics. Gaussian packets remain a supported analytic family, but the core state layer is not Gaussian-only; reusable packet families project through the same spectral engine and surface explicit domain-support and boundary-mismatch evidence instead of burying those assumptions in interface wrappers.

Advanced physics modules:

- `src/spectral_packet_engine/eigensolver.py` — arbitrary-potential Schrödinger eigenvalue solver
- `src/spectral_packet_engine/tensor_product.py` — explicit 2D tensor-product basis and Kronecker-sum operator primitives for structured dimensional lifts
- `src/spectral_packet_engine/split_operator.py` — split-operator propagation for time-dependent Schrödinger
- `src/spectral_packet_engine/wigner.py` — Wigner quasi-probability distribution
- `src/spectral_packet_engine/density_matrix.py` — density matrix formalism and quantum entropies
- `src/spectral_packet_engine/open_systems.py` — Lindblad/open-system and finite-resolution measurement contracts
- `src/spectral_packet_engine/greens_function.py` — spectral Green's function and LDOS
- `src/spectral_packet_engine/perturbation.py` — quantum perturbation theory engine
- `src/spectral_packet_engine/semiclassical.py` — WKB, Bohr-Sommerfeld, tunneling
- `src/spectral_packet_engine/operator_algebra.py` — commutators, uncertainty, BCH, ladder operators
- `src/spectral_packet_engine/symplectic.py` — structure-preserving Hamiltonian integrators
- `src/spectral_packet_engine/spectral_zeta.py` — spectral zeta, heat kernel, thermodynamics
- `src/spectral_packet_engine/scattering.py` — transfer matrix quantum scattering
- `src/spectral_packet_engine/berry_phase.py` — geometric phases and topology
- `src/spectral_packet_engine/quantum_info.py` — quantum information measures
- `src/spectral_packet_engine/pipelines.py` — auto-parameterized analysis pipelines
- `src/spectral_packet_engine/load_spectral.py` — spectral load modeling, adaptive throttling, anomaly detection
- `src/spectral_packet_engine/parametric_potentials.py` — explicit parameterized potential families for inference and design
- `src/spectral_packet_engine/uq.py` — shared local posterior, predictive-interval, and identifiability summaries
- `src/spectral_packet_engine/reduced_models.py` — separable, near-separable, block-coupled, structured-lift, coupled-channel, radial, and low-rank reduced-model surfaces
- `src/spectral_packet_engine/differentiable_physics.py` — differentiable calibration and inverse-design workflows
- `src/spectral_packet_engine/vertical_workflows.py` — domain-specific spectroscopy, transport, control, and tabular verticals

### 2. Workflow and artifact layer

Shared user-facing workflows live in:

- `src/spectral_packet_engine/workflows.py`
- `src/spectral_packet_engine/vertical_workflows.py`
- `src/spectral_packet_engine/diagnostics.py`
- `src/spectral_packet_engine/artifacts.py`
- `src/spectral_packet_engine/release_gate.py`
- `src/spectral_packet_engine/benchmark_registry.py`

This layer is the conductor between the engine and the public interfaces.
It is also where uncertainty/UQ, reduced-model orchestration, differentiable design, benchmark evidence, vertical workflows, and shared packet-state diagnostics such as density-matrix and Wigner phase-space summaries are surfaced before any interface wrapper sees them.

### 3. Data and storage bridge

Real user data enters through:

- `src/spectral_packet_engine/table_io.py`
- `src/spectral_packet_engine/tabular.py`
- `src/spectral_packet_engine/spectral_dataset.py`
- `src/spectral_packet_engine/database.py`

These modules exist to make file-backed and SQL-backed data usable by the same engine workflows.
`spectral_dataset.py` is the physics-aware dataset boundary for grid metadata, units, uncertainty, regime splits, content hashes, and artifact lineage; it does not replace the generic clean-table boundary.

### 4. Optional surrogate layer

Modal-surrogate workflows live in:

- `src/spectral_packet_engine/ml.py`
- `src/spectral_packet_engine/tf_surrogate.py`

This layer is subordinate to the spectral engine. It exists to learn over modal or profile-derived targets, not to turn the product into a generic ML framework.

### 5. Interface layer

The public wrappers live in:

- `src/spectral_packet_engine/cli.py`
- `src/spectral_packet_engine/mcp.py`
- `src/spectral_packet_engine/api.py`
- `src/spectral_packet_engine/interfaces.py`

These should stay thin. New product capability belongs in the workflow layer first, then in interface wrappers.

## Architectural Rules

- The spectral engine remains the center of gravity.
- CLI, MCP, and API are wrappers over shared workflows, not alternate implementations.
- Artifact writing belongs in shared artifact code.
- Backend routing belongs in the shared ML layer.
- Runtime inspection and service compatibility checks belong in shared runtime code.
- Public docs should describe current implemented truth, not internal ambition.

## Stability Model

Stable:

- core engine,
- Python library,
- CLI,
- file-backed workflows,
- SQLite workflows.

Beta:

- MCP,
- HTTP API,
- remote SQL backends,
- backend-aware JAX surface,
- separable 2D structured-dimensional-lift report workflow,
- backend-aware modal-surrogate orchestration.

Experimental:

- TensorFlow compatibility workflows,
- published transport-dataset benchmark path,
- official benchmark registry,
- open-system measurement contracts,
- SpectralDataset contract.

## What The Repository Should Feel Like

The repository should read as:

one bounded-domain spectral engine with practical data and service extensions.

It should not read as:

- a generic platform,
- a physics-for-everything repo,
- a manuscript shell,
- or a pile of wrappers around disconnected ideas.
