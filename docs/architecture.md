# Architecture

## Product Statement

Spectral Packet Engine is one scientific compute product with multiple interfaces.

The core is a bounded-domain spectral engine for:

- packet construction,
- modal projection,
- forward propagation,
- observables,
- truncation diagnostics,
- inverse reconstruction.

Everything else exists to move real data into that engine, run it through the same workflows, and expose the results through different integration surfaces.

## Runtime Statement

The runtime is one shared Python engine that validates environment assumptions explicitly, executes bounded in-process workflows, records task and artifact state, and leaves restart supervision to the host process manager.

## Product Spine

### 1. Core engine

The core mathematical engine lives in:

- `src/spectral_packet_engine/domain.py`
- `src/spectral_packet_engine/basis.py`
- `src/spectral_packet_engine/state.py`
- `src/spectral_packet_engine/projector.py`
- `src/spectral_packet_engine/dynamics.py`
- `src/spectral_packet_engine/observables.py`
- `src/spectral_packet_engine/simulation.py`
- `src/spectral_packet_engine/inference.py`

### 2. Workflow and artifact layer

Shared user-facing workflows live in:

- `src/spectral_packet_engine/workflows.py`
- `src/spectral_packet_engine/diagnostics.py`
- `src/spectral_packet_engine/artifacts.py`
- `src/spectral_packet_engine/release_gate.py`

This layer is the conductor between the engine and the public interfaces.

### 3. Data and storage bridge

Real user data enters through:

- `src/spectral_packet_engine/table_io.py`
- `src/spectral_packet_engine/tabular.py`
- `src/spectral_packet_engine/database.py`

These modules exist to make file-backed and SQL-backed data usable by the same engine workflows.

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
- backend-aware modal-surrogate orchestration.

Experimental:

- TensorFlow compatibility workflows,
- published transport-dataset benchmark path.

## What The Repository Should Feel Like

The repository should read as:

one bounded-domain spectral engine with practical data and service extensions.

It should not read as:

- a generic platform,
- a physics-for-everything repo,
- a manuscript shell,
- or a pile of wrappers around disconnected ideas.
