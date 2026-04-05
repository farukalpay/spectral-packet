# MCP Usage Guide

## Connecting to the Server

### Local Development

The repository ships with `.mcp.json` at the project root for local source-checkout use. Clients that support repo-local MCP discovery can use it directly, and other clients can paste the same command block manually:

```bash
# Install with MCP support
python3 -m pip install -e ".[mcp]"

# Verify before first use
spectral-packet-engine validate-install --device cpu
```

The `.mcp.json` config:

```json
{
  "mcpServers": {
    "spectral-packet-engine": {
      "command": "python3",
      "args": ["-m", "spectral_packet_engine", "serve-mcp", "--max-concurrent-tasks", "2", "--log-level", "warning"],
      "env": { "PYTHONPATH": "src" }
    }
  }
}
```

If you are wiring Claude Desktop, VS Code, Cursor, or another MCP client manually, generate the config block instead of copying docs by hand:

```bash
spectral-packet-engine generate-mcp-config --transport stdio
```

Paste that output into the client's MCP config file such as `claude_desktop_config.json` or `.vscode/mcp.json`.

### Remote Server

If the engine is deployed on a remote machine, you can connect through SSH:

```json
{
  "mcpServers": {
    "spectral-packet-engine-remote": {
      "command": "ssh",
      "args": [
        "user@example-host",
        "cd /srv/spectral-packet-engine && .venv/bin/python -m spectral_packet_engine serve-mcp --max-concurrent-tasks 2 --log-level warning"
      ]
    }
  }
}
```

Generate that block locally with:

```bash
spectral-packet-engine generate-mcp-config --transport ssh --host user@example-host --remote-cwd /srv/spectral-packet-engine
```

If you want the currently published hosted deployment directly, use [https://lightcap.ai/mcp](https://lightcap.ai/mcp) with an HTTP-capable MCP client.

For a self-hosted restartable remote deployment over streamable HTTP, keep the service private on the remote host and tunnel to it:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log

spectral-packet-engine plan-mcp-tunnel \
  --host user@example-host \
  --local-port 8765 \
  --remote-port 8765 \
  --streamable-http-path /mcp
```

Use the returned `local_endpoint_url` with the MCP client. The tunnel exists so the remote service can stay private on the host while the client still sees a stable local URL.

### After Connecting

Once connected, the AI agent should call `self_test` to verify the server is fully operational, then `server_info` to inspect:

- `transport`
- `bind_host`
- `bind_port`
- `streamable_http_path`
- `endpoint_url`
- `allowed_hosts`
- `allowed_origins`
- `scratch_dir`

Important runtime rule: `execute_python` is disabled by default. Only use the examples below when the operator intentionally started the server with `--allow-unsafe-python` for a trusted local session.

## Tool Overview

| Category | Key Tools |
| --- | --- |
| Environment & Status | `inspect_product`, `inspect_environment`, `validate_installation`, `self_test`, `server_info`, `probe_mcp_runtime` |
| Profile Table Workflows | `profile_table_report`, `compress_profile_table`, `fit_packet_to_profile_table` |
| SQL-Backed Workflows | `bootstrap_database`, `query_database`, `execute_database_script`, `report_database_profile_query` |
| Spectral Feature & Tree | `export_feature_table`, `train_tree_model`, `tune_tree_model` |
| Analysis Pipelines | `analyze_quantum_state_pipeline`, `analyze_potential_pipeline`, `analyze_scattering_pipeline` |
| Advanced Physics | `solve_eigenproblem`, `split_operator_propagate`, `compute_wigner_function`, etc. |
| Load Modeling | `analyze_server_load`, `compute_adaptive_throttle`, `detect_load_anomaly` |
| Code Execution & Scratch | `execute_python` (trusted opt-in), `create_scratch_database`, `generate_synthetic_profiles`, `demo_spectral_pipeline` |
| Artifact Inspection | `list_artifacts` |
| Audit | `write_audit_log` |

## Example Sessions

### Session 1: First Contact (Verify & Explore)

**Prompt to give an AI agent:**

> Connect to the spectral-packet-engine MCP server. Run `self_test` to verify everything works, then `inspect_product` to see what this engine does. After that, run `demo_spectral_pipeline` to see a complete end-to-end spectral analysis.

This gives the agent a complete onboarding in three tool calls.

### Session 2: Generate, Analyze, Report (No Input Files Needed)

**Prompt:**

> Use the spectral engine to generate 100 synthetic density profiles, write them to a scratch database, then run a full profile table report on them. Show me the convergence diagnostics and compression quality.

**What the agent does:**

1. `generate_synthetic_profiles` with `output_format="sqlite"`, `num_profiles=100`
2. `report_database_profile_query` on the scratch database with `SELECT * FROM profiles`
3. Returns convergence, compression quality, spectral analysis

### Session 3: Quantum Physics Analysis

**Prompt:**

> I want to study quantum tunneling through a double barrier. Set up two rectangular barriers using the scattering module, compute the transmission spectrum T(E) for energies from 0 to 100, then find the resonances. Compare with WKB tunneling estimates.

**What the agent does:**

1. `scattering_analysis` with double barrier segments
2. `wkb_analysis` on the same barrier shape
3. Compares exact transfer-matrix T(E) with WKB approximation

### Session 4: Custom Python Analysis via MCP (Trusted Opt-In)

**Prompt:**

> Use `execute_python` to create a Morse potential, solve for the first 20 eigenvalues, compute the Wigner function of the ground state, and analyze its density matrix. Print a summary of all results. Only do this if the server was intentionally started with `--allow-unsafe-python`.

**What the agent sends:**

```python
# This code runs on the MCP server
import torch

domain = spe.InfiniteWell1D.from_length(1.0, dtype=torch.float64)

# Solve eigenvalue problem for Morse potential
eigresult = spe.solve_eigenproblem(
    lambda x: spe.morse_potential(x, D=100.0, a=6.0, domain=domain),
    domain, num_points=256, num_states=20,
)
print(f"First 5 eigenvalues: {eigresult.eigenvalues[:5].tolist()}")

# Wigner function of ground state
grid = torch.linspace(float(domain.left), float(domain.right), 128, dtype=torch.float64)
psi_0 = eigresult.eigenstates[0]
wigner = spe.compute_wigner(psi_0, grid, num_p=64)
print(f"Wigner negativity: {float(wigner.negativity):.6f}")

# Density matrix analysis
rho = spe.pure_state_density_matrix(psi_0)
dm = spe.analyze_density_matrix(rho)
print(f"Purity: {float(dm.purity.real):.6f}")
print(f"Von Neumann entropy: {float(dm.von_neumann_entropy.real):.6f}")

result = {
    "eigenvalues": eigresult.eigenvalues[:20].tolist(),
    "wigner_negativity": float(wigner.negativity),
    "purity": float(dm.purity.real),
}
```

### Session 5: Server Load Analysis

**Prompt:**

> I have a web server getting ~80 requests per second with periodic spikes. Use the spectral engine to model this traffic pattern, classify it, and tell me the optimal throttling parameters. Also compare it against a 50 req/s baseline to see if the current pattern is anomalous.

**What the agent does:**

1. `execute_python` to generate realistic request timestamps on a trusted runtime with `--allow-unsafe-python`
2. `analyze_server_load` with those timestamps, `capacity_rps=100`, and baseline
3. Returns regime classification, cooldown, interval, anomaly assessment

### Session 6: SQL + Spectral Feature Pipeline

**Prompt:**

> Create a scratch SQLite database with 200 synthetic profiles. Export spectral features from it (16 modes), then train a random forest on the features to predict the profile center position. Show me the feature importances.

**What the agent does:**

1. `create_scratch_database` with `populate_synthetic=True`, `num_profiles=200`
2. `export_feature_table_from_sql` on the scratch database
3. `train_tree_model` with `target_column="moment_mean"`, `library="sklearn"`
4. Returns R-squared, feature importances

### Session 7: Berry Phase & Quantum Information

**Prompt:**

> Compute the Berry phase for a spin-1/2 particle as the magnetic field traces a cone on the Bloch sphere with half-angle theta from 0 to pi. Verify that it matches the theoretical solid angle. Then compute the quantum Fisher information and entanglement entropy for a Bell state.

**What the agent does:**

1. `berry_phase_analysis` with spin-half parameters
2. `quantum_info_analysis` with Bell state density matrix
3. Cross-validates Berry phase = half solid angle, concurrence = 1 for Bell state

### Session 8: Full Showcase (The Compelling Demo)

**This is the prompt you give someone to show what the engine can do:**

> I want to see the full power of this spectral engine. Do this sequence:
>
> 1. Run `self_test` to verify the server.
> 2. Solve the quantum harmonic oscillator for 20 eigenvalues and compare with exact E_n = (n+1/2)hbar*omega.
> 3. Propagate a Gaussian wavepacket through a double-well potential using split-operator time evolution. Report norm and energy conservation.
> 4. Compute the Wigner function of the propagated state and check for non-classical negativity.
> 5. Generate 100 synthetic density profiles, project them onto a spectral basis, analyze convergence, and compress them.
> 6. Export spectral features to a scratch SQL database, train a tree model, and report feature importances.
> 7. Analyze mock server traffic (generate 1000 request timestamps with burst patterns) using spectral load modeling. Report the traffic regime and recommended throttling.
> 8. Write all findings to an audit log entry.
>
> This should all happen through MCP tool calls. If `execute_python` is disabled, stay on the dedicated physics and load-modeling tools instead of assuming arbitrary Python is available.

This single prompt exercises: eigenvalue solver, split-operator propagation, Wigner function, density profiles, spectral compression, SQL, tree models, load modeling, and audit logging. Every tool call uses the same underlying spectral engine.

## CLI Equivalents

Every MCP tool has a CLI equivalent. The key mapping:

| MCP Tool | CLI Command |
| --- | --- |
| `inspect_product` | `spectral-packet-engine inspect-product` |
| `validate_installation` | `spectral-packet-engine validate-install --device cpu` |
| `profile_table_report` | `spectral-packet-engine profile-report <path>` |
| `train_tree_model` | `spectral-packet-engine tree-train <features> --target-column y` |
| `analyze_server_load` | `spectral-packet-engine load-analyze <timestamps>` |
| `compute_adaptive_throttle` | `spectral-packet-engine load-throttle <rates>` |
| `estimate_server_capacity` | `spectral-packet-engine load-capacity <rates>` |

## Scratch Directory

MCP tools that generate data write to a managed scratch directory:

- **Location:** a managed cache-backed scratch directory chosen by the runtime
- **Contents:** temporary CSV files, SQLite databases, synthetic profiles
- **Lifetime:** persists across MCP tool calls until the operator or cache policy removes it
- **Access:** the `execute_python` tool receives `scratch_dir` as a pre-set variable when unsafe execution was explicitly enabled

When connecting to a remote server, use `server_info` to see the scratch directory path on that machine.

## Connection Metadata

The `server_info` tool reports:

- `transport`
- `bind_host`
- `bind_port`
- `streamable_http_path`
- `endpoint_url`
- `best_effort_ipv4`

Use the bind and endpoint fields as the authoritative connection facts. `best_effort_ipv4` is only a best-effort hostname resolution and may not be remotely routable. This matters when:

- Generating file paths that need to be valid on the server side
- Configuring database URLs (SQLite paths are server-local)
- Understanding artifact locations

## Startup Checklist for New Deployments

When deploying the spectral engine MCP server on a new machine:

```bash
# 1. Install
python3 -m pip install -e ".[mcp,sql,ml-tree-core]"

# 2. Validate
spectral-packet-engine validate-install --device cpu
spectral-packet-engine inspect-environment --device cpu

# 3. Start MCP server
spectral-packet-engine serve-mcp --max-concurrent-tasks 2 --log-file logs/mcp.log

# 4. From the AI client, run:
#    self_test     → verify all modules
#    server_info   → confirm connection
#    probe_mcp_runtime(profile=\"smoke\") → confirm the real protocol surface
```

The `self_test` tool validates all engine subsystems (core, physics, load modeling, SQL, scratch directory) in one call. If any check fails, it reports exactly which module has a problem.
