# MCP Resilience Audit 2026-04

## Architectural Problem

The MCP surface had drifted into three conflicting states:

- the documented operational model assumed restartable supervised deployment,
- the default runtime was still mostly stdio-centric,
- the helper surface included trusted-only behaviors (`execute_python`, scratch synthesis) without a sufficiently explicit operational contract.

## Clean Structural Fix

Keep one MCP surface, but make the execution and deployment policy explicit:

- `stdio` remains the default client-launched transport,
- `streamable-http` is the supervised long-running transport,
- `execute_python` remains available only as a trusted opt-in capability,
- scratch helpers use one shared synthetic-profile generator and one managed scratch-path policy,
- the official MCP probe suite is the reproducible self-test and hardening loop.

## Findings Discovered And Fixed

1. `python -m spectral_packet_engine.mcp` was not the operationally preferred entrypoint and drifted from the generated client configuration.
   Fix:
   `.mcp.json`, generated configs, and probe tooling now use `python -m spectral_packet_engine serve-mcp`.

2. `execute_python` could read repository files and spawn subprocesses when enabled.
   Fix:
   the tool remains disabled by default and is now documented as trusted-only opt-in via `--allow-unsafe-python`.

3. `generate_synthetic_profiles` could emit incompatible CSV headers and non-finite data in edge cases.
   Fix:
   synthetic data generation now flows through a shared `ProfileTable` generator and the normal profile-table CSV writer.

4. Scratch helper paths accepted traversal-shaped names.
   Fix:
   scratch outputs are now constrained to the configured managed scratch directory.

5. Supervised deployment guidance lacked a transport distinction.
   Fix:
   streamable HTTP is now the restartable transport; stdio remains the client-launched transport.

6. `probe_mcp_runtime` could not safely probe itself while an event loop was already active.
   Fix:
   nested probing now runs through the real `probe-mcp` CLI path in a subprocess, and the lower-level probe suite falls back to a worker thread when called from an already-running event loop.

7. The high-level quantum-state report misreported position expectation by reusing the Pauli `sigma_x` expectation.
   Fix:
   the pipeline now computes spectral position expectation and variance directly from the reconstructed state instead of reusing an unrelated spin observable.

8. The tunneling workflow accepted `num_modes` but did not use it, and its comparison energy was effectively anchored to `V0/2`.
   Fix:
   the workflow now projects the packet into the spectral basis, derives a packet-energy interval from modal weights, compares exact and WKB transmission at the packet mean energy, and reports transmitted/reflected probability after propagation.

9. The double-well MCP wrappers passed the wrong keyword names into the shared potential builder.
   Fix:
   the wrappers now use the library contract (`a_param`, `b_param`) so the tool surface matches the underlying reusable physics layer.

## Reproducible Commands

Local MCP probe:

```bash
spectral-packet-engine probe-mcp --profile smoke --output-dir artifacts/mcp_probe
spectral-packet-engine inspect-artifacts artifacts/mcp_probe
```

Script wrapper over the same probe suite:

```bash
python3 scripts/mcp_stress_test.py --profile stress --output-dir artifacts/mcp_stress
```

Deeper protocol audit:

```bash
spectral-packet-engine probe-mcp \
  --profile audit \
  --output-dir artifacts/mcp_probe_audit
```

Restartable streamable-HTTP MCP endpoint:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --host localhost \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log
```

Service manifest preview:

```bash
spectral-packet-engine install-mcp-service --dry-run
```

SSH tunnel plan for a private remote service:

```bash
spectral-packet-engine plan-mcp-tunnel --host user@example-host --remote-port 8765 --streamable-http-path /mcp
```

Client config generation:

```bash
spectral-packet-engine generate-mcp-config --transport stdio
spectral-packet-engine generate-mcp-config --transport ssh --host user@example-host --remote-cwd /srv/spectral-packet-engine
```

Validation run used for this audit:

```bash
pytest -q
PYTHONPATH=src python3 -m spectral_packet_engine.cli profile-report examples/data/synthetic_profiles.csv --analyze-modes 8 --compress-modes 4 --device cpu --output-dir artifacts/final_profile_report
PYTHONPATH=src python3 -m spectral_packet_engine.cli probe-mcp --profile smoke --output-dir artifacts/final_mcp_probe --source-checkout
```

## Artifact Bundle

The probe suite writes:

- `mcp_probe_report.json`
- `mcp_probe_results.jsonl`
- `mcp_probe_summary.md`
- `mcp_tool_calls.jsonl`
- `server.log`
- `artifacts.json`

These files are sufficient to reproduce:

- which MCP calls were issued,
- which probes passed or failed,
- what the server reported,
- how the server was launched.

The latest validated smoke bundle lives at `artifacts/final_mcp_probe/` and currently reports `13` passed probes and `0` failures.

## Honest Limits After This Pass

- `execute_python` is still fundamentally a trusted-operator capability. The principled fix is explicit opt-in, not a pretend sandbox.
- automatic restart is meaningful only for the `streamable-http` transport, not for stdio.
- the repository does not hide remote defaults or silently auto-connect clients to a network service. Remote MCP use remains explicit by design.
