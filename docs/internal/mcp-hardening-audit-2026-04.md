# MCP Hardening Audit (2026-04)

## Architectural Problem

The MCP surface had drifted in three operationally important ways:

1. The documented module entrypoint and `.mcp.json` launch path were not aligned with the runtime behavior.
2. Trusted-only capabilities such as `execute_python` were too easy to expose or describe as if they were normal day-1 tools.
3. Restartable deployment and remote access were described, but the product surfaces for supervision, probing, and connection generation were not yet cohesive.

## Clean Structural Solution

- Keep the shared spectral engine unchanged and harden the MCP wrapper.
- Make arbitrary Python execution an explicit opt-in runtime policy instead of a silent default.
- Add a real self-probe workflow that uses the MCP protocol itself and writes reproducible artifacts.
- Support two operational modes explicitly:
  - `stdio` for client-launched local sessions
  - `streamable-http` for supervised long-running service mode
- Generate client configs and service manifests from shared library code instead of hardcoded doc snippets.

## What Was Implemented

- `serve-mcp` now exposes `stdio` and `streamable-http` transports through the shared `MCPServerConfig`.
- `python -m spectral_packet_engine.mcp --help` now works again and reflects the actual runtime options.
- `.mcp.json` now launches the package entrypoint instead of a broken module path.
- `execute_python` is disabled by default and returns a structured `PermissionError` payload unless the operator starts the server with `--allow-unsafe-python`.
- Managed scratch operations now run through the shared scratch-path policy, including traversal rejection for helper tools.
- `probe-mcp` now runs a real protocol-level self-probe suite and writes:
  - `mcp_probe_report.json`
  - `mcp_tool_calls.jsonl`
  - `mcp_probe_summary.md`
  - `artifacts.json`
- `install-mcp-service` now renders an opt-in restartable user service plan for `streamable-http`.
- `generate-mcp-config` now produces local or SSH-backed MCP client config blocks from the shared deployment layer.

## Reproduction

Baseline local probe:

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine probe-mcp --output-dir artifacts/mcp_probe
spectral-packet-engine inspect-artifacts artifacts/mcp_probe
```

Trusted local runtime for arbitrary Python:

```bash
spectral-packet-engine serve-mcp --allow-unsafe-python --max-concurrent-tasks 1 --log-level warning
```

Remote SSH client config generation:

```bash
spectral-packet-engine generate-mcp-config --transport ssh --host user@example-host --remote-cwd /srv/spectral-packet-engine
```

Restartable service plan:

```bash
spectral-packet-engine install-mcp-service --dry-run
```

## Probe Outcome

The current local probe run passed all built-in checks in the hardened default configuration:

- runtime bootstrap
- runtime policy inspection
- shared self-test
- malformed input handling
- scratch-path containment
- SQLite side-effect rejection
- tunneling pipeline numerical stability
- `execute_python` capability gate

The key positive signal is that the probe suite now verifies both product behavior and safety posture through the same MCP surface that external clients use.

## Residual Limits

- `execute_python` remains intentionally powerful when enabled. This is not a sandbox and should stay restricted to trusted local sessions.
- `python -m spectral_packet_engine.mcp --help` still emits a `runpy` warning because the package imports the MCP module before module execution. The supported production entrypoints remain `spectral-packet-engine serve-mcp` and `python -m spectral_packet_engine serve-mcp`.
- Restart behavior still belongs to the external supervisor. The repository now generates manifests and transport settings, but it does not implement an in-process restart manager.
