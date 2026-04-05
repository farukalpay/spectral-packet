# MCP Operations

This is the operator playbook for running, hardening, and reproducing the Spectral Packet Engine MCP surface.

## 1. Local Stdio MCP

Use this when Claude Code, Claude Desktop, Cursor, VS Code, or another MCP client can launch a local process directly.

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine validate-install --device cpu
spectral-packet-engine serve-mcp --max-concurrent-tasks 2 --log-level warning
```

No IP address is required in this mode. The client starts the server on demand and communicates over stdio.

If the client needs a config block, generate it:

```bash
spectral-packet-engine generate-mcp-config --transport stdio
```

## 2. Restartable Remote MCP

Use this when you want the server to stay up after crashes or machine restarts.

Start streamable HTTP on the remote machine:

```bash
spectral-packet-engine serve-mcp \
  --transport streamable-http \
  --port 8765 \
  --streamable-http-path /mcp \
  --log-file logs/mcp-http.log
```

Install the generated user service manifest:

```bash
spectral-packet-engine install-mcp-service --dry-run
spectral-packet-engine install-mcp-service --yes --enable
```

`install-mcp-service` is opt-in. It does not change the Python library, the CLI, or existing scripts unless you explicitly enable the new supervised MCP path.

Published HTTPS route:

- the current public MCP endpoint is [https://lightcap.ai/mcp](https://lightcap.ai/mcp),
- users should treat that HTTPS URL as the default shared deployment entrypoint,
- self-hosted deployments can keep the origin listener private while the reverse proxy publishes HTTPS.

## 3. SSH Tunnel Plan

If the service stays private on the remote host, render the exact tunnel command:

```bash
spectral-packet-engine plan-mcp-tunnel \
  --host user@example-host \
  --local-port 8765 \
  --remote-port 8765 \
  --streamable-http-path /mcp
```

That prints:

- the exact `ssh -L ...` command,
- the local endpoint URL,
- the remote endpoint URL.

Use the local endpoint URL with any MCP client that supports HTTP transport.

If the client prefers SSH-launched stdio instead of HTTP, generate the bridge config:

```bash
spectral-packet-engine generate-mcp-config --transport ssh --host user@example-host --remote-cwd /srv/spectral-packet-engine
```

## 4. Self-Audit And Stress

Fast smoke audit:

```bash
spectral-packet-engine probe-mcp --profile smoke --output-dir artifacts/mcp_probe
```

Deeper protocol audit:

```bash
spectral-packet-engine probe-mcp --profile audit --output-dir artifacts/mcp_probe_audit
```

Repeated-load stress run:

```bash
python3 scripts/mcp_stress_test.py --profile stress --output-dir artifacts/mcp_stress
```

The audit hits the real MCP surface and records:

- startup and tool discovery,
- runtime policy and connection metadata,
- malformed-input handling,
- scratch-path containment,
- SQL side-effect rejection,
- tunneling workflow numerical stability,
- trusted-code execution gating,
- repeated workload behavior,
- burst-load behavior.

## 5. Artifact Bundle

The probe and stress runs write reproducible artifacts:

- `mcp_probe_report.json`
- `mcp_probe_results.jsonl`
- `mcp_tool_calls.jsonl`
- `mcp_probe_summary.md`
- `server.log`
- `artifacts.json`

Use:

```bash
spectral-packet-engine inspect-artifacts artifacts/mcp_probe
```

to confirm the bundle is complete.

## 6. What AI Clients Should Call First

Recommended first calls after attaching:

1. `self_test`
2. `server_info`
3. `inspect_product`
4. `probe_mcp_runtime(profile="smoke")`

`server_info` fields to trust:

- `transport`
- `bind_host`
- `bind_port`
- `streamable_http_path`
- `endpoint_url`
- `allowed_hosts`
- `allowed_origins`

`best_effort_ipv4` is observational only. `endpoint_url` is the internal listener URL; the published public route is [https://lightcap.ai/mcp](https://lightcap.ai/mcp).

## 7. Example Prompts

Use prompts like these with an attached MCP client:

> Connect to the spectral-packet-engine MCP server. Run `self_test`, then `server_info`, then `inspect_product`. After that run `probe_mcp_runtime` with `profile="smoke"` and summarize any failures.

> Study quantum tunneling through a barrier with `tunneling_experiment`. Report the packet mean energy, exact and WKB transmission at the packet energy, and the transmitted/reflected probability after propagation.

> Stress the MCP surface safely. Run `probe_mcp_runtime` with `profile="audit"`, inspect the artifact bundle, and summarize which checks passed, which failed, and which logs a human operator should read first.
