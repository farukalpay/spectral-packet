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

### Container-first deployment

If you want the runtime contract and health check to stay identical across laptops and servers, use the repository Docker assets:

```bash
docker compose up -d --build
```

That publishes `http://127.0.0.1:8765/mcp`, persists scratch and logs under `docker-data/`, and health-checks the real callable bridge route at `/mcp/server_info` instead of a synthetic placeholder.
The compose contract binds to `127.0.0.1` by default so an nginx or Caddy front end can stay the only public edge; override `SPE_PUBLISHED_HOST` only when a direct bind is intentional.
The Docker image installs the CPU PyTorch wheel explicitly so a default Linux deployment does not pull CUDA-heavy runtimes unless you choose a different container build.
The persisted scratch volume now also carries `_storage_guard/`, which stores the managed SQLite mutation ledger plus guarded snapshots. If a managed scratch database disappears between restarts, the next container startup restores it automatically from that guard state unless `SPE_STORAGE_RESTORE_ON_STARTUP=false` was set deliberately.

If `.env` is populated with the remote host settings, the repository can push the same Docker deployment over SSH:

```bash
./scripts/deploy_mcp_docker.sh
```

The deploy script exports a stable `COMPOSE_PROJECT_NAME`, removes older managed containers that still publish the target port, and can stop legacy `spectral_packet_engine serve-mcp` systemd units on that same port before it starts Docker. If the port is still owned by an unrelated process, the script fails with ownership diagnostics instead of starting a half-published deployment.
If `SPE_PUBLIC_HOSTS` is configured, the container entrypoint auto-derives `SPE_ALLOWED_HOSTS` and `SPE_ALLOWED_ORIGINS` from the public ingress contract instead of leaving Host/Origin policy to manual drift.
If `SPE_INSTALL_NGINX_SITE=true`, the deploy script also renders a site file from the shared library plan, installs it under `/etc/nginx/sites-available`, enables it, validates `nginx -t`, reloads nginx, and verifies both the site root and `/mcp/server_info` through the local reverse proxy.
With `SPE_NGINX_SITE_MODE=detect` the deploy script first checks whether another enabled nginx site already owns one of the requested public hosts. When it detects a shared host, it refuses to install a competing server block, writes the generated MCP `location` snippet to `/etc/nginx/snippets/`, and exits with diagnostics instead of leaving hostname routing ambiguous.

### Managed SQLite protection

Managed SQLite databases are no longer treated like ordinary scratch files.

- Existing managed databases are sealed into the storage guard on first startup.
- SQLite mutation cost is measured from the actual page diff, so rewrites and deletes burn the same byte-denominated budget.
- The sustainable mutation rate is `protected_database_bytes / SPE_STORAGE_PROTECTION_WINDOW_SECONDS`.
- By default, that window is `86400` seconds, so mutating one current-database worth of pages again takes roughly one day of accumulated budget.
- `write_scratch_file` cannot overwrite `.db` / `.sqlite` files and `delete_scratch_file` cannot remove managed databases.
- `inspect_storage_economy` and `server_info` expose the current protected mass, spendable balance, refill rate, and guarded snapshot paths.

Useful environment variables for Docker and remote deploy:

- `SPE_STORAGE_PROTECTION_WINDOW_SECONDS`
- `SPE_STORAGE_SEED_BYTES`
- `SPE_STORAGE_MINIMUM_MUTATION_COST_BYTES`
- `SPE_STORAGE_SNAPSHOT_RETENTION`
- `SPE_STORAGE_RESTORE_ON_STARTUP`

Published HTTPS route:

- the current public MCP endpoint is [https://lightcap.ai/mcp](https://lightcap.ai/mcp),
- users should treat that HTTPS URL as the default shared deployment entrypoint,
- self-hosted deployments can keep the origin listener private while the reverse proxy publishes HTTPS.

### Reverse-proxy contract

Render the alignment plan before touching nginx:

```bash
spectral-packet-engine plan-mcp-ingress \
  --public-host lightcap.ai \
  --public-host www.lightcap.ai \
  --upstream-port 8765
```

That gives you:

- the canonical public endpoint URL,
- the exact `allowed_hosts` and `allowed_origins` values the MCP server should accept,
- the Docker env block for a loopback-bound container,
- the nginx server block for a dedicated hostname,
- the nginx location snippet for a hostname that is already owned by another site config.

If you publish the streamable-HTTP server behind nginx or another reverse proxy, proxy both the exact MCP mount and the scoped prefix. Publishing only the exact mount can make MCP `initialize` succeed while path-based compatibility routes still return `404`.

Example nginx shape:

```nginx
location = /mcp {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
}

location /mcp/ {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
}
```

After any reverse-proxy change, verify both:

- `GET /mcp`
- `GET /mcp/tool_registry`

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
- `http_bridge_tool_count`
- `http_bridge_fingerprint`

`best_effort_ipv4` is observational only. `endpoint_url` is the internal listener URL; the published public route is [https://lightcap.ai/mcp](https://lightcap.ai/mcp).

For streamable-HTTP deployments, every MCP tool is also mirrored through deterministic compatibility routes:

- `GET|POST /<tool_name>`
- `GET|POST /Lightcap/<tool_name>`
- `GET|POST /tool_registry`
- `GET|POST /Lightcap/tool_registry`
- `GET|POST /mcp/<tool_name>`
- `GET|POST /mcp/Lightcap/<tool_name>`
- `GET|POST /mcp/tool_registry`
- `GET|POST /mcp/Lightcap/tool_registry`

The path-scoped `/mcp/...` variants are the safest choice behind reverse proxies that only publish the MCP mount itself. All of these routes call the same shared tool implementations as MCP `call_tool`; they exist only to keep path-oriented clients and exported tool links aligned with the canonical MCP tool surface. `tool_registry` is the authoritative HTTP bridge manifest.

## 7. Example Prompts

Use prompts like these with an attached MCP client:

> Connect to the spectral-packet-engine MCP server. Run `self_test`, then `server_info`, then `inspect_product`. After that run `probe_mcp_runtime` with `profile="smoke"` and summarize any failures.

> Study quantum tunneling through a barrier with `tunneling_experiment`. Report the packet mean energy, exact and WKB transmission at the packet energy, and the transmitted/reflected probability after propagation.

> Stress the MCP surface safely. Run `probe_mcp_runtime` with `profile="audit"`, inspect the artifact bundle, and summarize which checks passed, which failed, and which logs a human operator should read first.
