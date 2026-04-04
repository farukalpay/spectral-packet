# MCP Usage

## Purpose

The MCP surface exists so an external tool client can call the same bounded-domain engine through explicit tools instead of shelling into the machine.

It is useful when you want:

- structured numerical tools,
- stable artifact bundles,
- explicit environment and health inspection,
- the same workflows the Python library and CLI already use.

It is not a separate product. MCP is a machine-facing wrapper over the shared workflow layer.

## Install

```bash
python3 -m pip install -e ".[mcp]"
```

Validate the local environment before wiring a client to the server:

```bash
spectral-packet-engine inspect-product
spectral-packet-engine inspect-environment --device cpu
spectral-packet-engine validate-install --device cpu
```

## Start

```bash
spectral-packet-engine serve-mcp
```

The server runs over stdio, which is the normal local-machine MCP transport.

Useful runtime controls:

```bash
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --slot-timeout-seconds 60 --log-level warning
spectral-packet-engine serve-mcp --log-file logs/mcp.log
```

Important rule: stdout is reserved for MCP protocol messages. Repository-managed logs go to stderr by default or to the log file you configure.

## Core Tool Families

- environment and install inspection
- service status inspection
- file-format and tabular capability inspection
- profile-table report, analysis, compression, comparison, and inverse fitting
- bounded-domain packet simulation and modal projection
- SQLite bootstrap, query, materialization, and SQL-backed report/analyze/compress/fit workflows
- backend-aware modal-surrogate training and evaluation

## Runtime Trust Signals

The MCP server exposes the same runtime model as the rest of the product:

- `inspect_product` for the shared product spine, runtime spine, and workflow map
- `guide_workflow` for the recommended high-value loop and defaults for file-backed or SQL-backed work
- `inspect_environment` for machine capability inspection
- `inspect_mcp_runtime` for transport, platform, bounded-execution, and logging policy inspection
- `validate_installation` for surface readiness
- `inspect_service_status` for uptime, task counters, and recent runs with canonical `workflow_id` plus raw `surface_action`

That lets an MCP client ask:

- is this machine ready,
- which workflow should I run instead of inventing a tool chain,
- how is the MCP runtime configured,
- what backends are installed,
- what just ran,
- where did artifacts go.

## Runtime Model

The repository guarantees a narrow in-process runtime model for MCP:

- stdio transport with stdout reserved for protocol traffic,
- repository-managed logging on stderr or an explicit file,
- bounded in-process execution slots for compute-heavy tools,
- shared workflow execution with the same spectral engine used by Python, CLI, and API,
- atomic artifact writes and output-directory locking for artifact-producing tools.

The repository does not promise:

- forced in-process cancellation of arbitrary running numerical work,
- automatic restarts,
- distributed scheduling,
- a generic job-queue platform.

If a heavy tool cannot acquire an execution slot before the configured timeout, it fails clearly so the client can retry or reduce concurrency expectations.

## Artifacts

Most compute tools can write artifact bundles when `output_dir` is provided.

Typical outputs include:

- JSON summaries
- top-level profile report overviews
- reconstructed profile tables
- coefficient tables
- predicted moments
- artifact indexes

Artifact-writing behavior is shared with the CLI and API instead of being reimplemented inside the MCP layer.

Artifact directories are restart-safe at the repository level:

- data files are staged and atomically moved into place,
- artifact directories are locked during writes to avoid cross-request collisions,
- temporary runtime files are cleaned up on the next managed write,
- a bundle is only considered complete when `artifacts.json` reports `"status": "complete"`.

Use `list_artifacts` to inspect:

- whether the directory exists,
- whether the bundle is complete,
- which files are present,
- what metadata was recorded.

For SQL-backed profile-table tools, MCP exposes the same explicit controls as Python and the CLI:

- `time_column`
- optional `position_columns`
- optional `sort_by_time`

Artifact indexes record the database URL, query text, bound parameters, and the profile-table materialization policy when those SQL-backed tools write outputs.

For the strongest bounded day-1 workflow through MCP, prefer:

- `inspect_product` before first use if the client needs the product and workflow map
- `profile_table_report` for a file-backed table
- `report_database_profile_query` for a SQL-backed table
- `list_artifacts` after the run if you asked the tool to write outputs

## Deployment Guidance

For local use, stdio MCP is enough:

```bash
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-file logs/mcp.log
```

For supervised long-running use, prefer an external service manager:

- Linux: `systemd` or a container restart policy
- macOS: `launchd`
- Windows: local stdio use is supported, but supervised long-running deployment is best-effort rather than a first-class promise

The process itself is restart-safe around artifact state, but restart policy belongs outside the repository process.

Minimal Linux `systemd` example:

```ini
[Unit]
Description=Spectral Packet Engine MCP
After=network.target

[Service]
ExecStart=/opt/spectral-packet-engine/.venv/bin/spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-file /var/log/spectral-packet-engine/mcp.log
Restart=on-failure
WorkingDirectory=/opt/spectral-packet-engine/app

[Install]
WantedBy=multi-user.target
```

Minimal macOS `launchd` example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>dev.spectral-packet-engine.mcp</string>
    <key>ProgramArguments</key>
    <array>
      <string>/opt/spectral-packet-engine/.venv/bin/spectral-packet-engine</string>
      <string>serve-mcp</string>
      <string>--max-concurrent-tasks</string><string>1</string>
      <string>--log-file</string><string>/opt/spectral-packet-engine/logs/mcp.log</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>WorkingDirectory</key><string>/opt/spectral-packet-engine/app</string>
  </dict>
</plist>
```

## Troubleshooting

- If `serve-mcp` reports a missing module, install the `mcp` extra in the active environment.
- If a request fails with a saturation error, reduce concurrent heavy requests or increase `--max-concurrent-tasks` conservatively.
- If artifact creation reports that an output directory is busy, avoid sharing one `output_dir` across overlapping heavy jobs.
- If you need logs, use stderr or `--log-file`; do not emit anything to stdout from wrapper code or shell glue around the MCP process.

## Status

MCP is currently a beta surface. Linux and macOS are the first-class operational targets for supervised MCP use. Windows local stdio use degrades gracefully, but it is not yet claimed as a first-class supervised deployment target. The Python library and CLI remain the most directly validated surfaces.
