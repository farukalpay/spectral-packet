# Platform Support

## Summary

The project is intentionally engineered for:

- Linux
- Windows
- macOS

The package uses `pathlib`, packaged entry points, explicit capability checks, and library-owned artifact paths so the same Python, CLI, MCP, and API workflows can run across all three without shell-specific assumptions.

Operational target distinction matters:

- Linux and macOS are the first-class targets for supervised MCP deployments.
- Windows is supported for the library, CLI, API, and local MCP stdio use, but long-running supervised MCP deployment is best-effort in this release.

Validate any machine with either:

- `spectral-packet-engine validate-install --device cpu`
- `python -m spectral_packet_engine validate-install --device cpu`

## Support Matrix

| Area | Linux | Windows | macOS |
| --- | --- | --- | --- |
| Core library | Supported | Supported | Supported |
| CLI | Supported | Supported | Supported |
| MCP server | Supported, including supervised deployments with external restart policy | Best-effort local stdio use; supervised long-running deployment is not a first-class promise | Supported, including launchd-supervised deployments |
| HTTP API | Supported | Supported | Supported |
| CSV / TSV / JSON | Supported | Supported | Supported |
| XLSX / Parquet (`.[files]`) | Supported when optional deps are installed | Supported when optional deps are installed | Supported when optional deps are installed |
| Local SQLite workflows | Supported | Supported | Supported |
| Remote SQL (`.[sql]`) | Beta | Beta | Beta |
| PyTorch CPU | Supported | Supported | Supported |
| PyTorch CUDA | Supported when available | Supported when available | Not applicable |
| PyTorch MPS | Not applicable | Not applicable | Supported when available |
| JAX (`.[ml-jax]`) | Supported when local JAX/JAXLIB install is compatible | Not a primary target in this release | Supported when local JAX/JAXLIB install is compatible |
| TensorFlow compatibility path (`.[ml]`) | Supported on compatible Python and wheels | CPU-oriented path; WSL2 recommended for GPU | Supported only on a compatible local TensorFlow stack |

## Practical Notes

Linux:

- strongest target for heavier numerical, SQL-backed batch, and CUDA-oriented workloads,
- recommended path for the optional `ml-cuda` extra,
- primary target for broader backend validation,
- preferred target for supervised MCP deployment with `systemd` or a container restart policy.

Windows:

- core package, CLI, API, tabular workflows, and local SQLite workflows are first-class,
- local stdio MCP usage is supported when the `mcp` extra is installed,
- supervised long-running MCP deployment is best-effort and should not be treated as release-grade parity with Linux or macOS,
- prefer `py -m spectral_packet_engine ...` if the script entrypoint is not on `PATH`,
- JAX is not a primary target in this release,
- TensorFlow GPU is not the primary target; WSL2 is the recommended route for GPU-oriented TensorFlow work.

macOS:

- core package is supported on bare metal,
- PyTorch can select MPS when available,
- JAX support depends on a compatible local wheel stack,
- TensorFlow depends on a compatible Python version and local TensorFlow installation,
- `launchd` is the preferred external supervisor for long-running MCP deployments.

## Capability Checks

Use the packaged inspection commands instead of guessing:

```bash
spectral-packet-engine validate-install --device cpu
spectral-packet-engine ml-backends --device cpu
spectral-packet-engine db-bootstrap artifacts/local.db
```

These commands report:

- Python and torch runtime state,
- available ML backends and backend-specific notes,
- optional API, MCP, SQL, XLSX, and Parquet capabilities,
- MCP runtime transport and supervision notes,
- local SQLite bootstrap readiness,
- which file and tabular formats are supported in the active environment.
