# Getting Started

This is the shortest validated path from install to a real artifact-producing spectral report.

If you are on Windows, replace `python3` with `py`.

## 1. Install The Package

```bash
python3 -m pip install -e .
```

## 2. Confirm The CLI Is Available

```bash
spectral-packet-engine --version
```

You can also use the module form:

```bash
python3 -m spectral_packet_engine --version
```

## 3. Inspect The Product And Environment

```bash
spectral-packet-engine inspect-product
spectral-packet-engine guide-workflow
spectral-packet-engine inspect-environment --device cpu
```

This tells you:

- what the product is,
- which killer workflow the product recommends for your input shape,
- what the runtime guarantees,
- which interface names map to the same shared workflows,
- which optional surfaces and backends are available.

## 4. Validate The Environment

```bash
spectral-packet-engine validate-install --device cpu
```

This reports:

- platform and Python version,
- torch runtime,
- optional API and MCP availability,
- MCP runtime transport and supervision notes,
- available ML backends,
- supported file formats.

## 5. Run The Hero Workflow

```bash
spectral-packet-engine profile-report examples/data/synthetic_profiles.csv --device cpu --output-dir artifacts/profile_report
```

This is the fastest way to see the product center of gravity: one validated profile table turned into an inspectable modal-analysis and compression bundle.

Outside a source checkout, replace `examples/data/synthetic_profiles.csv` with your own profile-table file.

That writes:

- `artifacts/profile_report/artifacts.json`
- `artifacts/profile_report/profile_table_report.json`
- `artifacts/profile_report/profile_table_summary.json`
- `artifacts/profile_report/analysis/spectral_analysis.json`
- `artifacts/profile_report/compression/compression_summary.json`

Notice three things in the output:

- the table summary tells you what was loaded,
- the report overview tells you which modes dominate and how many modes capture the profile mass,
- the compression summary tells you how much error you paid for the chosen mode budget.

## 6. Inspect The Artifact Bundle

```bash
spectral-packet-engine inspect-artifacts artifacts/profile_report
```

This confirms:

- whether the bundle is complete,
- which files were written,
- which workflow metadata was recorded.

## 7. Run The Local Release Gate

```bash
spectral-packet-engine release-gate --device cpu
```

## 8. Try The Direct Python Surface

```python
from spectral_packet_engine import load_profile_table_report

report = load_profile_table_report(
    "examples/data/synthetic_profiles.csv",
    analyze_num_modes=16,
    compress_num_modes=8,
    device="cpu",
)
artifacts = report.write_artifacts("artifacts/profile_report_python")

print(report.overview.dominant_modes)
print(report.overview.mean_relative_l2_error)
print(artifacts.is_complete)
```

## 9. Choose The Interface You Actually Need

- Python: direct integration into your own code
- CLI: reproducible local workflows and artifacts
- MCP: machine-side structured tools for external tool clients
- API: optional HTTP service over the same workflows

If you plan to use MCP locally, the shortest operational path is:

```bash
python3 -m pip install -e ".[mcp]"
spectral-packet-engine inspect-product
spectral-packet-engine serve-mcp --max-concurrent-tasks 1 --log-level warning
```

Use stderr or `--log-file` for logs. Stdout is reserved for MCP transport messages.

For SQL-backed spectral jobs, all three machine-facing surfaces now share the same explicit profile-table controls:

- `time_column`
- optional `position_columns`
- optional `sort_by_time`

For curated runnable examples, start with [../examples/README.md](../examples/README.md).
