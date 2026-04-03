# Getting Started

This is the shortest path from a fresh clone to a real first run.

## 1. Clone The Repository

Clone the repository normally through GitHub, then change into the project directory.

## 2. Install The Core Package

Linux and macOS:

```bash
python3 -m pip install -e .
```

Windows:

```powershell
py -m pip install -e .
```

## 3. Confirm The Installed Release Surface

Installed script:

```bash
spectral-packet-engine --version
```

Module form:

```bash
python3 -m spectral_packet_engine --version
```

## 4. Validate The Installation

Installed script:

```bash
spectral-packet-engine validate-install --device cpu
```

Module form:

```bash
python3 -m spectral_packet_engine validate-install --device cpu
```

Windows module form:

```powershell
py -m spectral_packet_engine validate-install --device cpu
```

## 5. Run One Practical Workflow

Compress the bundled profile table:

```bash
spectral-packet-engine compress-table examples/data/synthetic_profiles.csv --modes 8 --device cpu --output-dir artifacts/compression
```

This writes:

- `artifacts/compression/compression_summary.json`
- `artifacts/compression/reconstruction.csv`
- `artifacts/compression/coefficients.csv`
- `artifacts/compression/artifacts.json`

These example files live in the repository checkout. If you install the package somewhere else, use your own profile-table files instead of `examples/data/`.

Inspect the available ML backends on the same machine:

```bash
spectral-packet-engine ml-backends --device cpu
```

## 6. Choose Your Main Interface

Use Python when you are integrating the engine into your own code.

Use the CLI when you want reproducible local runs and stable artifact bundles.

Use MCP when you want an LLM client to delegate structured numerical jobs to the machine.

Use the API when another process needs HTTP access to the same workflows.
