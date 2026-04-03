# MCP Usage

## Why Use MCP Instead Of Direct Python

Use direct Python when the calling code already lives in Python and you want in-process tensors and objects.

Use MCP when you want an LLM client to delegate bounded, structured compute tasks to the machine through explicit domain tools instead of arbitrary shell execution.

The answer to "why connect my machine to this backend?" is straightforward:

- the backend can validate the install,
- inspect real tables on disk,
- run bounded numerical jobs,
- write reproducible artifact bundles,
- let the client retrieve results without turning the session into raw shell automation.

That makes the MCP path useful for:

- packet simulation,
- modal projection,
- SQL and file-backed profile ingestion,
- database bootstrap, inspection, query materialization, and table writes,
- backend-aware surrogate training and evaluation with explicit backend choice,
- inverse reconstruction from observed profiles,
- transport benchmark runs,
- compatibility TensorFlow training workflows when TensorFlow is installed.

## Install

```bash
python3 -m pip install -e ".[mcp]"
```

## Run

```bash
spectral-packet-engine serve-mcp
```

The server runs over stdio, which is the standard local-machine MCP integration path.

## Tool Surface

The tools are organized around machine-side jobs:

- inspect environment and validate installation,
- inspect ML backends,
- bootstrap and inspect databases,
- inspect supported file formats,
- load, inspect, and analyze profile tables,
- run packet simulation and modal projection,
- run packet sweeps and profile-table compression,
- compare reference and candidate tables,
- run compression sweeps and inverse reconstruction,
- run backend-aware modal-surrogate training and evaluation from files or SQL queries,
- run transport benchmark execution,
- run optional TensorFlow surrogate training and evaluation,
- list artifacts after a job completes.

Each tool delegates to the same workflow layer used by the Python package and CLI.

## Output Strategy

Most MCP tools can also write structured artifacts when an `output_dir` is provided. Typical outputs include:

- JSON summaries,
- reconstructed profile CSV files,
- coefficient tables,
- predicted density tables,
- residual profile tables,
- per-sample metric tables.
- artifact indexes.

The MCP surface also includes artifact listing so an LLM client can retrieve the output bundle after a compute run.

Database-query tools return redacted database URLs in their summaries rather than echoing raw connection credentials.
