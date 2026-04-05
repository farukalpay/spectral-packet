# API Usage

## Purpose

The HTTP API is an optional service layer over the same workflow surface exposed by the Python package, CLI, and MCP server.

Use it when another process needs:

- a stable JSON boundary,
- network access instead of a Python import,
- health, capability, and status endpoints,
- the same artifact-writing workflows behind HTTP.

It is not a separate backend product.

## Install

```bash
python3 -m pip install -e ".[api]"
```

## Start

```bash
spectral-packet-engine serve-api --host localhost --port 8000
```

If FastAPI is installed but incompatible in the current environment, the package reports that explicitly instead of pretending the API is usable.

For long-running deployment, use an external supervisor or container restart policy. The repository process reports health, status, and artifact completeness, but it does not implement its own restart manager.

## First Checks

```bash
curl http://localhost:8000/product
curl "http://localhost:8000/workflow/guide?input_kind=profile-table-file"
curl http://localhost:8000/health
curl http://localhost:8000/status
curl http://localhost:8000/validate-install
```

These endpoints answer:

- what product and runtime contract the service is exposing,
- which high-value workflow the product recommends for a file-backed or SQL-backed input,
- whether the process is alive, which product it is serving, and which version is running,
- what canonical workflows have been running and through which API route bindings,
- does the current machine satisfy the runtime expectations of the product.

## Main Endpoint Families

- runtime inspection: `/product`, `/workflow/guide`, `/health`, `/status`, `/capabilities`, `/validate-install`, `/ml/backends`, `/api/stack`
- tree runtime inspection: `/tree/backends`
- file and artifact inspection: `/file-formats`, `/tabular-formats`, `/artifacts`
- database workflows: inspect, bootstrap, query, write, materialize
- engine workflows: forward simulation, projection, packet sweeps
- profile-table workflows: inspect, report, analyze, compress, compare, inverse fit, feature export, and SQL-backed report/analyze/compress/fit/export
- tree-model workflows: `/tree/train` and `/tree/tune` over explicit feature tables
- ML workflows: backend-aware train and evaluate from files or SQL
- TensorFlow compatibility workflows: explicit `/tensorflow/*` endpoints when installed

## Minimal Example

See [../examples/api_workflow.py](../examples/api_workflow.py) for a small client that posts the hero profile-table report request and prints the JSON response.

## Artifacts

When an endpoint accepts `output_dir`, it writes the same artifact bundles used by the CLI and MCP surfaces.

That keeps service-mode outputs predictable:

- JSON summaries
- top-level profile report overviews
- reconstructed profiles
- coefficient tables
- feature tables with schema and provenance
- tree-model training or tuning summaries plus persisted model artifacts
- artifact indexes

`GET /artifacts` returns the shared artifact-directory report used by the Python and MCP surfaces, including:

- whether the directory exists,
- whether the bundle is complete,
- artifact metadata,
- the visible file list.

`GET /status` returns the shared service-status report. Recent task entries include both:

- `workflow_id` for the canonical product workflow, when the route maps to one,
- `surface_action` for the raw API route binding that executed it.

For SQL-backed profile-table endpoints such as `/profiles/analyze-from-sql`, `/profiles/compress-from-sql`, and `/inverse/fit-from-sql`, requests can pass:

- `time_column`
- `position_columns`
- `sort_by_time`

Artifact indexes record the database URL, query, parameters, and materialization settings for those runs.

If you want the clearest single API call for the core product workflow, use:

- `POST /profiles/report` for a file-backed or caller-supplied table payload
- `POST /profiles/report-from-sql` for a SQL-backed table payload

If you want the clearest feature-model path after spectral preparation, use:

- `GET /tree/backends` before model selection
- `POST /features/export` or `POST /features/export-from-sql` to materialize spectral features
- `POST /tree/train` or `POST /tree/tune` once the target column is present in the feature table

## Status

The API is currently beta. It is useful and intentionally thin, but the Python library and CLI remain the most directly validated surfaces.
