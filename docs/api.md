# API Usage

## Purpose

The HTTP API is an optional service layer for local or self-hosted deployment.

Status:

- beta interface,
- same workflow layer as the Python package and CLI,
- intended for service integration, not as a separate product.

It exists for cases where:

- another process needs network access instead of importing Python directly,
- you want a stable JSON boundary over the compute engine,
- you want to expose the same workflows behind an HTTP service,
- you want optional artifact bundles written by API jobs and retrievable afterward.

## Install

```bash
python3 -m pip install -e ".[api]"
```

## Start

```bash
spectral-packet-engine serve-api --host 127.0.0.1 --port 8000
```

If the local FastAPI stack is installed but incompatible, the package reports that explicitly instead of pretending API support is usable.

## Endpoints

- `GET /health`
- `GET /capabilities`
- `GET /validate-install`
- `GET /api/stack`
- `GET /ml/backends`
- `GET /file-formats`
- `GET /tabular-formats`
- `GET /artifacts`
- `GET /database/inspect`
- `GET /database/bootstrap`
- `GET /database/tables/{table_name}`
- `POST /database/query`
- `POST /database/write`
- `POST /database/materialize`
- `POST /forward`
- `POST /project`
- `POST /packet-sweep`
- `POST /profiles/inspect`
- `POST /profiles/analyze`
- `POST /profiles/analyze-from-sql`
- `POST /profiles/compress`
- `POST /profiles/compression-sweep`
- `POST /profiles/compare`
- `POST /inverse/fit`
- `POST /transport/benchmark`
- `POST /ml/train`
- `POST /ml/evaluate`
- `POST /ml/train-from-sql`
- `POST /ml/evaluate-from-sql`
- `POST /tensorflow/train`
- `POST /tensorflow/evaluate`

## Minimal Example

```bash
curl http://127.0.0.1:8000/health
```

Example client code is available in:

- [`../examples/api_client_demo.py`](../examples/api_client_demo.py)

When supported by the endpoint, include `output_dir` in the request body to write the same artifact bundles used by the CLI and MCP surfaces. The generic `/ml/*` endpoints use the backend-aware modal-surrogate workflow surface, while `/tensorflow/*` remains the explicit compatibility path.

Database-query responses return a redacted database URL instead of echoing raw connection credentials.

`GET /api/stack` reports the installed FastAPI, Starlette, Pydantic, and Uvicorn versions together with the compatibility decision the product uses before serving the API.
