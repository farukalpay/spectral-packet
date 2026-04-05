# AGENTS

Work on this repository as a product-minded engineering partner.

## Product Identity

This repository is a focused scientific compute product:

- bounded-domain spectral packet simulation,
- modal decomposition and compression,
- inverse reconstruction,
- Python-first workflows,
- CLI, MCP, and optional API interfaces over the same core.

Do not broaden the scope into generic AI platform work, manuscript-first theory expansion, or a general-purpose physics engine.

## Architectural Rules

- Keep the library core reusable and explicit.
- Keep CLI, MCP, and API layers thin wrappers over shared workflow functions.
- Keep artifact bundle creation in the shared artifact layer rather than reimplementing output logic per interface.
- When adding SQL support, prefer SQLAlchemy Core with parameterized operations over handwritten SQL concatenation or an ORM-heavy design.
- Keep generic tabular/data abstractions separate from `ProfileTable`; conversion into spectral workflows should be explicit.
- Keep raw-record normalization separate from both `TabularDataset` and `ProfileTable`. Normalize first, then materialize tables, then enter spectral or model workflows.
- Treat feature engineering as a product-aligned bridge into spectral analysis, inverse reconstruction, comparison, or surrogate modeling. Do not let it expand into a generic feature factory.
- Do not solve mixed-schema ingestion by forcing interface code to pre-clean rows manually. Put schema reconciliation and coercion policy in shared library code.
- Treat document ingestion as extraction, validation, and structured handoff. Do not imply semantic understanding when the code is only parsing text or tables.
- Any new ML subsystem must include real train/evaluate/persist behavior, not just training.
- Keep backend-routing policy centralized in the shared ML layer. CLI, MCP, API, and workflow wrappers must not invent their own backend fallback orders or platform rules.
- Keep optional-service compatibility checks centralized in shared runtime inspection code. Do not duplicate FastAPI or serving-stack checks across wrappers.
- Prefer model- and data-driven logic over hardcoded topic rules.
- Keep stable and experimental surfaces clearly labeled.
- Do not leak local filesystem paths into repository-facing docs.
- Keep the repository root product-first. Move legacy or historical materials under `docs/` or another explicit secondary location instead of leaving them at the top level.
- Preserve compatibility when possible; migrate cleanly instead of deleting useful work casually.

## File And Interface Conventions

- Use `pathlib` for paths.
- Design for Linux, Windows, and macOS intentionally.
- Keep file formats explicit and validated.
- Keep optional capability checks explicit for SQL drivers, Parquet, DOCX, PDF, and ML stacks.
- For table-style profile data, support real user file paths and deterministic artifact outputs.
- When adding schema inference, normalization, or reconciliation, make the policy explicit and inspectable. Do not hide coercion or dropping rules inside wrappers.
- Prefer shared, reusable normalization and feature-pipeline abstractions over interface-specific data cleanup code.
- Feature tables, analytical tables, and model inputs must remain traceable to source data and compatible with the spectral workflow layer. Do not create disconnected tabular side products.
- If you add a new user-facing workflow, expose it through the shared workflow layer first.
- Keep the repo explainable as one engine with multiple interfaces. If a change makes the product easier to describe as separate tools than as one coherent engine, redesign it.

## Validation Checklist

Before finishing:

1. Run `pytest -q`.
2. Run at least one CLI smoke command through `PYTHONPATH=src python3 -m spectral_packet_engine.cli ...` or the installed script.
3. Check docs for accidental absolute local paths or editor-only URIs before finishing.
4. Make sure stable vs beta vs experimental claims still match reality.
5. If you changed file handling, add or update tests for every supported format you touched.
6. If you changed CLI or MCP outputs, verify that artifact bundles are still consistent across both interfaces.
7. If you changed MCP discovery, validate tool names and descriptions from `list_tools()` instead of assuming the server metadata looks good.
8. For major workflow work, validate at least one human-user path and one AI-client path instead of relying only on unit tests.
9. For release-facing packaging work, run `python -m build --sdist --wheel` and `python -m twine check dist/*`.
10. For install-path changes, validate at least one isolated standard install and one isolated editable install.

## Docs Checklist

- README must answer what the project is, who it is for, why install it, and how to use Python vs CLI vs MCP vs API.
- Docs should prefer GitHub-relative links.
- If you add a new workflow, document:
  - input
  - command or code
  - outputs
  - artifact locations
  - what the user learns

## Release Readiness Checklist

- Package metadata and version reflect the current release state.
- New commands have tests or a documented reason they are optional.
- Optional dependency failures produce clear messages.
- Artifact bundles remain structured and predictable.
- Cross-platform notes remain honest and specific.
- Release claims only cover surfaces that have actually passed the current validation matrix.
