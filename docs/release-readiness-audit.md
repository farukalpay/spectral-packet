# Release Readiness Audit

This document records the final pre-release hardening pass.

## Trust-Breaking Issues Found

- SQLite `create_if_missing=False` was not enforced consistently.
- Unsupported-format errors listed optional formats as if they were always installed.
- XLSX readers and writers did not close workbook handles explicitly.
- Capability inspection still exposed some internal-looking flags instead of product-facing capabilities.
- CLI help was technically correct but understated optional-file support and the SQL plus backend-aware ML product spine.
- The README still surfaced too many internal phase-audit documents and featured a TensorFlow compatibility example ahead of the primary backend-aware ML workflow.

## Fixes Applied In This Pass

- enforced strict SQLite missing-database behavior in the database layer,
- stopped read-oriented database workflows from silently creating new SQLite files,
- redacted database URLs consistently across query-oriented CLI, MCP, and API responses,
- made unsupported-format errors distinguish installed formats from optional ones,
- closed workbook handles explicitly in XLSX load/save paths,
- tightened environment capability reporting around PyTorch, JAX, TensorFlow, MCP, SQLite, SQLAlchemy, XLSX, and Parquet,
- improved CLI descriptions and help text around SQL-backed and backend-aware workflows,
- added structured API error handling for missing files, missing optional modules, and invalid requests,
- added a real backend-aware ML example script,
- reduced README noise and centered the front page on current product docs instead of older phase audits,
- added regression tests for the SQLite lifecycle fix, updated format-error paths, redacted query summaries, and API error behavior.

## Remaining Honest Limits

- remote SQL backends remain beta and need broader backend-specific validation,
- TensorFlow remains a compatibility path, not the primary ML workflow,
- JAX is supported but is not the primary Windows target in this release,
- model reload and checkpoint-resume flows are still narrower than the training/evaluation/export surface,
- document ingestion beyond structured table formats is still out of scope for this release.
