# Final Engineering Standard

Date: 2026-04-04

This is the canonical internal standard for the final convergence pass.

Every important change must satisfy one rule:

The repository may only grow outward when the shared spectral engine, library contracts, runtime safety, and diagnostics first grow deeper.

## Required Properties

- Engine code must carry the mathematical and computational weight of the product. State, basis, propagation, inverse reconstruction, truncation, norms, and diagnostics must be explicit enough to inspect and trust.
- Library APIs must feel build-worthy. They must expose stable typed contracts, predictable configuration flow, composable result objects, and direct Python usage that looks like library code rather than script glue.
- Implementation depth matters more than feature count. If a choice exists between adding another surface and strengthening an existing one, strengthen the existing one.
- Wrappers must not pretend to be architecture. CLI, MCP, API, and façade modules are valid only when they expose real shared engine or runtime contracts underneath.
- Reliability and observability are first-class. Logging, health/status, failure context, startup checks, and overload behavior are part of correctness, not optional polish.
- Cross-platform and backend safety are part of the implementation. Python-version support, OS behavior, dependency compatibility, and backend routing must be explicit and testable.
- Originality must come from coherent engine and workflow design, not cosmetic novelty, naming tricks, or wrapper breadth.

## Automatic Rejection Conditions

Reject or rework a change if any of these are true:

- it mainly adds wrapper code without strengthening shared engine, workflow, runtime, or diagnostic contracts
- it introduces a new abstraction that does not reduce coupling or make decisions easier to inspect
- it duplicates numerical policy, schema-reconciliation logic, runtime inspection, or backend-routing policy that should exist in one place
- it hides unsupported platform or backend behavior behind implicit fallbacks or optimistic docs
- it makes the repo easier to describe as several adjacent tools instead of one spectral engine with extension layers
- it preserves shallow or fragile behavior only to keep a broad surface area alive
