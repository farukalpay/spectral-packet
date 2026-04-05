# Public Credibility Audit

Date: 2026-04-04

This audit asks one blunt question:

Would a skeptical GitHub visitor dismiss the repository before they try it?

## Fast Dismissal Risks

### 1. The examples surface looked like a lab bench, not a product surface

Before cleanup, `examples/` mixed:

- stable first-run demos,
- experimental scripts,
- reference-case scripts,
- an internal helper file with a leading underscore,
- a public-facing `sys.path` hack.

That makes the repository feel improvised even when the underlying engine is not.

### 2. The README was accurate but too long and too diffuse

The front page contained real information, but it still made a first-time reader do too much sorting:

- product explanation,
- install instructions,
- workflow catalog,
- compatibility notes,
- interface descriptions,
- example inventory,
- troubleshooting.

The result was breadth before clarity.

### 3. Public architecture docs mixed current truth with roadmap language

`docs/architecture.md` described the real engine, but it also carried forward-looking product layering and future-target language. That weakens credibility because outsiders cannot quickly tell what already exists versus what is merely intended.

### 4. The first 5 minutes path was not strong enough

The repo had enough capability for a strong first run, but the shortest visible path was still not opinionated enough:

- install,
- sanity check,
- release gate,
- one core-engine run,
- one artifact-producing workflow,
- one Python example,
- one service example.

Without that path, the project risks admiration without adoption.

### 5. The public tree still leaked residue

Even if the codebase is strong, outsiders lose trust quickly when they see:

- cache directories,
- build residue,
- inconsistent example naming,
- experimental files presented beside stable ones,
- internal-looking files at the public root of a surface.

### 6. API and MCP were real, but their public docs were still too inventory-like

A serious visitor needs to understand:

- why these interfaces exist,
- why they are not detached wrappers,
- how they relate to the same engine,
- how to verify health and status.

Raw endpoint and tool listings are not enough.

## Top Reasons An Outsider Would Dismiss The Repo

1. It looks like too many adjacent ideas instead of one engine with extensions.
2. The examples tree suggests research residue instead of a curated public surface.
3. The README takes too long to answer “why should I try this?”
4. Architecture docs risk sounding aspirational where they should sound implemented.
5. Stable, beta, experimental, and reference-only material were not separated sharply enough.
6. The first-run path was real but not visible enough.
7. The repo still lacks some external trust markers, especially a fully clean cross-platform validation story and an explicit license decision.

## Public Credibility Targets

The repository should make these points obvious within minutes:

- this is a bounded-domain spectral compute engine,
- the engine is the center of gravity,
- Python is the primary integration path,
- CLI, MCP, and API are thin wrappers over the same workflows,
- the repo has real tests and a real release gate,
- examples are curated and intentionally labeled,
- experimental work exists but is not pretending to be stable.

## Immediate Cleanup Rule

When a file, doc, example, or public description makes the repo feel more like:

- a research shell,
- a notebook dump,
- a generic platform,
- or a pile of ideas,

it should either be:

- clarified,
- moved behind a clear boundary,
- or removed from the public first-run path.
