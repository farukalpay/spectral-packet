# Advanced Physics Module Review — TUM × RWTH Cross-Audit

**Date:** 2026-04-04
**Context:** Code review session between a TU München theoretical physics senior and an RWTH Aachen data engineering student, evaluating the spectral packet engine's mathematical depth.

---

## Participants

- **Lukas** — TU München, Physics B.Sc. (accepted to M.Sc. Theoretical Physics). Expertise: quantum mechanics, spectral methods, mathematical physics.
- **Emre** — RWTH Aachen, Data Science & Engineering B.Sc. final year. Expertise: SQL pipelines, ML integration, MCP protocol, data validation.

## Findings Summary

The engine's existing physics layer (sine basis, Chebyshev basis, Parseval norm, Heisenberg uncertainty, energy conservation, spectral convergence) covers **Quantum Mechanics I** material correctly. All tests pass. However, the mathematical depth stops at undergraduate level. The following 13 modules were identified as necessary to bring the engine to graduate/PhD level.

---

## Approved Modules

### Tier 1 — Foundational (unlocks all downstream physics)

| Module | Gap | Formula Core |
|--------|-----|-------------|
| `eigensolver.py` | Only infinite well (V=0). No arbitrary potentials | Chebyshev collocation → `(-ℏ²/2m)D² + V(x))ψ = Eψ` → `torch.linalg.eigh` |
| `split_operator.py` | Only free-particle phase evolution | Trotter-Suzuki: `e^{-iHΔt} ≈ e^{-iVΔt/2} · F⁻¹ · e^{-iTΔt} · F · e^{-iVΔt/2}` |
| `wigner.py` | No phase-space representation at all | `W(x,p) = (1/πℏ) ∫ ψ*(x+y)ψ(x-y) e^{2ipy/ℏ} dy` |
| `density_matrix.py` | No mixed states, no decoherence, no entropy | `ρ = Σ p_i |ψ_i⟩⟨ψ_i|`, `S = -Tr(ρ ln ρ)`, `F(ρ,σ) = [Tr(√(√ρ σ √ρ))]²` |
| `greens_function.py` | No propagators, no local density of states | `G(x,x';E) = Σ_n ψ_n(x)ψ_n*(x')/(E - E_n ± iε)` |

### Tier 2 — Advanced (PhD-level methods)

| Module | Gap | Formula Core |
|--------|-----|-------------|
| `perturbation.py` | No perturbation theory at all | `E_n^(2) = Σ_{m≠n} |⟨m|V'|n⟩|²/(E_n⁰ - E_m⁰)` |
| `semiclassical.py` | No WKB, no tunneling | `T ≈ exp(-2/ℏ ∫√(2m(V-E)) dx)`, Bohr-Sommerfeld |
| `operator_algebra.py` | No commutators, no generalized uncertainty | `[A,B] = AB-BA`, `ΔA·ΔB ≥ ½|⟨[A,B]⟩|` |
| `symplectic.py` | No structure-preserving integrators | Störmer-Verlet, Yoshida 4th/6th order |
| `spectral_zeta.py` | No spectral zeta, no Casimir energy | `ζ_H(s) = Σ_n E_n^{-s}`, `K(t) = Σ_n e^{-E_n t}` |

### Tier 3 — Expert (publishable-level)

| Module | Gap | Formula Core |
|--------|-----|-------------|
| `scattering.py` | No transfer matrix, no T/R coefficients | `M = Π M_j`, `T = |t|²`, unitarity |
| `berry_phase.py` | No geometric phase, no topology | `γ_n = i∮⟨n(R)|∇_R n(R)⟩·dR` |
| `quantum_info.py` | No quantum Fisher information, no entanglement measures | `F_Q[ρ,A] = 2Σ (p_m-p_n)²|⟨m|A|n⟩|²/(p_m+p_n)` |

---

## Data Engineering Integration Requirements (Emre)

Each new physics module must:
1. Produce structured dict output compatible with `to_serializable()`
2. Be exposed as MCP tool(s) with bounded execution
3. Accept `device: str = "auto"` parameter for GPU/CPU flexibility
4. Return artifact-compatible output for `write_json()`
5. Handle edge cases (zero coefficients, degenerate eigenvalues) gracefully

## MCP Audit Tool Requirement

An `write_audit_log` MCP tool should be added so that AI clients developing the repo can write structured audit entries to `docs/internal/` without manual file creation.

---

## Decision

All 13 modules approved for implementation. Priority: Tier 1 → Tier 2 → Tier 3.
All modules must synchronize with the MCP layer.
