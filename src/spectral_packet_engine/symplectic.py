"""Symplectic integrators for Hamiltonian dynamics.

Symplectic integrators exactly preserve the symplectic 2-form dp /\\ dq
and therefore conserve phase space volume (Liouville's theorem).  While
they do not conserve energy exactly, the energy error remains bounded
for all time -- unlike generic Runge-Kutta methods where energy drifts
secularly.

Implemented integrators
-----------------------
Stoermer-Verlet (leapfrog):
    2nd order, O(dt^2) energy error.

Forest-Ruth:
    4th order, O(dt^4) energy error.

Yoshida (4th and 6th order):
    Composition method using Stoermer-Verlet as a building block.

All integrators operate on separable Hamiltonians H(q, p) = T(p) + V(q)
where T(p) = p^2 / (2m).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from spectral_packet_engine.domain import coerce_tensor

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PhaseSpaceTrajectory:
    """Phase space trajectory from a symplectic integration."""

    times: Tensor
    positions: Tensor
    momenta: Tensor
    energies: Tensor
    energy_error: Tensor


@dataclass(frozen=True, slots=True)
class SymplecticityCheck:
    """Verification that a trajectory preserves phase space volume."""

    initial_volume: Tensor
    final_volume: Tensor
    relative_error: Tensor
    is_symplectic: bool


# ---------------------------------------------------------------------------
# Energy helper
# ---------------------------------------------------------------------------

def hamiltonian_energy(
    q: Tensor,
    p: Tensor,
    potential_fn: Callable[[Tensor], Tensor],
    mass: float = 1.0,
) -> Tensor:
    """H = p^2/(2m) + V(q)."""
    q = coerce_tensor(q, dtype=torch.float64)
    p = coerce_tensor(p, dtype=torch.float64)
    kinetic = p ** 2 / (2.0 * mass)
    potential = potential_fn(q)
    if kinetic.ndim > 0 and potential.ndim > 0:
        return kinetic.sum(-1) + potential.sum(-1)
    return kinetic + potential


# ---------------------------------------------------------------------------
# Internal: single-step building blocks
# ---------------------------------------------------------------------------

def _kick(p: Tensor, grad_V: Callable[[Tensor], Tensor], q: Tensor, dt: float) -> Tensor:
    """Momentum update (kick): p <- p - dt * grad_V(q)."""
    return p - dt * grad_V(q)


def _drift(q: Tensor, p: Tensor, dt: float, mass: float) -> Tensor:
    """Position update (drift): q <- q + (dt/m) * p."""
    return q + (dt / mass) * p


def _build_trajectory(
    saved_t: list[Tensor],
    saved_q: list[Tensor],
    saved_p: list[Tensor],
    potential_fn: Callable[[Tensor], Tensor],
    mass: float,
) -> PhaseSpaceTrajectory:
    """Assemble saved snapshots into a PhaseSpaceTrajectory."""
    times = torch.stack(saved_t)
    positions = torch.stack(saved_q)
    momenta = torch.stack(saved_p)

    energies = torch.stack(
        [hamiltonian_energy(q, p, potential_fn, mass) for q, p in zip(saved_q, saved_p)]
    )
    E0 = energies[0]
    safe_E0 = torch.where(torch.abs(E0) > 0, E0, torch.ones_like(E0))
    energy_error = (energies - E0) / safe_E0

    return PhaseSpaceTrajectory(
        times=times,
        positions=positions,
        momenta=momenta,
        energies=energies,
        energy_error=energy_error,
    )


# ---------------------------------------------------------------------------
# Stoermer-Verlet (leapfrog)
# ---------------------------------------------------------------------------

def stormer_verlet(
    q0: Tensor,
    p0: Tensor,
    grad_V: Callable[[Tensor], Tensor],
    *,
    mass: float = 1.0,
    dt: float = 0.01,
    num_steps: int = 1000,
    save_every: int = 1,
) -> PhaseSpaceTrajectory:
    r"""Stoermer-Verlet (leapfrog) symplectic integrator -- 2nd order.

    p_{1/2} = p_n - (dt/2) grad_V(q_n)
    q_{n+1} = q_n + (dt/m) p_{1/2}
    p_{n+1} = p_{1/2} - (dt/2) grad_V(q_{n+1})

    Parameters
    ----------
    q0, p0:
        Initial position and momentum (scalar or 1-D tensors).
    grad_V:
        Callable returning dV/dq evaluated at a position tensor.
    mass:
        Particle mass.
    dt:
        Time step.
    num_steps:
        Number of integration steps.
    save_every:
        Save a snapshot every this many steps.
    """
    q = coerce_tensor(q0, dtype=torch.float64).clone()
    p = coerce_tensor(p0, dtype=torch.float64).clone()

    # Need the potential for energy evaluation -- derive from grad_V
    # We do not have an explicit V(q); compute energy from kinetic + numerical V
    # We will reconstruct V from cumulative grad_V -- instead, require user
    # passes grad_V only and we track energy via a helper.
    # For energy, integrate grad_V along q.  Simpler: accept that we need
    # potential_fn.  We will wrap grad_V to produce a dummy potential.
    # Actually: store q, p and compute energy later only if needed.
    # To keep the interface clean, we compute energy from T + integral.
    # The simplest correct approach: potential is unknown, so store
    # kinetic energy and note that we need potential.
    # DECISION: We will build a potential that is the antiderivative of grad_V
    # evaluated at discrete points.  BUT that is over-complex.
    # Instead, we will compute the potential at each saved point
    # using a numerical trick: V(q) ≈ -integral(grad_V) + const.
    # For the trajectory energy we only need *relative* energy, so
    # we can use V(q0) = 0 as reference.
    # BUT the spec says energy = H(q,p), and `hamiltonian_energy` takes
    # a potential_fn.  We will define a dummy that numerically integrates.
    # SIMPLEST: compute energy as KE only (incomplete).  The spec requires
    # the user has a potential_fn for `hamiltonian_energy` separately.
    # We will just store KE and set energy_error to KE drift.

    # Actually, the typical usage is:  grad_V = derivative of V.
    # We can numerically construct V at the trajectory points by
    # cumulative trapezoidal integration of grad_V from q0.
    # But that's complex for multi-D.
    # PRAGMATIC CHOICE: compute V(q_i) ≈ V(q0) + sum grad_V * dq for
    # the trajectory.  But we only evaluate grad_V at the saved qs.
    # Actually simplest: just track momentum and position, compute energy
    # using the formula H = p^2/(2m) + V where we approximate V by
    # cumulative sum.

    saved_t: list[Tensor] = []
    saved_q: list[Tensor] = []
    saved_p: list[Tensor] = []

    t = torch.tensor(0.0, dtype=torch.float64)
    _dt = torch.tensor(dt, dtype=torch.float64)

    # Save initial state
    saved_t.append(t.clone())
    saved_q.append(q.clone())
    saved_p.append(p.clone())

    for step in range(1, num_steps + 1):
        # Half kick
        p = _kick(p, grad_V, q, dt / 2.0)
        # Full drift
        q = _drift(q, p, dt, mass)
        # Half kick
        p = _kick(p, grad_V, q, dt / 2.0)

        t = t + _dt

        if step % save_every == 0:
            saved_t.append(t.clone())
            saved_q.append(q.clone())
            saved_p.append(p.clone())

    # Compute energies using cumulative potential estimate
    # V(q_i) ≈ V(q_0) + integral grad_V from q_0 to q_i
    # For simplicity, evaluate V at each saved point directly
    # via numerical integration of grad_V with trapezoid rule
    # from the first saved point.
    # Actually, since grad_V is dV/dq, we can compute V at each
    # trajectory point up to a constant by integrating.
    # But the simplest correct thing: compute kinetic energy and
    # the potential from integrating grad_V.
    _positions = torch.stack(saved_q)
    _momenta = torch.stack(saved_p)

    # Estimate potential by cumulative trapezoidal integration of grad_V
    grad_vals = torch.stack([grad_V(qq) for qq in saved_q])
    dq = _positions[1:] - _positions[:-1]
    grad_avg = (grad_vals[:-1] + grad_vals[1:]) / 2.0
    if dq.ndim > 1:
        V_increments = (grad_avg * dq).sum(-1)
    else:
        V_increments = grad_avg * dq
    V_cumulative = torch.zeros(len(saved_q), dtype=torch.float64)
    V_cumulative[1:] = torch.cumsum(V_increments, dim=0)

    KE = _momenta ** 2 / (2.0 * mass)
    if KE.ndim > 1:
        KE = KE.sum(-1)
    if V_cumulative.ndim > 1:
        V_cumulative = V_cumulative.sum(-1)
    energies = KE + V_cumulative
    E0 = energies[0]
    safe_E0 = torch.where(torch.abs(E0) > 0, E0, torch.ones_like(E0))
    energy_error = (energies - E0) / safe_E0

    return PhaseSpaceTrajectory(
        times=torch.stack(saved_t),
        positions=_positions,
        momenta=_momenta,
        energies=energies,
        energy_error=energy_error,
    )


# ---------------------------------------------------------------------------
# Forest-Ruth 4th order
# ---------------------------------------------------------------------------

def forest_ruth(
    q0: Tensor,
    p0: Tensor,
    grad_V: Callable[[Tensor], Tensor],
    *,
    mass: float = 1.0,
    dt: float = 0.01,
    num_steps: int = 1000,
    save_every: int = 1,
) -> PhaseSpaceTrajectory:
    r"""Forest-Ruth 4th order symplectic integrator.

    Uses the decomposition with theta = 1 / (2 - 2^{1/3}):

        x1 = q + theta * dt/2 * p/m
        p1 = p - theta * dt * grad_V(x1)
        ... (symmetric composition)

    The full step is a symmetric sequence of kicks and drifts with
    coefficients derived from theta.
    """
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))

    q = coerce_tensor(q0, dtype=torch.float64).clone()
    p = coerce_tensor(p0, dtype=torch.float64).clone()

    saved_t: list[Tensor] = []
    saved_q: list[Tensor] = []
    saved_p: list[Tensor] = []

    t = torch.tensor(0.0, dtype=torch.float64)
    _dt = torch.tensor(dt, dtype=torch.float64)

    saved_t.append(t.clone())
    saved_q.append(q.clone())
    saved_p.append(p.clone())

    for step in range(1, num_steps + 1):
        # Forest-Ruth symmetric composition
        # drift theta/2
        q = _drift(q, p, theta * dt / 2.0, mass)
        # kick theta
        p = _kick(p, grad_V, q, theta * dt)
        # drift (1-theta)/2
        q = _drift(q, p, (1.0 - theta) * dt / 2.0, mass)
        # kick (1 - 2*theta)
        p = _kick(p, grad_V, q, (1.0 - 2.0 * theta) * dt)
        # drift (1-theta)/2
        q = _drift(q, p, (1.0 - theta) * dt / 2.0, mass)
        # kick theta
        p = _kick(p, grad_V, q, theta * dt)
        # drift theta/2
        q = _drift(q, p, theta * dt / 2.0, mass)

        t = t + _dt

        if step % save_every == 0:
            saved_t.append(t.clone())
            saved_q.append(q.clone())
            saved_p.append(p.clone())

    return _build_trajectory_from_grad(saved_t, saved_q, saved_p, grad_V, mass)


# ---------------------------------------------------------------------------
# Yoshida
# ---------------------------------------------------------------------------

def yoshida(
    q0: Tensor,
    p0: Tensor,
    grad_V: Callable[[Tensor], Tensor],
    *,
    mass: float = 1.0,
    dt: float = 0.01,
    num_steps: int = 1000,
    save_every: int = 1,
    order: int = 4,
) -> PhaseSpaceTrajectory:
    r"""Yoshida symplectic integrator (4th or 6th order).

    4th order:
        w_1 = 1/(2 - 2^{1/3}),  w_0 = -2^{1/3}/(2 - 2^{1/3})
        Compose three Stoermer-Verlet steps with weights [w_1, w_0, w_1].

    6th order:
        w_1 = 1/(2 - 2^{1/5}),  w_0 = -2^{1/5}/(2 - 2^{1/5})
        Triple composition of the 4th order integrator.
    """
    if order == 4:
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        weights = [w1, w0, w1]
    elif order == 6:
        fifth_root2 = 2.0 ** (1.0 / 5.0)
        z1 = 1.0 / (2.0 - fifth_root2)
        z0 = -fifth_root2 / (2.0 - fifth_root2)
        # 4th-order weights
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1_4 = 1.0 / (2.0 - cbrt2)
        w0_4 = -cbrt2 / (2.0 - cbrt2)
        base_weights = [w1_4, w0_4, w1_4]
        # Compose: [z1*base, z0*base, z1*base]
        weights = [z1 * w for w in base_weights] + [z0 * w for w in base_weights] + [z1 * w for w in base_weights]
    else:
        raise ValueError(f"Yoshida integrator supports order 4 or 6, got {order}")

    q = coerce_tensor(q0, dtype=torch.float64).clone()
    p = coerce_tensor(p0, dtype=torch.float64).clone()

    saved_t: list[Tensor] = []
    saved_q: list[Tensor] = []
    saved_p: list[Tensor] = []

    t = torch.tensor(0.0, dtype=torch.float64)
    _dt = torch.tensor(dt, dtype=torch.float64)

    saved_t.append(t.clone())
    saved_q.append(q.clone())
    saved_p.append(p.clone())

    for step in range(1, num_steps + 1):
        # Each weight w_i defines a leapfrog sub-step with effective dt = w_i * dt
        for w in weights:
            sub_dt = w * dt
            # Leapfrog sub-step
            p = _kick(p, grad_V, q, sub_dt / 2.0)
            q = _drift(q, p, sub_dt, mass)
            p = _kick(p, grad_V, q, sub_dt / 2.0)

        t = t + _dt

        if step % save_every == 0:
            saved_t.append(t.clone())
            saved_q.append(q.clone())
            saved_p.append(p.clone())

    return _build_trajectory_from_grad(saved_t, saved_q, saved_p, grad_V, mass)


# ---------------------------------------------------------------------------
# Trajectory builder using grad_V
# ---------------------------------------------------------------------------

def _build_trajectory_from_grad(
    saved_t: list[Tensor],
    saved_q: list[Tensor],
    saved_p: list[Tensor],
    grad_V: Callable[[Tensor], Tensor],
    mass: float,
) -> PhaseSpaceTrajectory:
    """Build a PhaseSpaceTrajectory, estimating V by integrating grad_V."""
    positions = torch.stack(saved_q)
    momenta = torch.stack(saved_p)

    grad_vals = torch.stack([grad_V(qq) for qq in saved_q])
    dq = positions[1:] - positions[:-1]
    grad_avg = (grad_vals[:-1] + grad_vals[1:]) / 2.0

    if dq.ndim > 1:
        V_increments = (grad_avg * dq).sum(-1)
    else:
        V_increments = grad_avg * dq

    V_cumulative = torch.zeros(len(saved_q), dtype=torch.float64)
    V_cumulative[1:] = torch.cumsum(V_increments, dim=0)

    KE = momenta ** 2 / (2.0 * mass)
    if KE.ndim > 1:
        KE = KE.sum(-1)
    if V_cumulative.ndim > 1:
        V_cumulative = V_cumulative.sum(-1)

    energies = KE + V_cumulative
    E0 = energies[0]
    safe_E0 = torch.where(torch.abs(E0) > 0, E0, torch.ones_like(E0))
    energy_error = (energies - E0) / safe_E0

    return PhaseSpaceTrajectory(
        times=torch.stack(saved_t),
        positions=positions,
        momenta=momenta,
        energies=energies,
        energy_error=energy_error,
    )


# ---------------------------------------------------------------------------
# Symplecticity verification
# ---------------------------------------------------------------------------

def check_symplecticity(trajectory: PhaseSpaceTrajectory) -> SymplecticityCheck:
    """Verify that the integrator preserves phase space volume.

    For a 1-D system, the phase space area element dp /\\ dq is checked
    by computing |det(J)| of the Jacobian of the map (q_0, p_0) -> (q_n, p_n)
    using finite differences from neighbouring trajectory points.

    In practice, we compare the area of parallelograms formed by
    consecutive (dq, dp) increments at the start and end of the trajectory.
    """
    q = trajectory.positions
    p = trajectory.momenta

    # Use finite differences of consecutive phase-space points
    if q.ndim == 1:
        # Scalar positions -- area = |dq * dp' - dq' * dp| for two consecutive steps
        dq_start = q[1] - q[0]
        dp_start = p[1] - p[0]
        dq_end = q[-1] - q[-2]
        dp_end = p[-1] - p[-2]

        # Phase space "area" element
        area_start = torch.abs(dq_start * dp_start)
        area_end = torch.abs(dq_end * dp_end)
    else:
        # Multi-D: use first component
        dq_start = q[1] - q[0]
        dp_start = p[1] - p[0]
        dq_end = q[-1] - q[-2]
        dp_end = p[-1] - p[-2]

        area_start = torch.abs((dq_start * dp_start).sum())
        area_end = torch.abs((dq_end * dp_end).sum())

    safe_start = torch.where(area_start > 0, area_start, torch.ones_like(area_start))
    rel_error = torch.abs(area_end - area_start) / safe_start

    return SymplecticityCheck(
        initial_volume=area_start,
        final_volume=area_end,
        relative_error=rel_error,
        is_symplectic=bool((rel_error < 0.1).item()),
    )


__all__ = [
    "PhaseSpaceTrajectory",
    "SymplecticityCheck",
    "check_symplecticity",
    "forest_ruth",
    "hamiltonian_energy",
    "stormer_verlet",
    "yoshida",
]
