"""
planner/heuristics.py
=====================
Admissible heuristic functions for 3D grid path-planning.

All heuristics share the same signature::

    h = heuristic(current_pos: Coord3D, goal_pos: Coord3D) -> float

This is captured by the ``HeuristicFn`` type alias so callers can annotate
parameters cleanly (e.g. ``def solve(..., heuristic: HeuristicFn)``).

Admissibility & consistency
----------------------------
A heuristic is *admissible* if it never over-estimates the true cost.
With unit-step costs the true minimum cost equals the Euclidean distance
(for any continuous straight-line path), so *euclidean_3d* is the tightest
admissible bound and is the default for A*.

*manhattan_3d* and *chebyshev_3d* are provided for benchmarking:
  - Manhattan is inadmissible in 26-connectivity (over-estimates diagonals)
    but useful for verifying correctness by comparing path counts.
  - Chebyshev is admissible (equals the minimum number of Chebyshev steps)
    and is faster to compute — useful for large grids where sqrt is a
    non-trivial fraction of runtime.

Phase 3 note
------------
The weighted heuristic ``w * h(n)`` pattern (wA*, used in Enhanced A* for
speed vs. optimality trade-off) just wraps one of these:

    def weighted(pos, goal, w=1.5):
        return w * euclidean_3d(pos, goal)

No changes to this file are needed for that extension.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

Coord3D = Tuple[int, int, int]

# Protocol-style type alias — any callable matching this can be used as a heuristic.
HeuristicFn = Callable[[Coord3D, Coord3D], float]


# ---------------------------------------------------------------------------
# Built-in heuristics
# ---------------------------------------------------------------------------

def euclidean_3d(pos: Coord3D, goal: Coord3D) -> float:
    """
    3D Euclidean distance — the tightest admissible heuristic for
    26-connected grids with Euclidean step costs.

    h(n) = sqrt((x₂−x₁)² + (y₂−y₁)² + (z₂−z₁)²)
    """
    dx = pos[0] - goal[0]
    dy = pos[1] - goal[1]
    dz = pos[2] - goal[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def manhattan_3d(pos: Coord3D, goal: Coord3D) -> float:
    """
    3D Manhattan distance — inadmissible for diagonal movement but useful
    for debugging or 6-connectivity grids.

    h(n) = |x₂−x₁| + |y₂−y₁| + |z₂−z₁|
    """
    return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]) + abs(pos[2] - goal[2]))


def chebyshev_3d(pos: Coord3D, goal: Coord3D) -> float:
    """
    3D Chebyshev (L∞) distance — admissible for 26-connectivity with
    unit step cost; equals the minimum number of king-moves in 3D.

    h(n) = max(|x₂−x₁|, |y₂−y₁|, |z₂−z₁|)

    Slightly faster than euclidean_3d (no sqrt) at the cost of being a
    weaker bound (less informed).
    """
    return float(max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]), abs(pos[2] - goal[2])))


def octile_3d(pos: Coord3D, goal: Coord3D) -> float:
    """
    3D Octile distance — tighter than Chebyshev, cheaper than Euclidean.
    Admissible and consistent lower bound for 26-connected grids with
    Euclidean step costs (face=1, edge=√2, corner=√3).

    Derivation
    ----------
    Optimal 3D diagonal move decomposes into tri-diagonal, diagonal, and
    axis steps.  Sorting |dx| ≥ |dy| ≥ |dz| (i.e. d1 ≥ d2 ≥ d3):

        h = d1 + (√2 − 1)·d2 + (√3 − √2)·d3

    This is strictly ≤ Euclidean distance for all integer inputs, making
    it admissible.  It equals Euclidean only for pure axis-aligned moves.

    Proof of admissibility: the formula gives the exact cost to reach the
    goal by first taking d3 tri-diagonal steps, then (d2−d3) diagonal steps,
    then (d1−d2) axis steps — which is the minimum-cost path in the absence
    of obstacles.
    """
    _SQRT2 = math.sqrt(2.0)
    _SQRT3 = math.sqrt(3.0)
    D_DIAG  = _SQRT2 - 1.0      # marginal cost of a diagonal vs axis step  ≈ 0.4142
    D_TRI   = _SQRT3 - _SQRT2   # marginal cost of tri-diagonal vs diagonal ≈ 0.3178

    dx = abs(pos[0] - goal[0])
    dy = abs(pos[1] - goal[1])
    dz = abs(pos[2] - goal[2])

    # Sort so d1 ≥ d2 ≥ d3
    d1, d2, d3 = sorted((dx, dy, dz), reverse=True)
    return d1 + D_DIAG * d2 + D_TRI * d3


# ---------------------------------------------------------------------------
# Registry — maps string names to callables (used by CLI / config files)
# ---------------------------------------------------------------------------

HEURISTICS: dict[str, HeuristicFn] = {
    "euclidean": euclidean_3d,
    "manhattan": manhattan_3d,
    "chebyshev": chebyshev_3d,
    "octile":    octile_3d,
}


def get_heuristic(name: str) -> HeuristicFn:
    """
    Return a heuristic function by name.

    Parameters
    ----------
    name : str
        One of ``"euclidean"``, ``"manhattan"``, ``"chebyshev"``, ``"octile"``.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    try:
        return HEURISTICS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown heuristic '{name}'. "
            f"Available: {sorted(HEURISTICS.keys())}"
        )
