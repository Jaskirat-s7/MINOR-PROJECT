"""
planner/theta_star.py
=====================
Theta* — Any-Angle Path Planning  (Enhancement 1 of Enhanced A*).

Core idea
---------
Standard A* connects every successor to its **immediate grid neighbour**
(the node currently being expanded), forcing paths to follow grid edges
and producing zigzag sequences of short diagonal steps.

Theta* relaxes this by checking one extra condition per successor:

    s   = current node being expanded
    s'  = s.parent  ("grandparent" of the successor)
    n   = successor position being evaluated

    IF  line-of-sight(s'.pos, n)  is clear:
        connect n directly to s'          ← any-angle shortcut
        g(n) = g(s') + dist(s'.pos, n)
    ELSE:
        connect n to s                    ← standard A* behaviour
        g(n) = g(s)  + dist(s.pos,  n)

Because s' can be *anywhere* — not constrained to adjacent grid cells —
the resulting path segments can point in any direction, giving the
algorithm its "any-angle" name.

Implementation strategy
-----------------------
Only ONE method is overridden: ``_select_parent()``, which was extracted
from the A* main loop specifically as an extension point.

Everything else — open-list management, heuristic evaluation, closed-set
logic, path reconstruction, metrics — is inherited unchanged from
``AStarPlanner``.

Why this matters for UAVs
--------------------------
* Fewer waypoints → fewer control commands → less actuator wear
* Straight-line segments → no unnecessary turns → faster flight
* Line-of-sight check uses the existing Bresenham 3D voxel traversal
  already implemented in ``Grid3D.has_line_of_sight()``

Correctness guarantee
---------------------
Every segment  path[i] → path[i+1]  was explicitly verified obstacle-free
by ``has_line_of_sight`` before being committed.  The path cannot cross
an obstacle.
"""

from __future__ import annotations

from typing import Tuple

from .astar import AStarPlanner, PlanResult
from .node import SearchNode

Coord3D = Tuple[int, int, int]


class ThetaStarPlanner(AStarPlanner):
    """
    Theta* any-angle planner — minimal subclass of :class:`AStarPlanner`.

    Overrides only :meth:`_select_parent` to implement the grandparent
    line-of-sight bypass.  All other behaviour is inherited from A*.

    Usage
    -----
    >>> planner = ThetaStarPlanner(grid)
    >>> result  = planner.solve(start, goal)
    >>> print(result.planner_name)      # "Theta*"
    >>> print(len(result.path))         # far fewer waypoints than A*
    """

    # ------------------------------------------------------------------
    # Single override — the entire algorithmic difference from A*
    # ------------------------------------------------------------------

    def _select_parent(
        self,
        current: SearchNode,
        nb_pos: Coord3D,
    ) -> Tuple[SearchNode, float]:
        """
        Theta* parent selection — grandparent line-of-sight bypass.

        Steps
        -----
        1. If *current* has a parent (grandparent gp exists):
               cost_via_gp = gp.g + dist(gp.pos, nb_pos)
               if has_line_of_sight(gp.pos, nb_pos):
                   return (gp, cost_via_gp)      ← any-angle shortcut
        2. Fall back to standard A*:
               return (current, current.g + dist(current.pos, nb_pos))

        Complexity: O(max(dx, dy, dz)) for the LoS check — negligible
        compared to heap operations.
        """
        gp = current.parent  # grandparent node

        if gp is not None:
            gp_cost = gp.g + self._step_cost(gp.pos, nb_pos)
            if self._grid.has_line_of_sight(gp.pos, nb_pos):
                return gp, gp_cost  # ← skip current, connect direct

        # LoS blocked or no grandparent → behave exactly like A*
        return current, current.g + self._step_cost(current.pos, nb_pos)

    # ------------------------------------------------------------------
    # Stamp planner_name on every result (only override of solve)
    # ------------------------------------------------------------------

    def solve(self, start, goal, *, heuristic=None, weight=1.0) -> PlanResult:  # type: ignore[override]
        result = super().solve(start, goal, heuristic=heuristic, weight=weight)
        result.planner_name = "Theta*"
        return result
