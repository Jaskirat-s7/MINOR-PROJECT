"""
planner/astar.py
================
Standard A* path planner for 3D UAV navigation.

Architecture overview
---------------------

    AStarPlanner(grid)          ← constructed once, reused for many queries
        └─ solve(start, goal)   ← stateless per call; returns PlanResult

    PlanResult                  ← plain dataclass carrying everything a caller
                                   needs (path, cost, metrics, success flag)

Extension points (Phase 3)
--------------------------
The implementation is structured around three clearly labelled hooks:

  [HOOK: NEIGHBOR EXPANSION]
      Located inside the main loop.  Theta* replaces the plain neighbour
      list with a filtered list that checks line-of-sight to the grandparent.

  [HOOK: COST FUNCTION]
      ``_step_cost()`` computes the 3D Euclidean distance between two adjacent
      cells.  Enhanced A* can inject an anisotropic terrain cost here.

  [HOOK: PARENT ASSIGNMENT]
      A single call site sets ``node.parent``.  D* Lite will augment this
      with a "predecessor list" for backward-search support.

Performance characteristics (50×50×50 grid)
--------------------------------------------
* Worst-case open-list size  : O(X·Y·Z) = 125 000 nodes
* Memory per node            : 6 slots × 8 bytes ≈ 48 bytes → ~6 MB worst case
* Per-iteration complexity   : O(26 × log N) ≈ O(26 × 17) ≈ O(440) ops
* Empirically on dense maps  : < 100 ms on a modern laptop (CPython 3.9)
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from environment.grid3d import Grid3D
from .heuristics import HeuristicFn, euclidean_3d
from .node import SearchNode

Coord3D = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """
    Immutable result returned by :meth:`AStarPlanner.solve`.

    Attributes
    ----------
    success : bool
        True if a path was found.
    path : list of Coord3D
        Ordered waypoints from start (inclusive) to goal (inclusive).
        Empty if no path exists.
    path_length : float
        Accumulated Euclidean cost along *path*.  0.0 if no path.
    nodes_expanded : int
        Number of nodes popped from the open list (closed set size).
        Primary performance indicator: lower → more efficient search.
    nodes_generated : int
        Total nodes pushed onto the open list (including duplicates from
        g-value updates).  Tracks heap pressure.
    elapsed_seconds : float
        Wall-clock time consumed by :meth:`AStarPlanner.solve`.
    start : Coord3D
        Query start position.
    goal : Coord3D
        Query goal position.
    heuristic_name : str
        Name of the heuristic used (for reporting).

    Notes
    -----
    *path_length* is always the **exact** Euclidean path cost, regardless
    of which heuristic was used.  This lets you compare path quality across
    different heuristics on the same map.
    """

    success: bool
    path: List[Coord3D]
    path_length: float
    nodes_expanded: int
    nodes_generated: int
    elapsed_seconds: float
    start: Coord3D
    goal: Coord3D
    heuristic_name: str = "euclidean"

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.success:
            return (
                f"A* SUCCESS | "
                f"path_length={self.path_length:.3f}  "
                f"waypoints={len(self.path)}  "
                f"expanded={self.nodes_expanded}  "
                f"generated={self.nodes_generated}  "
                f"time={self.elapsed_seconds * 1000:.2f}ms  "
                f"heuristic={self.heuristic_name}"
            )
        return (
            f"A* FAILED  | No path from {self.start} → {self.goal}  "
            f"expanded={self.nodes_expanded}  "
            f"time={self.elapsed_seconds * 1000:.2f}ms"
        )

    def metrics(self) -> dict:
        """Return all performance metrics as a plain dict (easy to log/plot)."""
        return {
            "success":          self.success,
            "path_length":      self.path_length,
            "waypoints":        len(self.path),
            "nodes_expanded":   self.nodes_expanded,
            "nodes_generated":  self.nodes_generated,
            "elapsed_ms":       self.elapsed_seconds * 1000,
            "heuristic":        self.heuristic_name,
            "start":            self.start,
            "goal":             self.goal,
        }

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class AStarPlanner:
    """
    Standard A* planner operating on a :class:`~environment.grid3d.Grid3D`.

    The planner holds only a reference to the grid — it is **stateless per
    query** and thread-safe as long as the grid is not concurrently mutated.

    Parameters
    ----------
    grid : Grid3D
        The environment to search.  Must remain unchanged for the duration
        of a :meth:`solve` call.
    heuristic : HeuristicFn, optional
        Callable ``(pos, goal) → float``.  Defaults to :func:`euclidean_3d`.
        Must be admissible for optimality guarantees.
    allow_diagonal : bool, optional
        If True (default) uses 26-connectivity; if False uses 6-face
        connectivity.  Passed directly to :meth:`Grid3D.get_neighbors`.

    Examples
    --------
    >>> from environment import MapGenerator
    >>> from planner import AStarPlanner
    >>> gen = MapGenerator(seed=42)
    >>> grid, start, goal = gen.generate_cuboid()
    >>> planner = AStarPlanner(grid)
    >>> result = planner.solve(start, goal)
    >>> print(result)
    """

    def __init__(
        self,
        grid: Grid3D,
        heuristic: HeuristicFn = euclidean_3d,
        allow_diagonal: bool = True,
    ) -> None:
        self._grid = grid
        self._heuristic = heuristic
        self._allow_diagonal = allow_diagonal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_cost(a: Coord3D, b: Coord3D) -> float:
        """
        [HOOK: COST FUNCTION]

        3D Euclidean distance between two *adjacent* grid cells.
        Returns one of three possible values:
            1.0       – face-adjacent  (axis-aligned move)
            √2 ≈ 1.414 – edge-adjacent  (one diagonal)
            √3 ≈ 1.732 – corner-adjacent (full space-diagonal)

        Enhanced A* can replace this with a terrain-weighted cost:
            base_cost * grid.terrain_weight(b)
        """
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _reconstruct_path(goal_node: SearchNode) -> Tuple[List[Coord3D], float]:
        """
        Walk parent pointers from *goal_node* back to the start and
        return the reversed path along with its accumulated cost.
        """
        path: List[Coord3D] = []
        node: Optional[SearchNode] = goal_node
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        return path, goal_node.g

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        start: Coord3D,
        goal: Coord3D,
        *,
        heuristic: Optional[HeuristicFn] = None,
        weight: float = 1.0,
    ) -> PlanResult:
        """
        Run A* from *start* to *goal*.

        Parameters
        ----------
        start : Coord3D
            Starting position.  Must be free and in-bounds.
        goal : Coord3D
            Goal position.  Must be free and in-bounds.
        heuristic : HeuristicFn, optional
            Per-call heuristic override.  Falls back to the planner-level
            heuristic if not provided.
        weight : float, optional
            Heuristic weight (default 1.0 = standard A*).
            Set ``weight > 1.0`` for weighted A* (wA*) — faster but
            sub-optimal.  Prepared for Enhanced A* integration.

        Returns
        -------
        PlanResult
            Contains path, metrics, and success flag.

        Raises
        ------
        ValueError
            If start or goal are out-of-bounds or inside obstacles.
        """
        # ---- Validate --------------------------------------------------
        self._grid.validate_position(start, "start")
        self._grid.validate_position(goal, "goal")

        t_start = time.perf_counter()

        h_fn = heuristic if heuristic is not None else self._heuristic
        h_name = getattr(h_fn, "__name__", "custom")

        # ---- Early exit: trivial case ----------------------------------
        if start == goal:
            elapsed = time.perf_counter() - t_start
            return PlanResult(
                success=True,
                path=[start],
                path_length=0.0,
                nodes_expanded=0,
                nodes_generated=1,
                elapsed_seconds=elapsed,
                start=start,
                goal=goal,
                heuristic_name=h_name,
            )

        # ---- Data structures -------------------------------------------

        # g_values[pos] → best known cost from start to pos
        # Maintained separately from nodes so we can check g without
        # scanning the heap (O(1) lookup vs O(N) heap scan).
        # [HOOK: D* Lite] — this dict becomes the "rhs" table in D* Lite.
        g_values: Dict[Coord3D, float] = {start: 0.0}

        # closed_set: positions whose optimal g is finalised.
        # [HOOK: Theta*] — Theta* does NOT skip re-expansion for LoS updates,
        # so it removes the closed-set guard; track that change here.
        closed_set: Set[Coord3D] = set()

        # Monotone counter provides stable heap ordering (avoids Coord3D comparison).
        _seq = 0

        # Open list: min-heap of SearchNode.
        start_h = weight * h_fn(start, goal)
        start_node = SearchNode(pos=start, g=0.0, h=start_h, parent=None, seq=_seq)
        open_list: List[SearchNode] = [start_node]
        heapq.heapify(open_list)

        nodes_expanded = 0
        nodes_generated = 1   # start node counts

        # ---- Main loop -------------------------------------------------
        while open_list:
            current = heapq.heappop(open_list)

            # Skip stale nodes (a cheaper g was found and this entry is
            # a duplicate left in the heap).
            if current.pos in closed_set:
                continue
            if current.g > g_values.get(current.pos, math.inf):
                continue

            # ---- Goal check --------------------------------------------
            if current.pos == goal:
                path, length = self._reconstruct_path(current)
                elapsed = time.perf_counter() - t_start
                return PlanResult(
                    success=True,
                    path=path,
                    path_length=length,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    elapsed_seconds=elapsed,
                    start=start,
                    goal=goal,
                    heuristic_name=h_name,
                )

            closed_set.add(current.pos)
            nodes_expanded += 1

            # ---- Neighbour expansion -----------------------------------
            # [HOOK: NEIGHBOR EXPANSION]
            # D* Lite will iterate predecessors instead of successors here.
            neighbors = self._grid.get_neighbors(
                current.pos, allow_diagonal=self._allow_diagonal
            )

            for nb_pos in neighbors:
                if nb_pos in closed_set:
                    continue

                # ---- Cost computation ----------------------------------
                # [HOOK: COST FUNCTION]
                # Replace _step_cost with a terrain-aware version here.
                tentative_g = current.g + self._step_cost(current.pos, nb_pos)

                if tentative_g >= g_values.get(nb_pos, math.inf):
                    continue    # existing path to nb_pos is at least as good

                # ---- Relaxation ----------------------------------------
                # [HOOK: PARENT ASSIGNMENT]
                # D* Lite augments this with predecessor tracking:
                #     predecessors[nb_pos].add(current.pos)
                g_values[nb_pos] = tentative_g
                nb_h = weight * h_fn(nb_pos, goal)
                _seq += 1
                nb_node = SearchNode(
                    pos=nb_pos,
                    g=tentative_g,
                    h=nb_h,
                    parent=current,          # ← parent pointer set here
                    seq=_seq,
                )
                heapq.heappush(open_list, nb_node)
                nodes_generated += 1

        # ---- No path found --------------------------------------------
        elapsed = time.perf_counter() - t_start
        return PlanResult(
            success=False,
            path=[],
            path_length=0.0,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            elapsed_seconds=elapsed,
            start=start,
            goal=goal,
            heuristic_name=h_name,
        )

    # ------------------------------------------------------------------
    # Convenience: batch solving & heuristic comparison
    # ------------------------------------------------------------------

    def compare_heuristics(
        self,
        start: Coord3D,
        goal: Coord3D,
        heuristic_names: Optional[List[str]] = None,
    ) -> Dict[str, PlanResult]:
        """
        Run A* with multiple heuristics on the same start/goal pair and
        return a dict mapping heuristic name → PlanResult.

        Useful for benchmarking and picking the best heuristic for a map.

        Parameters
        ----------
        heuristic_names : list of str, optional
            Subset of ``["euclidean", "octile", "chebyshev", "manhattan"]``.
            Defaults to all four.
        """
        from .heuristics import HEURISTICS

        names = heuristic_names or list(HEURISTICS.keys())
        results: Dict[str, PlanResult] = {}
        for name in names:
            fn = HEURISTICS[name]
            results[name] = self.solve(start, goal, heuristic=fn)
        return results
