"""
dynamic_obstacles.py
====================
Registry for dynamic obstacles — designed to plug directly into the
replanning loop of D* Lite / Enhanced A* in later phases.

Design Goals
------------
* Obstacles are identified by a string or int *ID* so callers can refer to
  moving objects (drones, vehicles) by a stable key.
* Adding/removing an obstacle fires a callback list so a running planner
  can be notified without polling.
* The registry holds a *weak* reference back to the Grid3D so it can write
  directly into the voxel array; no second copy of the grid is needed.

Phase 3 integration note
------------------------
When the dynamic replanning module is implemented it should:

  1. Subscribe to ``DynamicObstacleRegistry.on_change`` callbacks.
  2. On each callback, mark the changed cells as "locally inconsistent"
     in the D* Lite key-queue and trigger a re-expansion.
"""

from __future__ import annotations

import weakref
from typing import Callable, Dict, List, Optional, Set, Tuple

from .grid3d import CellState, Grid3D

Coord3D = Tuple[int, int, int]
ObstacleId = str
ChangeCallback = Callable[[ObstacleId, str, Set[Coord3D]], None]
#                                         ^^^          "added" | "removed"


class DynamicObstacle:
    """Represents one dynamic agent occupying a set of cells."""

    def __init__(self, oid: ObstacleId, cells: Set[Coord3D]) -> None:
        self.id: ObstacleId = oid
        self.cells: Set[Coord3D] = set(cells)


class DynamicObstacleRegistry:
    """
    Manages dynamic obstacles on a Grid3D.

    Parameters
    ----------
    grid :
        The Grid3D to mutate.  Stored as a weak reference so the registry
        does not prevent garbage collection.

    Example
    -------
    >>> registry = DynamicObstacleRegistry(grid)
    >>> registry.on_change.append(lambda oid, action, cells: print(oid, action))
    >>> registry.add("drone_2", {(10, 10, 5), (10, 11, 5)})
    >>> registry.move("drone_2", {(10, 12, 5), (10, 13, 5)})
    >>> registry.remove("drone_2")
    """

    def __init__(self, grid: Grid3D) -> None:
        self._grid_ref: weakref.ref[Grid3D] = weakref.ref(grid)
        self._obstacles: Dict[ObstacleId, DynamicObstacle] = {}
        self.on_change: List[ChangeCallback] = []
        """Callbacks fired on every add/remove/move.  Thread-safety is the
        caller's responsibility for now."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid(self) -> Grid3D:
        g = self._grid_ref()
        if g is None:
            raise RuntimeError("The Grid3D this registry points to has been garbage-collected.")
        return g

    def _notify(self, oid: ObstacleId, action: str, cells: Set[Coord3D]) -> None:
        for cb in self.on_change:
            cb(oid, action, cells)

    def _write_cells(self, cells: Set[Coord3D], state: CellState) -> None:
        g = self._grid()
        for cell in cells:
            g.set_cell(cell, state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, oid: ObstacleId, cells: Set[Coord3D]) -> None:
        """
        Register a new dynamic obstacle occupying *cells*.

        Raises
        ------
        KeyError
            If *oid* is already registered.  Use :meth:`move` to update.
        """
        if oid in self._obstacles:
            raise KeyError(f"Obstacle '{oid}' is already registered. Use move() to update.")
        obs = DynamicObstacle(oid, cells)
        self._obstacles[oid] = obs
        self._write_cells(cells, CellState.DYNAMIC)
        self._notify(oid, "added", cells)

    def remove(self, oid: ObstacleId) -> None:
        """
        Unregister *oid* and free its cells back to FREE.

        Raises
        ------
        KeyError
            If *oid* is not registered.
        """
        if oid not in self._obstacles:
            raise KeyError(f"Obstacle '{oid}' is not registered.")
        obs = self._obstacles.pop(oid)
        self._write_cells(obs.cells, CellState.FREE)
        self._notify(oid, "removed", obs.cells)

    def move(self, oid: ObstacleId, new_cells: Set[Coord3D]) -> None:
        """
        Atomically move *oid* from its current cells to *new_cells*.

        Old cells are freed before new cells are marked, so the net cost is
        bounded by |old_cells| + |new_cells| writes.

        Raises
        ------
        KeyError
            If *oid* is not registered.
        """
        if oid not in self._obstacles:
            raise KeyError(f"Obstacle '{oid}' is not registered.")
        obs = self._obstacles[oid]
        old_cells = obs.cells

        # Free old cells that are NOT also in new_cells (avoid double-write)
        to_free = old_cells - new_cells
        self._write_cells(to_free, CellState.FREE)

        # Mark newly occupied cells
        to_mark = new_cells - old_cells
        self._write_cells(to_mark, CellState.DYNAMIC)

        obs.cells = set(new_cells)
        # Notify with the union of changed cells
        self._notify(oid, "moved", to_free | to_mark)

    def update_position(self, oid: ObstacleId, new_center: Coord3D) -> None:
        """
        Convenience method: shift a single-cell obstacle to *new_center*.
        For multi-cell agents, use :meth:`move` directly.
        """
        if oid not in self._obstacles:
            raise KeyError(f"Obstacle '{oid}' is not registered.")
        self.move(oid, {new_center})

    def get_cells(self, oid: ObstacleId) -> Set[Coord3D]:
        """Return current cells occupied by *oid*."""
        return set(self._obstacles[oid].cells)

    def all_ids(self) -> List[ObstacleId]:
        """Return a list of all registered obstacle IDs."""
        return list(self._obstacles.keys())

    def __len__(self) -> int:
        return len(self._obstacles)

    def __contains__(self, oid: ObstacleId) -> bool:
        return oid in self._obstacles

    def __repr__(self) -> str:
        return f"DynamicObstacleRegistry(n={len(self._obstacles)} obstacles)"
