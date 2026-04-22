"""
grid3d.py
=========
Core 3D voxel grid representing the UAV navigation environment.

Cell encoding
-------------
  0  (CellState.FREE)     – navigable free space
  1  (CellState.OBSTACLE) – static obstacle
  2  (CellState.DYNAMIC)  – dynamic obstacle (reserved for Phase 3)

26-connectivity
---------------
A UAV can move to any of the 26 face / edge / corner neighbours in one step,
which enables diagonal and vertical-diagonal movement that any-angle planning
(Theta*) later exploits.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Coord3D = Tuple[int, int, int]

# ---------------------------------------------------------------------------
# Precomputed 26-direction offset table (excludes (0,0,0))
# ---------------------------------------------------------------------------
_NEIGHBOR_OFFSETS: List[Coord3D] = [
    (dx, dy, dz)
    for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
    if (dx, dy, dz) != (0, 0, 0)
]


class CellState(IntEnum):
    """Semantic labels for grid cells."""
    FREE = 0
    OBSTACLE = 1
    DYNAMIC = 2   # reserved for dynamic obstacle support in Phase 3


@dataclass
class GridMetadata:
    """
    Lightweight metadata attached to every Grid3D instance.
    Preserved through serialisation so reproducibility is guaranteed.
    """
    name: str = "unnamed"
    env_type: str = "unknown"   # "cuboid" | "dense_mixed" | "maze"
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)


class Grid3D:
    """
    A 50×50×50 (default) 3D voxel environment for UAV path-planning.

    Axes convention (right-hand, Z-up)
    -----------------------------------
      x → width   (columns)
      y → depth   (rows)
      z → height  (layers)

    Indexing
    --------
    Internal NumPy array is shaped (x_size, y_size, z_size) and stored as
    ``uint8`` to minimise memory (~125 KB for default size).  All public
    methods accept (x, y, z) tuples so callers never have to think about
    axis order.

    Parameters
    ----------
    size : tuple of int
        Grid dimensions as (x_size, y_size, z_size).  Defaults to (50, 50, 50).
    metadata : GridMetadata, optional
        Descriptive metadata; auto-created if omitted.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        size: Tuple[int, int, int] = (50, 50, 50),
        metadata: Optional[GridMetadata] = None,
    ) -> None:
        if any(s < 2 for s in size):
            raise ValueError(f"Each grid dimension must be ≥ 2, got {size}.")

        self._size: Tuple[int, int, int] = tuple(size)  # type: ignore[assignment]
        self._data: np.ndarray = np.zeros(self._size, dtype=np.uint8)
        self.metadata: GridMetadata = metadata or GridMetadata()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> Tuple[int, int, int]:
        """Grid dimensions (x_size, y_size, z_size) — immutable."""
        return self._size

    @property
    def x_size(self) -> int:
        return self._size[0]

    @property
    def y_size(self) -> int:
        return self._size[1]

    @property
    def z_size(self) -> int:
        return self._size[2]

    @property
    def data(self) -> np.ndarray:
        """Read-only view of the raw uint8 grid array."""
        view = self._data.view()
        view.flags.writeable = False
        return view

    @property
    def obstacle_density(self) -> float:
        """Fraction of cells occupied by static obstacles."""
        return float(np.sum(self._data == CellState.OBSTACLE)) / self._data.size

    # ------------------------------------------------------------------
    # Coordinate validation
    # ------------------------------------------------------------------

    def in_bounds(self, coord: Coord3D) -> bool:
        """Return True if *coord* lies within grid boundaries."""
        x, y, z = coord
        return 0 <= x < self._size[0] and 0 <= y < self._size[1] and 0 <= z < self._size[2]

    def is_free(self, coord: Coord3D) -> bool:
        """Return True if *coord* is in-bounds and not occupied."""
        if not self.in_bounds(coord):
            return False
        return self._data[coord] == CellState.FREE

    def is_obstacle(self, coord: Coord3D) -> bool:
        """Return True if *coord* holds a static or dynamic obstacle."""
        if not self.in_bounds(coord):
            return False
        return self._data[coord] != CellState.FREE

    def validate_position(
        self, coord: Coord3D, label: str = "position"
    ) -> None:
        """
        Raise ``ValueError`` if *coord* is out-of-bounds or inside an obstacle.
        Used to sanity-check start/goal before running planners.
        """
        if not self.in_bounds(coord):
            raise ValueError(
                f"{label} {coord} is outside grid bounds {self._size}."
            )
        if not self.is_free(coord):
            raise ValueError(
                f"{label} {coord} is inside an obstacle "
                f"(cell value={self._data[coord]})."
            )

    # ------------------------------------------------------------------
    # Cell read / write
    # ------------------------------------------------------------------

    def get_cell(self, coord: Coord3D) -> int:
        """Return the raw cell value at *coord*."""
        if not self.in_bounds(coord):
            raise IndexError(f"Coordinate {coord} out of bounds {self._size}.")
        return int(self._data[coord])

    def set_cell(self, coord: Coord3D, value: CellState) -> None:
        """Write *value* to *coord*.  Out-of-bounds writes are silently skipped."""
        if self.in_bounds(coord):
            self._data[coord] = int(value)

    def set_obstacle(self, coord: Coord3D) -> None:
        """Convenience alias: mark *coord* as a static obstacle."""
        self.set_cell(coord, CellState.OBSTACLE)

    def clear_obstacle(self, coord: Coord3D) -> None:
        """Convenience alias: remove obstacle at *coord* (set FREE)."""
        self.set_cell(coord, CellState.FREE)

    def set_dynamic(self, coord: Coord3D) -> None:
        """Mark *coord* as occupied by a dynamic obstacle (Phase 3 use)."""
        self.set_cell(coord, CellState.DYNAMIC)

    # ------------------------------------------------------------------
    # Bulk obstacle operations
    # ------------------------------------------------------------------

    def add_cuboid(
        self,
        corner: Coord3D,
        dimensions: Tuple[int, int, int],
        state: CellState = CellState.OBSTACLE,
    ) -> None:
        """
        Fill a rectangular cuboid with *state*.

        Parameters
        ----------
        corner :
            Minimum (x, y, z) corner of the cuboid.
        dimensions :
            (dx, dy, dz) extents.  Cells in range [corner, corner+dimensions)
            are set.
        """
        x0, y0, z0 = corner
        dx, dy, dz = dimensions

        # Clamp to grid boundaries
        x1 = min(x0 + dx, self._size[0])
        y1 = min(y0 + dy, self._size[1])
        z1 = min(z0 + dz, self._size[2])
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        z0 = max(z0, 0)

        if x0 < x1 and y0 < y1 and z0 < z1:
            self._data[x0:x1, y0:y1, z0:z1] = int(state)

    def fill_layer(self, z: int, state: CellState = CellState.OBSTACLE) -> None:
        """Fill an entire horizontal layer at height *z*."""
        if 0 <= z < self._size[2]:
            self._data[:, :, z] = int(state)

    # ------------------------------------------------------------------
    # Neighbour enumeration
    # ------------------------------------------------------------------

    def get_neighbors(
        self, coord: Coord3D, allow_diagonal: bool = True
    ) -> List[Coord3D]:
        """
        Return all valid, free neighbours of *coord*.

        Parameters
        ----------
        coord :
            Source cell.
        allow_diagonal :
            If True (default) uses 26-connectivity.
            If False falls back to 6-face connectivity only.

        Returns
        -------
        list of Coord3D
            Each neighbour is guaranteed in-bounds and free.
        """
        offsets = _NEIGHBOR_OFFSETS if allow_diagonal else [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
        x, y, z = coord
        result: List[Coord3D] = []
        for dx, dy, dz in offsets:
            nb: Coord3D = (x + dx, y + dy, z + dz)
            if self.is_free(nb):
                result.append(nb)
        return result

    def iter_free_cells(self) -> Iterator[Coord3D]:
        """Yield every free cell coordinate — useful for random sampling."""
        xs, ys, zs = np.where(self._data == CellState.FREE)
        for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
            yield (x, y, z)

    # ------------------------------------------------------------------
    # Path validation helpers (used by planners)
    # ------------------------------------------------------------------

    def has_line_of_sight(self, a: Coord3D, b: Coord3D) -> bool:
        """
        3D rasterised line-of-sight check (Bresenham-style voxel traversal).
        Returns True if no obstacle voxel lies strictly between *a* and *b*.
        This is the key primitive for Theta* any-angle optimisation.
        """
        x0, y0, z0 = a
        x1, y1, z1 = b
        dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        sz = 1 if z1 > z0 else -1

        if dx >= dy and dx >= dz:
            ey, ez = 2 * dy - dx, 2 * dz - dx
            while x0 != x1:
                x0 += sx
                if ey > 0:
                    y0 += sy
                    ey -= 2 * dx
                if ez > 0:
                    z0 += sz
                    ez -= 2 * dx
                ey += 2 * dy
                ez += 2 * dz
                if not self.is_free((x0, y0, z0)):
                    return False
        elif dy >= dx and dy >= dz:
            ex, ez = 2 * dx - dy, 2 * dz - dy
            while y0 != y1:
                y0 += sy
                if ex > 0:
                    x0 += sx
                    ex -= 2 * dy
                if ez > 0:
                    z0 += sz
                    ez -= 2 * dy
                ex += 2 * dx
                ez += 2 * dz
                if not self.is_free((x0, y0, z0)):
                    return False
        else:
            ex, ey = 2 * dx - dz, 2 * dy - dz
            while z0 != z1:
                z0 += sz
                if ex > 0:
                    x0 += sx
                    ex -= 2 * dz
                if ey > 0:
                    y0 += sy
                    ey -= 2 * dz
                ex += 2 * dx
                ey += 2 * dy
                if not self.is_free((x0, y0, z0)):
                    return False
        return True

    # ------------------------------------------------------------------
    # Convenience / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        total = self._data.size
        obstacle_count = int(np.sum(self._data == CellState.OBSTACLE))
        dynamic_count = int(np.sum(self._data == CellState.DYNAMIC))
        free_count = total - obstacle_count - dynamic_count
        return (
            f"Grid3D(size={self._size}, "
            f"free={free_count} ({100 * free_count / total:.1f}%), "
            f"obstacles={obstacle_count} ({100 * obstacle_count / total:.1f}%), "
            f"dynamic={dynamic_count}, "
            f"env_type='{self.metadata.env_type}')"
        )

    def __repr__(self) -> str:
        return self.summary()
