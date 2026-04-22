"""
map_generator.py
================
Generates three classes of 3D UAV environments, each designed to stress a
different aspect of path-planning:

  1. ``cuboid``     – sparse, axis-aligned box obstacles (baseline)
  2. ``dense_mixed``– random scatter + structured pillars (realistic indoor)
  3. ``maze``       – corridor-and-wall maze with dead-end traps
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np

from .grid3d import CellState, Grid3D, GridMetadata

Coord3D = Tuple[int, int, int]


class MapGenerator:
    """
    Stateless factory for procedurally generated Grid3D environments.

    All methods return a *new* Grid3D instance and are safe to call in
    parallel (no shared mutable state).

    Parameters
    ----------
    size :
        Grid dimensions.  Defaults to (50, 50, 50).
    seed :
        RNG seed for reproducibility.  Pass ``None`` for non-deterministic.
    """

    def __init__(
        self,
        size: Tuple[int, int, int] = (50, 50, 50),
        seed: Optional[int] = None,
    ) -> None:
        self.size = size
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal RNG helpers
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def _safe_start_goal(
        self,
        grid: Grid3D,
        start: Optional[Coord3D],
        goal: Optional[Coord3D],
        rng: np.random.Generator,
        margin: int = 2,
    ) -> Tuple[Coord3D, Coord3D]:
        """
        Resolve start/goal positions.  If not provided, sample random free
        cells that are at least *margin* voxels from the grid boundary and
        are sufficiently far apart.
        """
        X, Y, Z = self.size

        def _random_free(preferred_region: str) -> Coord3D:
            """Sample a free cell in a preferred half of the grid."""
            for _ in range(10_000):
                if preferred_region == "start":
                    x_hi = max(margin + 1, X // 3)
                    y_hi = max(margin + 1, Y // 3)
                    x = int(rng.integers(margin, x_hi))
                    y = int(rng.integers(margin, y_hi))
                else:
                    x_lo = min(2 * X // 3, X - margin - 1)
                    y_lo = min(2 * Y // 3, Y - margin - 1)
                    x = int(rng.integers(max(0, x_lo), X - margin))
                    y = int(rng.integers(max(0, y_lo), Y - margin))
                z_hi = max(margin + 1, Z - margin)
                z = int(rng.integers(margin, z_hi))
                coord: Coord3D = (x, y, z)
                if grid.is_free(coord):
                    return coord
            raise RuntimeError(
                "Could not find a free start/goal position after 10 000 attempts. "
                "Obstacle density may be too high."
            )

        s = start if (start and grid.is_free(start)) else _random_free("start")
        g = goal if (goal and grid.is_free(goal)) else _random_free("goal")
        return s, g

    # ------------------------------------------------------------------
    # Environment 1: Cuboid obstacles
    # ------------------------------------------------------------------

    def generate_cuboid(
        self,
        num_obstacles: int = 20,
        min_dim: int = 2,
        max_dim: int = 8,
        start: Optional[Coord3D] = None,
        goal: Optional[Coord3D] = None,
    ) -> Tuple[Grid3D, Coord3D, Coord3D]:
        """
        Place *num_obstacles* randomly sized axis-aligned cuboid obstacles.

        Mimics an open warehouse / atrium with scattered structural pillars
        and storage units — a common UAV delivery scenario.

        Returns
        -------
        grid, start, goal
        """
        rng = self._make_rng()
        meta = GridMetadata(name="cuboid", env_type="cuboid", seed=self.seed)
        grid = Grid3D(size=self.size, metadata=meta)
        X, Y, Z = self.size

        for _ in range(num_obstacles):
            dx = int(rng.integers(min_dim, max_dim + 1))
            dy = int(rng.integers(min_dim, max_dim + 1))
            dz = int(rng.integers(min_dim, max_dim + 1))
            x0 = int(rng.integers(0, X - dx))
            y0 = int(rng.integers(0, Y - dy))
            z0 = int(rng.integers(0, Z - dz))
            grid.add_cuboid((x0, y0, z0), (dx, dy, dz))

        s, g = self._safe_start_goal(grid, start, goal, rng)
        return grid, s, g

    # ------------------------------------------------------------------
    # Environment 2: Dense mixed obstacles
    # ------------------------------------------------------------------

    def generate_dense_mixed(
        self,
        random_obstacle_ratio: float = 0.12,
        num_pillars: int = 15,
        pillar_radius: int = 1,
        num_walls: int = 8,
        wall_thickness: int = 1,
        start: Optional[Coord3D] = None,
        goal: Optional[Coord3D] = None,
    ) -> Tuple[Grid3D, Coord3D, Coord3D]:
        """
        Mix three obstacle classes for a realistic indoor environment:

        * **Random scatter** – individual obstacle voxels (clutter, cables)
        * **Pillars**        – thin vertical columns (structural supports)
        * **Partial walls**  – flat panels with deliberate gaps (room dividers)

        Parameters
        ----------
        random_obstacle_ratio :
            Fraction of total cells to mark as scattered obstacles (0–1).
        num_pillars :
            Number of vertical pillar obstacles.
        pillar_radius :
            Pillar half-width in voxels (1 = 3×3 cross-section).
        num_walls :
            Number of partial walls (each has a randomly placed gap).
        wall_thickness :
            Wall panel thickness in voxels.
        """
        rng = self._make_rng()
        meta = GridMetadata(name="dense_mixed", env_type="dense_mixed", seed=self.seed)
        grid = Grid3D(size=self.size, metadata=meta)
        X, Y, Z = self.size

        # ---- 1. Random scatter ----------------------------------------
        n_random = int(random_obstacle_ratio * grid._data.size)
        coords = rng.integers(0, [X, Y, Z], size=(n_random, 3))
        grid._data[coords[:, 0], coords[:, 1], coords[:, 2]] = CellState.OBSTACLE

        # ---- 2. Vertical pillars --------------------------------------
        for _ in range(num_pillars):
            cx = int(rng.integers(pillar_radius + 1, X - pillar_radius - 1))
            cy = int(rng.integers(pillar_radius + 1, Y - pillar_radius - 1))
            z_base = int(rng.integers(0, Z // 4))
            z_top = int(rng.integers(3 * Z // 4, Z))
            grid.add_cuboid(
                (cx - pillar_radius, cy - pillar_radius, z_base),
                (2 * pillar_radius + 1, 2 * pillar_radius + 1, z_top - z_base),
            )

        # ---- 3. Partial walls with gaps --------------------------------
        for _ in range(num_walls):
            axis = int(rng.integers(0, 2))   # 0 → wall along Y, 1 → wall along X
            z_lo = min(Z // 3, Z - 1)
            z_hi_lo = min(2 * Z // 3, Z - 1)
            z_floor = int(rng.integers(0, max(1, z_lo)))
            z_ceil = int(rng.integers(max(z_floor + 1, z_hi_lo), Z))
            wall_height = z_ceil - z_floor

            # Gap position — guarantees a navigable corridor
            span = X if axis == 0 else Y
            gap_lo = min(2, span - 2)
            gap_hi = max(gap_lo + 1, span - 6)
            if gap_lo >= gap_hi:
                continue   # grid too small to fit a wall with gap; skip
            gap_start = int(rng.integers(gap_lo, gap_hi))
            gap_max = min(8, span - gap_start - 1)
            gap_width = int(rng.integers(min(4, gap_max), max(gap_max, gap_max + 1)))

            if axis == 0:        # wall perpendicular to X axis
                if X <= 10:
                    continue   # grid too small; skip
                wall_pos = int(rng.integers(5, X - 5))
                # Bottom section before gap
                grid.add_cuboid((wall_pos, 0, z_floor), (wall_thickness, gap_start, wall_height))
                # Top section after gap
                end = gap_start + gap_width
                grid.add_cuboid((wall_pos, end, z_floor), (wall_thickness, max(0, Y - end), wall_height))
            else:                # wall perpendicular to Y axis
                if Y <= 10:
                    continue   # grid too small; skip
                wall_pos = int(rng.integers(5, Y - 5))
                grid.add_cuboid((0, wall_pos, z_floor), (gap_start, wall_thickness, wall_height))
                end = gap_start + gap_width
                grid.add_cuboid((end, wall_pos, z_floor), (max(0, X - end), wall_thickness, wall_height))

        s, g = self._safe_start_goal(grid, start, goal, rng)
        return grid, s, g

    # ------------------------------------------------------------------
    # Environment 3: Maze-like with traps
    # ------------------------------------------------------------------

    def generate_maze(
        self,
        corridor_width: int = 3,
        wall_height_ratio: float = 0.75,
        num_traps: int = 6,
        start: Optional[Coord3D] = None,
        goal: Optional[Coord3D] = None,
    ) -> Tuple[Grid3D, Coord3D, Coord3D]:
        """
        Iterative DFS on a 2D cell graph to carve a perfect 2D maze,
        then extrude it vertically to form a 3D flyable maze.

        **Traps**: dead-end pockets whose only entrance is a narrow gap,
        designed to stress-test planners that blindly follow heuristics.

        Parameters
        ----------
        corridor_width :
            Voxel width of carved corridors.
        wall_height_ratio :
            Fraction of Z range occupied by maze walls (ceiling remains open
            so the UAV can in principle fly over — but planners should still
            try to stay *inside* the maze).
        num_traps :
            Number of dead-end trap alcoves to inject.
        """
        rng = self._make_rng()
        meta = GridMetadata(name="maze", env_type="maze", seed=self.seed)
        grid = Grid3D(size=self.size, metadata=meta)
        X, Y, Z = self.size

        # ---- Step 1: Start with a fully solid base slice ---------------
        wall_z = max(2, int(Z * wall_height_ratio))
        grid._data[:, :, :wall_z] = CellState.OBSTACLE

        # ---- Step 2: DFS maze carving in 2D (cells = corridor blocks) --
        cw = corridor_width
        # Number of maze cells that fit
        nx = max(2, (X - cw) // (cw + 1))
        ny = max(2, (Y - cw) // (cw + 1))

        visited = np.zeros((nx, ny), dtype=bool)
        stack: list[Tuple[int, int]] = []
        cx, cy = int(rng.integers(0, nx)), int(rng.integers(0, ny))
        visited[cx, cy] = True
        stack.append((cx, cy))

        def _cell_origin(ci: int, cj: int) -> Tuple[int, int]:
            """Top-left voxel of maze cell (ci, cj)."""
            return (ci * (cw + 1), cj * (cw + 1))

        def _carve(ci: int, cj: int, ni: int, nj: int) -> None:
            """Carve corridors between cells (ci,cj) and (ni,nj)."""
            ox, oy = _cell_origin(ci, cj)
            nnx, nny = _cell_origin(ni, nj)
            # Carve both cells' rooms
            for ix, iy in [(ox, oy), (nnx, nny)]:
                grid._data[ix:ix + cw, iy:iy + cw, :wall_z] = CellState.FREE
            # Carve the wall voxels between them
            mx = min(ox, nnx) + cw if abs(ci - ni) == 1 else ox
            my = min(oy, nny) + cw if abs(cj - nj) == 1 else oy
            wall_dx = abs(nnx - ox) - cw + cw if abs(ci - ni) == 1 else cw
            wall_dy = abs(nny - oy) - cw + cw if abs(cj - nj) == 1 else cw
            # Simpler: just clear the bounding box between the two rooms
            lx = min(ox, nnx)
            rx = max(ox, nnx) + cw
            ly = min(oy, nny)
            ry = max(oy, nny) + cw
            grid._data[lx:rx, ly:ry, :wall_z] = CellState.FREE

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while stack:
            ci, cj = stack[-1]
            shuffled = rng.permutation(len(dirs))
            moved = False
            for di in shuffled:
                dci, dcj = dirs[di]
                ni, nj = ci + dci, cj + dcj
                if 0 <= ni < nx and 0 <= nj < ny and not visited[ni, nj]:
                    visited[ni, nj] = True
                    _carve(ci, cj, ni, nj)
                    stack.append((ni, nj))
                    moved = True
                    break
            if not moved:
                stack.pop()

        # ---- Step 3: Inject traps (narrow dead-end pockets) ------------
        free_cells = list(grid.iter_free_cells())
        # Filter to cells in maze Z range
        trap_candidates = [c for c in free_cells if c[2] < wall_z // 2]
        rng.shuffle(trap_candidates)
        for trap_cell in trap_candidates[:num_traps]:
            tx, ty, tz = trap_cell
            # Build a small room around the cell and seal off 3 of 4 horizontal exits
            room_dx, room_dy = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            rx0 = max(0, tx - room_dx // 2)
            ry0 = max(0, ty - room_dy // 2)
            rz0, rz1 = 0, min(wall_z, tz + cw + 2)
            # Carve the room
            grid._data[rx0:rx0 + room_dx, ry0:ry0 + room_dy, rz0:rz1] = CellState.FREE
            # Re-seal 3 walls to create a near-dead-end (tiny gap left open)
            if rx0 > 0:
                grid._data[rx0 - 1, ry0:ry0 + room_dy, rz0:rz1] = CellState.OBSTACLE
            if ry0 > 0:
                grid._data[rx0:rx0 + room_dx, ry0 - 1, rz0:rz1] = CellState.OBSTACLE

        s, g = self._safe_start_goal(grid, start, goal, rng)
        return grid, s, g
