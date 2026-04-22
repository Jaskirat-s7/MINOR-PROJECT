"""
test_environment.py
===================
Self-contained test suite for Phase 1 of the UAV Path Planning project.

Run with:
    python test_environment.py

No external test framework required — the script prints a pass/fail summary.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

# Ensure the project root (parent of this file) is on the path
sys.path.insert(0, str(Path(__file__).parent))

from environment import (
    CellState,
    DynamicObstacleRegistry,
    Grid3D,
    GridMetadata,
    MapGenerator,
    MapIO,
)

# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------

_PASS: List[str] = []
_FAIL: List[str] = []


def test(name: str) -> Callable:
    """Decorator that registers a named test function."""
    def decorator(fn: Callable) -> Callable:
        try:
            fn()
            _PASS.append(name)
            print(f"  ✓  {name}")
        except Exception as exc:
            _FAIL.append(name)
            print(f"  ✗  {name}")
            traceback.print_exc()
        return fn
    return decorator


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ===========================================================================
# Section 1 – Grid3D core
# ===========================================================================

section("1. Grid3D – Core functionality")


@test("Default grid is 50×50×50 and fully free")
def _():
    g = Grid3D()
    assert g.size == (50, 50, 50)
    assert g.obstacle_density == 0.0


@test("Custom size is respected")
def _():
    g = Grid3D(size=(10, 20, 15))
    assert g.size == (10, 20, 15)
    assert g.x_size == 10 and g.y_size == 20 and g.z_size == 15


@test("in_bounds rejects out-of-range coordinates")
def _():
    g = Grid3D(size=(10, 10, 10))
    assert g.in_bounds((0, 0, 0))
    assert g.in_bounds((9, 9, 9))
    assert not g.in_bounds((10, 0, 0))
    assert not g.in_bounds((-1, 5, 5))
    assert not g.in_bounds((5, 5, 10))


@test("set_obstacle / is_free / is_obstacle round-trip")
def _():
    g = Grid3D(size=(10, 10, 10))
    coord = (5, 5, 5)
    assert g.is_free(coord)
    g.set_obstacle(coord)
    assert g.is_obstacle(coord)
    assert not g.is_free(coord)
    g.clear_obstacle(coord)
    assert g.is_free(coord)


@test("add_cuboid fills correct voxels")
def _():
    g = Grid3D(size=(20, 20, 20))
    g.add_cuboid((2, 2, 2), (4, 4, 4))
    # All interior cells must be obstacles
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(2, 6):
                assert g.is_obstacle((x, y, z)), f"Expected obstacle at {(x,y,z)}"
    # Cell just outside must be free
    assert g.is_free((6, 2, 2))


@test("add_cuboid clips correctly at grid boundary")
def _():
    g = Grid3D(size=(10, 10, 10))
    g.add_cuboid((8, 8, 8), (10, 10, 10))   # extends well beyond the grid
    assert g.is_obstacle((9, 9, 9))          # last valid cell is an obstacle
    # No IndexError should occur


@test("validate_position raises for obstacle cell")
def _():
    g = Grid3D(size=(10, 10, 10))
    g.set_obstacle((5, 5, 5))
    try:
        g.validate_position((5, 5, 5), "start")
        assert False, "Expected ValueError"
    except ValueError:
        pass


@test("validate_position raises for out-of-bounds cell")
def _():
    g = Grid3D(size=(10, 10, 10))
    try:
        g.validate_position((15, 0, 0), "goal")
        assert False, "Expected ValueError"
    except ValueError:
        pass


@test("get_neighbors returns ≤ 26 cells in 26-connectivity")
def _():
    g = Grid3D(size=(10, 10, 10))
    nbs = g.get_neighbors((5, 5, 5))
    assert len(nbs) <= 26
    assert all(g.is_free(nb) for nb in nbs)


@test("get_neighbors returns ≤ 6 cells in face-only connectivity")
def _():
    g = Grid3D(size=(10, 10, 10))
    nbs = g.get_neighbors((5, 5, 5), allow_diagonal=False)
    assert len(nbs) <= 6


@test("get_neighbors excludes out-of-bounds and obstacle cells")
def _():
    g = Grid3D(size=(5, 5, 5))
    g.set_obstacle((1, 1, 1))
    nbs = g.get_neighbors((0, 0, 0))
    assert (1, 1, 1) not in nbs
    assert all(g.in_bounds(nb) for nb in nbs)


@test("obstacle_density is accurate after bulk obstacle placement")
def _():
    g = Grid3D(size=(10, 10, 10))
    # Place 100 obstacles in a 10×10×1 slice → 10% density
    g.add_cuboid((0, 0, 0), (10, 10, 1))
    assert abs(g.obstacle_density - 0.10) < 1e-6


@test("iter_free_cells yields correct count")
def _():
    g = Grid3D(size=(5, 5, 5))
    g.add_cuboid((0, 0, 0), (5, 5, 1))   # bottom layer blocked
    free = list(g.iter_free_cells())
    assert len(free) == 5 * 5 * 4


@test("has_line_of_sight returns True for unobstructed path")
def _():
    g = Grid3D(size=(20, 20, 20))
    assert g.has_line_of_sight((0, 0, 0), (19, 19, 19))


@test("has_line_of_sight returns False when wall blocks path")
def _():
    g = Grid3D(size=(20, 20, 5))
    # Build a solid wall at x=10
    g.add_cuboid((10, 0, 0), (1, 20, 5))
    assert not g.has_line_of_sight((5, 10, 2), (15, 10, 2))


# ===========================================================================
# Section 2 – MapGenerator
# ===========================================================================

section("2. MapGenerator – Environment generation")


@test("Cuboid environment: start and goal are free")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=42)
    grid, start, goal = gen.generate_cuboid(num_obstacles=10)
    assert grid.is_free(start), f"Start {start} is not free"
    assert grid.is_free(goal), f"Goal {goal} is not free"


@test("Cuboid environment: grid has obstacles")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=42)
    grid, _, _ = gen.generate_cuboid(num_obstacles=15)
    assert grid.obstacle_density > 0.0


@test("Dense-mixed environment: start and goal are free")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=7)
    grid, start, goal = gen.generate_dense_mixed(random_obstacle_ratio=0.08)
    assert grid.is_free(start)
    assert grid.is_free(goal)


@test("Dense-mixed environment: metadata env_type is correct")
def _():
    gen = MapGenerator(seed=0)
    grid, _, _ = gen.generate_dense_mixed()
    assert grid.metadata.env_type == "dense_mixed"


@test("Maze environment: start and goal are free")
def _():
    gen = MapGenerator(size=(30, 30, 20), seed=99)
    grid, start, goal = gen.generate_maze()
    assert grid.is_free(start)
    assert grid.is_free(goal)


@test("Maze environment: has significant obstacle density")
def _():
    gen = MapGenerator(size=(30, 30, 20), seed=123)
    grid, _, _ = gen.generate_maze()
    # A valid maze should block a good portion of the lower layers
    assert grid.obstacle_density > 0.05


@test("Seed reproducibility: two identical seeds produce identical grids")
def _():
    gen1 = MapGenerator(size=(20, 20, 20), seed=55)
    gen2 = MapGenerator(size=(20, 20, 20), seed=55)
    grid1, s1, g1 = gen1.generate_cuboid(num_obstacles=10)
    grid2, s2, g2 = gen2.generate_cuboid(num_obstacles=10)
    import numpy as np
    assert (grid1.data == grid2.data).all(), "Grids differ despite same seed"
    assert s1 == s2 and g1 == g2


@test("Different seeds produce different grids")
def _():
    import numpy as np
    gen1 = MapGenerator(size=(20, 20, 20), seed=1)
    gen2 = MapGenerator(size=(20, 20, 20), seed=2)
    g1, _, _ = gen1.generate_cuboid()
    g2, _, _ = gen2.generate_cuboid()
    assert not (g1.data == g2.data).all(), "Different seeds produced identical grids"


# ===========================================================================
# Section 3 – MapIO (serialisation)
# ===========================================================================

section("3. MapIO – Save / Load")

_TMP = Path("_test_maps")
_TMP.mkdir(exist_ok=True)


@test("NPZ round-trip preserves grid data and metadata")
def _():
    import numpy as np
    gen = MapGenerator(size=(15, 15, 15), seed=42)
    grid, start, goal = gen.generate_cuboid(num_obstacles=5)

    path = _TMP / "test_cuboid.npz"
    MapIO.save_npz(grid, path)
    loaded = MapIO.load_npz(path)

    assert (grid.data == loaded.data).all()
    assert loaded.metadata.name == grid.metadata.name
    assert loaded.metadata.env_type == grid.metadata.env_type
    assert loaded.metadata.seed == grid.metadata.seed


@test("JSON round-trip preserves grid data and metadata")
def _():
    import numpy as np
    gen = MapGenerator(size=(15, 15, 15), seed=7)
    grid, _, _ = gen.generate_dense_mixed(random_obstacle_ratio=0.05)

    path = _TMP / "test_dense.json"
    MapIO.save_json(grid, path, indent=2)
    loaded = MapIO.load_json(path)

    assert (grid.data == loaded.data).all()
    assert loaded.metadata.env_type == "dense_mixed"


@test("Auto-dispatch save/load (npz)")
def _():
    import numpy as np
    gen = MapGenerator(size=(12, 12, 12), seed=3)
    grid, _, _ = gen.generate_maze()
    path = _TMP / "test_maze.npz"
    MapIO.save(grid, path)
    loaded = MapIO.load(path)
    assert (grid.data == loaded.data).all()


@test("Auto-dispatch save/load (json)")
def _():
    import numpy as np
    gen = MapGenerator(size=(15, 15, 15), seed=111)
    grid, _, _ = gen.generate_cuboid(num_obstacles=3)
    path = _TMP / "test_cuboid.json"
    MapIO.save(grid, path)
    loaded = MapIO.load(path)
    assert (grid.data == loaded.data).all()


@test("load_npz raises FileNotFoundError for missing file")
def _():
    try:
        MapIO.load_npz(_TMP / "nonexistent.npz")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ===========================================================================
# Section 4 – DynamicObstacleRegistry
# ===========================================================================

section("4. DynamicObstacleRegistry – Dynamic obstacles")


@test("add marks cells as DYNAMIC in grid")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    cells = {(3, 3, 3), (3, 4, 3)}
    reg.add("drone_1", cells)
    for c in cells:
        assert g.get_cell(c) == CellState.DYNAMIC, f"Expected DYNAMIC at {c}"


@test("remove frees cells back to FREE")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    cells = {(2, 2, 2)}
    reg.add("drone_1", cells)
    reg.remove("drone_1")
    assert g.is_free((2, 2, 2))
    assert "drone_1" not in reg


@test("move atomically updates cells")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    reg.add("drone_1", {(1, 1, 1)})
    reg.move("drone_1", {(2, 2, 2)})
    assert g.is_free((1, 1, 1)), "Old cell should be freed"
    assert g.get_cell((2, 2, 2)) == CellState.DYNAMIC, "New cell should be DYNAMIC"


@test("on_change callback is fired on add / remove / move")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    events = []
    reg.on_change.append(lambda oid, action, cells: events.append((oid, action)))

    reg.add("d1", {(5, 5, 5)})
    reg.move("d1", {(6, 6, 6)})
    reg.remove("d1")

    assert ("d1", "added") in events
    assert ("d1", "moved") in events
    assert ("d1", "removed") in events


@test("duplicate add raises KeyError")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    reg.add("d1", {(1, 1, 1)})
    try:
        reg.add("d1", {(2, 2, 2)})
        assert False, "Expected KeyError"
    except KeyError:
        pass


@test("remove unknown id raises KeyError")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    try:
        reg.remove("ghost")
        assert False, "Expected KeyError"
    except KeyError:
        pass


@test("len() and __contains__ are consistent")
def _():
    g = Grid3D(size=(10, 10, 10))
    reg = DynamicObstacleRegistry(g)
    assert len(reg) == 0
    reg.add("a", {(0, 0, 0)})
    reg.add("b", {(9, 9, 9)})
    assert len(reg) == 2
    assert "a" in reg
    assert "c" not in reg


# ===========================================================================
# Summary
# ===========================================================================

def _cleanup() -> None:
    import shutil
    if _TMP.exists():
        shutil.rmtree(_TMP)


_cleanup()

total = len(_PASS) + len(_FAIL)
print(f"\n{'═' * 60}")
print(f"  Results: {len(_PASS)}/{total} tests passed", end="")
if _FAIL:
    print(f"  |  FAILED: {', '.join(_FAIL)}")
else:
    print("  ✓ All tests passed!")
print(f"{'═' * 60}\n")

sys.exit(0 if not _FAIL else 1)
