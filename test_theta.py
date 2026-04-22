"""
test_theta.py
=============
Correctness and improvement tests for ThetaStarPlanner (Enhancement 1).

Run:
    python3 test_theta.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Callable, List

sys.path.insert(0, str(Path(__file__).parent))

from environment import Grid3D, MapGenerator
from planner import AStarPlanner, ThetaStarPlanner

_PASS: List[str] = []
_FAIL: List[str] = []


def test(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        try:
            fn()
            _PASS.append(name)
            print(f"  ✓  {name}")
        except Exception:
            _FAIL.append(name)
            print(f"  ✗  {name}")
            traceback.print_exc()
        return fn
    return decorator


def section(title: str) -> None:
    print(f"\n{'─' * 64}\n  {title}\n{'─' * 64}")


def _env(kind: str, seed: int = 42, size=(30, 30, 30)):
    gen = MapGenerator(size=size, seed=seed)
    return getattr(gen, f"generate_{kind}")()


# ═══════════════════════════════════════════════════════════════════════
# 1 – Correctness
# ═══════════════════════════════════════════════════════════════════════
section("1. Correctness — Theta* must satisfy all A* invariants")


@test("planner_name is 'Theta*'")
def _():
    g, s, goal = _env("cuboid")
    r = ThetaStarPlanner(g).solve(s, goal)
    assert r.planner_name == "Theta*"


@test("A* planner_name still 'A*' after Theta* import")
def _():
    g, s, goal = _env("cuboid")
    r = AStarPlanner(g).solve(s, goal)
    assert r.planner_name == "A*"


@test("Theta* finds a valid path — cuboid")
def _():
    g, s, goal = _env("cuboid")
    assert ThetaStarPlanner(g).solve(s, goal).success


@test("Theta* path starts at start and ends at goal")
def _():
    g, s, goal = _env("cuboid")
    r = ThetaStarPlanner(g).solve(s, goal)
    assert r.path[0] == s and r.path[-1] == goal


@test("Theta* path is entirely obstacle-free")
def _():
    g, s, goal = _env("dense_mixed", seed=7)
    r = ThetaStarPlanner(g).solve(s, goal)
    assert r.success
    for coord in r.path:
        assert g.is_free(coord), f"Obstacle at {coord}"


@test("Every Theta* segment has line-of-sight  (key invariant)")
def _():
    g, s, goal = _env("cuboid", seed=5)
    r = ThetaStarPlanner(g).solve(s, goal)
    assert r.success
    for a, b in zip(r.path, r.path[1:]):
        assert g.has_line_of_sight(a, b), f"Segment {a}→{b} blocked!"


@test("Theta* trivial case: start == goal")
def _():
    g = Grid3D(size=(10, 10, 10))
    r = ThetaStarPlanner(g).solve((5, 5, 5), (5, 5, 5))
    assert r.success and r.path == [(5, 5, 5)] and r.path_length == 0.0


@test("Theta* raises ValueError for obstacle start")
def _():
    g = Grid3D(size=(10, 10, 10))
    g.set_obstacle((3, 3, 3))
    try:
        ThetaStarPlanner(g).solve((3, 3, 3), (8, 8, 8))
        assert False
    except ValueError:
        pass


@test("Theta* returns failure when no path exists")
def _():
    g = Grid3D(size=(10, 10, 10))
    for x in range(3, 8):
        for y in range(3, 8):
            for z in range(3, 8):
                if (x, y, z) != (5, 5, 5):
                    g.set_obstacle((x, y, z))
    r = ThetaStarPlanner(g).solve((0, 0, 0), (5, 5, 5))
    assert not r.success


@test("Theta* path_length matches manual segment sum")
def _():
    g, s, goal = _env("cuboid")
    r = ThetaStarPlanner(g).solve(s, goal)
    assert r.success
    manual = sum(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
        for p, q in zip(r.path, r.path[1:])
    )
    assert math.isclose(r.path_length, manual, rel_tol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 2 – Improvement guarantees
# ═══════════════════════════════════════════════════════════════════════
section("2. Improvement — Theta* must improve or equal A* waypoints")


def _compare(kind: str, seed: int = 42, size=(30, 30, 30)):
    gen = MapGenerator(size=size, seed=seed)
    grid, s, g = getattr(gen, f"generate_{kind}")()
    return AStarPlanner(grid).solve(s, g), ThetaStarPlanner(grid).solve(s, g)


@test("Theta* ≤ waypoints vs A*  — cuboid")
def _():
    ra, rt = _compare("cuboid")
    assert rt.success and ra.success
    assert len(rt.path) <= len(ra.path)


@test("Theta* ≤ waypoints vs A*  — dense mixed")
def _():
    ra, rt = _compare("dense_mixed")
    assert rt.success and ra.success
    assert len(rt.path) <= len(ra.path)


@test("Theta* ≤ waypoints vs A*  — maze")
def _():
    ra, rt = _compare("maze")
    assert rt.success and ra.success
    assert len(rt.path) <= len(ra.path)


@test("Theta* reduces waypoints by ≥20% on 50³ cuboid (seed=42)")
def _():
    gen = MapGenerator(size=(50, 50, 50), seed=42)
    grid, s, g = gen.generate_cuboid(num_obstacles=20)
    ra = AStarPlanner(grid).solve(s, g)
    rt = ThetaStarPlanner(grid).solve(s, g)
    reduction = 1.0 - len(rt.path) / len(ra.path)
    assert reduction >= 0.20, f"Only {reduction:.1%} reduction"


@test("Theta* path_length within 10% of A* cost")
def _():
    gen = MapGenerator(size=(40, 40, 40), seed=99)
    grid, s, g = gen.generate_cuboid(num_obstacles=15)
    ra = AStarPlanner(grid).solve(s, g)
    rt = ThetaStarPlanner(grid).solve(s, g)
    assert rt.path_length / ra.path_length <= 1.10


@test("All Theta* segments LoS-clear on maze  (strictest env)")
def _():
    gen = MapGenerator(size=(40, 40, 40), seed=7)
    grid, s, g = gen.generate_maze()
    r = ThetaStarPlanner(grid).solve(s, g)
    assert r.success
    for a, b in zip(r.path, r.path[1:]):
        assert grid.has_line_of_sight(a, b)


@test("Theta* accepts weight parameter (weighted Theta*)")
def _():
    g, s, goal = _env("cuboid", size=(25, 25, 25))
    r = ThetaStarPlanner(g).solve(s, goal, weight=1.5)
    assert r.success


# ═══════════════════════════════════════════════════════════════════════
# 3 – Regression: A* identical after refactor
# ═══════════════════════════════════════════════════════════════════════
section("3. Regression — A* behaviour unchanged")


@test("A* paths still obstacle-free after _select_parent() refactor")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=11)
    grid, s, g = gen.generate_dense_mixed()
    r = AStarPlanner(grid).solve(s, g)
    assert r.success
    for coord in r.path:
        assert grid.is_free(coord)


@test("A* path_length still optimal on empty grid")
def _():
    g = Grid3D(size=(20, 20, 20))
    r = AStarPlanner(g).solve((0, 0, 0), (5, 5, 5))
    assert math.isclose(r.path_length, math.sqrt(3) * 5, rel_tol=1e-6)


@test("A* nodes_expanded identical to pre-refactor (1943, seed=42, cuboid)")
def _():
    gen = MapGenerator(size=(50, 50, 50), seed=42)
    grid, s, g = gen.generate_cuboid(num_obstacles=20)
    r = AStarPlanner(grid).solve(s, g)
    assert r.nodes_expanded == 1943, f"Got {r.nodes_expanded}, expected 1943"


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

total = len(_PASS) + len(_FAIL)
print(f"\n{'═' * 64}")
print(f"  Results: {len(_PASS)}/{total} tests passed", end="")
if _FAIL:
    print(f"\n  FAILED: {', '.join(_FAIL)}")
else:
    print("  ✓ All tests passed!")
print(f"{'═' * 64}\n")

sys.exit(0 if not _FAIL else 1)
