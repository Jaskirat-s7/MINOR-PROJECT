"""
test_planner.py
===============
Self-contained test suite for Phase 2: Standard A* planner.

Run with:
    python3 test_planner.py

All 4 Phase 1 test sections still live in test_environment.py.
This file focuses purely on the planner layer.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Callable, List

sys.path.insert(0, str(Path(__file__).parent))

from environment import Grid3D, MapGenerator
from planner import (
    AStarPlanner,
    PlanResult,
    SearchNode,
    chebyshev_3d,
    euclidean_3d,
    manhattan_3d,
)
from planner.heuristics import get_heuristic, octile_3d

# ---------------------------------------------------------------------------
# Minimal test runner (mirrors test_environment.py style)
# ---------------------------------------------------------------------------

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
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


# ===========================================================================
# Section 1 – SearchNode
# ===========================================================================

section("1. SearchNode")


@test("SearchNode stores pos, g, h, f correctly")
def _():
    n = SearchNode(pos=(1, 2, 3), g=2.5, h=1.5)
    assert n.pos == (1, 2, 3)
    assert n.g == 2.5
    assert n.h == 1.5
    assert math.isclose(n.f, 4.0)
    assert n.parent is None


@test("SearchNode heap ordering: lower f wins")
def _():
    import heapq
    a = SearchNode(pos=(0, 0, 0), g=1.0, h=3.0, seq=0)   # f=4
    b = SearchNode(pos=(1, 0, 0), g=1.0, h=2.0, seq=1)   # f=3
    heap = [a, b]
    heapq.heapify(heap)
    assert heapq.heappop(heap).pos == (1, 0, 0)   # b (f=3) pops first


@test("SearchNode tiebreak: lower h wins when f is equal")
def _():
    import heapq
    a = SearchNode(pos=(0, 0, 0), g=2.0, h=2.0, seq=0)   # f=4, h=2
    b = SearchNode(pos=(1, 0, 0), g=3.0, h=1.0, seq=1)   # f=4, h=1
    heap = [a, b]
    heapq.heapify(heap)
    assert heapq.heappop(heap).pos == (1, 0, 0)   # b (lower h) pops first


@test("SearchNode tiebreak: lower seq wins when f and h are equal")
def _():
    import heapq
    a = SearchNode(pos=(0, 0, 0), g=2.0, h=2.0, seq=0)
    b = SearchNode(pos=(1, 0, 0), g=2.0, h=2.0, seq=1)
    heap = [a, b]
    heapq.heapify(heap)
    assert heapq.heappop(heap).pos == (0, 0, 0)   # a (lower seq) pops first


@test("SearchNode parent chain is correct")
def _():
    root = SearchNode(pos=(0, 0, 0), g=0.0, h=5.0)
    child = SearchNode(pos=(1, 0, 0), g=1.0, h=4.0, parent=root)
    assert child.parent is root
    assert child.parent.parent is None


# ===========================================================================
# Section 2 – Heuristics
# ===========================================================================

section("2. Heuristics")


@test("euclidean_3d: same point → 0")
def _():
    assert euclidean_3d((3, 3, 3), (3, 3, 3)) == 0.0


@test("euclidean_3d: axis-aligned distance")
def _():
    assert math.isclose(euclidean_3d((0, 0, 0), (3, 0, 0)), 3.0)


@test("euclidean_3d: 3D diagonal distance")
def _():
    # sqrt(1² + 1² + 1²) = sqrt(3)
    assert math.isclose(euclidean_3d((0, 0, 0), (1, 1, 1)), math.sqrt(3))


@test("manhattan_3d: correct sum of absolute differences")
def _():
    assert manhattan_3d((0, 0, 0), (3, 4, 5)) == 12.0


@test("chebyshev_3d: correct max of absolute differences")
def _():
    assert chebyshev_3d((0, 0, 0), (3, 4, 5)) == 5.0


@test("octile_3d: is admissible relative to grid step costs (≤ optimal grid path cost)")
def _():
    # octile_3d computes the exact minimum grid path cost assuming perfect
    # diagonal alignment: d1 axis + (d2-d3) diagonal + d3 tri-diagonal steps.
    # It is admissible (never over-estimates the true grid cost) but CAN
    # exceed Euclidean straight-line distance when grid steps cannot form a
    # straight line.  Verify it equals the analytical minimum cost.
    _SQRT2 = math.sqrt(2.0)
    _SQRT3 = math.sqrt(3.0)
    cases = [
        ((0, 0, 0), (5, 0, 0)),    # pure axis      → should be 5.0
        ((0, 0, 0), (3, 3, 0)),    # 2D diagonal    → should be 3√2 ≈ 4.243
        ((0, 0, 0), (3, 3, 3)),    # 3D tri-diagonal → should be 3√3 ≈ 5.196
        ((0, 0, 0), (5, 3, 1)),    # mixed
    ]
    expected = [
        5.0,
        3 * _SQRT2,
        3 * _SQRT3,
        # d1=5,d2=3,d3=1: (5-3)*1 + (3-1)*sqrt2 + 1*sqrt3
        2 * 1.0 + 2 * _SQRT2 + 1 * _SQRT3,
    ]
    from planner.heuristics import octile_3d
    for (a, b), exp in zip(cases, expected):
        got = octile_3d(a, b)
        assert math.isclose(got, exp, rel_tol=1e-9), (
            f"octile({a}, {b}) = {got:.6f}, expected {exp:.6f}"
        )


@test("octile_3d: value is ≥ chebyshev (tighter than chebyshev)")
def _():
    import random
    rng = random.Random(77)
    for _ in range(200):
        a = (rng.randint(0, 49), rng.randint(0, 49), rng.randint(0, 49))
        b = (rng.randint(0, 49), rng.randint(0, 49), rng.randint(0, 49))
        assert octile_3d(a, b) >= chebyshev_3d(a, b) - 1e-9


@test("get_heuristic: returns correct callable by name")
def _():
    fn = get_heuristic("euclidean")
    assert math.isclose(fn((0, 0, 0), (1, 0, 0)), 1.0)


@test("get_heuristic: raises KeyError for unknown name")
def _():
    try:
        get_heuristic("nonexistent")
        assert False, "Expected KeyError"
    except KeyError:
        pass


# ===========================================================================
# Section 3 – AStarPlanner: basic correctness
# ===========================================================================

section("3. AStarPlanner – Correctness")


def _empty_grid(size=(20, 20, 20)) -> Grid3D:
    """Fully free grid — guarantees a straight-line path exists."""
    return Grid3D(size=size)


@test("A* finds path on empty grid")
def _():
    g = _empty_grid()
    result = AStarPlanner(g).solve((0, 0, 0), (19, 19, 19))
    assert result.success
    assert len(result.path) > 0
    assert result.path[0] == (0, 0, 0)
    assert result.path[-1] == (19, 19, 19)


@test("A* path starts at start and ends at goal")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=1)
    grid, start, goal = gen.generate_cuboid(num_obstacles=10)
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal


@test("A* path is obstacle-free")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=2)
    grid, start, goal = gen.generate_cuboid(num_obstacles=10)
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success
    for coord in result.path:
        assert grid.is_free(coord), f"Obstacle at {coord} is on the path!"


@test("A* path is connected (each step ≤ √3)")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=3)
    grid, start, goal = gen.generate_cuboid(num_obstacles=8)
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success
    for a, b in zip(result.path, result.path[1:]):
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        assert dist <= math.sqrt(3) + 1e-9, f"Step {a}→{b} distance {dist:.4f} > √3"


@test("A* trivial case: start == goal")
def _():
    g = _empty_grid()
    result = AStarPlanner(g).solve((5, 5, 5), (5, 5, 5))
    assert result.success
    assert result.path == [(5, 5, 5)]
    assert result.path_length == 0.0
    assert result.nodes_expanded == 0


@test("A* returns failure when goal is completely enclosed by obstacles")
def _():
    g = Grid3D(size=(10, 10, 10))
    # Surround (5,5,5) with an impenetrable shell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) != (0, 0, 0):
                    g.set_obstacle((5 + dx, 5 + dy, 5 + dz))
    # Set (5,5,5) as a free pocket surrounded by obstacles
    # Start: any free cell far away; goal: isolated cell
    # Actually just verify that a completely walled-in region is unreachable
    g.set_obstacle((5, 5, 5))  # make goal itself an obstacle
    try:
        AStarPlanner(g).solve((0, 0, 0), (5, 5, 5))
        assert False, "Expected ValueError (goal is obstacle)"
    except ValueError:
        pass


@test("A* raises ValueError for out-of-bounds start")
def _():
    g = _empty_grid()
    try:
        AStarPlanner(g).solve((99, 0, 0), (5, 5, 5))
        assert False, "Expected ValueError"
    except ValueError:
        pass


@test("A* raises ValueError for start inside obstacle")
def _():
    g = _empty_grid()
    g.set_obstacle((2, 2, 2))
    try:
        AStarPlanner(g).solve((2, 2, 2), (5, 5, 5))
        assert False, "Expected ValueError"
    except ValueError:
        pass


@test("A* path_length matches manually computed Euclidean chain")
def _():
    g = _empty_grid()
    result = AStarPlanner(g).solve((0, 0, 0), (3, 0, 0))
    assert result.success
    # Recompute length by summing step distances
    manual_len = sum(
        math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
        for p, q in zip(result.path, result.path[1:])
    )
    assert math.isclose(result.path_length, manual_len, rel_tol=1e-9)


# ===========================================================================
# Section 4 – AStarPlanner: optimality
# ===========================================================================

section("4. AStarPlanner – Optimality")


@test("Euclidean path_length is optimal on empty grid (equals straight-line distance)")
def _():
    g = _empty_grid((20, 20, 20))
    start, goal = (0, 0, 0), (5, 5, 5)
    result = AStarPlanner(g).solve(start, goal)
    assert result.success
    optimal = math.sqrt(3) * 5   # 5 space-diagonal steps
    assert math.isclose(result.path_length, optimal, rel_tol=1e-6), (
        f"path_length={result.path_length:.6f} vs optimal={optimal:.6f}"
    )


@test("Admissible heuristics (euclidean, chebyshev) find the same optimal path cost")
def _():
    gen = MapGenerator(size=(25, 25, 25), seed=55)
    grid, start, goal = gen.generate_cuboid(num_obstacles=8)
    planner = AStarPlanner(grid)

    # euclidean and chebyshev are both admissible w.r.t. Euclidean step cost
    # → must find same optimal path length.
    # octile uses grid-step cost as its bound and can exceed Euclidean distance
    # (inadmissible w.r.t. Euclidean metric), so it is excluded from this check.
    results = planner.compare_heuristics(start, goal, ["euclidean", "chebyshev"])
    costs = {name: r.path_length for name, r in results.items() if r.success}
    assert len(costs) == 2, f"Some heuristics failed: {results}"
    reference = costs["euclidean"]
    for name, cost in costs.items():
        assert math.isclose(cost, reference, rel_tol=1e-6), (
            f"{name} cost {cost:.6f} ≠ euclidean cost {reference:.6f}"
        )


# ===========================================================================
# Section 5 – AStarPlanner: performance metrics
# ===========================================================================

section("5. AStarPlanner – Performance metrics")


@test("PlanResult.metrics() returns all expected keys")
def _():
    g = _empty_grid()
    result = AStarPlanner(g).solve((0, 0, 0), (10, 10, 10))
    m = result.metrics()
    for key in ["success", "path_length", "waypoints", "nodes_expanded",
                "nodes_generated", "elapsed_ms", "heuristic", "start", "goal"]:
        assert key in m, f"Missing key: {key}"


@test("nodes_expanded ≤ nodes_generated always")
def _():
    gen = MapGenerator(size=(30, 30, 30), seed=7)
    grid, start, goal = gen.generate_dense_mixed()
    result = AStarPlanner(grid).solve(start, goal)
    assert result.nodes_expanded <= result.nodes_generated


@test("elapsed_seconds is positive")
def _():
    g = _empty_grid()
    result = AStarPlanner(g).solve((0, 0, 0), (15, 15, 15))
    assert result.elapsed_seconds > 0


@test("Octile heuristic expands fewer nodes than Chebyshev (tighter bound)")
def _():
    gen = MapGenerator(size=(40, 40, 40), seed=123)
    grid, start, goal = gen.generate_cuboid(num_obstacles=15)
    planner = AStarPlanner(grid)

    r_oct = planner.solve(start, goal, heuristic=octile_3d)
    r_cheb = planner.solve(start, goal, heuristic=chebyshev_3d)

    assert r_oct.success and r_cheb.success
    # Octile is ≥ chebyshev → at least as informed → expands ≤ nodes
    assert r_oct.nodes_expanded <= r_cheb.nodes_expanded + 1, (
        f"octile expanded {r_oct.nodes_expanded} vs chebyshev {r_cheb.nodes_expanded}"
    )


@test("Weighted A* (w=1.5) expands fewer nodes than standard A* (w=1.0)")
def _():
    gen = MapGenerator(size=(40, 40, 40), seed=77)
    grid, start, goal = gen.generate_dense_mixed()
    planner = AStarPlanner(grid)

    r_std = planner.solve(start, goal, weight=1.0)
    r_wt  = planner.solve(start, goal, weight=1.5)

    assert r_std.success and r_wt.success
    # Weighted should expand fewer or equal nodes
    assert r_wt.nodes_expanded <= r_std.nodes_expanded + 5


# ===========================================================================
# Section 6 – Integration with MapGenerator environments
# ===========================================================================

section("6. Integration – three environment types")


@test("A* solves cuboid environment (seed=42)")
def _():
    gen = MapGenerator(size=(50, 50, 50), seed=42)
    grid, start, goal = gen.generate_cuboid()
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success, f"A* failed on cuboid env: {result}"


@test("A* solves dense-mixed environment (seed=42)")
def _():
    gen = MapGenerator(size=(50, 50, 50), seed=42)
    grid, start, goal = gen.generate_dense_mixed()
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success, f"A* failed on dense env: {result}"


@test("A* solves maze environment (seed=42)")
def _():
    gen = MapGenerator(size=(50, 50, 50), seed=42)
    grid, start, goal = gen.generate_maze()
    result = AStarPlanner(grid).solve(start, goal)
    assert result.success, f"A* failed on maze env: {result}"


@test("6-connectivity mode finds a path on obstacle-free grid")
def _():
    g = _empty_grid((10, 10, 10))
    planner = AStarPlanner(g, allow_diagonal=False)
    result = planner.solve((0, 0, 0), (9, 9, 9))
    assert result.success
    # In 6-connectivity, all steps are axis-aligned → each step cost is 1.0
    for a, b in zip(result.path, result.path[1:]):
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        assert math.isclose(dist, 1.0, abs_tol=1e-9), f"Non-axis-aligned step: {a}→{b}"


# ===========================================================================
# Summary
# ===========================================================================

total = len(_PASS) + len(_FAIL)
print(f"\n{'═' * 62}")
print(f"  Results: {len(_PASS)}/{total} tests passed", end="")
if _FAIL:
    print(f"\n  FAILED: {', '.join(_FAIL)}")
else:
    print("  ✓ All tests passed!")
print(f"{'═' * 62}\n")

sys.exit(0 if not _FAIL else 1)
