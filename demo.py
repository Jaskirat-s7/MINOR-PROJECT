"""
demo.py
=======
Quick demonstration of Phase 1 environment module.
Shows grid creation, map generation, serialisation, and dynamic obstacles.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment import (
    CellState,
    DynamicObstacleRegistry,
    Grid3D,
    MapGenerator,
    MapIO,
)


def separator(title: str) -> None:
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


# ---------------------------------------------------------------------------
# 1. Create a grid manually
# ---------------------------------------------------------------------------
separator("1. Manual Grid Creation")

grid = Grid3D(size=(10, 10, 10))
grid.add_cuboid(corner=(3, 3, 3), dimensions=(4, 4, 4))
print(grid)
print(f"  (5,5,5) is obstacle? : {grid.is_obstacle((5, 5, 5))}")
print(f"  (0,0,0) is free?     : {grid.is_free((0, 0, 0))}")
print(f"  Neighbours of (0,0,0): {grid.get_neighbors((0, 0, 0))}")


# ---------------------------------------------------------------------------
# 2. Generate the three environment types
# ---------------------------------------------------------------------------
separator("2. Map Generation")

gen = MapGenerator(size=(50, 50, 50), seed=42)

grid_c, start_c, goal_c = gen.generate_cuboid(num_obstacles=20)
print(f"\n  Cuboid env   → {grid_c}")
print(f"    Start: {start_c}  →  Goal: {goal_c}")

grid_d, start_d, goal_d = gen.generate_dense_mixed()
print(f"\n  Dense-mixed  → {grid_d}")
print(f"    Start: {start_d}  →  Goal: {goal_d}")

grid_m, start_m, goal_m = gen.generate_maze()
print(f"\n  Maze env     → {grid_m}")
print(f"    Start: {start_m}  →  Goal: {goal_m}")


# ---------------------------------------------------------------------------
# 3. Line-of-sight check (Theta* primitive)
# ---------------------------------------------------------------------------
separator("3. Line-of-Sight (Theta* primitive)")

los = grid_c.has_line_of_sight(start_c, goal_c)
print(f"  Direct LoS from {start_c} → {goal_c}: {los}")


# ---------------------------------------------------------------------------
# 4. Serialisation round-trip
# ---------------------------------------------------------------------------
separator("4. Serialisation – NPZ & JSON")

import tempfile, os, numpy as np
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    npz_path = Path(tmpdir) / "cuboid.npz"
    json_path = Path(tmpdir) / "maze.json"

    MapIO.save(grid_c, npz_path)
    loaded_c = MapIO.load(npz_path)
    print(f"  NPZ  save/load OK : data match = {(grid_c.data == loaded_c.data).all()}")

    MapIO.save(grid_m, json_path)
    loaded_m = MapIO.load(json_path)
    print(f"  JSON save/load OK : data match = {(grid_m.data == loaded_m.data).all()}")


# ---------------------------------------------------------------------------
# 5. Dynamic obstacles
# ---------------------------------------------------------------------------
separator("5. Dynamic Obstacles (Phase 3 stub)")

g = Grid3D(size=(20, 20, 20))
reg = DynamicObstacleRegistry(g)

# Subscribe a planner stub that would trigger replanning
def replan_trigger(oid, action, cells):
    print(f"    [Planner hook] obstacle '{oid}' {action} at cells: "
          f"{sorted(cells)[:3]}{'…' if len(cells) > 3 else ''}")

reg.on_change.append(replan_trigger)

# Simulate a drone entering the scene
reg.add("drone_2", {(10, 10, 5), (10, 11, 5)})

# Drone moves one step
reg.move("drone_2", {(10, 12, 5), (10, 13, 5)})

# Drone leaves
reg.remove("drone_2")
print(f"\n  Registry after removal: {reg}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
separator("Done — Phase 1 environment module is ready")
print("""
  Modules created:
    environment/
      __init__.py           – public API
      grid3d.py             – Grid3D, CellState, GridMetadata
      map_generator.py      – MapGenerator (cuboid / dense_mixed / maze)
      map_io.py             – MapIO  (NPZ + JSON serialisation)
      dynamic_obstacles.py  – DynamicObstacleRegistry

  Next steps:
    Phase 2 → Implement A* on Grid3D (get_neighbors + has_line_of_sight ready)
    Phase 3 → Implement Enhanced A* (Theta* LoS + D* Lite replanning hooks)
    Phase 4 → Visualise with Matplotlib / Plotly / Open3D
""")
