# UAV Path Planning — Internet of Drones (IoD) Minor Project

A modular, extensible **3D UAV path planning system** built on a 50×50×50 voxel grid.

## Project Structure

```
minor/
├── environment/              # Phase 1 — 3D Grid Environment
│   ├── grid3d.py             # Grid3D class: 50³ voxel grid, 26-connectivity, LoS checks
│   ├── map_generator.py      # MapGenerator: cuboid / dense-mixed / maze environments
│   ├── map_io.py             # NPZ + JSON serialization
│   └── dynamic_obstacles.py  # DynamicObstacleRegistry with callback hooks
├── planner/                  # Phase 2 — Standard A* Planner
│   ├── node.py               # SearchNode: heap-compatible, slot-based
│   ├── heuristics.py         # euclidean / manhattan / chebyshev / octile
│   └── astar.py              # AStarPlanner + PlanResult (with Phase 3 hooks)
├── demo.py                   # Phase 1 environment demo
├── demo_planner.py           # Phase 2 baseline A* evaluation (5 figures)
├── test_environment.py       # 35 environment tests
├── test_planner.py           # 34 planner tests
└── results/                  # Generated figures (gitignored PNGs)
```

## Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | 3D Grid Environment | ✅ Complete (35/35 tests) |
| Phase 2 | Standard A* Planner | ✅ Complete (34/34 tests) |
| Phase 3 | Enhanced A* (Theta* + D* Lite + Beam Pruning) | 🔜 In Progress |

## Quick Start

```bash
# Install dependency
pip install numpy matplotlib

# Run tests
python3 test_environment.py   # 35/35 pass
python3 test_planner.py       # 34/34 pass

# Run baseline A* evaluation
python3 demo_planner.py       # saves 5 figures to results/
```

## Algorithm — Standard A*

- **Grid**: 50×50×50 voxel space, 26-connectivity (face + edge + corner neighbours)
- **Heuristics**: Euclidean, Octile, Chebyshev, Manhattan
- **Metrics tracked**: path length, waypoints, nodes expanded, computation time, detour factor
- **Weighted A\***: configurable weight parameter (w=1.0 → optimal, w>1.0 → faster)

## Baseline Results (seed=42, 50³ grid)

| Environment | Density | Path Length | Waypoints | Nodes Expanded | Time |
|---|---|---|---|---|---|
| Cuboid | 2% | 49.112 | 36 | 1,943 | ~70ms |
| Dense Mixed | 20% | 66.676 | 50 | 22,444 | ~754ms |
| Maze | 31% | 89.050 | 67 | 32,531 | ~1049ms |

## Requirements

- Python 3.8+
- numpy
- matplotlib (for demo figures)
