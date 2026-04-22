# UAV Path Planning — IoD Minor Project
## Team Documentation & Getting Started Guide

---

## 1. Project Goal

We are building an **Enhanced A\* path planner** for UAV navigation in a 3D grid environment (Internet of Drones).

The project is split into phases:

| Phase | What we built | Status |
|---|---|---|
| Phase 1 | 3D Grid Environment | ✅ Done |
| Phase 2 | Standard A\* (baseline) | ✅ Done |
| Enhancement 1 | Theta\* — Any-Angle Movement | ✅ Done |
| Enhancement 2 | D\* Lite — Dynamic Replanning | 🔜 Next |
| Enhancement 3 | Beam Pruning — Memory Limit | 🔜 Next |

---

## 2. Setup — Run This First

```bash
# Clone the repo
git clone https://github.com/Jaskirat-s7/MINOR-PROJECT.git
cd MINOR-PROJECT

# Install dependencies (only 2)
pip install numpy matplotlib

# Switch to the enhanced branch
git checkout enhanced-astar
```

---

## 3. Project File Map — What Every File Does

```
MINOR-PROJECT/
│
├── environment/                 ← PHASE 1: The 3D world the UAV flies in
│   ├── grid3d.py               ← THE GRID: 50×50×50 voxel space
│   ├── map_generator.py        ← Creates 3 types of maps
│   ├── map_io.py               ← Save/load maps to disk
│   └── dynamic_obstacles.py   ← Moving obstacles (used in D* Lite later)
│
├── planner/                     ← PHASE 2 + ENHANCEMENT 1: The algorithms
│   ├── node.py                 ← One search node (position + cost + parent)
│   ├── heuristics.py           ← 4 distance functions (euclidean, octile, etc.)
│   ├── astar.py                ← Standard A* planner
│   └── theta_star.py          ← Enhanced A* (Theta*) — any-angle movement
│
├── demo.py                      ← Quick Phase 1 environment demo
├── demo_planner.py              ← A* full evaluation → saves fig1–fig5
├── demo_enhanced.py             ← Theta* full evaluation → saves fig8–fig12
├── demo_theta.py                ← A* vs Theta* side-by-side comparison
│
├── test_environment.py          ← 35 tests for Phase 1
├── test_planner.py              ← 34 tests for A*
├── test_theta.py                ← 20 tests for Theta*
│
└── results/                     ← All generated figures (auto-created)
    ├── fig1_env_comparison.png          ┐
    ├── fig2_heuristic_compare.png       │  A* figures
    ├── fig3_path_projections.png        │  (from demo_planner.py)
    ├── fig4_weighted_astar.png          │
    ├── fig5_heuristic_heatmap.png       ┘
    ├── fig6_astar_vs_theta_bars.png     ┐  Comparison figures
    ├── fig7_path_smoothness.png         ┘  (from demo_theta.py)
    ├── fig8_theta_env_comparison.png    ┐
    ├── fig9_theta_heuristic_compare.png │  Theta* figures
    ├── fig10_theta_path_projections.png │  (from demo_enhanced.py)
    ├── fig11_theta_weighted.png         │
    └── fig12_theta_heatmap.png          ┘
```

---

## 4. Understanding the Grid (`environment/grid3d.py`)

The world is a **50×50×50 cube** where each cell is:
- `0` = FREE (UAV can fly here)
- `1` = OBSTACLE (wall/building)
- `2` = DYNAMIC (moving obstacle, reserved for later)

The UAV can move to any of **26 neighbours** from any cell (face + edge + corner).

```python
# How to use it in code:
from environment import Grid3D, MapGenerator

gen  = MapGenerator(size=(50, 50, 50), seed=42)
grid, start, goal = gen.generate_cuboid()   # sparse obstacles
grid, start, goal = gen.generate_dense_mixed()  # dense obstacles
grid, start, goal = gen.generate_maze()     # maze-like walls
```

**3 environment types:**

| Type | Obstacle Density | Difficulty |
|---|---|---|
| Cuboid | ~2% | Easy |
| Dense Mixed | ~20% | Medium |
| Maze | ~31% | Hard |

---

## 5. Understanding A\* (`planner/astar.py`)

Standard A\* finds the shortest obstacle-free path from start to goal.

```python
from planner import AStarPlanner

planner = AStarPlanner(grid)
result  = planner.solve(start, goal)

print(result.path_length)      # total Euclidean distance
print(len(result.path))        # number of waypoints
print(result.nodes_expanded)   # search effort
print(result.elapsed_seconds)  # time taken
```

**4 heuristics available:**

| Name | Admissible? | Best for |
|---|---|---|
| `euclidean` | ✅ Yes | Optimal paths |
| `octile` | ✅ Yes | Fewest expansions |
| `chebyshev` | ✅ Yes | Simple/fast |
| `manhattan` | ❌ No | Risky but fast |

```python
from planner.heuristics import HEURISTICS
result = planner.solve(start, goal, heuristic=HEURISTICS["octile"])
```

---

## 6. Understanding Theta\* (`planner/theta_star.py`)

Theta\* is A\* with **one extra rule**:

> Before connecting a successor to its immediate parent, check if the  
> **grandparent** has a clear line-of-sight to the successor.  
> If YES → skip the parent, connect directly (any-angle shortcut).  
> If NO → connect normally (standard A\* behaviour).

```python
from planner import ThetaStarPlanner

planner = ThetaStarPlanner(grid)   # drop-in replacement for AStarPlanner
result  = planner.solve(start, goal)

print(result.planner_name)   # "Theta*"
print(len(result.path))      # far fewer waypoints than A*
```

**Results:**
| Environment | A\* Waypoints | Theta\* Waypoints | Saved |
|---|---|---|---|
| Cuboid | 36 | 3 | −91.7% |
| Dense Mixed | 50 | 9 | −82.0% |
| Maze | 67 | 11 | −83.6% |

---

## 7. Commands to Run — In Order

### ✅ Step 1 — Verify everything works
```bash
python3 test_environment.py   # should print: 35/35 tests passed
python3 test_planner.py       # should print: 34/34 tests passed
python3 test_theta.py         # should print: 20/20 tests passed
```

### 📊 Step 2 — Generate A\* baseline results
```bash
python3 demo_planner.py
```
Saves 5 figures to `results/` and prints 2 tables in the terminal.

### 📊 Step 3 — Generate Theta\* results
```bash
python3 demo_enhanced.py
```
Saves 5 more figures to `results/` using the exact same format as A\*.

### 📊 Step 4 — Generate A\* vs Theta\* direct comparison
```bash
python3 demo_theta.py
```
Prints the trade-off table and saves fig6 + fig7.

### 🖼️ Step 5 — Open all figures
```bash
open results/
```

---

## 8. Figure Guide — What Each Figure Shows

### A\* Figures (from `demo_planner.py`)

| Figure | Shows |
|---|---|
| `fig1_env_comparison.png` | 4 bar charts: path length, nodes expanded, time, detour factor — for all 3 environments |
| `fig2_heuristic_compare.png` | How each of 4 heuristics performs (nodes + time) per environment |
| `fig3_path_projections.png` | The actual A\* path drawn on XY, XZ, YZ planes for all 3 environments |
| `fig4_weighted_astar.png` | How weight parameter trades optimality for speed |
| `fig5_heuristic_heatmap.png` | Heat map grid: heuristic × environment (darker = worse) |

### Theta\* Figures (from `demo_enhanced.py`)

| Figure | Shows |
|---|---|
| `fig8_theta_env_comparison.png` | Same as fig1 but for Theta\* — compare bar heights directly |
| `fig9_theta_heuristic_compare.png` | Same as fig2 but for Theta\* |
| `fig10_theta_path_projections.png` | Same as fig3 but Theta\* paths — **visually fewer dots = fewer waypoints** |
| `fig11_theta_weighted.png` | Same as fig4 but for weighted Theta\* |
| `fig12_theta_heatmap.png` | Same as fig5 but for Theta\* |

### Comparison Figures (from `demo_theta.py`)

| Figure | Shows |
|---|---|
| `fig6_astar_vs_theta_bars.png` | 4 grouped bars: A\* vs Theta\* for all metrics, all environments |
| `fig7_path_smoothness.png` | Side-by-side paths: A\* left, Theta\* right — smoothness is obvious |

---

## 9. How to Explain to Professor

### Showing A\* first:
> *"fig1 shows A\* works correctly. Nodes expanded grows 16× from cuboid to maze — same grid size, different topology. fig3 shows the actual path — each dot is a waypoint, A\* produces 36–67 waypoints. fig4 shows weighted A\* can trade 0–5% sub-optimality for 5–30× speed-up."*

### Showing Theta\* second:
> *"fig8 shows the same metrics for Theta\*. Path length is 4–7% shorter. fig10 shows the same environments — notice far fewer dots on the paths. Theta\* gives 82–92% fewer waypoints. fig6 puts them side by side for direct comparison."*

### Answering the trade-off question:
> *"Theta\* adds one Bresenham 3D line-of-sight check per neighbour per expansion. On 32,000 expansions this adds millions of voxel checks → +55–178% more time. But: planning happens once offline. A UAV benefits from fewer waypoints through fewer motor commands, less energy, and smoother turns."*

---

## 10. Git Branch Structure

```
baseline-astar-v1  ← TAG: permanent snapshot, always recoverable
baseline-astar     ← BRANCH: frozen pure A* — demo this for "before"
enhanced-astar     ← BRANCH: current active — all enhancements here
main               ← original base commit
```

**Switch to baseline to show pure A\*:**
```bash
git checkout baseline-astar
python3 demo_planner.py
```

**Switch back to enhanced:**
```bash
git checkout enhanced-astar
python3 demo_enhanced.py
```

---

## 11. Key Numbers to Remember

| Fact | Value |
|---|---|
| Grid size | 50 × 50 × 50 voxels |
| Connectivity | 26-directional |
| Reproducibility seed | 42 |
| A\* waypoints (cuboid) | 36 |
| Theta\* waypoints (cuboid) | **3** (−91.7%) |
| A\* waypoints (maze) | 67 |
| Theta\* waypoints (maze) | **11** (−83.6%) |
| Total tests passing | **89/89** (35 + 34 + 20) |
