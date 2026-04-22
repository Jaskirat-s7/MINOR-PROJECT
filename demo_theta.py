"""
demo_theta.py
=============
A* vs Theta* (Any-Angle) — head-to-head comparison.

Produces:
  results/fig6_astar_vs_theta_bars.png
  results/fig7_path_smoothness.png

Run:
    python3 demo_theta.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment import MapGenerator
from planner import AStarPlanner, ThetaStarPlanner

# ── style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor":   "#3a3f5c", "axes.labelcolor": "#c8cce8",
    "axes.titlecolor":  "#e8eaf6", "xtick.color":     "#8890b8",
    "ytick.color":      "#8890b8", "grid.color":       "#2a2f4a",
    "grid.linewidth":   0.6,       "text.color":       "#c8cce8",
    "legend.facecolor": "#1e2235", "legend.edgecolor": "#3a3f5c",
    "font.family":      "monospace", "font.size":       9,
})

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEED    = 42
SIZE    = (50, 50, 50)
C_ASTAR = "#00b4d8"
C_THETA = "#ffd166"
C_START = "#00e5ff"
C_GOAL  = "#ff6b6b"
C_OBS   = "#2a2f4a"


def sep(t): print(f"\n{'═'*66}\n  {t}\n{'═'*66}")


# ═══════════════════════════════════════════════════════════════════════
# 1. Run both planners on all 3 environments
# ═══════════════════════════════════════════════════════════════════════
sep("A* vs Theta*  (seed=42, 50³)")

gen  = MapGenerator(size=SIZE, seed=SEED)
envs = {
    "Cuboid":      gen.generate_cuboid(num_obstacles=20),
    "Dense Mixed": gen.generate_dense_mixed(),
    "Maze":        gen.generate_maze(),
}

rows = []
for name, (grid, start, goal) in envs.items():
    ra = AStarPlanner(grid).solve(start, goal)
    rt = ThetaStarPlanner(grid).solve(start, goal)
    rows.append(dict(env=name, grid=grid, start=start, goal=goal, astar=ra, theta=rt))

# ═══════════════════════════════════════════════════════════════════════
# 2. Console table
# ═══════════════════════════════════════════════════════════════════════
sep("Results Table")
HDR = f"{'Environment':<14}  {'Planner':<8}  {'Waypts':>6}  {'PathLen':>8}  {'Expanded':>9}  {'Time(ms)':>9}"
print(f"\n  {HDR}\n  {'─'*len(HDR)}")
for row in rows:
    for r in (row["astar"], row["theta"]):
        print(f"  {row['env']:<14}  {r.planner_name:<8}  "
              f"{len(r.path):>6}  {r.path_length:>8.3f}  "
              f"{r.nodes_expanded:>9,}  {r.elapsed_seconds*1000:>9.2f}")
    print()

sep("Improvement Summary")
print(f"\n  {'Environment':<14}  {'WayptΔ':>7}  {'WayptΔ%':>8}  {'LenΔ%':>8}  {'TimeΔ%':>9}")
print(f"  {'─'*52}")
for row in rows:
    ra, rt = row["astar"], row["theta"]
    wd  = len(rt.path) - len(ra.path)
    wp  = 100.0 * wd / len(ra.path)
    lp  = 100.0 * (rt.path_length - ra.path_length) / ra.path_length
    tp  = 100.0 * (rt.elapsed_seconds - ra.elapsed_seconds) / ra.elapsed_seconds
    print(f"  {row['env']:<14}  {wd:>+7}  {wp:>+7.1f}%  {lp:>+7.2f}%  {tp:>+8.1f}%")

sep("Why Theta* Reduces Waypoints")
print("""
  A* must step grid-edge by grid-edge — one waypoint per voxel step.
  Even a perfectly straight 10-voxel corridor needs 11 waypoints.

  Theta* asks one extra question per neighbour:
      "Can my GRANDPARENT see this cell directly?"
      YES → skip me, connect straight through      ← any-angle
      NO  → connect normally                        ← standard A*

  Result: that 10-voxel corridor becomes 2 waypoints (start + end).
  Path length also improves because straight diagonals are more direct
  than grid zig-zags.
""")

# ═══════════════════════════════════════════════════════════════════════
# 3. Figure 6 — grouped bar comparison
# ═══════════════════════════════════════════════════════════════════════
sep("Generating Figure 6 — astar_vs_theta_bars.png")

metrics = [
    ("Waypoints",      lambda r: len(r.path),               True),
    ("Path Length",    lambda r: r.path_length,              False),
    ("Nodes Expanded", lambda r: r.nodes_expanded,           True),
    ("Time (ms)",      lambda r: r.elapsed_seconds * 1000,   False),
]

fig6, axes = plt.subplots(1, 4, figsize=(20, 5))
fig6.suptitle("A*  vs  Theta* — Head-to-Head Comparison  (seed=42, 50³)",
              fontsize=11, fontweight="bold", color="#e8eaf6", y=1.02)

x  = np.arange(3)
bw = 0.35
env_labels = [r["env"] for r in rows]

for ax, (metric, fn, annotate) in zip(axes, metrics):
    av = [fn(r["astar"]) for r in rows]
    tv = [fn(r["theta"]) for r in rows]

    b1 = ax.bar(x - bw/2, av, bw, color=C_ASTAR, label="A*",     edgecolor="#0f1117", zorder=3)
    b2 = ax.bar(x + bw/2, tv, bw, color=C_THETA, label="Theta*", edgecolor="#0f1117", zorder=3)

    ax.set_title(metric, fontsize=9, fontweight="bold", pad=8)
    ax.set_xticks(x); ax.set_xticklabels(env_labels, fontsize=8)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(fontsize=7)

    for bar, val in zip(list(b1)+list(b2), av+tv):
        lbl = f"{val:,.0f}" if val > 10 else f"{val:.2f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                lbl, ha="center", va="bottom", fontsize=7, color="#e8eaf6")

    if annotate:
        for xi, (a, t) in enumerate(zip(av, tv)):
            pct = 100*(t-a)/a if a else 0
            ax.text(xi, max(a,t)*1.12, f"{pct:+.0f}%",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color="#06d6a0" if pct < 0 else "#ef476f")

fig6.tight_layout()
fig6.savefig(RESULTS_DIR/"fig6_astar_vs_theta_bars.png", dpi=150,
             bbox_inches="tight", facecolor=fig6.get_facecolor())
plt.close(fig6)
print("  Saved → results/fig6_astar_vs_theta_bars.png")

# ═══════════════════════════════════════════════════════════════════════
# 4. Figure 7 — side-by-side path projections (XY plane)
# ═══════════════════════════════════════════════════════════════════════
sep("Generating Figure 7 — path_smoothness.png")

fig7, axes7 = plt.subplots(3, 2, figsize=(14, 16))
fig7.suptitle("Path Smoothness — A* (left)  vs  Theta* (right)  [XY projection]",
              fontsize=11, fontweight="bold", color="#e8eaf6", y=1.01)

rng = np.random.default_rng(SEED)

for ri, row in enumerate(rows):
    grid, start, goal = row["grid"], row["start"], row["goal"]
    ox, oy, _ = np.where(grid.data == 1)
    n = min(1500, len(ox))
    idx = rng.choice(len(ox), n, replace=False)
    obs_x, obs_y = ox[idx], oy[idx]

    for ci, (r, lbl) in enumerate([(row["astar"], "A*"), (row["theta"], "Theta*")]):
        ax = axes7[ri, ci]
        ax.scatter(obs_x, obs_y, s=1.5, color=C_OBS, alpha=0.5, zorder=1, rasterized=True)

        px = [p[0] for p in r.path]
        py = [p[1] for p in r.path]
        col = C_ASTAR if lbl == "A*" else C_THETA
        ax.plot(px, py, color=col, linewidth=1.6, zorder=3)
        ax.scatter(px, py, s=18, color=col, zorder=4, alpha=0.7)
        ax.scatter(start[0], start[1], s=100, color=C_START, marker="^", zorder=5)
        ax.scatter(goal[0],  goal[1],  s=100, color=C_GOAL,  marker="*", zorder=5)

        ax.set_xlim(0, grid.size[0]); ax.set_ylim(0, grid.size[1])
        ax.set_title(f"{row['env']}  |  {lbl}  |  {len(r.path)} waypoints  |  len={r.path_length:.1f}",
                     fontsize=8.5, fontweight="bold", color=col, pad=6)
        ax.set_xlabel("X", fontsize=8)
        if ci == 0: ax.set_ylabel("Y", fontsize=8)
        ax.xaxis.grid(True, lw=0.4, zorder=0); ax.yaxis.grid(True, lw=0.4, zorder=0)

        if lbl == "Theta*":
            reduction = 1.0 - len(row["theta"].path) / len(row["astar"].path)
            ax.text(0.97, 0.03, f"▼ {reduction:.0%} fewer waypoints",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="#06d6a0", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d27",
                              edgecolor="#06d6a0", alpha=0.85))

from matplotlib.lines import Line2D
fig7.legend(handles=[
    Line2D([0],[0], color=C_ASTAR, lw=2, label="A* path"),
    Line2D([0],[0], color=C_THETA, lw=2, label="Theta* path"),
    Line2D([0],[0], marker="^", color="none", markerfacecolor=C_START, ms=8, label="Start"),
    Line2D([0],[0], marker="*", color="none", markerfacecolor=C_GOAL,  ms=8, label="Goal"),
], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig7.tight_layout(rect=[0, 0.02, 1, 1])
fig7.savefig(RESULTS_DIR/"fig7_path_smoothness.png", dpi=150,
             bbox_inches="tight", facecolor=fig7.get_facecolor())
plt.close(fig7)
print("  Saved → results/fig7_path_smoothness.png")

sep("Enhancement 1 Complete — Theta* Any-Angle Movement")
print("""
  Algorithm change  (13 lines of logic):
    BEFORE:  parent = current
    AFTER:   if LoS(grandparent, successor):
                 parent = grandparent   ← any-angle
             else:
                 parent = current       ← standard A*

  Files:
    planner/astar.py       +_select_parent() hook
    planner/theta_star.py  ThetaStarPlanner  (new)
    planner/__init__.py    ThetaStarPlanner exported
    test_theta.py          20 tests
    demo_theta.py          this file
    results/fig6_astar_vs_theta_bars.png
    results/fig7_path_smoothness.png
""")
