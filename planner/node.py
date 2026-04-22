"""
planner/node.py
===============
SearchNode — the fundamental unit of the A* search frontier.

Design notes
------------
* ``__lt__`` is the only comparison needed by ``heapq``; it compares ``f``
  first, then uses a monotone counter (``seq``) to break ties without ever
  comparing positions.  This matters because tuples like (x, y, z) compare
  element-by-element and can raise TypeErrors in Python 3 if the float
  comparison is equal.

* The ``parent`` pointer is a plain object reference (not an ID).  This is
  intentional: path reconstruction is a single O(depth) traversal with no
  dict lookup.  For D* Lite the pointer will be replaced by a bi-directional
  link structure, but that change lives entirely inside ``astar.py``, not here.

* ``g``, ``h``, and ``f`` are all stored as ``float`` so diagonal and
  volumetric costs (√2, √3) are represented exactly.

* The class is a plain Python class (not a dataclass) so that ``__lt__``
  can be hand-tuned and ``__hash__`` / ``__eq__`` can be deliberately
  excluded — we never want nodes in sets; only positions are hashed.
"""

from __future__ import annotations

from typing import Optional, Tuple

Coord3D = Tuple[int, int, int]


class SearchNode:
    """
    One node in the A* search tree.

    Parameters
    ----------
    pos :
        (x, y, z) grid coordinate this node represents.
    g :
        Exact cost from start to this node.
    h :
        Heuristic estimate of cost from this node to goal.
    parent :
        The node from which this node was reached, or ``None`` for the
        start node.
    seq :
        Monotone insertion counter — used as a tiebreaker in the heap so
        that nodes with equal ``f`` values are expanded in FIFO order
        (first inserted → first expanded).  Set by :class:`AStarPlanner`.
    """

    __slots__ = ("pos", "g", "h", "f", "parent", "seq")

    def __init__(
        self,
        pos: Coord3D,
        g: float,
        h: float,
        parent: Optional["SearchNode"] = None,
        seq: int = 0,
    ) -> None:
        self.pos: Coord3D = pos
        self.g: float = g
        self.h: float = h
        self.f: float = g + h
        self.parent: Optional[SearchNode] = parent
        self.seq: int = seq

    # ------------------------------------------------------------------
    # Heap ordering
    # ------------------------------------------------------------------

    def __lt__(self, other: "SearchNode") -> bool:
        """Primary sort by f; break ties in favour of lower h (greedy nudge),
        then by insertion order (FIFO for stable, deterministic traversal)."""
        if self.f != other.f:
            return self.f < other.f
        if self.h != other.h:
            return self.h < other.h
        return self.seq < other.seq

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SearchNode(pos={self.pos}, g={self.g:.3f}, "
            f"h={self.h:.3f}, f={self.f:.3f})"
        )
