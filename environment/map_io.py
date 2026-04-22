"""
map_io.py
=========
Serialisation and deserialisation for Grid3D.

Formats supported
-----------------
* ``.npz`` (NumPy compressed)  — fast, compact, preferred for large grids
* ``.json``                    — human-readable, useful for small grids or CI

Both formats embed the GridMetadata so a loaded grid is self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from .grid3d import Grid3D, GridMetadata

PathLike = Union[str, Path]


class MapIO:
    """
    Static-method collection for saving and loading Grid3D environments.

    Usage
    -----
    >>> MapIO.save_npz(grid, "maps/my_env.npz")
    >>> grid2 = MapIO.load_npz("maps/my_env.npz")

    >>> MapIO.save_json(grid, "maps/my_env.json")
    >>> grid3 = MapIO.load_json("maps/my_env.json")
    """

    # ------------------------------------------------------------------
    # NPZ  –  preferred for large grids
    # ------------------------------------------------------------------

    @staticmethod
    def save_npz(grid: Grid3D, path: PathLike) -> None:
        """
        Save *grid* to a compressed NumPy ``.npz`` archive.

        Stored arrays
        -------------
        * ``data``     – uint8 voxel array
        * ``meta``     – 1-element object array containing a JSON-encoded
                         metadata dict (avoids introducing a custom dtype)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta_json = json.dumps({
            "name": grid.metadata.name,
            "env_type": grid.metadata.env_type,
            "seed": grid.metadata.seed,
            "extra": grid.metadata.extra,
        })
        meta_arr = np.array([meta_json], dtype=object)

        np.savez_compressed(path, data=grid._data, meta=meta_arr)

    @staticmethod
    def load_npz(path: PathLike) -> Grid3D:
        """Load a Grid3D from a ``.npz`` archive previously saved by :meth:`save_npz`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")

        archive = np.load(path, allow_pickle=True)
        data: np.ndarray = archive["data"]
        meta_dict: dict = json.loads(str(archive["meta"][0]))

        metadata = GridMetadata(
            name=meta_dict.get("name", "unnamed"),
            env_type=meta_dict.get("env_type", "unknown"),
            seed=meta_dict.get("seed"),
            extra=meta_dict.get("extra", {}),
        )
        grid = Grid3D(size=data.shape, metadata=metadata)
        grid._data[:] = data
        return grid

    # ------------------------------------------------------------------
    # JSON  –  human-readable, good for debugging / small grids
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(grid: Grid3D, path: PathLike, *, indent: int = None) -> None:
        """
        Save *grid* to a JSON file.

        .. warning::
            JSON serialisation flattens the 3D array to a nested list, which
            is much slower and ~4–10× larger than ``.npz``.  Use only for
            small grids (≤ 20³) or debugging.

        Parameters
        ----------
        indent :
            JSON indent level.  ``None`` for compact (default), ``2`` for pretty.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": {
                "name": grid.metadata.name,
                "env_type": grid.metadata.env_type,
                "seed": grid.metadata.seed,
                "extra": grid.metadata.extra,
                "size": list(grid.size),
            },
            "data": grid._data.tolist(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent)

    @staticmethod
    def load_json(path: PathLike) -> Grid3D:
        """Load a Grid3D from a JSON file previously saved by :meth:`save_json`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        meta_dict = payload["metadata"]
        metadata = GridMetadata(
            name=meta_dict.get("name", "unnamed"),
            env_type=meta_dict.get("env_type", "unknown"),
            seed=meta_dict.get("seed"),
            extra=meta_dict.get("extra", {}),
        )
        data = np.array(payload["data"], dtype=np.uint8)
        grid = Grid3D(size=tuple(meta_dict["size"]), metadata=metadata)
        grid._data[:] = data
        return grid

    # ------------------------------------------------------------------
    # Convenience round-trip helper
    # ------------------------------------------------------------------

    @staticmethod
    def save(grid: Grid3D, path: PathLike, **kwargs) -> None:
        """
        Auto-dispatch to :meth:`save_npz` or :meth:`save_json` based on file
        extension.  Unknown extensions default to ``.npz``.
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            MapIO.save_json(grid, path, **kwargs)
        else:
            MapIO.save_npz(grid, path)

    @staticmethod
    def load(path: PathLike) -> Grid3D:
        """
        Auto-dispatch to :meth:`load_npz` or :meth:`load_json` based on file
        extension.
        """
        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return MapIO.load_json(path)
        return MapIO.load_npz(path)
