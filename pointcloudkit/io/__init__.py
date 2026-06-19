# Copyright (C) 2026 Mindkosh technologies private limited

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pointcloudkit.point_cloud import PointCloud

# Populated lazily on first call to avoid import overhead for unused formats.
_READERS: dict = {}
_WRITERS: dict = {}


def _ensure_loaded() -> None:
    if _READERS:
        return
    from pointcloudkit.io import pcd, ply, las
    from pointcloudkit.io import bin as bin_

    _READERS['.pcd'] = pcd.PCDFile.read
    _READERS['.ply'] = ply.PLYFile.read
    _READERS['.las'] = las.LASFile.read
    _READERS['.bin'] = bin_.BINFile.read

    _WRITERS['.pcd'] = pcd.PCDFile.write
    _WRITERS['.ply'] = ply.PLYFile.write
    _WRITERS['.las'] = las.LASFile.write
    # .bin is read-only; users convert via .pcd or .ply


def read(path: str) -> PointCloud:
    """Read a point cloud from *path*. Format is inferred from the file suffix.

    Supported: .pcd, .ply, .las, .bin
    """
    _ensure_loaded()
    ext    = Path(path).suffix.lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported format for reading: {ext!r}. "
            f"Supported extensions: {sorted(_READERS)}"
        )
    return reader(path)


def write(pc: PointCloud, path: str, binary: bool = True) -> None:
    """Write *pc* to *path*. Format is inferred from the file suffix.

    Supported: .pcd, .ply, .las
    """
    _ensure_loaded()
    ext    = Path(path).suffix.lower()
    writer = _WRITERS.get(ext)
    if writer is None:
        raise ValueError(
            f"Unsupported format for writing: {ext!r}. "
            f"Supported extensions: {sorted(_WRITERS)}"
        )
    writer(pc, path, binary=binary)
