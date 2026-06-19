# Copyright (C) 2026 Mindkosh technologies private limited

from pointcloudkit.point_cloud import PointCloud


def convert(src: str, dst: str, binary: bool = True) -> None:
    """Convert a point cloud file from one format to another.

    Example::

        convert("scan.las", "scan.pcd")
        convert("scan.bin", "scan.ply", binary=False)
    """
    PointCloud.read(src).write(dst, binary=binary)


__all__ = ['PointCloud', 'convert']
