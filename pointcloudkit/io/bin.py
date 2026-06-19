# Copyright (C) 2026 Mindkosh technologies private limited

import numpy as np

from pointcloudkit.point_cloud import PointCloud


class BINFile:
    """Read Velodyne binary (.bin) point cloud files (read-only).

    BIN files store points as a flat sequence of float32 values.
    The default layout is KITTI format: x, y, z, intensity (4 values/point).
    """

    @staticmethod
    def read(path: str, num_attrs: int = 4) -> PointCloud:
        """Read a Velodyne BIN file and return a PointCloud.

        :param path: path to the .bin file
        :param num_attrs: number of float32 values per point (default 4 = x, y, z, intensity)
        """
        data = np.fromfile(path, dtype=np.float32).reshape(-1, num_attrs)

        position = data[:, :3].astype(np.float64)
        pc       = PointCloud(position=position)

        if num_attrs > 3:
            pc.intensity = data[:, 3].copy()  # already float32

        return pc
