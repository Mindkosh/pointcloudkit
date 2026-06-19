# Copyright (C) 2026 Mindkosh technologies private limited

import numpy as np
import laspy

from pointcloudkit.point_cloud import PointCloud


def _has_rgb(las) -> bool:
    return all(hasattr(las, c) for c in ('red', 'green', 'blue'))


def _has_intensity(las) -> bool:
    return hasattr(las, 'intensity')


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert LAS uint16 colour channel (0-65535) to uint8 (0-255).

    LAS files produced by different scanners may store 8-bit values in the
    low byte *or* the high byte of a uint16 field. Values that fit in a byte
    (≤ 255) are kept as-is; values that don't are right-shifted by 8.
    """
    arr = np.asarray(arr, dtype=np.uint32)
    return np.where(arr > 255, arr >> 8, arr).astype(np.uint8)


def _should_use_double(position: np.ndarray) -> bool:
    """Return True if any coordinate magnitude exceeds float32 precision."""
    return float(np.abs(position).max()) > 1e6


class LASFile:
    """Read and write LAS point cloud files."""

    @staticmethod
    def read(path: str) -> PointCloud:
        """Read a LAS file and return a PointCloud."""
        las = laspy.read(path)

        position = np.column_stack([
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ])
        pc = PointCloud(position=position)

        if _has_intensity(las):
            pc.intensity = np.asarray(las.intensity, dtype=np.float32)

        if _has_rgb(las):
            r = _normalize_to_uint8(np.asarray(las.red))
            g = _normalize_to_uint8(np.asarray(las.green))
            b = _normalize_to_uint8(np.asarray(las.blue))
            pc.rgb = np.column_stack([r, g, b]).astype(np.uint8)

        return pc

    @staticmethod
    def write(pc: PointCloud, path: str, binary: bool = True) -> None:
        """Write a PointCloud to a LAS file.

        LAS is always a binary format; the *binary* parameter is accepted for
        API consistency but has no effect.

        Point format 2 is chosen when RGB data is present; format 0 otherwise.
        Intensity (float32) is scaled to uint16 via clipping to [0, 65535].
        RGB (uint8) is scaled to uint16 by shifting left 8 bits.
        """
        n = len(pc)
        has_rgb_data       = len(pc.rgb)       == n
        has_intensity_data = len(pc.intensity) == n

        point_format = 2 if has_rgb_data else 0
        header = laspy.LasHeader(point_format=point_format, version="1.2")
        las    = laspy.LasData(header=header)

        las.x = pc.position[:, 0]
        las.y = pc.position[:, 1]
        las.z = pc.position[:, 2]

        if has_intensity_data:
            # Intensity is stored as uint16 in LAS; clip to valid range
            las.intensity = np.clip(pc.intensity, 0, 65535).astype(np.uint16)

        if has_rgb_data:
            # Scale uint8 (0-255) → uint16 (0-65535) by shifting left 8 bits
            las.red   = pc.rgb[:, 0].astype(np.uint16) << 8
            las.green = pc.rgb[:, 1].astype(np.uint16) << 8
            las.blue  = pc.rgb[:, 2].astype(np.uint16) << 8

        las.write(path)
