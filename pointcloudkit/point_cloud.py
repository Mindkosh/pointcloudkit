# Copyright (C) 2026 Mindkosh technologies private limited

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class PointCloud:
    """Central data model for a 3-D point cloud.

    Attributes
    ----------
    position  : (N, 3) float64 – required
    intensity : (N,)   float32 – empty array if absent
    rgb       : (N, 3) uint8   – empty array if absent
    extra     : name → (N,) ndarray for any additional per-point fields
    """

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.position  = position  if position  is not None else np.empty((0, 3), dtype=np.float64)
        self.intensity = intensity if intensity is not None else np.empty(0,      dtype=np.float32)
        self.rgb       = rgb       if rgb       is not None else np.empty((0, 3), dtype=np.uint8)
        self.extra: Dict[str, np.ndarray] = extra if extra is not None else {}

    def __len__(self) -> int:
        return len(self.position)

    @classmethod
    def read(cls, path: str, drop_nan: bool = True) -> PointCloud:
        """Load a point cloud from *path*; format is inferred from the file suffix.

        By default, points with NaN/Inf x/y/z are removed and NaN in intensity
        or extra float fields is zeroed out.  Pass ``drop_nan=False`` to keep
        raw values as-is.
        """
        from pointcloudkit.io import read as _read
        pc = _read(path)
        if drop_nan:
            pc.drop_nan()
        return pc

    def write(self, path: str, binary: bool = True) -> None:
        """Save this cloud to *path*; format is inferred from the file suffix."""
        from pointcloudkit.io import write as _write
        _write(self, path, binary=binary)

    def drop_nan(self) -> PointCloud:
        """Remove points where any of x, y, z is NaN or Inf; zero out NaN in intensity and extra float fields."""
        n = len(self.position)
        valid = np.isfinite(self.position).all(axis=1)

        self.position = self.position[valid]

        if len(self.intensity) == n:
            self.intensity = self.intensity[valid]
            nan_mask = ~np.isfinite(self.intensity)
            self.intensity[nan_mask] = 0.0

        if len(self.rgb) == n:
            self.rgb = self.rgb[valid]
            # rgb is uint8; NaN cannot occur in integer arrays

        for key, arr in self.extra.items():
            if len(arr) == n:
                self.extra[key] = arr[valid]
                if np.issubdtype(arr.dtype, np.floating):
                    nan_mask = ~np.isfinite(self.extra[key])
                    self.extra[key][nan_mask] = 0

        return self

    def center(self) -> PointCloud:
        """Subtract the centroid from all positions in-place. Returns self."""
        self.position -= np.mean(self.position, axis=0)
        return self

    def make_upright(self) -> PointCloud:
        """Align the cloud so its dominant plane is parallel to XY (Z points up).

        Uses PCA: the eigenvector corresponding to the *smallest* eigenvalue of
        the covariance matrix is the plane normal. That normal is rotated onto
        [0, 0, 1] using the Rodrigues rotation formula. The cloud is also
        centered as a side-effect.

        Returns self.
        """
        centroid = self.position.mean(axis=0)
        Pc = self.position - centroid                      # centered copy

        C = (Pc.T @ Pc) / len(Pc)                         # 3×3 covariance
        _, vecs = np.linalg.eigh(C)
        normal = vecs[:, 0]                                # smallest eigenvalue → plane normal

        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, z)
        s = np.linalg.norm(v)
        c = np.dot(normal, z)

        if s == 0.0:
            # normal already aligned with Z (or anti-aligned); no rotation needed
            R = np.eye(3)
        else:
            # Rodrigues / cross-product matrix formula
            vx = np.array([
                [ 0.0,  -v[2],  v[1]],
                [ v[2],  0.0,  -v[0]],
                [-v[1],  v[0],  0.0],
            ])
            R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))

        self.position = (R @ Pc.T).T   # Pc is already centered
        return self
