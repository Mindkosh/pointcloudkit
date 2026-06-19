# pointcloudkit

Lightweight point cloud I/O and processing library. Pure numpy — no open3d dependency.

## Requirements

```
numpy
laspy   # LAS format only
```

Python ≥ 3.9.

## Supported Formats

| Extension | Read | Write |
|-----------|------|-------|
| `.pcd`    | ✓    | ✓     |
| `.ply`    | ✓    | ✓     |
| `.las`    | ✓    | ✓     |
| `.bin`    | ✓    | —     |

PCD and PLY support both ASCII and binary modes. Binary writes are chunked and memory-efficient regardless of cloud size.

---

## Operations

### Reading and writing

```python
from pointcloudkit import PointCloud

pc = PointCloud.read("scan.pcd")
pc.write("scan.ply")                  # binary by default
pc.write("scan_ascii.pcd", binary=False)
```

### Format conversion

```python
from pointcloudkit import convert

convert("scan.las", "scan.pcd")
convert("scan.bin", "scan.ply")
convert("scan.pcd", "scan_ascii.ply", binary=False)
```

### Center

Subtracts the centroid from all positions in-place.

```python
pc = PointCloud.read("scan.pcd")
pc.center()
pc.write("centered.pcd")
```

### Make upright

Rotates the cloud so its dominant plane is parallel to XY (Z axis points up).
Uses PCA: the eigenvector with the smallest eigenvalue is the plane normal,
which is then aligned to `[0, 0, 1]` via the Rodrigues formula.

```python
pc = PointCloud.read("scan.pcd")
pc.make_upright()
pc.write("upright.pcd")
```

Operations return `self` and can be chained:

```python
PointCloud.read("scan.las").center().make_upright().write("processed.pcd")
```

---

## PointCloud data model

```python
pc.position   # (N, 3) float64  — always present
pc.intensity  # (N,)   float32  — empty array if absent
pc.rgb        # (N, 3) uint8    — empty array if absent
pc.extra      # dict: name → (N,) ndarray for any additional fields
```

Constructing manually:

```python
import numpy as np
from pointcloudkit import PointCloud

pc = PointCloud(
    position=np.random.rand(1000, 3),
    intensity=np.random.rand(1000).astype(np.float32),
    rgb=np.random.randint(0, 255, (1000, 3), dtype=np.uint8),
)
print(len(pc))  # 1000
```

### Reading BIN files

Velodyne BIN files default to KITTI layout (x, y, z, intensity). Pass `num_attrs`
to override the number of float32 values per point.

```python
from pointcloudkit.io.bin import BINFile

pc = BINFile.read("velodyne.bin")              # 4 attrs (default)
pc = BINFile.read("custom.bin", num_attrs=6)   # 6 attrs; first 3 → position
```

### Accessing format-specific classes directly

```python
from pointcloudkit.io.pcd import PCDFile
from pointcloudkit.io.ply import PLYFile
from pointcloudkit.io.las import LASFile

pc = PCDFile.read("scan.pcd")
PLYFile.write(pc, "scan.ply", binary=True)
```
