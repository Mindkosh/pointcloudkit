# Copyright (C) 2026 Mindkosh technologies private limited

import numpy as np

from pointcloudkit.point_cloud import PointCloud

_CHUNK = 65_536


class PLYFile:
    """Read and write PLY (Polygon File Format) point cloud files."""

    _KNOWN_PROPS = {'x', 'y', 'z', 'intensity', 'red', 'green', 'blue'}

    # PLY type name → numpy dtype char (endian-neutral)
    _PLY_TO_NUMPY = {
        'char':    'i1', 'int8':    'i1',
        'uchar':   'u1', 'uint8':   'u1',
        'short':   'i2', 'int16':   'i2',
        'ushort':  'u2', 'uint16':  'u2',
        'int':     'i4', 'int32':   'i4',
        'uint':    'u4', 'uint32':  'u4',
        'float':   'f4', 'float32': 'f4',
        'double':  'f8', 'float64': 'f8',
    }

    # numpy (kind, itemsize) → PLY type string
    _NUMPY_TO_PLY = {
        ('f', 4): 'float',  ('f', 8): 'double',
        ('u', 1): 'uchar',  ('u', 2): 'ushort', ('u', 4): 'uint',  ('u', 8): 'uint',
        ('i', 1): 'char',   ('i', 2): 'short',  ('i', 4): 'int',   ('i', 8): 'int',
    }

    # ── Reader ─────────────────────────────────────────────────────────────

    @classmethod
    def read(cls, path: str) -> PointCloud:
        """Read a PLY file (ASCII or binary) and return a PointCloud."""
        with open(path, 'rb') as f:
            hdr   = cls._parse_header(f)
            fmt   = hdr['format']      # 'ascii' | 'binary_little_endian' | 'binary_big_endian'
            props = hdr['properties']  # list of (name, ply_type_str)
            n_pts = hdr['count']

            if fmt == 'ascii':
                raw = cls._read_ascii(f, props, n_pts)
            else:
                endian = '<' if 'little' in fmt else '>'
                raw = cls._read_binary(f, props, n_pts, endian)

        pc = PointCloud()

        if 'x' in raw:
            pc.position = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(np.float64)
        if 'intensity' in raw:
            pc.intensity = raw['intensity'].astype(np.float32)
        if 'red' in raw:
            pc.rgb = np.column_stack([raw['red'], raw['green'], raw['blue']]).astype(np.uint8)

        for name, arr in raw.items():
            if name not in cls._KNOWN_PROPS:
                pc.extra[name] = arr

        return pc

    @staticmethod
    def _parse_header(f) -> dict:
        """Parse PLY header; leaves f positioned at the first data byte."""
        magic = f.readline().decode('ascii', errors='ignore').strip()
        if magic != 'ply':
            raise ValueError("Not a PLY file (missing 'ply' magic line)")

        hdr       = {'format': 'ascii', 'count': 0, 'properties': []}
        in_vertex = False

        while True:
            line  = f.readline().decode('ascii', errors='ignore').strip()
            if not line or line.startswith('comment'):
                continue
            parts = line.split()
            kw    = parts[0]

            if kw == 'format':
                hdr['format'] = parts[1]
            elif kw == 'element':
                in_vertex = (parts[1] == 'vertex')
                if in_vertex:
                    hdr['count'] = int(parts[2])
            elif kw == 'property' and in_vertex:
                if parts[1] == 'list':
                    continue  # skip face index lists
                hdr['properties'].append((parts[2], parts[1]))
            elif kw == 'end_header':
                break

        return hdr

    @classmethod
    def _read_ascii(cls, f, props, n_pts) -> dict:
        bufs = {name: [] for name, _ in props}

        for _ in range(n_pts):
            raw  = f.readline()
            line = raw.decode('ascii', errors='ignore').strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            tokens = line.split()

            for i, (name, ply_type) in enumerate(props):
                nc = cls._PLY_TO_NUMPY[ply_type]
                bufs[name].append(float(tokens[i]) if 'f' in nc else int(float(tokens[i])))

        return {
            name: np.array(bufs[name], dtype=np.dtype(cls._PLY_TO_NUMPY[ply_type]))
            for name, ply_type in props
        }

    @classmethod
    def _read_binary(cls, f, props, n_pts, endian) -> dict:
        dt   = np.dtype([(name, endian + cls._PLY_TO_NUMPY[ply_type]) for name, ply_type in props])
        data = np.frombuffer(f.read(n_pts * dt.itemsize), dtype=dt)
        return {name: data[name].copy() for name, _ in props}

    # ── Writer ─────────────────────────────────────────────────────────────

    @classmethod
    def write(cls, pc: PointCloud, output_path: str, binary: bool = True) -> None:
        """Write a PointCloud to a PLY file.

        All non-empty fields are written: position (x, y, z), intensity, rgb
        (as separate red/green/blue uchar properties), and any 1-D arrays in
        pc.extra.

        Binary mode uses chunked writes so that peak extra memory stays at
        _CHUNK × bytes_per_point regardless of total cloud size.
        """
        n = len(pc)

        # Each entry: (ply_prop_name, ply_type_str, little-endian np.dtype, 1-D array)
        props_info = []

        xyz_ply = cls._NUMPY_TO_PLY.get((pc.position.dtype.kind, pc.position.dtype.itemsize), 'double')
        xyz_ndt = np.dtype('<' + cls._PLY_TO_NUMPY[xyz_ply])
        props_info.append(('x', xyz_ply, xyz_ndt, pc.position[:, 0]))
        props_info.append(('y', xyz_ply, xyz_ndt, pc.position[:, 1]))
        props_info.append(('z', xyz_ply, xyz_ndt, pc.position[:, 2]))

        if len(pc.intensity) == n:
            ply_t = cls._NUMPY_TO_PLY.get((pc.intensity.dtype.kind, pc.intensity.dtype.itemsize), 'float')
            props_info.append(('intensity', ply_t, np.dtype('<' + cls._PLY_TO_NUMPY[ply_t]), pc.intensity))

        if len(pc.rgb) == n:
            u1 = np.dtype('<u1')
            props_info.append(('red',   'uchar', u1, pc.rgb[:, 0]))
            props_info.append(('green', 'uchar', u1, pc.rgb[:, 1]))
            props_info.append(('blue',  'uchar', u1, pc.rgb[:, 2]))

        for name, arr in pc.extra.items():
            if arr.ndim == 1 and len(arr) == n:
                ply_t = cls._NUMPY_TO_PLY.get((arr.dtype.kind, arr.dtype.itemsize), 'float')
                props_info.append((name, ply_t, np.dtype('<' + cls._PLY_TO_NUMPY[ply_t]), arr))

        fmt_str    = 'binary_little_endian' if binary else 'ascii'
        prop_lines = "\n".join(f"property {ply_t} {name}" for name, ply_t, _, _ in props_info)
        header     = (
            "ply\n"
            f"format {fmt_str} 1.0\n"
            "comment Mindkosh point cloud\n"
            f"element vertex {n}\n"
            f"{prop_lines}\n"
            "end_header\n"
        )

        with open(output_path, 'wb' if binary else 'w') as out:
            out.write(header.encode('ascii') if binary else header)

            if n == 0:
                return

            if binary:
                cls._write_binary(out, props_info, n)
            else:
                arrays = [pi[3] for pi in props_info]
                for i in range(n):
                    out.write(" ".join(str(arr[i]) for arr in arrays) + "\n")

    @staticmethod
    def _write_binary(out, props_info, n: int) -> None:
        struct_dt = np.dtype([(pi[0], pi[2]) for pi in props_info])
        buf       = np.empty(_CHUNK, dtype=struct_dt)
        for start in range(0, n, _CHUNK):
            end        = min(start + _CHUNK, n)
            chunk_size = end - start
            for name, _, _, arr in props_info:
                buf[name][:chunk_size] = arr[start:end]
            out.write(buf[:chunk_size].tobytes())
