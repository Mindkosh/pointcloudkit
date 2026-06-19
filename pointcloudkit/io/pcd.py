# Copyright (C) 2026 Mindkosh technologies private limited

import math
import struct

import numpy as np

from pointcloudkit.point_cloud import PointCloud

# Points written per iteration during binary streaming writes.
# Keeps peak extra memory at CHUNK × bytes_per_point (~8 MB for 128 B/pt).
_CHUNK = 65_536

# Beyond this absolute coordinate value float32 loses sub-millimetre precision
# (~7 significant digits → 1 mm resolution at 10 000 m).
_F32_POSITION_THRESHOLD = 1e4


class PCDFile:
    """Read and write PCD (Point Cloud Data) files."""

    _KNOWN_FIELDS = {'x', 'y', 'z', 'intensity', 'rgb'}

    _STRUCT_FMT = {
        ('F', 4): 'f',  ('F', 8): 'd',
        ('U', 1): 'B',  ('U', 2): 'H',  ('U', 4): 'I',  ('U', 8): 'Q',
        ('I', 1): 'b',  ('I', 2): 'h',  ('I', 4): 'i',  ('I', 8): 'q',
    }

    # ── Reader ─────────────────────────────────────────────────────────────

    @classmethod
    def read(cls, path: str) -> PointCloud:
        """Read a PCD file (ASCII or binary) and return a PointCloud."""
        pc = PointCloud()

        # Temporary list buffers; replaced by numpy arrays after all points are read
        pc._buf_x         = []
        pc._buf_y         = []
        pc._buf_z         = []
        pc._buf_intensity = []
        pc._buf_rgb       = []

        with open(path, 'rb') as f:
            hdr    = cls._parse_header(f)
            fields = hdr['fields']
            types  = hdr['types']
            sizes  = hdr['sizes']
            counts = hdr['counts']
            n_pts  = hdr.get('points', 0)
            fmt    = hdr.get('data', 'ascii')

            for field in fields:
                if field not in cls._KNOWN_FIELDS:
                    pc.extra[field] = []

            field_type = dict(zip(fields, types))

            if fmt == 'ascii':
                cls._read_ascii(f, pc, fields, field_type, counts)
            elif fmt == 'binary':
                cls._read_binary(f, pc, fields, field_type, types, sizes, counts, n_pts)
            else:
                raise ValueError(f"Unsupported PCD data format: {fmt!r}")

        if pc._buf_x:
            pc.position = np.column_stack([pc._buf_x, pc._buf_y, pc._buf_z]).astype(np.float64)
        pc.intensity = np.array(pc._buf_intensity, dtype=np.float32)
        pc.rgb       = np.array(pc._buf_rgb, dtype=np.uint8).reshape(-1, 3)
        pc.extra     = {k: np.array(v) for k, v in pc.extra.items()}

        del pc._buf_x, pc._buf_y, pc._buf_z, pc._buf_intensity, pc._buf_rgb

        return pc

    @staticmethod
    def _parse_header(f) -> dict:
        hdr = {'fields': [], 'types': [], 'sizes': [], 'counts': []}
        while True:
            line = f.readline()
            if isinstance(line, bytes):
                line = line.decode('ascii', errors='ignore')
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts     = line.split()
            key, vals = parts[0].upper(), parts[1:]
            if   key == 'FIELDS': hdr['fields'] = vals
            elif key == 'TYPE':   hdr['types']  = vals
            elif key == 'SIZE':   hdr['sizes']  = [int(v) for v in vals]
            elif key == 'COUNT':  hdr['counts'] = [int(v) for v in vals]
            elif key == 'POINTS': hdr['points'] = int(vals[0])
            elif key == 'DATA':
                hdr['data'] = vals[0].lower()
                break
        return hdr

    @classmethod
    def _read_ascii(cls, f, pc, fields, field_type, counts):
        expected_tokens = sum(counts)
        for line in f:
            if isinstance(line, bytes):
                line = line.decode('ascii', errors='ignore')
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            if len(tokens) != expected_tokens:
                continue

            parsed = {}
            valid  = True
            tok_i  = 0

            for field, count in zip(fields, counts):
                typ = field_type[field]
                raw = tokens[tok_i:tok_i + count]
                tok_i += count
                try:
                    if typ == 'F':
                        v = [float(t) for t in raw]
                        if any(not math.isfinite(x) for x in v):
                            if field in ('x', 'y', 'z'):
                                valid = False
                                break
                            v = [0.0] * count  # zero out NaN/Inf in non-position fields
                        v = v[0] if count == 1 else v
                    else:
                        v = [int(float(t)) for t in raw]
                        v = v[0] if count == 1 else v
                except (ValueError, OverflowError):
                    valid = False
                    break
                parsed[field] = v

            if valid:
                cls._append_point(pc, fields, field_type, parsed)

    @classmethod
    def _read_binary(cls, f, pc, fields, field_type, types, sizes, counts, n_pts):
        fmt_str = '<' + ''.join(
            cls._STRUCT_FMT[(typ, size)] * count
            for typ, size, count in zip(types, sizes, counts)
        )
        packer     = struct.Struct(fmt_str)
        point_size = packer.size

        for _ in range(n_pts):
            raw = f.read(point_size)
            if len(raw) < point_size:
                break

            unpacked = packer.unpack(raw)
            parsed   = {}
            valid    = True
            idx      = 0

            for field, count in zip(fields, counts):
                typ = field_type[field]
                v   = unpacked[idx] if count == 1 else list(unpacked[idx:idx + count])
                idx += count

                if typ == 'F':
                    check = [v] if count == 1 else v
                    if any(not math.isfinite(x) for x in check):
                        if field in ('x', 'y', 'z'):
                            valid = False
                            break
                        v = 0.0 if count == 1 else [0.0] * count  # zero out NaN/Inf in non-position fields

                parsed[field] = v

            if valid:
                cls._append_point(pc, fields, field_type, parsed)

    @classmethod
    def _append_point(cls, pc, fields, field_type, parsed):
        if 'x'         in parsed: pc._buf_x.append(parsed['x'])
        if 'y'         in parsed: pc._buf_y.append(parsed['y'])
        if 'z'         in parsed: pc._buf_z.append(parsed['z'])
        if 'intensity' in parsed: pc._buf_intensity.append(parsed['intensity'])
        if 'rgb'       in parsed:
            pc._buf_rgb.append(cls._decode_rgb(parsed['rgb'], field_type['rgb']))
        for field, val in parsed.items():
            if field in pc.extra:
                pc.extra[field].append(val)

    @staticmethod
    def _decode_rgb(val, typ: str) -> tuple:
        """Decode a packed RGB value (integer or reinterpreted float) into (r, g, b)."""
        if typ == 'F':
            # PCL packs RGB into a float's bit pattern
            packed = struct.unpack('I', struct.pack('f', float(val)))[0]
        else:
            packed = int(val)
        return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)

    # ── Writer ─────────────────────────────────────────────────────────────

    _KIND_TO_PCD_TYPE = {'f': 'F', 'u': 'U', 'i': 'I'}

    @classmethod
    def write(cls, pc: PointCloud, output_path: str, binary: bool = True) -> None:
        """Write a PointCloud to a PCD file.

        All non-empty fields are written: position (x, y, z), intensity, rgb,
        and any 1-D arrays in pc.extra.

        Binary mode uses chunked writes so that peak extra memory stays at
        _CHUNK × bytes_per_point regardless of total cloud size.
        """
        n = len(pc)

        # Each entry: (field_name, little-endian np.dtype, PCD type char, PCD size in bytes, 1-D array)
        fields_info = []

        use_f64 = n > 0 and np.max(np.abs(pc.position)) > _F32_POSITION_THRESHOLD
        xyz_dt  = np.dtype('<f8') if use_f64 else np.dtype('<f4')
        fields_info.append(('x', xyz_dt, 'F', xyz_dt.itemsize, pc.position[:, 0].astype(xyz_dt)))
        fields_info.append(('y', xyz_dt, 'F', xyz_dt.itemsize, pc.position[:, 1].astype(xyz_dt)))
        fields_info.append(('z', xyz_dt, 'F', xyz_dt.itemsize, pc.position[:, 2].astype(xyz_dt)))

        if len(pc.intensity) == n:
            idt = pc.intensity.dtype.newbyteorder('<')
            fields_info.append(('intensity', idt, 'F', pc.intensity.dtype.itemsize, pc.intensity))

        if len(pc.rgb) == n:
            r = pc.rgb[:, 0].astype(np.uint32)
            g = pc.rgb[:, 1].astype(np.uint32)
            b = pc.rgb[:, 2].astype(np.uint32)
            packed_rgb = (r << 16) | (g << 8) | b
            fields_info.append(('rgb', np.dtype('<u4'), 'U', 4, packed_rgb))

        for name, arr in pc.extra.items():
            if arr.ndim == 1 and len(arr) == n:
                pcd_type = cls._KIND_TO_PCD_TYPE.get(arr.dtype.kind, 'F')
                fields_info.append((name, arr.dtype.newbyteorder('<'), pcd_type, arr.dtype.itemsize, arr))

        field_names = [fi[0] for fi in fields_info]
        field_types = [fi[2] for fi in fields_info]
        field_sizes = [str(fi[3]) for fi in fields_info]

        header = "\n".join([
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f"FIELDS {' '.join(field_names)}",
            f"SIZE {' '.join(field_sizes)}",
            f"TYPE {' '.join(field_types)}",
            f"COUNT {' '.join(['1'] * len(field_names))}",
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {n}",
            "DATA binary" if binary else "DATA ascii",
        ]) + "\n"

        with open(output_path, 'wb' if binary else 'w') as out:
            out.write(header.encode('ascii') if binary else header)

            if n == 0:
                return

            if binary:
                cls._write_binary(out, fields_info, n)
            else:
                arrays = [fi[4] for fi in fields_info]
                for i in range(n):
                    out.write(" ".join(str(arr[i]) for arr in arrays) + "\n")

    @staticmethod
    def _write_binary(out, fields_info, n: int) -> None:
        struct_dt = np.dtype([(fi[0], fi[1]) for fi in fields_info])
        buf       = np.empty(_CHUNK, dtype=struct_dt)
        for start in range(0, n, _CHUNK):
            end        = min(start + _CHUNK, n)
            chunk_size = end - start
            for fi in fields_info:
                buf[fi[0]][:chunk_size] = fi[4][start:end]
            out.write(buf[:chunk_size].tobytes())
