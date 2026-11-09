"""Minimal binary PLY reading (vertices only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """XYZ of a binary little-endian PLY with float xyz and optional uchar colours."""
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
        lines = header.split(b"\n")
        n = int(next(x for x in lines if x.startswith(b"element vertex")).split()[-1])
        types = {b"float": "<f4", b"double": "<f8", b"uchar": "u1", b"int": "<i4"}
        dt = []
        for x in lines:
            if x.startswith(b"property"):
                parts = x.split()
                dt.append((parts[-1].decode(), types[parts[1]]))
        a = np.frombuffer(fh.read(n * np.dtype(dt).itemsize), dtype=dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
