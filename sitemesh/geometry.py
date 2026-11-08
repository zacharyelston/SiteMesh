"""Small closed-form geometry: frames, similarity fits, ray intersection, triangulation."""

from __future__ import annotations

import numpy as np

#: ARKit camera axes are +X right, +Y up, +Z back (looks down -Z).
#: COLMAP camera axes are +X right, +Y down, +Z forward. Same X, flipped Y and Z.
ARKIT_TO_COLMAP_CAM = np.diag([1.0, -1.0, -1.0])

#: ARKit world (+Y up, gravity-aligned) to a Z-up frame: x->x, y->z, z->-y.
#: Right-handed, and it keeps heading arbitrary — only "up" is fixed.
DEVICE_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def arkit_world_from_colmap_cam(rotation_xyzw) -> np.ndarray:
    """Rotation taking COLMAP-convention camera axes into the ARKit world frame."""
    return quat_xyzw_to_matrix(rotation_xyzw) @ ARKIT_TO_COLMAP_CAM


def optical_axis(rotation_xyzw) -> np.ndarray:
    """Unit viewing direction in the device world frame."""
    return arkit_world_from_colmap_cam(rotation_xyzw)[:, 2]


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1.0
    rot = u @ sgn @ vt
    scale = float(np.trace(np.diag(d) @ sgn) / ((xs**2).sum() / len(src))) if with_scale else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def yaw_fit(src_zup: np.ndarray, dst_zup: np.ndarray):
    """Rigid fit about the vertical axis only: (R, t) with R a pure yaw.

    Both frames already agree on "up" (ARKit gravity vs ENU vertical). Letting GPS rotate the
    other two axes tipped an indoor model 105.9 degrees in hydra-vision; here tilt is simply not a
    free parameter, so altitude noise cannot tilt anything.
    """
    a = src_zup[:, :2] - src_zup[:, :2].mean(0)
    b = dst_zup[:, :2] - dst_zup[:, :2].mean(0)
    theta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot, dst_zup.mean(0) - rot @ src_zup.mean(0)


def nearest_point_to_rays(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray | None:
    """Least-squares point closest to a bundle of rays (where the cameras were looking)."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = d / np.linalg.norm(d)
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ o
    if np.linalg.cond(a) > 1e6:
        return None
    return np.linalg.solve(a, b)


def triangulate_dlt(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """Linear N-view triangulation from 3x4 projection matrices and pixel observations."""
    rows = []
    for p, (u, v) in zip(projections, pixels):
        rows.append(u * p[2] - p[0])
        rows.append(v * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    x = vt[-1]
    return x[:3] / x[3]


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def _ecef(lat, lon, alt):
    lat, lon = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1 - WGS84_E2) + alt) * np.sin(lat),
        ]
    )


def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt) -> np.ndarray:
    d = _ecef(lat, lon, alt) - _ecef(ref_lat, ref_lon, ref_alt)
    la, lo = np.radians(ref_lat), np.radians(ref_lon)
    r = np.array(
        [
            [-np.sin(lo), np.cos(lo), 0.0],
            [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)],
        ]
    )
    return r @ d
