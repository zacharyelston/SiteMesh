import numpy as np

from sitemesh.geometry import (
    arkit_world_from_colmap_cam,
    optical_axis,
    quat_xyzw_to_matrix,
    triangulate_dlt,
    umeyama,
    wgs84_to_enu,
    yaw_fit,
)


def _rand_rot(rng):
    q = rng.normal(size=4)
    return quat_xyzw_to_matrix(q / np.linalg.norm(q))


def test_identity_arkit_camera_looks_down_minus_z():
    # ARKit cameras look along their own -Z; with identity pose that is world -Z.
    assert np.allclose(optical_axis((0, 0, 0, 1)), [0, 0, -1])
    r = arkit_world_from_colmap_cam((0, 0, 0, 1))
    assert np.allclose(r[:, 1], [0, -1, 0])  # COLMAP +Y (down) is ARKit world -Y
    assert np.isclose(np.linalg.det(r), 1.0)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    rot, s, t = _rand_rot(rng), 3.7, np.array([1.0, -2.0, 5.0])
    dst = s * (rot @ src.T).T + t
    s2, r2, t2 = umeyama(src, dst)
    assert np.isclose(s2, s) and np.allclose(r2, rot) and np.allclose(t2, t)


def test_yaw_fit_ignores_vertical_noise():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(20, 3)) * [10, 10, 1]
    th = np.radians(37)
    rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    dst = (rz @ src.T).T + [5, 6, 7]
    dst[:, 2] += rng.normal(scale=20, size=20)  # GPS altitude is terrible
    r, _ = yaw_fit(src, dst)
    assert np.allclose(r, rz, atol=1e-9)
    assert np.allclose(r[:, 2], [0, 0, 1])  # cannot tilt


def test_triangulate_dlt():
    point = np.array([0.3, -0.2, 4.0])
    projs, pix = [], []
    for cx in (-0.5, 0.0, 0.6):
        p = np.hstack([np.eye(3), [[-cx], [0], [0]]])
        x = p @ np.append(point, 1)
        projs.append(p)
        pix.append(x[:2] / x[2])
    assert np.allclose(triangulate_dlt(projs, pix), point)


def test_wgs84_to_enu_north():
    d = wgs84_to_enu(41.3009, -72.784, 6.0, 41.3, -72.784, 6.0)
    assert abs(d[0]) < 0.01 and 99 < d[1] < 101 and abs(d[2]) < 0.01
