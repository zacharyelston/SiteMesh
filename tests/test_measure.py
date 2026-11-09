import numpy as np
import pytest

from sitemesh.contracts import DevicePoseFrame, DevicePoseTrack, GpsFix
from sitemesh.geometry import DEVICE_TO_ZUP
from sitemesh.gpscheck import gps_scale_check
from sitemesh.measure import measure_points

M_PER_DEG = 111_320.0


def _room(length=8.0, width=3.0, height=2.7, table=0.75, n=4000, seed=0):
    """Floor, ceiling, two long walls and a table, Z up, floor at z = 1.0."""
    rng = np.random.default_rng(seed)
    z0 = 1.0
    u = rng.uniform
    floor = np.c_[u(0, length, n), u(0, width, n), np.full(n, z0)]
    ceil = np.c_[u(0, length, n // 2), u(0, width, n // 2), np.full(n // 2, z0 + height)]
    wall1 = np.c_[u(0, length, n // 2), np.zeros(n // 2), u(z0, z0 + height, n // 2)]
    wall2 = np.c_[u(0, length, n // 2), np.full(n // 2, width), u(z0, z0 + height, n // 2)]
    tab = np.c_[u(3, 4, n // 4), u(1, 2, n // 4), np.full(n // 4, z0 + table)]
    pts = np.vstack([floor, ceil, wall1, wall2, tab])
    pts += rng.normal(0, 0.005, pts.shape)
    cams = np.c_[np.linspace(0.5, length - 0.5, 20), np.full(20, width / 2), np.full(20, z0 + 1.5)]
    return pts, cams


def test_box_room_dimensions():
    pts, cams = _room()
    m = measure_points(pts, cams)
    # extents come from 25 cm voxel centres: true size within about one voxel
    assert m["extent"]["length_m"] == pytest.approx(8.0, abs=0.35)
    assert m["extent"]["width_m"] == pytest.approx(3.0, abs=0.35)
    assert m["ground_z_m"] == pytest.approx(1.0, abs=0.01)
    assert m["ceiling_above_ground_m"] == pytest.approx(2.7, abs=0.02)
    heights = sorted(round(lv["above_ground_m"], 2) for lv in m["levels"])
    assert heights[0] == 0.0 and any(abs(h - 0.75) < 0.02 for h in heights)
    assert all(lv["tilt_deg"] < 1.0 for lv in m["levels"])
    assert m["cameras"]["height_above_ground_m"]["median"] == pytest.approx(1.5, abs=0.01)
    s = m["sections"]
    assert s["height"]["median_m"] == pytest.approx(2.7, abs=0.05)
    assert s["width"]["median_m"] == pytest.approx(3.0, abs=0.1)
    assert m["cameras"]["walk_length_m"] == pytest.approx(7.0, abs=1e-6)


def _gps_track(scale_true=1.0, extent=25.0, n=200, acc=4.0, seed=0):
    """VIO walks a loop; GPS sees the same loop scaled by scale_true, rotated, with noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 200, n)
    zup = np.c_[extent / 2 * np.cos(t / 30), extent / 2 * np.sin(t / 45), np.zeros(n)]
    device = (DEVICE_TO_ZUP.T @ zup.T).T
    frames = [DevicePoseFrame(timestamp=float(ti), position_m=tuple(p),
                              rotation_xyzw=(0.0, 0.0, 0.0, 1.0)) for ti, p in zip(t, device)]
    c, s = np.cos(0.7), np.sin(0.7)
    enu = (np.array([[c, -s], [s, c]]) @ (scale_true * zup[:, :2]).T).T
    enu += rng.normal(0, acc / 2, enu.shape)
    fixes = [GpsFix(timestamp=float(ti), lat_deg=41.3 + e[1] / M_PER_DEG,
                    lon_deg=-72.78 + e[0] / (M_PER_DEG * np.cos(np.radians(41.3))),
                    alt_m=10.0, horizontal_accuracy_m=acc) for ti, e in zip(t, enu)]
    return DevicePoseTrack(source="arkit", frames=frames, gps_track=fixes)


def test_gps_confirms_correct_scale():
    out = gps_scale_check(_gps_track(scale_true=1.0))
    assert out["status"] == "consistent"
    lo, hi = out["scale_95ci"]
    assert lo <= 1.0 <= hi and hi - lo < 0.2


def test_gps_flags_wrong_scale():
    out = gps_scale_check(_gps_track(scale_true=1.3))
    assert out["status"] == "inconsistent"
    assert out["gps_over_vio_scale"] == pytest.approx(1.3, abs=0.06)


@pytest.mark.parametrize("kwargs, why", [
    ({"extent": 3.0}, "walk extent"),
    ({"acc": 15.0}, "accuracy"),
    ({"n": 10}, "fixes"),
])
def test_gps_refuses_when_it_cannot_tell(kwargs, why):
    out = gps_scale_check(_gps_track(**kwargs))
    assert out["status"] == "not_applicable" and why in out["reason"]
    assert "gps_over_vio_scale" not in out
