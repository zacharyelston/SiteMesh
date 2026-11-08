"""georegister on a synthetic model: metric scale, +Z up, and GPS can only yaw the model.

Ground truth is a Z-up metric scene. The phone saw it in ARKit's frame (+Y up), and SfM returned
it in an arbitrary tilted, scaled gauge. georegister must undo the gauge using only the VIO
centres, level it with the fixed ARKit->Z-up map, and let GPS choose nothing but heading.
"""

import numpy as np
import pycolmap
import pytest

from sitemesh.contracts import DevicePoseFrame, DevicePoseTrack, GpsFix
from sitemesh.geometry import DEVICE_TO_ZUP
from sitemesh.georef import georegister

REF = (41.30, -72.78, 10.0)  # lat, lon, alt of the first fix
M_PER_DEG_LAT = 111_320.0


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _scene(span_m):
    """Synthetic reconstruction whose camera centres span ~span_m, in true Z-up metres."""
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_frames_per_rig = 12
    opts.num_points3D = 50
    rec = pycolmap.synthesize_dataset(opts)
    c = np.array([im.projection_center() for im in rec.images.values()])
    s = span_m / np.ptp(c, axis=0).max()
    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(np.eye(3)), -s * c.mean(0)))
    return rec


def _centres(rec):
    names = sorted(im.name for im in rec.images.values())
    by = {im.name: im for im in rec.images.values()}
    return names, np.array([by[n].projection_center() for n in names])


def _track(names, truth_zup, heading_deg=None, alt_noise_m=0.0, seed=0):
    """ARKit pose track for the truth scene; GPS (if heading given) from a yawed ENU copy."""
    device = (DEVICE_TO_ZUP.T @ truth_zup.T).T  # ARKit world: +Y up
    frames = [
        DevicePoseFrame(timestamp=float(i), image=n, position_m=tuple(device[i]),
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        for i, n in enumerate(names)
    ]
    gps = []
    if heading_deg is not None:
        rng = np.random.default_rng(seed)
        enu = (_rot([0, 0, 1], heading_deg) @ truth_zup.T).T
        enu -= enu[0]
        for i, (e, n, u) in enumerate(enu):
            gps.append(GpsFix(
                timestamp=float(i),
                lat_deg=REF[0] + n / M_PER_DEG_LAT,
                lon_deg=REF[1] + e / (M_PER_DEG_LAT * np.cos(np.radians(REF[0]))),
                alt_m=REF[2] + u + rng.normal(0, alt_noise_m),
            ))
    return DevicePoseTrack(source="arkit", frames=frames, gps_track=gps)


def _tilt_into_gauge(rec, scale=0.37):
    """What SfM hands back: arbitrary scale, a 50-degree tilt and an offset."""
    r = _rot([1.0, 0.4, -0.3], 50.0)
    rec.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(r), np.array([3.0, -2.0, 7.0])))
    return scale


def test_no_gps_recovers_scale_and_z_up():
    rec = _scene(span_m=2.0)
    names, truth = _centres(rec)
    points_truth = np.array([p.xyz for p in rec.points3D.values()])
    scale = _tilt_into_gauge(rec)

    out = georegister(rec, _track(names, truth))

    assert out["output_frame"] == "device_zup"
    assert out["sfm_to_device_scale"] == pytest.approx(1 / scale, rel=1e-6)
    assert out["device_alignment_rms_m"] < 1e-6
    _, got = _centres(rec)
    np.testing.assert_allclose(got, truth, atol=1e-6)  # metric, and +Z is the true up
    points_got = np.array([p.xyz for p in rec.points3D.values()])
    np.testing.assert_allclose(points_got, points_truth, atol=1e-6)


def test_short_walk_reports_position_only():
    rec = _scene(span_m=1.5)  # < 3 m of walking: GPS heading is noise
    names, truth = _centres(rec)
    _tilt_into_gauge(rec)

    out = georegister(rec, _track(names, truth, heading_deg=70.0))

    assert out["output_frame"] == "enu_position_only"
    assert out["heading"].startswith("arbitrary")
    _, got = _centres(rec)
    # heading not applied: only a translation away from the levelled truth
    d = got - truth
    np.testing.assert_allclose(d, np.broadcast_to(d.mean(0), d.shape), atol=1e-6)


def test_gps_heading_cannot_tilt_the_model():
    rec = _scene(span_m=20.0)
    names, truth = _centres(rec)
    _tilt_into_gauge(rec)
    heading = 35.0

    # 8 m of altitude noise: an unconstrained similarity fit to these fixes would tilt the model.
    out = georegister(rec, _track(names, truth, heading_deg=heading, alt_noise_m=8.0))

    assert out["output_frame"] == "enu"
    _, got = _centres(rec)
    # heights are untouched by GPS altitude: z differs from truth only by one constant offset
    dz = got[:, 2] - truth[:, 2]
    np.testing.assert_allclose(dz, dz.mean(), atol=1e-6)
    # and the horizontal heading is the GPS one, up to lat/lon linearisation error
    expect = (_rot([0, 0, 1], heading) @ truth.T).T
    a = got[:, :2] - got[:, :2].mean(0)
    b = expect[:, :2] - expect[:, :2].mean(0)
    np.testing.assert_allclose(a, b, atol=0.05)
