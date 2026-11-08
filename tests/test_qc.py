import numpy as np

from sitemesh.contracts import DevicePoseFrame, DevicePoseTrack
from sitemesh.qc import stance_qc


def _look_at_quat(centre, target):
    """ARKit-convention rotation (camera looks down -Z, +Y up) as xyzw."""
    back = centre - target
    back /= np.linalg.norm(back)
    right = np.cross([0.0, 1.0, 0.0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    m = np.column_stack([right, up, back])
    w = np.sqrt(max(0.0, 1 + np.trace(m))) / 2
    return (
        (m[2, 1] - m[1, 2]) / (4 * w),
        (m[0, 2] - m[2, 0]) / (4 * w),
        (m[1, 0] - m[0, 1]) / (4 * w),
        w,
    )


def _track(centres, target=(0.0, 1.5, -2.0)):
    target = np.asarray(target, float)
    frames = [
        DevicePoseFrame(
            timestamp=float(i), image=f"IMG_{i:04d}.jpg", position_m=tuple(c),
            rotation_xyzw=_look_at_quat(np.array(c, float), target),
        )
        for i, c in enumerate(centres)
    ]
    return DevicePoseTrack(source="arkit", frames=frames)


def test_stance_triangles_at_d_over_6_pass():
    # Subject 2 m away, stance-card spacing 33 cm: left / right / above, repeated sideways.
    base = 2.0 / 6
    centres = []
    for k in range(6):
        x0 = k * base * 2
        centres += [(x0 - base / 2, 1.5, 0.0), (x0 + base / 2, 1.5, 0.0), (x0, 1.5 + base, 0.0)]
    result = stance_qc(_track(centres, target=np.array([1.5, 1.5, -2.0])))
    s = result["summary"]
    assert s["verdict"] == "ok"
    assert 3 < s["median_d_over_b"] < 12
    assert s["flag_counts"].get("wrist_pivot", 0) == 0
    assert s["flag_counts"].get("too_wide", 0) == 0


def test_wrist_pivots_are_flagged():
    centres = [(0.0, 1.5, 0.0)] * 10
    targets = [np.array([x, 1.5, -2.0]) for x in np.linspace(-1.5, 1.5, 10)]
    frames = [
        DevicePoseFrame(timestamp=float(i), image=f"IMG_{i:04d}.jpg", position_m=c,
                        rotation_xyzw=_look_at_quat(np.array(c), t))
        for i, (c, t) in enumerate(zip(centres, targets))
    ]
    result = stance_qc(DevicePoseTrack(source="arkit", frames=frames))
    assert result["summary"]["flag_counts"]["wrist_pivot"] == 9
    assert result["summary"]["verdict"].startswith("re-shoot")


def test_post_sfm_stance_gives_unregistered_shots_a_neighbour_d_over_b():
    import pycolmap

    from sitemesh.qc import post_sfm_stance

    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_frames_per_rig = 6
    opts.num_points3D = 60
    rec = pycolmap.synthesize_dataset(opts)
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    centres = [np.array(im.projection_center()) for im in ims]
    frames = [
        DevicePoseFrame(timestamp=float(i), image=im.name, position_m=tuple(c),
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        for i, (im, c) in enumerate(zip(ims, centres))
    ]
    track = DevicePoseTrack(source="arkit", frames=frames)
    dropped = ims[3]
    rec.deregister_frame(dropped.frame_id)

    qc = {"shots": [{"image": ims[3].name, "d_over_b": 1.5, "flags": ["too_wide"]}]}
    rows = {r["image"]: r for r in post_sfm_stance(track, rec, qc)}

    gone = rows[dropped.name]
    assert not gone["registered"] and "measured_depth_m" not in gone
    assert gone["qc_d_over_b"] == 1.5 and gone["qc_flags"] == ["too_wide"]
    expect = np.median([rows[ims[j].name]["measured_depth_m"] for j in (1, 2, 4, 5)])
    assert gone["neighbour_depth_m"] == expect
    assert gone["neighbour_d_over_b"] == expect / gone["baseline_m"]
    # registered shots get the same neighbour-based estimate, excluding themselves
    assert "neighbour_d_over_b" in rows[ims[2].name]
