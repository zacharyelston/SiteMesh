from sitemesh.report import Gate, evaluate


def _check(report, name):
    return next(c for c in evaluate(report)["checks"] if c["check"] == name)


def test_alignment_is_judged_relative_to_scene_size():
    # 90 mm on a 45.8 m site (b8b8ebd7, merged) passes; it failed the old absolute 10 mm target
    ok = _check({"device_alignment_rms_m": 0.090, "scene_diagonal_m": 45.8}, "device_alignment_rel")
    assert ok["status"] == "pass" and abs(ok["value"] - 0.090 / 45.8) < 1e-12
    bad = _check({"device_alignment_rms_m": 0.10, "scene_diagonal_m": 10.0}, "device_alignment_rel")
    assert bad["status"] == "fail"  # 1% > 0.5%


def test_alignment_without_scene_size_is_not_measured_never_passed():
    for report in ({"device_alignment_rms_m": 0.01}, {"scene_diagonal_m": 5.0},
                   {"device_alignment_rms_m": 0.01, "scene_diagonal_m": 0.0}):
        assert _check(report, "device_alignment_rel")["status"] == "not_measured"
    verdict = evaluate({"n_photos": 10, "n_registered": 10, "mean_reprojection_px": 0.5,
                        "device_alignment_rms_m": 0.001, "device_rotation_rms_deg": 0.5})
    assert verdict["verdict"] == "pass_unverified"


def test_no_absolute_alignment_check_remains():
    assert not hasattr(Gate(), "max_device_alignment_rms_m")
    names = {c["check"] for c in evaluate({})["checks"]}
    assert "device_alignment_rms_m" not in names and "device_alignment_rel" in names
