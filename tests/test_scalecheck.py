"""End-to-end scale check on a synthetic scene: render a printed tag into two posed views."""

import cv2
import numpy as np
import pycolmap

from sitemesh.report import evaluate
from sitemesh.scalecheck import scale_check

TAG_ID, TAG_M = 205, 0.1651  # room-far kit tag


def _render(k, cam_from_world, size=(1920, 1440)):
    """Tag lying in the world plane z=2 (facing the cameras), centred at the origin."""
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), TAG_ID, 400
    )
    # Quiet zone so the detector sees a border.
    canvas = np.full((600, 600), 255, np.uint8)
    canvas[100:500, 100:500] = marker
    half = TAG_M / 2 * 600 / 400  # canvas edge in metres
    world = np.array([[-half, -half, 2], [half, -half, 2], [half, half, 2], [-half, half, 2]])
    r, t = cam_from_world[:, :3], cam_from_world[:, 3]
    pix = []
    for xw in world:
        x = k @ (r @ xw + t)
        pix.append(x[:2] / x[2])
    src = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
    h = cv2.getPerspectiveTransform(src, np.float32(pix))
    return cv2.warpPerspective(canvas, h, size, borderValue=200)


def test_scale_check_recovers_tag_size(tmp_path):
    f, cx, cy = 1400.0, 960.0, 720.0
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_name(1, "SIMPLE_RADIAL", f, 1920, 1440)
    cam.params = [f, cx, cy, 0.0]
    rec.add_camera_with_trivial_rig(cam)
    for i, x in enumerate((-0.25, 0.0, 0.3), start=1):
        cfw = np.hstack([np.eye(3), [[-x], [0.0], [0.0]]])
        cv2.imwrite(str(tmp_path / f"IMG_{i}.jpg"), _render(k, cfw), [cv2.IMWRITE_JPEG_QUALITY, 95])
        im = pycolmap.Image(name=f"IMG_{i}.jpg", camera_id=1, image_id=i)
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)),
                                                              cfw[:, 3]))
    result = scale_check(rec, tmp_path)
    tag = result["tags"][0]
    assert tag["tag_id"] == TAG_ID and tag["n_views"] == 3
    assert abs(tag["error_pct"]) < 0.5, tag


def test_gate_never_passes_unmeasured():
    report = {"n_photos": 50, "n_registered": 50, "mean_reprojection_px": 0.8}
    gate = evaluate(report)
    assert gate["verdict"] == "pass_unverified"
    report["n_registered"] = 40
    assert evaluate(report)["verdict"] == "fail"
