"""Slow end-to-end check on a real capture subset. Excluded by default; run with `pytest -m slow`.

Needs a capture folder (raw/ + poses.json). Set SITEMESH_TEST_CAPTURE to its path; the default is
data/object-orbit under the repo root. Skips when the capture is absent.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from sitemesh.contracts import Survey
from sitemesh.geometry import arkit_world_from_colmap_cam
from sitemesh.pipeline import run_sparse
from sitemesh.sfm import SfmOptions

REPO = Path(__file__).resolve().parents[1]
CAPTURE = Path(os.environ.get("SITEMESH_TEST_CAPTURE", REPO / "data" / "object-orbit"))
N_IMAGES = 15


@pytest.mark.slow
@pytest.mark.skipif(not (CAPTURE / "poses.json").exists(), reason=f"no capture at {CAPTURE}")
def test_sparse_on_real_subset_is_registered_metric_and_z_up(tmp_path):
    names = sorted(p.name for p in (CAPTURE / "raw").iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:N_IMAGES]
    (tmp_path / "raw").mkdir()
    for n in names:
        (tmp_path / "raw" / n).symlink_to(CAPTURE / "raw" / n)
    shutil.copy(CAPTURE / "poses.json", tmp_path / "poses.json")
    survey = Survey(tmp_path)

    report = run_sparse(survey, SfmOptions(seed=0), scale_tags=False)

    # registered, with the priors in play
    assert report["n_photos"] == N_IMAGES
    assert report["n_registered"] >= N_IMAGES - 1
    assert report["n_priors"] >= 3 and report["matcher"] == "spatial"
    # metric: SfM came out at device scale and agrees with VIO to centimetres
    assert report["sfm_to_device_scale"] == pytest.approx(1.0, abs=0.02)
    assert report["device_alignment_rms_m"] < 0.02
    assert report["device_rotation_rms_deg"] < 3.0
    assert report["output_frame"] in {"device_zup", "enu_position_only", "enu"}

    # Z-up: each camera's gravity direction, seen through the written model, matches ARKit's.
    # Heading-independent, so it holds whether or not GPS set the yaw.
    rec = pycolmap.Reconstruction(survey.sparse)
    frames = {f.image: f for f in survey.poses().shots() if f.tracking_state == "normal"}
    errs = []
    for im in rec.images.values():
        f = frames.get(im.name)
        if f is None or not im.has_pose:
            continue
        down_model = im.cam_from_world().rotation.matrix() @ np.array([0.0, 0.0, -1.0])
        down_arkit = arkit_world_from_colmap_cam(f.rotation_xyzw).T @ np.array([0.0, -1.0, 0.0])
        cos = np.clip(down_model @ down_arkit, -1.0, 1.0)
        errs.append(np.degrees(np.arccos(cos)))
    assert len(errs) >= N_IMAGES - 3
    assert np.median(errs) < 2.0, f"gravity mismatch {np.median(errs):.2f} deg: model not Z-up"
    for name in ("model.ply", "cameras.json", "report.json"):
        assert (survey.model / name).exists()
