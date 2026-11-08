"""merge_submodels: pieces solved in different gauges come back as one metric model."""

import numpy as np
import pycolmap
import pytest

from sitemesh.contracts import DevicePoseFrame, DevicePoseTrack, Survey
from sitemesh.sfm import SfmOptions, _vio_fit, merge_submodels


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


@pytest.fixture
def split_survey(tmp_path):
    """A synthetic capture whose database has matches and VIO priors, split in two gauges."""
    survey = Survey(tmp_path)
    survey.raw.mkdir()
    survey.model.mkdir()
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_frames_per_rig = 10
    opts.num_points3D = 200
    opts.prior_position = True
    db = pycolmap.Database.open(survey.database)
    truth = pycolmap.synthesize_dataset(opts, db)
    db.close()
    names = sorted(im.name for im in truth.images.values())
    centre = {im.name: np.array(im.projection_center()) for im in truth.images.values()}
    track = DevicePoseTrack(source="arkit", frames=[
        DevicePoseFrame(timestamp=float(i), image=n, position_m=tuple(centre[n]),
                        rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        for i, n in enumerate(names)
    ])
    survey.poses_path.write_text(track.model_dump_json())

    halves = []
    for keep, (scale, axis, deg, t) in [
        (names[:5], (0.4, [1, 0, 0], 30, [1.0, 2.0, 3.0])),
        (names[5:], (2.5, [0, 1, 1], -70, [-4.0, 0.0, 1.0])),
    ]:
        part = pycolmap.Reconstruction(truth)
        for im in list(part.images.values()):
            if im.name not in keep:
                part.deregister_frame(im.frame_id)
        part.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(_rot(axis, deg)), np.array(t)))
        halves.append(part)
    return survey, truth, {0: halves[0], 1: halves[1]}


def test_merges_two_gauges_into_one_metric_model(split_survey):
    survey, _truth, maps = split_survey
    rec, report = merge_submodels(survey, maps, SfmOptions())

    assert report["merged"] and report["n_segments_merged"] == 2
    assert report["n_registered_largest"] == 5
    assert rec.num_reg_images() == 10 == report["n_registered_merged"]
    assert all(seg["kept"] for seg in report["segments"])
    # pieces were 0.4x and 2.5x in their own gauges; merged agrees with VIO, at scale 1
    frames = {f.image: f for f in survey.poses().shots()}
    s, _, _, rms, n = _vio_fit(rec, frames)
    assert n == 10 and rms < 1e-3 and s == pytest.approx(1.0, abs=1e-3)
    # the halves overlap in the synthetic matches, so some points span the join
    assert report["n_joint_points"] > 0
    assert rec.num_points3D() > 0


def test_single_model_is_left_alone(split_survey):
    survey, truth, _maps = split_survey
    rec, report = merge_submodels(survey, {0: truth}, SfmOptions())
    assert rec is None and not report["merged"]


def test_segment_that_disagrees_with_vio_is_dropped(split_survey):
    survey, _truth, maps = split_survey
    bad = maps[1]
    # bend the second piece non-rigidly so no similarity can fit it to VIO
    rng = np.random.default_rng(0)
    for im in bad.images.values():
        if im.has_pose:
            pose = im.cam_from_world()
            bad.frame(im.frame_id).set_cam_from_world(
                im.camera_id, pycolmap.Rigid3d(pose.rotation, pose.translation + rng.normal(0, 1, 3)))
    rec, report = merge_submodels(survey, maps, SfmOptions(max_segment_rms_m=0.25))
    dropped = [seg for seg in report["segments"] if not seg["kept"]]
    assert len(dropped) == 1 and "VIO RMS" in dropped[0]["reason"]
    assert rec is None  # one good piece left: nothing to merge
