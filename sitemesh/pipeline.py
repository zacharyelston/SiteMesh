"""One survey, end to end: QC → sparse (with priors) → georegister → scale check → gate → dense."""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from sitemesh import __version__
from sitemesh.contracts import Survey
from sitemesh.dense import run_dense
from sitemesh.georef import georegister
from sitemesh.gpscheck import gps_scale_check
from sitemesh.measure import run_measure
from sitemesh.qc import post_sfm_stance, stance_qc
from sitemesh.report import evaluate
from sitemesh.scalecheck import scale_check
from sitemesh.sfm import SfmOptions, device_agreement, run_sfm

log = logging.getLogger(__name__)


def run_qc(survey: Survey) -> dict:
    track = survey.poses()
    if track is None:
        return {"summary": {"verdict": "no poses.json — cannot check stance"}}
    result = stance_qc(track)
    survey.write_json("qc.json", result)
    return result


def scene_diagonal_m(rec) -> float | None:
    """Horizontal size of the scene: 2nd-98th percentile XY extent of the output cloud, in m."""
    if rec.num_points3D() < 10:
        return None
    xy = np.array([p.xyz[:2] for p in rec.points3D.values()])
    lo, hi = np.percentile(xy, [2, 98], axis=0)
    return float(np.linalg.norm(hi - lo))


def run_sparse(survey: Survey, opts: SfmOptions | None = None, scale_tags: bool = True) -> dict:
    t0 = time.time()
    qc = run_qc(survey)
    rec, info = run_sfm(survey, opts)
    agree = device_agreement(survey, rec)
    agree.pop("_fit", None)
    geo = georegister(rec, survey.poses())

    # The reconstruction is now in the output frame; overwrite sparse/ so dense inherits it.
    rec.write(survey.sparse)
    rec.export_PLY(str(survey.model / "model.ply"))
    survey.write_json(
        "cameras.json",
        [{"image": im.name, "centre_m": list(map(float, im.projection_center()))}
         for im in rec.images.values() if im.has_pose],
    )

    track = survey.poses()
    report = {
        "sitemesh_version": __version__,
        "survey": survey.root.name,
        "created_at": time.time(),
        **{k: v for k, v in info.items() if k != "options"},
        "sfm_options": info["options"],
        **{k: v for k, v in agree.items() if k.startswith("device_rotation")},
        **geo,
        "scene_diagonal_m": scene_diagonal_m(rec) if geo.get("scale_source") else None,
        "qc": qc.get("summary"),
        "gps_scale_check": gps_scale_check(track),
    }
    if track is not None:
        survey.write_json("stance_vs_registration.json", post_sfm_stance(track, rec, qc))
    if scale_tags:
        report["scale_check"] = scale_check(rec, survey.raw)
    report["gate"] = evaluate(report)
    report["seconds_total"] = time.time() - t0
    survey.write_json("report.json", report)
    try:
        run_measure(survey)  # bulk measures: measures.json / measures.md
    except (ValueError, RuntimeError, OSError, IndexError, np.linalg.LinAlgError) as exc:
        # measures must never sink a finished reconstruction
        log.warning("bulk measures failed: %s", exc)
    return report


def run_dense_stage(survey: Survey, **kwargs) -> dict:
    import pycolmap

    rec = pycolmap.Reconstruction(survey.sparse)
    result = run_dense(survey, rec, **kwargs)
    report_path = survey.model / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["dense"] = result
    survey.write_json("report.json", report)
    return result
