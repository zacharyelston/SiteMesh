"""sitemesh qc | sparse | dense | run | scalecheck  <survey-folder>"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from sitemesh.contracts import Survey
from sitemesh.sfm import SfmOptions


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sitemesh", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, help_ in [
        ("qc", "stance-card check from poses.json alone (seconds; run on site)"),
        ("sparse", "QC + prior-guided SfM + georegister + scale check + gate"),
        ("dense", "patch-match + fusion + Poisson on an existing sparse model (CUDA)"),
        ("run", "sparse, then dense if CUDA is available"),
        ("scalecheck", "re-run the AprilTag scale check on an existing sparse model"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("survey")
        if name in ("sparse", "run"):
            p.add_argument("--matcher", default="auto", choices=["auto", "spatial", "exhaustive"])
            p.add_argument("--no-priors", action="store_true", help="ignore poses.json in SfM")
            p.add_argument("--max-image-size", type=int, default=2400)
            p.add_argument("--neighbors", type=int, default=40)
            p.add_argument("--prior-sigma-m", type=float, default=0.05)
            p.add_argument("--camera-model", default="SIMPLE_RADIAL",
                           choices=["SIMPLE_RADIAL", "PINHOLE", "OPENCV"])
            p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    survey = Survey(args.survey)

    if args.cmd == "qc":
        from sitemesh.pipeline import run_qc

        result = run_qc(survey)
        _print(result["summary"])
        return 0

    if args.cmd in ("sparse", "run"):
        from sitemesh.pipeline import run_dense_stage, run_sparse

        opts = SfmOptions(
            max_image_size=args.max_image_size,
            matcher=args.matcher,
            spatial_neighbors=args.neighbors,
            prior_sigma_m=args.prior_sigma_m,
            use_priors=not args.no_priors,
            camera_model=args.camera_model,
            seed=args.seed,
        )
        report = run_sparse(survey, opts)
        _print({k: report.get(k) for k in (
            "n_photos", "n_registered", "n_points3d", "mean_reprojection_px", "matcher",
            "device_alignment_rms_m", "device_rotation_rms_deg", "output_frame", "gate")})
        if args.cmd == "run":
            _print(run_dense_stage(survey))
        return 0 if report["gate"]["verdict"] != "fail" else 2

    if args.cmd == "dense":
        from sitemesh.pipeline import run_dense_stage

        _print(run_dense_stage(survey))
        return 0

    if args.cmd == "scalecheck":
        import pycolmap

        from sitemesh.scalecheck import scale_check

        _print(scale_check(pycolmap.Reconstruction(survey.sparse), survey.raw))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
