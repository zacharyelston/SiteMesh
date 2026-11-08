"""Dense reconstruction and meshing.

Runs on the already-georegistered sparse model, so the fused cloud and the mesh come out in the
output frame (metres, Z up / ENU) with no refit.

COLMAP's patch-match stereo is CUDA-only — `pycolmap.has_cuda` is False on Apple Silicon — so this
stage runs on a CUDA host (the hydra-v2 GPU nodes, or any NVIDIA box with a CUDA pycolmap build).
On a Mac, use tools/objectcapture for an on-site preview mesh; it is in its own frame, not ours.

Lessons carried over from hydra-vision:
  * explicit depth bounds per run, from the sparse points — a blank view otherwise aborts patch-match;
  * geometric fusion for stills; frame sets only fuse geometrically when they share one camera,
    which sfm.py now guarantees;
  * Poisson output carries a float `value` between position and colour, and its winding needs
    both sides rendered in viewers that cull back faces.
"""

from __future__ import annotations

import time

import numpy as np
import pycolmap

from sitemesh.contracts import Survey


def depth_bounds(rec, lo_pct: float = 1.0, hi_pct: float = 99.0, margin: float = 0.5):
    """Scene depth range across all registered views, in model units (metres here)."""
    depths = []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        cfw = im.cam_from_world()
        for p in im.points2D:
            if p.has_point3D():
                depths.append((cfw * rec.points3D[p.point3D_id].xyz)[2])
    depths = np.array([d for d in depths if d > 0])
    if len(depths) == 0:
        return None, None
    lo, hi = np.percentile(depths, [lo_pct, hi_pct])
    return float(lo * (1 - margin)), float(hi * (1 + margin))


def run_dense(
    survey: Survey,
    rec,
    max_image_size: int = 2000,
    geom_consistency: bool = True,
    min_num_pixels: int = 5,
    poisson_depth: int = 11,
    poisson_trim: float = 7.0,
) -> dict:
    if not pycolmap.has_cuda:
        return {
            "status": "skipped",
            "reason": "patch-match stereo needs CUDA (pycolmap.has_cuda is False). Run "
            "`sitemesh dense <survey>` on a CUDA host; for a Mac preview use tools/objectcapture.",
        }
    work = survey.model / "densework"
    out = survey.model / "dense"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pycolmap.undistort_images(work, survey.sparse, survey.raw)
    t1 = time.time()

    dmin, dmax = depth_bounds(rec)
    pm = pycolmap.PatchMatchOptions()
    pm.max_image_size = max_image_size
    pm.geom_consistency = geom_consistency
    pm.filter = True
    if dmin is not None:
        pm.depth_min, pm.depth_max = dmin, dmax
    pycolmap.patch_match_stereo(work, options=pm)
    t2 = time.time()

    fusion = pycolmap.StereoFusionOptions()
    fusion.min_num_pixels = min_num_pixels
    fused = out / "fused.ply"
    cloud = pycolmap.stereo_fusion(
        fused, work, input_type="geometric" if geom_consistency else "photometric",
        options=fusion, output_type="ply",
    )
    t3 = time.time()

    mesh = out / "meshed-poisson.ply"
    pois = pycolmap.PoissonMeshingOptions()
    pois.depth = poisson_depth
    pois.trim = poisson_trim
    pycolmap.poisson_meshing(fused, mesh, pois)
    t4 = time.time()

    return {
        "status": "ok",
        "depth_bounds_m": [dmin, dmax],
        "n_fused_points": cloud.num_points3D() if cloud is not None else None,
        "fused": str(fused.relative_to(survey.root)),
        "mesh": str(mesh.relative_to(survey.root)),
        "seconds": {"undistort": t1 - t0, "patch_match": t2 - t1, "fusion": t3 - t2,
                    "mesh": t4 - t3},
    }
