"""Sparse reconstruction that uses the ARKit data instead of ignoring it.

hydra-vision ran COLMAP blind (AUTO cameras, exhaustive matching) and only afterwards fitted one
similarity to the VIO centres. Here the pose track goes *into* the reconstruction:

1. **Known intrinsics.** One camera per lens/format, seeded from ARKit's per-frame pinhole
   intrinsics with a prior focal length. Never one self-calibrated camera per image — that is
   what collapsed dense fusion on hydra-vision video surveys.
2. **Position priors.** Every tracked shot's VIO centre is written as a COLMAP pose prior, and the
   mapper runs with `use_prior_position`, so bundle adjustment pulls toward metric, gravity-aligned
   positions and the output is already in the device frame — no post-hoc gauge.
3. **Pose-guided matching.** Spatial matching over the priors (k nearest viewpoints) instead of
   O(n^2) exhaustive; exhaustive stays available for small sets or when priors are missing.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pycolmap

from sitemesh.contracts import DevicePoseFrame, Survey
from sitemesh.geometry import arkit_world_from_colmap_cam, rotation_angle_deg

log = logging.getLogger(__name__)


@dataclass
class SfmOptions:
    max_image_size: int = 2400
    matcher: str = "auto"  # auto | spatial | exhaustive
    spatial_neighbors: int = 40
    exhaustive_below: int = 0  # use exhaustive when n_images <= this (0 = never in auto)
    prior_sigma_m: float = 0.05  # VIO position std-dev handed to bundle adjustment
    use_priors: bool = True
    camera_model: str = "SIMPLE_RADIAL"
    focal_group_tol: float = 0.01  # ARKit fx varies with focus; group within 1%
    seed: int = 0
    num_threads: int = -1


def _intrinsics_groups(survey: Survey, frames: dict[str, DevicePoseFrame], tol: float):
    """Group images by lens: (rounded fx, cx, cy) from ARKit. Images without intrinsics → None."""
    groups: dict[object, list[str]] = defaultdict(list)
    keys: list[tuple[float, float, float]] = []
    for name in survey.images():
        f = frames.get(name)
        k = f.intrinsics if f else None
        if k is None:
            groups[None].append(name)
            continue
        for key in keys:
            if abs(key[0] - k.fx) / key[0] < tol and abs(key[1] - k.cx) < 8 and abs(key[2] - k.cy) < 8:
                groups[key].append(name)
                break
        else:
            key = (k.fx, k.cx, k.cy)
            keys.append(key)
            groups[key].append(name)
    out = {}
    for key, names in groups.items():
        if key is None:
            out[None] = names
            continue
        fxs = [frames[n].intrinsics.fx for n in names]
        cxs = [frames[n].intrinsics.cx for n in names]
        cys = [frames[n].intrinsics.cy for n in names]
        out[(float(np.median(fxs)), float(np.median(cxs)), float(np.median(cys)))] = names
    return out


def extract(survey: Survey, opts: SfmOptions) -> dict:
    survey.model.mkdir(parents=True, exist_ok=True)
    if survey.database.exists():
        survey.database.unlink()
    track = survey.poses()
    frames = {f.image: f for f in track.shots()} if track else {}

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = opts.max_image_size
    ext.num_threads = opts.num_threads

    groups = _intrinsics_groups(survey, frames, opts.focal_group_tol)
    cameras = []
    for key, names in groups.items():
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = opts.camera_model
        if key is not None:
            fx, cx, cy = key
            if opts.camera_model == "SIMPLE_RADIAL":
                reader.camera_params = f"{fx},{cx},{cy},0"
            elif opts.camera_model == "PINHOLE":
                reader.camera_params = f"{fx},{fx},{cx},{cy}"
            elif opts.camera_model == "OPENCV":
                reader.camera_params = f"{fx},{fx},{cx},{cy},0,0,0,0"
            else:
                raise ValueError(f"unsupported camera model {opts.camera_model}")
            mode = pycolmap.CameraMode.SINGLE
        else:
            # No ARKit intrinsics: still one camera per folder, never per image. A frame set from
            # one lens self-calibrating per frame is the failure hydra-vision measured (20 → 20).
            mode = pycolmap.CameraMode.SINGLE
        pycolmap.extract_features(
            survey.database,
            survey.raw,
            image_names=names,
            camera_mode=mode,
            reader_options=reader,
            extraction_options=ext,
        )
        cameras.append({"intrinsics": key, "n_images": len(names)})

    # Mark ARKit focal lengths as priors so BA starts from — and is judged against — them.
    db = pycolmap.Database.open(survey.database)
    try:
        for cam in db.read_all_cameras():
            if any(c["intrinsics"] is not None for c in cameras):
                cam.has_prior_focal_length = True
                db.update_camera(cam)
    finally:
        db.close()
    return {"n_images": len(survey.images()), "cameras": cameras}


def write_priors(survey: Survey, opts: SfmOptions) -> int:
    """VIO centres → COLMAP pose priors (CARTESIAN, metres, device world frame)."""
    track = survey.poses()
    if track is None or not opts.use_priors:
        return 0
    frames = {f.image: f for f in track.shots() if f.tracking_state == "normal"}
    cov = np.eye(3) * opts.prior_sigma_m**2
    n = 0
    db = pycolmap.Database.open(survey.database)
    try:
        db.clear_pose_priors()
        for image in db.read_all_images():
            f = frames.get(image.name)
            if f is None:
                continue
            prior = pycolmap.PosePrior()
            prior.corr_data_id = image.data_id
            prior.position = np.array(f.position_m, float)
            prior.position_covariance = cov
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            db.write_pose_prior(prior)
            n += 1
    finally:
        db.close()
    return n


def _match_unprioritized(survey: Survey) -> int:
    """Pair every image that has no pose prior with every other image.

    Spatial matching only pairs images that have a prior position. Shots ARKit marked
    `limited` get no prior (their VIO centre is not trusted), so spatial matching gave them zero
    pairs and they could never register: object-orbit-2 stopped at 83/85 with the mapper log
    reporting "connected 83". Matching them exhaustively costs at most n_missing * n pairs.
    """
    db = pycolmap.Database.open(survey.database)
    try:
        images = db.read_all_images()
        with_prior = {p.corr_data_id.id for p in db.read_all_pose_priors()}
    finally:
        db.close()
    missing = [im for im in images if im.data_id.id not in with_prior]
    if not missing:
        return 0
    pairs = set()
    for a in missing:
        for b in images:
            if a.image_id != b.image_id:
                pairs.add(tuple(sorted((a.name, b.name))))
    path = survey.model / "pairs_unprioritized.txt"
    path.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    imported = pycolmap.ImportedPairingOptions()
    imported.match_list_path = str(path)
    pycolmap.match_image_pairs(survey.database, pairing_options=imported)
    log.info("matched %d images without pose priors exhaustively (%d pairs)", len(missing), len(pairs))
    return len(missing)


def match(survey: Survey, opts: SfmOptions, n_priors: int) -> str:
    n = len(survey.images())
    method = opts.matcher
    if method == "auto":
        method = "spatial" if n_priors >= 3 and n > opts.exhaustive_below else "exhaustive"
    if method == "spatial":
        pairing = pycolmap.SpatialPairingOptions()
        pairing.max_num_neighbors = min(opts.spatial_neighbors, max(n - 1, 1))
        # Neighbour count is the real limit. A huge radius (1e6) aborts COLMAP 4.1.1 with
        # std::out_of_range, so keep a finite one that still covers any walkable site.
        pairing.max_distance = 500.0
        # Default ignore_z=True assumes Z is vertical (GPS). ARKit's vertical is Y, so ignoring Z
        # would throw away a horizontal axis. Keep all three.
        pairing.ignore_z = False
        pycolmap.match_spatial(survey.database, pairing_options=pairing)
        _match_unprioritized(survey)
    else:
        pycolmap.match_exhaustive(survey.database)
    return method


def map_sparse(survey: Survey, opts: SfmOptions, n_priors: int):
    survey.sparse.mkdir(parents=True, exist_ok=True)
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.random_seed = opts.seed
    mapper.num_threads = opts.num_threads
    mapper.use_prior_position = n_priors >= 3
    maps = pycolmap.incremental_mapping(survey.database, survey.raw, survey.sparse, mapper)
    if not maps:
        raise RuntimeError("SfM produced no reconstruction")
    _, rec = max(maps.items(), key=lambda kv: kv[1].num_reg_images())
    # incremental_mapping writes each model to sparse/<idx>; keep the best at sparse/ itself.
    rec.write(survey.sparse)
    return rec, len(maps)


def device_agreement(survey: Survey, rec) -> dict:
    """How well the reconstruction agrees with VIO: position RMS (after a fit) and rotation RMS.

    The rotation check is also the guard on the ARKit→COLMAP axis convention: a wrong convention
    shows up here as a large constant angle, not as a silently bad mesh.
    """
    from sitemesh.geometry import umeyama

    track = survey.poses()
    if track is None:
        return {}
    frames = {f.image: f for f in track.shots() if f.tracking_state == "normal"}
    names, src, dst, r_sfm, r_dev = [], [], [], [], []
    for im in rec.images.values():
        f = frames.get(im.name)
        if f is None or not im.has_pose:
            continue
        names.append(im.name)
        src.append(np.array(im.projection_center()))
        dst.append(np.array(f.position_m))
        r_sfm.append(im.cam_from_world().rotation.matrix().T)  # world_from_cam
        r_dev.append(arkit_world_from_colmap_cam(f.rotation_xyzw))
    if len(src) < 3:
        return {"n_pose_matched": len(src)}
    src, dst = np.array(src), np.array(dst)
    s, r, t = umeyama(src, dst)
    res = np.linalg.norm(s * (r @ src.T).T + t - dst, axis=1)
    rot_err = [rotation_angle_deg(r @ a, b) for a, b in zip(r_sfm, r_dev)]
    return {
        "n_pose_matched": len(src),
        "sfm_to_device_scale": float(s),
        "device_alignment_rms_m": float(np.sqrt((res**2).mean())),
        "device_alignment_max_m": float(res.max()),
        "device_rotation_rms_deg": float(np.sqrt(np.mean(np.square(rot_err)))),
        "device_rotation_max_deg": float(np.max(rot_err)),
        "_fit": (s, r, t),
    }


def run_sfm(survey: Survey, opts: SfmOptions | None = None) -> tuple[object, dict]:
    opts = opts or SfmOptions()
    t0 = time.time()
    info = {"options": vars(opts).copy()}
    info["extract"] = extract(survey, opts)
    t1 = time.time()
    info["n_priors"] = write_priors(survey, opts)
    info["matcher"] = match(survey, opts, info["n_priors"])
    t2 = time.time()
    rec, n_models = map_sparse(survey, opts, info["n_priors"])
    t3 = time.time()
    cams = []
    for cam in rec.cameras.values():
        cams.append({"model": cam.model_name, "params": list(map(float, cam.params))})
    info.update(
        n_models=n_models,
        n_photos=len(survey.images()),
        n_registered=rec.num_reg_images(),
        n_points3d=rec.num_points3D(),
        mean_reprojection_px=float(rec.compute_mean_reprojection_error()),
        refined_cameras=cams,
        seconds={"extract": t1 - t0, "match": t2 - t1, "map": t3 - t2},
    )
    return rec, info
