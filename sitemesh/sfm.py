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
    # Feature/matching knobs, COLMAP defaults unless a measurement says otherwise.
    sift_max_features: int = 8192
    # estimate_affine_shape: on object-orbit, IMG_0027-0030 (blurred, oblique) only register with
    # it: 40/44 -> 44/44 on seeds 0-2, 0.873 -> 0.94 px. Cost: extraction 26 s -> 48 s on the
    # 85-shot object-orbit-2 (M6 CPU), ~unchanged on object-orbit.
    # object-orbit-2 stays 85/85. Measurements in docs/results.md.
    sift_affine: bool = True
    sift_dsp: bool = False  # domain_size_pooling
    guided_matching: bool = False
    # Merge disconnected sub-models through the shared ARKit frame (needs pose priors). A segment
    # whose own VIO fit is worse than max_segment_rms_m is left out, with the reason reported.
    merge_submodels: bool = True
    max_segment_rms_m: float = 0.25
    min_segment_images: int = 3


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
    ext.sift.max_num_features = opts.sift_max_features
    ext.sift.estimate_affine_shape = opts.sift_affine
    ext.sift.domain_size_pooling = opts.sift_dsp

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


def _matching_options(opts: SfmOptions):
    m = pycolmap.FeatureMatchingOptions()
    m.num_threads = opts.num_threads
    m.guided_matching = opts.guided_matching
    return m


def spatial_pairs(
    names: list[str],
    positions: dict[str, np.ndarray],
    k: int,
    max_distance_m: float = 500.0,
) -> set[tuple[str, str]]:
    """Image pairs to match: k nearest prior positions, plus all pairs for images without one.

    Replaces COLMAP 4.1.1's spatial pair generator, which aborted the process with
    `std::out_of_range: vector` in about half of full pipeline runs (exit 134 in 3 of 6 identical
    object-orbit runs), even after max_distance was cut from 1e6 to 500. The same neighbour rule
    in numpy is deterministic and cheap: n is at most a few hundred stills.

    Images with no prior (ARKit `limited` tracking: VIO centre not trusted) are paired with every
    other image. COLMAP's generator gave them zero pairs, so they could never register
    (object-orbit-2 stopped at 83/85, mapper log "connected 83").
    """
    pairs: set[tuple[str, str]] = set()
    with_pos = [n for n in names if n in positions]
    if with_pos:
        xyz = np.array([positions[n] for n in with_pos], float)
        dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        kk = min(k, len(with_pos) - 1)
        for i, name in enumerate(with_pos):
            for j in np.argsort(dist[i], kind="stable")[:kk]:
                if dist[i, j] <= max_distance_m:
                    pairs.add(tuple(sorted((name, with_pos[j]))))
    for a in names:
        if a in positions:
            continue
        for b in names:
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def match(survey: Survey, opts: SfmOptions, n_priors: int) -> str:
    n = len(survey.images())
    method = opts.matcher
    if method == "auto":
        method = "spatial" if n_priors >= 3 and n > opts.exhaustive_below else "exhaustive"
    if method == "spatial":
        # Pairs from the priors actually written to the database, in ARKit's frame. All three
        # axes count: ARKit's vertical is Y, so a GPS-style "ignore Z" would drop a horizontal.
        db = pycolmap.Database.open(survey.database)
        try:
            images = db.read_all_images()
            by_data = {im.data_id.id: im.name for im in images}
            positions = {
                by_data[p.corr_data_id.id]: np.asarray(p.position, float)
                for p in db.read_all_pose_priors()
            }
        finally:
            db.close()
        names = sorted(im.name for im in images)
        pairs = spatial_pairs(names, positions, opts.spatial_neighbors)
        path = survey.model / "pairs.txt"
        path.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
        imported = pycolmap.ImportedPairingOptions()
        imported.match_list_path = str(path)
        pycolmap.match_image_pairs(
            survey.database, matching_options=_matching_options(opts), pairing_options=imported
        )
        log.info(
            "spatial matching: %d pairs, %d images without a prior paired exhaustively",
            len(pairs), len(names) - len(positions),
        )
    else:
        pycolmap.match_exhaustive(survey.database, matching_options=_matching_options(opts))
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
    return rec, maps


def _vio_fit(rec, frames: dict):
    """Similarity from a reconstruction's camera centres to the VIO centres, and its residual."""
    from sitemesh.geometry import umeyama

    src, dst = [], []
    for im in rec.images.values():
        f = frames.get(im.name)
        if f is not None and im.has_pose:
            src.append(np.array(im.projection_center()))
            dst.append(np.array(f.position_m))
    if len(src) < 3:
        return None
    src, dst = np.array(src), np.array(dst)
    s, r, t = umeyama(src, dst)
    res = np.linalg.norm(s * (r @ src.T).T + t - dst, axis=1)
    return s, r, t, float(np.sqrt((res**2).mean())), len(src)


def _combine(segments):
    """One reconstruction holding every segment's cameras, rigs, posed frames and images.

    Images keep their keypoints but drop their 3D links; points are re-triangulated afterwards.
    Database ids are shared across segments, so they carry over unchanged.
    """
    import copy

    merged = pycolmap.Reconstruction()
    for seg in segments:
        for cid, cam in seg.cameras.items():
            if not merged.exists_camera(cid):
                merged.add_camera(cam)
        for rid, rig in seg.rigs.items():
            if not merged.exists_rig(rid):
                merged.add_rig(rig)
        for fid, frame in seg.frames.items():
            if merged.exists_frame(fid) or not frame.has_pose():  # Frame.has_pose is a method
                continue
            f = copy.copy(frame)
            f.reset_rig_ptr()
            merged.add_frame(f)
        for iid, image in seg.images.items():
            if merged.exists_image(iid) or not image.has_pose:
                continue
            im = copy.copy(image)
            im.reset_camera_ptr()
            im.reset_frame_ptr()
            for p in im.points2D:
                p.point3D_id = pycolmap.INVALID_POINT3D_ID
            merged.add_image(im)
    return merged


def merge_submodels(survey: Survey, maps: dict, opts: SfmOptions):
    """Merge disconnected sub-models through the ARKit frame they were all solved in.

    incremental_mapping splits a capture wherever consecutive shots share too few features.
    sitemesh used to keep only the largest piece: d4737a5d reconstructed 80/86 shots in three
    pieces but kept 53, and b8b8ebd7 kept 22 of 47. Every piece was solved with the same metric
    VIO pose priors, so each is aligned to VIO on its own (a similarity, which also exposes a
    piece that disagrees with VIO), then all are combined. Points are re-triangulated from the
    database with poses fixed, so any matches across the joins become shared points, and a
    pose-prior bundle adjustment settles the joins against images and VIO together.

    Returns (reconstruction or None, report). None means nothing to merge: keep the largest.
    """
    track = survey.poses()
    report: dict = {"n_submodels": len(maps), "segments": [], "merged": False}
    if track is None or len(maps) < 2:
        return None, report
    frames = {f.image: f for f in track.shots() if f.tracking_state == "normal"}
    kept = []
    for idx, seg in sorted(maps.items(), key=lambda kv: -kv[1].num_reg_images()):
        row = {"submodel": int(idx), "n_images": seg.num_reg_images()}
        fit = _vio_fit(seg, frames)
        if seg.num_reg_images() < opts.min_segment_images or fit is None:
            row.update(kept=False, reason=f"fewer than {opts.min_segment_images} images with VIO")
        else:
            s, r, t, rms, n = fit
            row.update(vio_rms_m=rms, sfm_to_device_scale=float(s), n_vio=n)
            if rms > opts.max_segment_rms_m:
                row.update(kept=False, reason=f"VIO RMS {rms:.3f} m > {opts.max_segment_rms_m} m")
            else:
                seg = pycolmap.Reconstruction(seg)
                seg.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(r), t))
                kept.append(seg)
                row["kept"] = True
        report["segments"].append(row)
    if len(kept) < 2:
        return None, report

    import tempfile

    combined = _combine(kept)
    seg_of = {}
    for i, seg in enumerate(kept):
        for im in seg.images.values():
            if im.has_pose:
                seg_of.setdefault(im.image_id, i)
    with tempfile.TemporaryDirectory() as tmp:
        rec = pycolmap.triangulate_points(combined, survey.database, survey.raw, tmp)

    db = pycolmap.Database.open(survey.database)
    try:
        priors = db.read_all_pose_priors()
    finally:
        db.close()
    cfg = pycolmap.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        cfg.add_image(iid)
    ba = pycolmap.create_pose_prior_bundle_adjuster(
        pycolmap.BundleAdjustmentOptions(), pycolmap.PosePriorBundleAdjustmentOptions(),
        cfg, priors, rec,
    )
    ba.solve()
    rec.update_point_3d_errors()  # BA does not refresh the stored per-point errors

    # Points observed from more than one segment: real feature links across a join.
    joint = sum(
        1 for p in rec.points3D.values()
        if len({seg_of.get(el.image_id) for el in p.track.elements}) > 1
    )
    fit = _vio_fit(rec, frames)
    report.update(
        merged=True,
        n_segments_merged=len(kept),
        n_registered_largest=max(seg.num_reg_images() for seg in kept),
        n_registered_merged=rec.num_reg_images(),
        n_joint_points=joint,
        merged_vio_rms_m=fit[3] if fit else None,
    )
    log.info("merged %d sub-models: %d -> %d images, %d points across joins",
             len(kept), report["n_registered_largest"], rec.num_reg_images(), joint)
    return rec, report


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
    rec, maps = map_sparse(survey, opts, info["n_priors"])
    n_models = len(maps)
    t3 = time.time()
    if opts.merge_submodels and info["n_priors"] >= 3:
        merged, info["merge"] = merge_submodels(survey, maps, opts)
        if merged is not None:
            rec = merged
            rec.write(survey.sparse)
    t4 = time.time()
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
        seconds={"extract": t1 - t0, "match": t2 - t1, "map": t3 - t2, "merge": t4 - t3},
    )
    return rec, info
