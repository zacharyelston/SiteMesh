"""Experimental CPU dense-lite: OpenCV SGBM on rectified, pose-known image pairs (issue #18).

COLMAP patch-match is CUDA-only, and the field agent is an Apple M6. This tries the cheapest
dense path that runs anywhere. For each registered image it picks a partner with a sideways
baseline from the sparse model's co-visibility, rectifies the pair with the known poses, runs
semi-global matching, lifts the disparities into the output frame, and checks each pair against
the sparse points it can see (those are accurate). Pairs that disagree are dropped, and only
voxels supported by at least `--min-support` pairs are kept.

    python scripts/dense_cpu.py data/object-orbit-2 [--scale 0.5] [--voxel 0.02]

Writes <survey>/model/dense_cpu/fused.ply and prints a JSON summary. Not a pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from sitemesh.contracts import Survey


def intr(cam, scale):
    f, cx, cy = cam.params[0] * scale, cam.params[1] * scale, cam.params[2] * scale
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    k1 = cam.params[3] if len(cam.params) > 3 else 0.0
    return k, np.array([k1, 0, 0, 0], float)


def pose(im):
    p = im.cam_from_world()
    return p.rotation.matrix(), np.asarray(p.translation, float)


def pick_pairs(rec, min_ratio, max_ratio, max_angle, min_shared, k=3):
    obs = defaultdict(set)
    for pid, p in rec.points3D.items():
        for el in p.track.elements:
            obs[el.image_id].add(pid)
    ims = {i: rec.images[i] for i in rec.reg_image_ids()}
    centre = {i: np.asarray(im.projection_center()) for i, im in ims.items()}
    axis = {i: pose(im)[0][2] for i, im in ims.items()}
    depth = {}
    for i, im in ims.items():
        r, t = pose(im)
        z = [(r @ rec.points3D[p].xyz + t)[2] for p in obs[i]]
        depth[i] = float(np.median(z)) if z else None
    pairs = []
    for i in ims:
        if not depth[i]:
            continue
        cands = []
        for j in ims:
            if j == i:
                continue
            b = centre[j] - centre[i]
            bl = np.linalg.norm(b)
            if bl < 1e-6:
                continue
            ratio = bl / depth[i]
            ang = np.degrees(np.arccos(np.clip(axis[i] @ axis[j], -1, 1)))
            side = abs(b @ axis[i]) / bl  # 0 = sideways, 1 = along the view (bad for rectification)
            shared = len(obs[i] & obs[j])
            if not (min_ratio <= ratio <= max_ratio and ang <= max_angle and side < 0.6
                    and shared >= min_shared):
                continue
            cands.append((shared * (1 - side), j))
        for _, j in sorted(cands, reverse=True)[:k]:
            pairs.append((i, j))
    return pairs, obs, depth


def run_pair(rec, survey, i, j, obs, scale, max_disp_cap, swapped=False):
    a, b = rec.images[i], rec.images[j]
    ka, da = intr(rec.cameras[a.camera_id], scale)
    kb, db = intr(rec.cameras[b.camera_id], scale)
    ia = cv2.imread(str(survey.raw / a.name))
    ib = cv2.imread(str(survey.raw / b.name))
    if scale != 1.0:
        ia = cv2.resize(ia, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ib = cv2.resize(ib, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    size = (ia.shape[1], ia.shape[0])
    ra, ta = pose(a)
    rb, tb = pose(b)
    r_rel = rb @ ra.T
    t_rel = tb - r_rel @ ta
    r1, r2, p1, p2, _q, _, _ = cv2.stereoRectify(ka, da, kb, db, size, r_rel, t_rel.reshape(3, 1), alpha=0,
                                                flags=cv2.CALIB_ZERO_DISPARITY)
    m1 = cv2.initUndistortRectifyMap(ka, da, r1, p1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(kb, db, r2, p2, size, cv2.CV_32FC1)
    ga = cv2.remap(ia, *m1, cv2.INTER_LINEAR)
    gb = cv2.remap(ib, *m2, cv2.INTER_LINEAR)
    # disparity range from the sparse depths seen in image a
    # horizontal or vertical rectification: OpenCV puts the baseline on x when |tx| > |ty|
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    f = p1[0, 0]
    base = abs((p2[1, 3] if vertical else p2[0, 3]) / f)
    z = [(ra @ rec.points3D[p].xyz + ta)[2] for p in obs[i]]
    zmin = max(np.percentile(z, 2) * 0.7, 0.2)
    ndisp = int(np.ceil(min(f * base / zmin, max_disp_cap) / 16) * 16)
    if ndisp < 16:
        return None
    if vertical:
        ga, gb = cv2.rotate(ga, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(gb, cv2.ROTATE_90_CLOCKWISE)
    # OpenCV: P2[k,3] = -f*B when the partner sits at +x (or +y). After a clockwise rotation, +y
    # becomes -x. Image a must be the left view; otherwise run the pair the other way round.
    t2 = p2[1, 3] if vertical else p2[0, 3]
    a_is_left = (t2 > 0) if vertical else (t2 < 0)
    if not a_is_left:
        if swapped:
            return None
        return run_pair(rec, survey, j, i, obs, scale, max_disp_cap, swapped=True)
    left, right = ga, gb
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=ndisp, blockSize=5,
                                 P1=8 * 3 * 25, P2=32 * 3 * 25, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
    if vertical:
        disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        left = cv2.rotate(left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    valid = disp > 0.5
    zr = np.zeros_like(disp)
    zr[valid] = f * base / disp[valid]  # depth in the rectified frame of a
    # Keep depths inside what the sparse points show for this view (+-20%). SGBM fills
    # untextured pixels with confident-looking garbage; it shows up as streaks along view rays.
    zlo, zhi = np.percentile(z, 2) * 0.8, np.percentile(z, 98) * 1.2
    valid &= (zr > zlo) & (zr < zhi)
    h, w = disp.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = p1[0, 2], p1[1, 2]
    x = (uu - cx) * zr / f
    y = (vv - cy) * zr / f
    rect = np.stack([x, y, zr], -1)[valid]
    cam = rect @ r1  # rect = r1 @ cam  ->  cam = r1.T @ rect (row vectors)
    world = (cam - ta) @ ra  # cam = ra @ w + ta  ->  w = ra.T @ (cam - ta)
    colours = cv2.remap(ia, *m1, cv2.INTER_LINEAR)[valid][:, ::-1]

    # validate against sparse points visible in a: project into the rectified view of a
    errs = []
    for p in obs[i]:
        xc = ra @ rec.points3D[p].xyz + ta
        xr = r1 @ xc
        if xr[2] <= 0:
            continue
        u, v = round(f * xr[0] / xr[2] + cx), round(f * xr[1] / xr[2] + cy)
        if 0 <= u < w and 0 <= v < h and valid[v, u]:
            errs.append(abs(zr[v, u] - xr[2]) / xr[2])
    return {
        "world": world.astype(np.float32), "colours": colours.astype(np.uint8),
        "n": int(valid.sum()), "coverage": float(valid.mean()),
        "n_check": len(errs), "median_rel_err": float(np.median(errs)) if errs else None,
        "vertical": bool(vertical), "ref": int(i), "ndisp": ndisp,
    }


def read_ply(path) -> np.ndarray:
    """XYZ from a binary PLY written by write_ply (or COLMAP's export_PLY)."""
    with open(path, "rb") as fh:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += fh.readline()
        lines = header.split(b"\n")
        n = int(next(x for x in lines if x.startswith(b"element vertex")).split()[-1])
        props = [x.split()[-1].decode() for x in lines if x.startswith(b"property")]
        dt = [(k, "<f4") if k in ("x", "y", "z") else (k, "u1") for k in props]
        a = np.frombuffer(fh.read(n * np.dtype(dt).itemsize), dtype=dt)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(float)


def write_ply(path, pts, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                  ).encode())
        rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        fh.write(rec.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("survey", type=Path)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--max-rel-err", type=float, default=0.03, help="drop pairs worse than this")
    ap.add_argument("--min-ratio", type=float, default=0.03)
    ap.add_argument("--max-ratio", type=float, default=0.15)
    ap.add_argument("--max-angle", type=float, default=25.0)
    ap.add_argument("--max-disp", type=int, default=384)
    ap.add_argument("--partners", type=int, default=3, help="partners per reference image")
    args = ap.parse_args()
    s = Survey(args.survey)
    rec = pycolmap.Reconstruction(s.sparse)
    t0 = time.time()
    pairs, obs, _ = pick_pairs(rec, args.min_ratio, args.max_ratio, args.max_angle, 30,
                               args.partners)
    results, rows = [], []
    for i, j in pairs:
        r = run_pair(rec, s, i, j, obs, args.scale, args.max_disp)
        if r is None:
            rows.append({"pair": [i, j], "status": "skipped (no usable disparity range)"})
            continue
        ok = r["median_rel_err"] is not None and r["median_rel_err"] <= args.max_rel_err
        rows.append({"pair": [i, j], "ref": r["ref"], "vertical": r["vertical"], "n": r["n"],
                     "coverage": r["coverage"],
                     "median_rel_err": r["median_rel_err"], "n_check": r["n_check"], "kept": ok})
        if ok:
            results.append(r)
    t1 = time.time()
    if not results:
        print(json.dumps({"status": "no pair passed validation", "pairs": rows}, indent=1))
        return 1
    pts = np.concatenate([r["world"] for r in results])
    cols = np.concatenate([r["colours"] for r in results])
    # support counts distinct reference views, so two pairs sharing a view can't confirm each other
    src = np.concatenate([np.full(len(r["world"]), r["ref"]) for r in results])
    key = np.floor(pts / args.voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    # support = number of distinct pairs per voxel
    pair_vox = np.unique(np.stack([inv, src], 1), axis=0)
    support = np.bincount(pair_vox[:, 0])
    keep_vox = support >= args.min_support
    n_vox = len(support)
    sums = np.zeros((n_vox, 3))
    np.add.at(sums, inv, pts)
    csum = np.zeros((n_vox, 3))
    np.add.at(csum, inv, cols.astype(float))
    cnt = np.bincount(inv, minlength=n_vox)[:, None]
    fused = (sums / cnt)[keep_vox]
    fcol = (csum / cnt)[keep_vox].astype(np.uint8)
    out = s.model / "dense_cpu" / "fused.ply"
    write_ply(out, fused, fcol)
    t2 = time.time()
    errs = [r["median_rel_err"] for r in results]
    print(json.dumps({
        "status": "ok", "pairs_tried": len(pairs), "pairs_kept": len(results),
        "pair_median_rel_err": {"median": float(np.median(errs)), "max": float(np.max(errs))},
        "raw_points": len(pts), "voxels": int(n_vox),
        "fused_points": len(fused), "sparse_points": rec.num_points3D(),
        "voxel_m": args.voxel, "min_support": args.min_support, "scale": args.scale,
        "seconds": {"pairs": t1 - t0, "fuse": t2 - t1}, "out": str(out),
        "pairs": rows,
    }, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
