"""Experiments on existing captures, no tags and no CUDA (issue #18). Results: docs/experiments.md.

    python scripts/experiments.py timing    <capture> <model>   # VIO time offset sweep
    python scripts/experiments.py drift     <capture> <model>   # is the VIO residual smooth?
    python scripts/experiments.py gps-scale <capture>           # GPS vs VIO scale, outdoors
    python scripts/experiments.py waterline <model>             # is the pool waterline level?
    python scripts/experiments.py corridor  <model>             # corridor width / height slices
    python scripts/experiments.py recover   <survey>            # PnP for never-registered shots

<capture> is data/<name> (raw/ + poses.json). <model> is a sparse model directory. <survey> is a
folder with model/{sparse,database.db}, e.g. bench-gate/<capture>__priors_spatial__s0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

from sitemesh.contracts import Survey
from sitemesh.geometry import (
    DEVICE_TO_ZUP,
    arkit_world_from_colmap_cam,
    rotation_angle_deg,
    umeyama,
    wgs84_to_enu,
)


def _track(capture: Path):
    d = json.loads((capture / "poses.json").read_text())
    fr = sorted(d["frames"], key=lambda f: f["timestamp"])
    t = np.array([f["timestamp"] for f in fr])
    x = np.array([f["position_m"] for f in fr], float)
    stills = {f["image"]: f for f in fr if f.get("image") and f["tracking_state"] == "normal"}
    return d, t, x, stills


def _interp(t, x, q):
    return np.array([np.interp(q, t, x[:, k]) for k in range(3)])


def _centres(model: Path):
    rec = pycolmap.Reconstruction(model)
    return {im.name: np.asarray(im.projection_center()) for im in rec.images.values()
            if im.has_pose}


def _residuals(src, dst):
    s, r, t = umeyama(src, dst)
    return s * (r @ src.T).T + t - dst


def cmd_timing(a):
    _, t, x, stills = _track(a.capture)
    c = _centres(a.model)
    names = [n for n in c if n in stills]
    src = np.array([c[n] for n in names])

    def rms(delta):
        dst = np.array([_interp(t, x, stills[n]["timestamp"] + delta) for n in names])
        return float(np.sqrt((np.linalg.norm(_residuals(src, dst), axis=1) ** 2).mean()))

    speed = [np.linalg.norm(_interp(t, x, stills[n]["timestamp"] + 0.05)
                            - _interp(t, x, stills[n]["timestamp"] - 0.05)) / 0.1 for n in names]
    deltas = np.round(np.arange(-0.3, 0.31, 0.05), 3)
    vals = [rms(d) * 1000 for d in deltas]
    print(json.dumps({
        "track_sample_gap_s": float(np.median(np.diff(t))), "n": len(names),
        "speed_at_shutter_m_s": {"median": float(np.median(speed)),
                                 "p90": float(np.percentile(speed, 90))},
        "rms_mm_at_0": rms(0.0) * 1000, "best_delta_s": float(deltas[int(np.argmin(vals))]),
        "rms_mm_best": min(vals), "curve_mm": dict(zip(map(float, deltas), vals)),
    }, indent=1))


def cmd_drift(a):
    _, t, x, stills = _track(a.capture)
    c = _centres(a.model)
    names = sorted((n for n in c if n in stills), key=lambda n: stills[n]["timestamp"])
    src = np.array([c[n] for n in names])
    dst = np.array([stills[n]["position_m"] for n in names], float)
    res = _residuals(src, dst)

    def lag1(r):
        u = r / np.linalg.norm(r, axis=1, keepdims=True)
        return float(np.mean((u[:-1] * u[1:]).sum(1)))

    rng = np.random.default_rng(0)
    shuffled = [lag1(res[rng.permutation(len(res))]) for _ in range(200)]
    k = 7
    smooth = np.array([res[max(0, i - k // 2):i + k // 2 + 1].mean(0) for i in range(len(res))])

    def rms(r):
        return float(np.sqrt((np.linalg.norm(r, axis=1) ** 2).mean()) * 1000)

    speed = np.array([np.linalg.norm(_interp(t, x, stills[n]["timestamp"] + 0.05)
                                     - _interp(t, x, stills[n]["timestamp"] - 0.05)) / 0.1
                      for n in names])
    mag = np.linalg.norm(res, axis=1)
    rank = lambda v: np.argsort(np.argsort(v))
    print(json.dumps({
        "n": len(names), "rms_mm": rms(res), "rms_mm_after_smooth_drift_removed": rms(res - smooth),
        "lag1_cosine": lag1(res), "lag1_cosine_shuffled": [float(np.mean(shuffled)),
                                                           float(np.std(shuffled))],
        "spearman_speed_vs_residual": float(np.corrcoef(rank(speed), rank(mag))[0, 1]),
    }, indent=1))


def cmd_gps_scale(a):
    d, t, x, _ = _track(a.capture)
    normal = np.array([f["tracking_state"] == "normal"
                       for f in sorted(d["frames"], key=lambda f: f["timestamp"])])
    t, x = t[normal], x[normal]
    g = sorted(d["gps_track"], key=lambda f: f["timestamp"])
    gt = np.array([f["timestamp"] for f in g])
    acc = np.array([f.get("horizontal_accuracy_m") or np.nan for f in g])
    ref = g[0]
    enu = np.array([wgs84_to_enu(f["lat_deg"], f["lon_deg"], f.get("alt_m") or 0.0,
                                 ref["lat_deg"], ref["lon_deg"], ref.get("alt_m") or 0.0)
                    for f in g])[:, :2]
    ok = (gt >= t[0]) & (gt <= t[-1])
    xz = (DEVICE_TO_ZUP @ x.T).T
    dev = np.array([_interp(t, xz, q)[:2] for q in gt[ok]])
    e, w = enu[ok], 1 / np.clip(acc[ok], 1, None) ** 2

    def fit(dv, en, ww):
        za = (dv - (dv * ww[:, None]).sum(0) / ww.sum()) @ [1, 1j]
        zb = (en - (en * ww[:, None]).sum(0) / ww.sum()) @ [1, 1j]
        k = (ww * np.conj(za) * zb).sum() / (ww * np.abs(za) ** 2).sum()
        return float(abs(k)), float(np.degrees(np.angle(k)))

    s, heading = fit(dev, e, w)
    rng = np.random.default_rng(0)
    blocks = [np.arange(i, min(i + 10, len(e))) for i in range(0, len(e), 10)]
    boot = []
    for _ in range(500):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        boot.append(fit(dev[idx], e[idx], w[idx])[0])
    print(json.dumps({
        "n_fixes": int(ok.sum()), "gps_hacc_median_m": float(np.nanmedian(acc[ok])),
        "vio_extent_m": np.ptp(dev, 0).tolist(), "gps_over_vio_scale": s,
        "scale_95ci_block_bootstrap": [float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))],
        "heading_deg": heading,
    }, indent=1))


def cmd_waterline(a):
    import cv2

    rec = pycolmap.Reconstruction(a.model)
    p = np.array([q.xyz for q in rec.points3D.values()])
    col = np.array([q.color for q in rec.points3D.values()], np.uint8)
    hsv = cv2.cvtColor(col.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(int)
    sel = ((hsv[:, 0] >= 82) & (hsv[:, 0] <= 98) & (hsv[:, 1] > 120) & (hsv[:, 2] > 100)
           & (p[:, 2] < a.max_z))
    q = p[sel]
    m = q.mean(0)
    d = np.linalg.svd(q - m)[2][0]
    h = d[:2] / np.linalg.norm(d[:2])
    s = (q[:, :2] - m[:2]) @ h
    (b, c0), cov = np.polyfit(s, q[:, 2], 1, cov=True)
    print(json.dumps({
        "n_points": int(sel.sum()), "length_m": float(np.ptp(s)),
        "slope_deg": float(np.degrees(np.arctan(b))),
        "slope_1sigma_deg": float(np.degrees(np.sqrt(cov[0, 0]))),
        "z_scatter_mm": float((q[:, 2] - (c0 + b * s)).std() * 1000),
    }, indent=1))


def cmd_corridor(a):
    rec = pycolmap.Reconstruction(a.model)
    p = np.array([q.xyz for q in rec.points3D.values()])
    if a.extra_ply:
        sys.path.insert(0, str(Path(__file__).parent))
        from dense_cpu import read_ply

        p = np.vstack([p, read_ply(a.extra_ply)])
    cams = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    m = cams[:, :2].mean(0)
    ax = np.linalg.svd(cams[:, :2] - m)[2][0]
    perp = np.array([-ax[1], ax[0]])
    along, across, z = (p[:, :2] - m) @ ax, (p[:, :2] - m) @ perp, p[:, 2]
    ca = (cams[:, :2] - m) @ ax
    widths, heights = [], []
    for s0 in np.arange(ca.min(), ca.max(), a.slice_m):
        sl = (along >= s0) & (along < s0 + a.slice_m) & (np.abs(across) < 4)
        if sl.sum() < 50:
            continue
        w, zz = across[sl], z[sl]
        heights.append(float(np.percentile(zz, 98) - np.percentile(zz, 2)))
        mid = (zz > np.percentile(zz, 20)) & (zz < np.percentile(zz, 80))
        if (mid & (w < 0)).sum() > 10 and (mid & (w > 0)).sum() > 10:
            widths.append(float(np.percentile(w[mid & (w > 0)], 95)
                                - np.percentile(w[mid & (w < 0)], 5)))
    print(json.dumps({
        "n_points": len(p), "camera_track_length_m": float(np.ptp(ca)), "slices": len(heights),
        "height_m": {"median": float(np.median(heights)),
                     "iqr": np.percentile(heights, [25, 75]).tolist()},
        "width_m": {"median": float(np.median(widths)),
                    "iqr": np.percentile(widths, [25, 75]).tolist()} if widths else None,
    }, indent=1))


def cmd_recover(a):
    s = Survey(a.survey)
    rec = pycolmap.Reconstruction(s.sparse)
    frames = {f.image: f for f in s.poses().shots()}
    reg = {im.name: im for im in rec.images.values() if im.has_pose}
    names = [n for n in reg if n in frames and frames[n].tracking_state == "normal"]
    sc, r, t = umeyama(np.array([frames[n].position_m for n in names], float),
                       np.array([reg[n].projection_center() for n in names]))
    db = pycolmap.Database.open(s.database)
    try:
        images = {im.name: im for im in db.read_all_images()}
        cams = {c.camera_id: c for c in db.read_all_cameras()}
        id2name = {im.image_id: im.name for im in images.values()}
        pair_ids, geoms = db.read_two_view_geometries()
        rows = []
        for n in (n for n in s.images() if n not in reg):
            im = images[n]
            kp = db.read_keypoints(im.image_id)
            corr: dict[int, int] = {}
            for pid, g in zip(pair_ids, geoms):
                i, j = pycolmap.pair_id_to_image_pair(pid)
                if im.image_id not in (i, j):
                    continue
                other = id2name[j if i == im.image_id else i]
                if other not in reg or len(g.inlier_matches) == 0:
                    continue
                me, them = (0, 1) if i == im.image_id else (1, 0)
                for row in g.inlier_matches:
                    p2 = reg[other].points2D[int(row[them])]
                    if p2.has_point3D():
                        corr.setdefault(int(row[me]), p2.point3D_id)
            f = frames.get(n)
            out = {"image": n, "tracking": f.tracking_state if f else None, "n_2d3d": len(corr)}
            if len(corr) >= 6:
                idx = np.array(list(corr))
                res = pycolmap.estimate_and_refine_absolute_pose(
                    kp[idx, :2].astype(np.float64),
                    np.array([rec.points3D[corr[k]].xyz for k in idx]), cams[im.camera_id])
                if res is not None:
                    pose = res["cam_from_world"]
                    c = -pose.rotation.matrix().T @ np.asarray(pose.translation)
                    out["inliers"] = int(res["num_inliers"])
                    if f is not None:
                        out["d_pos_m"] = float(np.linalg.norm(
                            c - (sc * r @ np.array(f.position_m) + t)))
                        out["d_rot_deg"] = rotation_angle_deg(
                            pose.rotation.matrix().T, r @ arkit_world_from_colmap_cam(f.rotation_xyzw))
            rows.append(out)
    finally:
        db.close()
    ok = [x for x in rows if x.get("inliers", 0) >= 15 and x.get("d_pos_m", 9) < 0.3
          and x.get("d_rot_deg", 99) < 5]
    print(json.dumps({
        "missing": len(rows), "with_6plus_2d3d": sum(x["n_2d3d"] >= 6 for x in rows),
        "with_zero_2d3d": sum(x["n_2d3d"] == 0 for x in rows),
        "accepted_15_inliers_0p3m_5deg": [x["image"] for x in ok], "rows": rows,
    }, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("timing")
    p.add_argument("capture", type=Path)
    p.add_argument("model", type=Path)
    p = sub.add_parser("drift")
    p.add_argument("capture", type=Path)
    p.add_argument("model", type=Path)
    p = sub.add_parser("gps-scale")
    p.add_argument("capture", type=Path)
    p = sub.add_parser("waterline")
    p.add_argument("model", type=Path)
    p.add_argument("--max-z", type=float, default=1.5)
    p = sub.add_parser("corridor")
    p.add_argument("model", type=Path)
    p.add_argument("--extra-ply", type=Path)
    p.add_argument("--slice-m", type=float, default=2.0)
    p = sub.add_parser("recover")
    p.add_argument("survey", type=Path)
    a = ap.parse_args()
    {"timing": cmd_timing, "drift": cmd_drift, "gps-scale": cmd_gps_scale,
     "waterline": cmd_waterline, "corridor": cmd_corridor, "recover": cmd_recover}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
