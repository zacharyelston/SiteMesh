"""Capture QC: the stance card, checked against the pose track before any reconstruction.

Everything here needs only poses.json, so it can run on the phone or laptop in seconds and tell the
operator to re-shoot before leaving the site.

Rule (hydra-vision stance card, #121): baseline ~ subject distance / 6. The /3 limit on the card is
derived, not measured — corner-bi-focal-exp-1 registered 94% at D/B ~1.8 — so the "too wide" flag
here is at D/B < 2 and the thresholds are parameters, meant to be re-fit from the post-SfM
per-image table this module also produces (see `post_sfm_stance`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from sitemesh.contracts import DevicePoseTrack
from sitemesh.geometry import (
    arkit_world_from_colmap_cam,
    nearest_point_to_rays,
    optical_axis,
    rotation_angle_deg,
)


@dataclass
class StanceLimits:
    target_d_over_b: float = 6.0
    min_d_over_b: float = 2.0  # wider than this: view change too large (card says 3; data says ~1.8 ok)
    max_d_over_b: float = 15.0  # tighter than this: depth goes soft
    pivot_translation_m: float = 0.02  # below this, a rotation is a wrist pivot
    pivot_rotation_deg: float = 5.0
    collinear_ratio: float = 0.15  # 2nd/1st singular value of a 3-shot set below this = a line
    window: int = 5  # shots used to estimate where the operator was looking


@dataclass
class ShotQC:
    image: str
    index: int
    tracking: str
    subject_distance_m: float | None
    subject_source: str | None
    baseline_m: float | None
    rotation_deg: float | None
    d_over_b: float | None
    flags: list[str] = field(default_factory=list)


def _subject_distance(i, centres, axes, frames, limits) -> tuple[float | None, str | None]:
    if frames[i].subject_distance_m:
        return float(frames[i].subject_distance_m), "device"
    lo, hi = max(0, i - limits.window // 2), min(len(centres), i + limits.window // 2 + 1)
    target = nearest_point_to_rays(centres[lo:hi], axes[lo:hi])
    if target is None:
        return None, None
    depth = float(np.dot(target - centres[i], axes[i]))
    # A target behind the camera or absurdly far means the rays were near-parallel.
    if depth <= 0.1 or depth > 200:
        return None, None
    return depth, "ray_convergence"


def stance_qc(track: DevicePoseTrack, limits: StanceLimits | None = None) -> dict:
    limits = limits or StanceLimits()
    frames = track.shots()
    if not frames:
        return {"n_shots": 0, "shots": [], "summary": {"verdict": "no still shots in poses.json"}}

    centres = np.array([f.position_m for f in frames], float)
    axes = np.array([optical_axis(f.rotation_xyzw) for f in frames])
    rots = [arkit_world_from_colmap_cam(f.rotation_xyzw) for f in frames]

    shots: list[ShotQC] = []
    for i, f in enumerate(frames):
        dist, src = _subject_distance(i, centres, axes, frames, limits)
        base = rot = dob = None
        flags: list[str] = []
        if f.tracking_state != "normal":
            flags.append(f"tracking_{f.tracking_state}")
        if i > 0:
            base = float(np.linalg.norm(centres[i] - centres[i - 1]))
            rot = rotation_angle_deg(rots[i - 1], rots[i])
            if base < limits.pivot_translation_m and rot > limits.pivot_rotation_deg:
                flags.append("wrist_pivot")
            elif base < limits.pivot_translation_m:
                flags.append("duplicate_position")
            if dist and base > 1e-6:
                dob = dist / base
                if dob < limits.min_d_over_b:
                    flags.append("too_wide")
                elif dob > limits.max_d_over_b and "duplicate_position" not in flags:
                    flags.append("too_tight")
        shots.append(ShotQC(f.image, i, f.tracking_state, dist, src, base, rot, dob, flags))

    # Triangle, not line: a viewpoint's spatial neighbours should spread in BOTH image axes, or
    # depth along the missing axis is poorly constrained. Consecutive shots on an orbit are always
    # locally a line, so this looks at the k nearest viewpoints (any capture order) projected onto
    # the shot's own image plane. Informational — an orbit at one height still registers fine.
    collinear = 0
    k = min(6, len(frames) - 1)
    for i in range(len(frames)):
        if k < 2:
            break
        d = np.linalg.norm(centres - centres[i], axis=1)
        nbr = centres[np.argsort(d)[: k + 1]] - centres[i]
        plane = nbr @ rots[i][:, :2]  # camera x (right) and y (down) axes
        s = np.linalg.svd(plane - plane.mean(0), compute_uv=False)
        if s[0] > limits.pivot_translation_m and s[1] / s[0] < limits.collinear_ratio:
            collinear += 1
            shots[i].flags.append("no_cross_parallax")

    dobs = [s.d_over_b for s in shots if s.d_over_b]
    flagged = [s for s in shots if s.flags]
    counts: dict[str, int] = {}
    for s in shots:
        for fl in s.flags:
            counts[fl] = counts.get(fl, 0) + 1
    summary = {
        "n_shots": len(shots),
        "median_d_over_b": float(np.median(dobs)) if dobs else None,
        "target_d_over_b": limits.target_d_over_b,
        "frac_in_band": (
            float(np.mean([limits.min_d_over_b <= d <= limits.max_d_over_b for d in dobs]))
            if dobs
            else None
        ),
        "n_flagged": len(flagged),
        "flag_counts": counts,
        "frac_cross_parallax": 1.0 - collinear / len(shots),
        "verdict": _verdict(counts, len(shots)),
    }
    return {"limits": asdict(limits), "summary": summary, "shots": [asdict(s) for s in shots]}


def _verdict(counts: dict[str, int], n: int) -> str:
    bad = counts.get("wrist_pivot", 0) + counts.get("too_wide", 0)
    if n < 8:
        return "too few shots (hydra-vision skipped captures under 8 images)"
    if bad / n > 0.25:
        return "re-shoot: over a quarter of shots are wrist pivots or too-wide jumps"
    if bad / n > 0.10:
        return "marginal: expect some unregistered shots"
    return "ok"


def post_sfm_stance(track: DevicePoseTrack, rec, qc: dict | None = None, window: int = 2) -> list[dict]:
    """Per-shot table joining capture geometry to what SfM made of it.

    `measured_depth_m` is the median depth of the shot's own triangulated points, i.e. the real
    subject distance, so the D/B envelope can be re-fit from data instead of the card's
    derivation. A shot that failed to register has no points of its own, which would hide exactly
    the shots the fit is about. So every row also gets `neighbour_depth_m`: the median measured
    depth of registered shots within `window` places in capture order (never the shot itself).
    `neighbour_d_over_b` uses it, and is defined the same way for registered and unregistered
    shots. `qc_d_over_b` / `qc_flags` are what the on-site check predicted from poses.json alone.
    """
    reg = {im.name: im for im in rec.images.values() if im.has_pose}
    qc_by = {s["image"]: s for s in (qc or {}).get("shots", [])}
    shots = track.shots()
    rows = []
    prev_centre = None
    for f in shots:
        centre = np.array(f.position_m)
        base = float(np.linalg.norm(centre - prev_centre)) if prev_centre is not None else None
        prev_centre = centre
        im = reg.get(f.image)
        row = {"image": f.image, "registered": im is not None, "baseline_m": base,
               "tracking": f.tracking_state}
        q = qc_by.get(f.image)
        if q is not None:
            row["qc_d_over_b"] = q.get("d_over_b")
            row["qc_flags"] = q.get("flags", [])
        if im is not None:
            cam_from_world = im.cam_from_world()
            depths = [
                (cam_from_world * rec.points3D[p.point3D_id].xyz)[2]
                for p in im.points2D
                if p.has_point3D()
            ]
            row["n_points3d"] = len(depths)
            if depths:
                row["measured_depth_m"] = float(np.median(depths))
                if base:
                    row["measured_d_over_b"] = row["measured_depth_m"] / base
        rows.append(row)
    for i, row in enumerate(rows):
        lo, hi = max(0, i - window), min(len(rows), i + window + 1)
        nbr = [rows[j]["measured_depth_m"] for j in range(lo, hi)
               if j != i and "measured_depth_m" in rows[j]]
        if nbr:
            row["neighbour_depth_m"] = float(np.median(nbr))
            if row["baseline_m"]:
                row["neighbour_d_over_b"] = row["neighbour_depth_m"] / row["baseline_m"]
    return rows
