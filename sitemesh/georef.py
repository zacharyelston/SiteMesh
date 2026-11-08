"""Put the reconstruction in metres, Z up, and — when GPS allows — ENU.

One similarity is composed and applied to the reconstruction itself, before anything is written,
so every downstream artifact (sparse, dense, mesh) is born in the output frame. hydra-vision ran
dense against the raw SfM gauge and had to refit afterwards; its clouds showed up tilted 45–90°.

    SfM gauge --(Umeyama to VIO centres: metric scale)--> ARKit world (+Y up)
              --(DEVICE_TO_ZUP: fixed)-->                 levelled device frame (+Z up)
              --(yaw-only fit to GPS: heading)-->         ENU (+Z up, +Y north)

GPS never touches scale or tilt: it is metres-grade and, indoors, wildly so.
"""

from __future__ import annotations

import numpy as np
import pycolmap

from sitemesh.contracts import DevicePoseTrack
from sitemesh.geometry import DEVICE_TO_ZUP, umeyama, wgs84_to_enu, yaw_fit


def _gps_at(track: DevicePoseTrack, t: float):
    fixes = sorted(track.gps_track, key=lambda f: f.timestamp)
    if not fixes or t < fixes[0].timestamp or t > fixes[-1].timestamp:
        return None
    times = [f.timestamp for f in fixes]
    hi = int(np.searchsorted(times, t))
    if hi == 0:
        return fixes[0]
    lo = hi - 1
    span = times[hi] - times[lo]
    w = (t - times[lo]) / span if span else 0.0
    a, b = fixes[lo], fixes[hi]
    alt = (1 - w) * (a.alt_m or 0.0) + w * (b.alt_m or 0.0)
    return ((1 - w) * a.lat_deg + w * b.lat_deg, (1 - w) * a.lon_deg + w * b.lon_deg, alt)


def georegister(rec, track: DevicePoseTrack | None, min_gps_spread_m: float = 3.0) -> dict:
    """Transform `rec` in place to the output frame; return the report fields."""
    out: dict = {"scale_source": None, "output_frame": "sfm_gauge", "enu_reference": None}
    if track is None:
        out["warning"] = "no poses.json — model has arbitrary scale and orientation"
        return out

    frames = {f.image: f for f in track.shots() if f.tracking_state == "normal"}
    names, src, dst, times = [], [], [], []
    for im in rec.images.values():
        f = frames.get(im.name)
        if f is not None and im.has_pose:
            names.append(im.name)
            src.append(np.array(im.projection_center()))
            dst.append(np.array(f.position_m))
            times.append(f.timestamp)
    if len(src) < 3:
        out["warning"] = f"only {len(src)} registered images have VIO poses — not georegistered"
        return out

    src, dst = np.array(src), np.array(dst)
    s, r, t = umeyama(src, dst)
    res = np.linalg.norm(s * (r @ src.T).T + t - dst, axis=1)
    out.update(
        scale_source=track.source,
        sfm_to_device_scale=float(s),
        device_alignment_rms_m=float(np.sqrt((res**2).mean())),
        device_alignment_max_m=float(res.max()),
    )

    # Level: ARKit +Y up -> +Z up.
    r_total = DEVICE_TO_ZUP @ r
    t_total = DEVICE_TO_ZUP @ t
    out["output_frame"] = "device_zup"

    # Heading from GPS, if the GPS track says anything useful.
    anchor = [(DEVICE_TO_ZUP @ d, _gps_at(track, tm)) for d, tm in zip(dst, times)]
    anchor = [(d, g) for d, g in anchor if g is not None]
    if len(anchor) >= 3:
        ref = anchor[0][1]
        dev = np.array([d for d, _ in anchor])
        enu = np.array([wgs84_to_enu(*g, *ref) for _, g in anchor])
        spread = float(np.linalg.norm(dev[:, :2] - dev[:, :2].mean(0), axis=1).max())
        out["device_horizontal_spread_m"] = spread
        out["enu_reference"] = {"lat_deg": ref[0], "lon_deg": ref[1], "alt_m": ref[2]}
        ry, ty = yaw_fit(dev, enu)
        gres = np.linalg.norm((ry @ dev.T).T + ty - enu, axis=1)
        out["gps_alignment_rms_m"] = float(np.sqrt((gres**2).mean()))
        if spread >= min_gps_spread_m:
            r_total, t_total = ry @ r_total, ry @ t_total + ty
            out["output_frame"] = "enu"
        else:
            # Anchor position only. Heading from GPS fixes over a few metres of walking is noise;
            # better an honest arbitrary heading than a confident wrong one.
            ty_pos = enu.mean(0) - dev.mean(0)
            t_total = t_total + ty_pos
            out["output_frame"] = "enu_position_only"
            out["heading"] = f"arbitrary (walked {spread:.1f} m < {min_gps_spread_m} m)"

    rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(r_total), t_total))
    out["model_to_output"] = {
        "scale": float(s),
        "rotation": r_total.tolist(),
        "translation": list(map(float, t_total)),
    }
    return out
