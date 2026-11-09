"""GPS as independent evidence for the VIO metric scale (outdoor sites).

ARKit's VIO sets the model's scale; nothing else in a capture checks it. Over a walk of tens of
metres, consumer GPS (metres-grade per fix, but unbiased-ish outdoors) can: fit a 2D similarity
from the VIO track to GPS ENU and read off its scale. On the b8b8ebd7 backyard (256 fixes, ~25 m
walk, 4.7 m median accuracy) this gave 1.033 [0.977, 1.077]. Indoors, or over a few metres, the
estimate is meaningless (d4737a5d: 1.29; object-orbit-2: 0.21), so the check refuses those cases
instead of reporting a number (docs/experiments.md §7).

Evidence only: it never changes the model and is not a gate check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sitemesh.contracts import DevicePoseTrack
from sitemesh.geometry import DEVICE_TO_ZUP, wgs84_to_enu


@dataclass
class GpsCheckLimits:
    min_fixes: int = 30
    min_extent_m: float = 10.0  # VIO track extent (larger horizontal side)
    max_median_accuracy_m: float = 10.0
    block: int = 10  # bootstrap block length (GPS errors are correlated in time)
    n_boot: int = 500
    consistent_tol: float = 0.05  # |scale - 1| within this, or 1 inside the CI, is consistent


def _fit(dev: np.ndarray, enu: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted 2D similarity dev -> enu; returns (scale, heading_deg)."""
    za = (dev - (dev * w[:, None]).sum(0) / w.sum()) @ np.array([1, 1j])
    zb = (enu - (enu * w[:, None]).sum(0) / w.sum()) @ np.array([1, 1j])
    k = (w * np.conj(za) * zb).sum() / (w * np.abs(za) ** 2).sum()
    return float(abs(k)), float(np.degrees(np.angle(k)))


def gps_scale_check(track: DevicePoseTrack | None, limits: GpsCheckLimits | None = None,
                    seed: int = 0) -> dict:
    lim = limits or GpsCheckLimits()
    out: dict = {"status": "not_applicable", "limits": asdict(lim)}
    if track is None or not track.gps_track:
        out["reason"] = "no GPS track"
        return out
    frames = sorted((f for f in track.frames if f.tracking_state == "normal"),
                    key=lambda f: f.timestamp)
    if len(frames) < 3:
        out["reason"] = "fewer than 3 tracked VIO samples"
        return out
    t = np.array([f.timestamp for f in frames])
    xy = (DEVICE_TO_ZUP @ np.array([f.position_m for f in frames], float).T).T[:, :2]
    fixes = sorted((g for g in track.gps_track if t[0] <= g.timestamp <= t[-1]),
                   key=lambda g: g.timestamp)
    acc = np.array([g.horizontal_accuracy_m if g.horizontal_accuracy_m else np.nan for g in fixes])
    dev = np.array([[np.interp(g.timestamp, t, xy[:, k]) for k in range(2)] for g in fixes])
    extent = float(np.ptp(dev, 0).max()) if len(fixes) else 0.0
    med_acc = float(np.nanmedian(acc)) if len(fixes) and np.isfinite(acc).any() else None
    out.update(n_fixes=len(fixes), vio_extent_m=extent, gps_median_accuracy_m=med_acc)
    if len(fixes) < lim.min_fixes:
        out["reason"] = f"{len(fixes)} GPS fixes during tracking < {lim.min_fixes}"
        return out
    if extent < lim.min_extent_m:
        out["reason"] = f"walk extent {extent:.1f} m < {lim.min_extent_m} m"
        return out
    if med_acc is None or med_acc > lim.max_median_accuracy_m:
        acc_txt = "unknown" if med_acc is None else f"{med_acc:.1f}"
        out["reason"] = (f"median GPS accuracy {acc_txt} m > {lim.max_median_accuracy_m} m "
                         "(indoors or poor sky view)")
        return out

    ref = fixes[0]
    enu = np.array([wgs84_to_enu(g.lat_deg, g.lon_deg, g.alt_m or 0.0,
                                 ref.lat_deg, ref.lon_deg, ref.alt_m or 0.0)[:2] for g in fixes])
    w = 1.0 / np.clip(np.nan_to_num(acc, nan=med_acc), 1.0, None) ** 2
    scale, heading = _fit(dev, enu, w)
    rng = np.random.default_rng(seed)
    blocks = [np.arange(i, min(i + lim.block, len(fixes))) for i in range(0, len(fixes), lim.block)]
    boot = []
    for _ in range(lim.n_boot):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        boot.append(_fit(dev[idx], enu[idx], w[idx])[0])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    consistent = (lo <= 1.0 <= hi) or abs(scale - 1.0) <= lim.consistent_tol
    out.update(
        status="consistent" if consistent else "inconsistent",
        gps_over_vio_scale=scale, scale_95ci=[lo, hi], heading_deg=heading,
        reason=None if consistent else
        f"GPS says the model is {scale:.3f}x VIO scale, 95% CI [{lo:.3f}, {hi:.3f}] excludes 1",
    )
    return out
