"""Bulk measures of a surveyed site from the written model (metres, Z up).

For site surveys the deliverable is bulk measures, not fine detail: how big is the site, where
are its horizontal levels (ground, tables, decks, ceilings), how tall is it, and how wide and high
is it along its main axis. Everything is computed from the georegistered output model, so it's
in metres only as far as the VIO scale is right. The report says what the scale rests on: ARKit
VIO always; GPS evidence outdoors when the walk is long enough (`sitemesh.gpscheck`). No measure
here is tape-verified.

Robust statistics throughout (percentiles, slab medians) because sparse SfM clouds carry outliers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from sitemesh.contracts import Survey


@dataclass
class MeasureOptions:
    trim_pct: float = 1.0  # extents drop this share of occupied voxels at each end
    cell_m: float = 0.25  # voxel / footprint grid
    min_voxel_points: int = 2  # a voxel is occupied with at least this many points
    level_bin_m: float = 0.05
    level_min_frac: float = 0.03  # a level holds at least this share of the points
    level_merge_m: float = 0.20  # peaks closer than this are one level
    level_slab_m: float = 0.05  # half-thickness of a level's slab
    section_m: float = 2.0
    section_min_points: int = 50
    wall_min_cover: float = 0.4  # a wall's 10 cm distance bin covers this share of the heights


def _oriented_extent(xy: np.ndarray, trim: float):
    centre = xy.mean(0)
    axes = np.linalg.svd(xy - centre, full_matrices=False)[2]
    proj = (xy - centre) @ axes.T
    lo, hi = np.percentile(proj, trim, 0), np.percentile(proj, 100 - trim, 0)
    length, width = (hi - lo)
    major = axes[0]
    if width > length:  # after trimming the second PCA axis can be the longer one
        length, width, major = width, length, axes[1]
    return {
        "length_m": float(length), "width_m": float(width),
        "diagonal_m": float(np.hypot(length, width)),
        "major_axis_heading_deg": float(np.degrees(np.arctan2(major[0], major[1])) % 180),
    }, axes, centre


def _footprint_m2(xy: np.ndarray, cell: float) -> float:
    cells = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    return float(len(cells) * cell * cell)


def _levels(p: np.ndarray, o: MeasureOptions) -> list[dict]:
    z = p[:, 2]
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    edges = np.arange(lo, hi + o.level_bin_m, o.level_bin_m)
    if len(edges) < 3:
        return []
    hist, edges = np.histogram(z, bins=edges)
    # mass within +-2 bins: a level smeared over a few bins still counts as one
    mass = np.convolve(hist, np.ones(5), mode="same")
    order = np.argsort(hist)[::-1]
    peaks: list[float] = []
    for i in order:
        if mass[i] / len(z) < o.level_min_frac:
            break
        zc = edges[i] + o.level_bin_m / 2
        if all(abs(zc - q) >= o.level_merge_m for q in peaks):
            peaks.append(zc)
    levels = []
    for zc in sorted(peaks):
        slab = p[np.abs(z - zc) < o.level_slab_m]
        if len(slab) < 10 or len(slab) / len(p) < o.level_min_frac:
            continue
        normal = np.linalg.svd(slab - slab.mean(0), full_matrices=False)[2][2]
        normal *= np.sign(normal[2]) or 1.0
        levels.append({
            "z_m": float(np.median(slab[:, 2])), "n_points": len(slab),
            "share": len(slab) / len(p),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
            "footprint_m2": _footprint_m2(slab[:, :2], o.cell_m),
        })
    return levels


def _nearest_wall(dist: np.ndarray, z: np.ndarray, zlo: float, zhi: float,
                  o: MeasureOptions) -> float | None:
    """Distance to the first wall on one side of the camera line.

    A wall is vertical: at its distance, points cover a large share of the slice's height range.
    Floors, tables and clutter cover many distances but few heights, and the farthest points
    are views through doorways (d4737a5d read 11.9 m wide using them), so the first
    distance bin, walking outward, whose height coverage reaches `wall_min_cover` is the wall.
    """
    if len(dist) < 20 or zhi - zlo < 0.3:
        return None
    zbins = max(int((zhi - zlo) / 0.1), 1)
    db = np.floor(dist / 0.1).astype(int)
    zb = np.clip(((z - zlo) / 0.1).astype(int), 0, zbins - 1)
    for k in range(int(np.percentile(db, 99)) + 1):
        sel = db == k
        if sel.sum() >= 5 and len(np.unique(zb[sel])) / zbins >= o.wall_min_cover:
            return (k + 0.5) * 0.1
    return None


def _sections(p, cams, o: MeasureOptions):
    """Width and height in slices perpendicular to the main axis of the camera track."""
    if len(cams) < 3:
        return None
    m = cams[:, :2].mean(0)
    ax = np.linalg.svd(cams[:, :2] - m, full_matrices=False)[2][0]
    perp = np.array([-ax[1], ax[0]])
    along, across, z = (p[:, :2] - m) @ ax, (p[:, :2] - m) @ perp, p[:, 2]
    ca = (cams[:, :2] - m) @ ax
    rows = []
    for s0 in np.arange(ca.min(), ca.max(), o.section_m):
        sel = (along >= s0) & (along < s0 + o.section_m)
        if sel.sum() < o.section_min_points:
            continue
        w, zz = across[sel], z[sel]
        row = {"from_m": float(s0 - ca.min()), "n_points": int(sel.sum()),
               "height_m": float(np.percentile(zz, 98) - np.percentile(zz, 2))}
        zlo, zhi = np.percentile(zz, 2), np.percentile(zz, 98)
        left = _nearest_wall(-w[w < 0], zz[w < 0], zlo, zhi, o)
        right = _nearest_wall(w[w > 0], zz[w > 0], zlo, zhi, o)
        if left is not None and right is not None:
            row["width_m"] = left + right  # clear width between the first walls either side
        rows.append(row)
    if not rows:
        return None

    def summary(key):
        v = [r[key] for r in rows if key in r]
        return ({"median_m": float(np.median(v)), "iqr_m": np.percentile(v, [25, 75]).tolist(),
                 "n_slices": len(v)} if v else None)

    width = summary("width_m")
    width_note = None
    if width is None:
        width_note = "no wall found in any slice"
    else:
        lo, hi = width["iqr_m"]
        if (width["n_slices"] < max(3, len(rows) / 2)
                or (hi - lo) > 0.25 * width["median_m"]):
            width_note = (f"walls found in {width['n_slices']}/{len(rows)} slices, IQR "
                          f"{lo:.2f}-{hi:.2f} m: too sparse or inconsistent to report")
            width = None
    return {"axis_length_m": float(np.ptp(ca)), "slice_m": o.section_m,
            "height": summary("height_m"), "width": width, "width_not_measured": width_note,
            "slices": rows}


def measure_points(p: np.ndarray, cams: np.ndarray, o: MeasureOptions | None = None) -> dict:
    """All bulk measures from a Z-up metric point cloud and camera centres."""
    o = o or MeasureOptions()
    # Extents come from occupied voxels, not raw points: percentiles of raw points are weighted
    # by point density, so adding dense points in one area shrank b8b8ebd7's length 44 -> 23 m.
    vox, counts = np.unique(np.floor(p / o.cell_m).astype(np.int64), axis=0, return_counts=True)
    vox = (vox[counts >= o.min_voxel_points] + 0.5) * o.cell_m
    lo, hi = np.percentile(vox, o.trim_pct, 0), np.percentile(vox, 100 - o.trim_pct, 0)
    core = vox[np.all((vox >= lo) & (vox <= hi), axis=1)]
    extent, _, _ = _oriented_extent(core[:, :2], 0.0)
    levels = _levels(p, o)
    cam_z = float(np.median(cams[:, 2])) if len(cams) else None
    ground = levels[0] if levels else None
    ceiling = None
    if levels and cam_z is not None:
        above = [lv for lv in levels if lv["z_m"] > cam_z + 0.3]
        ceiling = above[-1] if above else None
    gz = ground["z_m"] if ground else None
    for lv in levels:
        lv["above_ground_m"] = lv["z_m"] - gz if gz is not None else None
    walk = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum()) if len(cams) > 1 else 0.0
    return {
        "n_points": len(p), "n_occupied_voxels": len(vox),
        "extent": extent,
        "footprint_m2": _footprint_m2(core[:, :2], o.cell_m),
        "height_range_m": float(hi[2] - lo[2]),
        "ground_z_m": gz,
        "max_height_above_ground_m": float(hi[2] - gz) if gz is not None else None,
        "ceiling_above_ground_m": (ceiling["z_m"] - gz) if ceiling and gz is not None else None,
        "levels": levels,
        "cameras": {
            "n": len(cams), "walk_length_m": walk,
            "extent_m": np.ptp(cams[:, :2], 0).tolist() if len(cams) else None,
            "height_above_ground_m": (
                {"min": float(cams[:, 2].min() - gz), "median": float(cam_z - gz),
                 "max": float(cams[:, 2].max() - gz)} if gz is not None and len(cams) else None),
        },
        "sections": _sections(p, cams, o),
        "options": asdict(o),
    }


def _md(name: str, m: dict, scale: dict) -> str:
    e = m["extent"]
    lines = [f"# Bulk measures: {name}", "",
             f"**Scale basis:** {scale['basis']}. {scale['note']}", "",
             "| measure | value |", "|---|---|",
             (f"| site length × width (main axes, occupied 25 cm voxels) | {e['length_m']:.1f} × "
              f"{e['width_m']:.1f} m |"),
             f"| site diagonal | {e['diagonal_m']:.1f} m |",
             f"| occupied footprint | {m['footprint_m2']:.0f} m² |"]
    if m["max_height_above_ground_m"] is not None:
        lines.append(f"| tallest occupied voxel above ground (99th pct) | "
                     f"{m['max_height_above_ground_m']:.2f} m |")
    if m["ceiling_above_ground_m"] is not None:
        lines.append(f"| ceiling above ground | {m['ceiling_above_ground_m']:.2f} m |")
    s = m["sections"]
    if s and s["height"]:
        h = s["height"]
        lines.append(f"| height along main axis (median, IQR) | {h['median_m']:.2f} m "
                     f"({h['iqr_m'][0]:.2f}–{h['iqr_m'][1]:.2f}) |")
    if s and s.get("width_not_measured"):
        lines.append(f"| clear width along main axis | not measured: {s['width_not_measured']} |")
    if s and s["width"]:
        w = s["width"]
        lines.append(f"| clear width along main axis (median, IQR) | {w['median_m']:.2f} m "
                     f"({w['iqr_m'][0]:.2f}–{w['iqr_m'][1]:.2f}) |")
    c = m["cameras"]
    lines.append(f"| camera path, shot to shot | {c['walk_length_m']:.1f} m |")
    if m["levels"]:
        lines += ["", ("Horizontal levels; *ground* is the lowest (a floor, deck, or the table "
                       "an object sits on)."), "",
                  "| level | above ground | tilt | footprint | share of points |",
                  "|---|---|---|---|---|"]
        for i, lv in enumerate(m["levels"]):
            ag = lv["above_ground_m"]
            lines.append(f"| {i} | {ag:+.2f} m | {lv['tilt_deg']:.2f}° | "
                         f"{lv['footprint_m2']:.1f} m² | {lv['share']:.0%} |")
    lines += ["", "No measure here is tape-verified."]
    return "\n".join(lines) + "\n"


def _scale_basis(report: dict, gps: dict | None) -> dict:
    frame = report.get("output_frame")
    if report.get("scale_source") is None:
        return {"basis": "none", "note": "Model is not metric (no VIO); measures are in SfM "
                "units.", "metric": False}
    note = f"Metric from {report.get('scale_source')} VIO; output frame `{frame}`."
    if gps and gps.get("status") == "consistent":
        lo, hi = gps["scale_95ci"]
        note += (f" GPS independently agrees: scale {gps['gps_over_vio_scale']:.3f} "
                 f"(95% CI {lo:.3f}–{hi:.3f}).")
        basis = "VIO, GPS-consistent"
    elif gps and gps.get("status") == "inconsistent":
        note += f" **GPS disagrees:** {gps['reason']}."
        basis = "VIO, GPS-inconsistent"
    else:
        note += (" No independent scale check" +
                 (f" (GPS: {gps['reason']})" if gps and gps.get("reason") else "") + ".")
        basis = "VIO only"
    return {"basis": basis, "note": note, "metric": True}


def run_measure(survey: Survey, dense: bool = False, options: MeasureOptions | None = None) -> dict:
    import pycolmap

    from sitemesh.gpscheck import gps_scale_check

    rec = pycolmap.Reconstruction(survey.sparse)
    p = np.array([q.xyz for q in rec.points3D.values()])
    sources = {"sparse": len(p)}
    dense_path = survey.model / "dense_cpu" / "fused.ply"
    if dense and dense_path.exists():
        from sitemesh.plyio import read_ply_xyz

        d = read_ply_xyz(dense_path)
        p = np.vstack([p, d])
        sources["dense_cpu"] = len(d)
    # camera centres in capture order (poses.json timestamps), so walk length means something
    track = survey.poses()
    order = {f.image: f.timestamp for f in track.shots()} if track else {}
    posed = sorted((im for im in rec.images.values() if im.has_pose),
                   key=lambda im: (order.get(im.name, float("inf")), im.name))
    cams = np.array([im.projection_center() for im in posed]).reshape(-1, 3)
    report_path = survey.model / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    gps = report.get("gps_scale_check") or gps_scale_check(track)
    scale = _scale_basis(report, gps)
    m = measure_points(p, cams, options)
    out = {"survey": survey.root.name, "point_sources": sources, "scale": scale,
           "gps_scale_check": gps, **m}
    survey.write_json("measures.json", out)
    (survey.model / "measures.md").write_text(_md(survey.root.name, m, scale))
    return out
