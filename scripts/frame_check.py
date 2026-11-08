"""Is a written SiteMesh model metric and Z-up? (issue #3)

Reads <survey>/model/{report.json,cameras.json,sparse/} and prints: output frame and heading,
the dominant horizontal plane (assumed to be the floor or table) and its tilt from +Z, camera
heights above it, and mean camera pitch. Optionally writes a side-view PNG of the cloud.

    python scripts/frame_check.py data/object-orbit-2 [--png side.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

from sitemesh.contracts import Survey


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("survey", type=Path)
    ap.add_argument("--band-m", type=float, default=0.03, help="half-width of the plane band")
    ap.add_argument("--png", type=Path)
    args = ap.parse_args()
    s = Survey(args.survey)
    report = json.loads((s.model / "report.json").read_text())
    rec = pycolmap.Reconstruction(s.sparse)
    pts = np.array([p.xyz for p in rec.points3D.values()])
    cams = np.array([c["centre_m"] for c in json.loads((s.model / "cameras.json").read_text())])

    # Dominant horizontal plane: the densest 2*band slab in z, then a PCA normal.
    zs = np.sort(pts[:, 2])
    w = 2 * args.band_m
    counts = np.searchsorted(zs, zs + w) - np.arange(len(zs))
    z0 = zs[int(np.argmax(counts))] + args.band_m
    slab = pts[np.abs(pts[:, 2] - z0) < args.band_m]
    normal = np.linalg.svd(slab - slab.mean(0), full_matrices=False)[2][2]
    normal *= np.sign(normal[2]) or 1.0
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    heights = cams[:, 2] - z0
    views = np.array([im.cam_from_world().rotation.matrix().T[:, 2]
                      for im in rec.images.values() if im.has_pose])
    pitch = np.degrees(np.arcsin(np.clip(views[:, 2], -1, 1)))
    out = {
        "output_frame": report.get("output_frame"),
        "heading": report.get("heading"),
        "device_horizontal_spread_m": report.get("device_horizontal_spread_m"),
        "gps_alignment_rms_m": report.get("gps_alignment_rms_m"),
        "sfm_to_device_scale": report.get("sfm_to_device_scale"),
        "n_points": len(pts),
        "plane_z_m": float(z0),
        "plane_points": len(slab),
        "plane_tilt_from_z_deg": tilt,
        "camera_height_above_plane_m": {
            "min": float(heights.min()), "median": float(np.median(heights)),
            "max": float(heights.max()),
        },
        "mean_camera_pitch_deg": float(pitch.mean()),
    }
    print(json.dumps(out, indent=2))
    if args.png:
        import cv2

        cols = np.array([p.color for p in rec.points3D.values()])
        allp = np.vstack([pts, cams])
        lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
        sc = 800 / max(hi[0] - lo[0], hi[2] - lo[2])
        h, wid = int((hi[2] - lo[2]) * sc) + 40, int((hi[0] - lo[0]) * sc) + 40
        img = np.full((h, wid, 3), 255, np.uint8)
        for p, c in zip(pts, cols):
            x, y = int((p[0] - lo[0]) * sc) + 20, h - int((p[2] - lo[2]) * sc) - 20
            if 0 <= x < wid and 0 <= y < h:
                img[y, x] = c[::-1]
        for p in cams:
            x, y = int((p[0] - lo[0]) * sc) + 20, h - int((p[2] - lo[2]) * sc) - 20
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite(str(args.png), img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
