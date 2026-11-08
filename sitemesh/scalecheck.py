"""Independent scale check against printed AprilTags of known size.

Every hydra-vision phone-survey metric was internal consistency (registration, reprojection, VIO
agreement). This is the one number that is not: triangulate the four corners of a tag from the
reconstruction's own camera poses and compare its edge length with the printed size.

Tag sizes default to hydra-vision's room kit (data/markers/room-kit.json): ids 200–219 are
165.1 mm, ids 300–319 are 76.2 mm. Print at 100% and ruler-check the printed edge before trusting.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sitemesh.geometry import triangulate_dlt

ROOM_KIT_MM = {**{i: 165.1 for i in range(200, 220)}, **{i: 76.2 for i in range(300, 320)}}


def detect_tags(image_path: Path, family: int = cv2.aruco.DICT_APRILTAG_36h11):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def scale_check(rec, raw_dir: Path, tag_sizes_mm: dict[int, float] | None = None) -> dict:
    """Triangulate tag corners in the (already metric) reconstruction and measure their edges."""
    sizes = tag_sizes_mm or ROOM_KIT_MM
    obs: dict[int, list] = defaultdict(list)  # tag id -> [(normalized P, 4x2 normalized corners)]
    for im in rec.images.values():
        if not im.has_pose:
            continue
        tags = detect_tags(Path(raw_dir) / im.name)
        if not tags:
            continue
        pose = im.cam_from_world().matrix()  # 3x4, normalized camera
        cam = rec.cameras[im.camera_id]
        for tid, px in tags.items():
            if tid in sizes:
                obs[tid].append((pose, np.asarray(cam.cam_from_img(px))))

    tags_out = []
    for tid, views in sorted(obs.items()):
        if len(views) < 2:
            tags_out.append({"tag_id": tid, "n_views": len(views), "status": "needs >= 2 views"})
            continue
        corners = []
        for k in range(4):
            corners.append(triangulate_dlt([v[0] for v in views], [v[1][k] for v in views]))
        corners = np.array(corners)
        edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1) * 1000.0
        known = sizes[tid]
        tags_out.append(
            {
                "tag_id": tid,
                "n_views": len(views),
                "known_mm": known,
                "measured_mm": float(edges.mean()),
                "edge_spread_mm": float(edges.max() - edges.min()),
                "error_pct": float((edges.mean() - known) / known * 100.0),
            }
        )
    measured = [t for t in tags_out if "error_pct" in t]
    summary = {
        "n_tags_measured": len(measured),
        "scale_check_error_pct": (
            float(np.median([t["error_pct"] for t in measured])) if measured else None
        ),
    }
    return {"summary": summary, "tags": tags_out}
