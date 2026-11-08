"""Survey folder contract.

Compatible with hydra-vision's capture output, so existing HailHydra captures run unchanged:

    <survey>/
        raw/            stills (jpg/png/heic converted to jpg)
        poses.json      DevicePoseTrack — ARKit/ARCore VIO, the metric authority
        model/          everything SiteMesh writes

Conventions: metres, quaternions (x, y, z, w). The device world frame is ARKit's
gravity-aligned frame (+Y up). Output frame is ENU (+Z up) when GPS can anchor heading,
otherwise a levelled device frame (+Z up, arbitrary heading).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class PinholeIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class DevicePoseFrame(BaseModel):
    """world_from_camera in the device world frame; position_m is the camera centre."""

    timestamp: float
    image: str | None = None
    position_m: Vec3
    rotation_xyzw: Quat
    intrinsics: PinholeIntrinsics | None = None
    tracking_state: Literal["normal", "limited", "not_available"] = "normal"
    subject_distance_m: float | None = None


class GpsFix(BaseModel):
    timestamp: float
    lat_deg: float
    lon_deg: float
    alt_m: float | None = None
    horizontal_accuracy_m: float | None = None


class DevicePoseTrack(BaseModel):
    schema_version: str = "0.1.0"
    source: Literal["arkit", "arcore"]
    app_version: str | None = None
    os_version: str | None = None
    frames: list[DevicePoseFrame]
    gps_track: list[GpsFix] = Field(default_factory=list)

    def shots(self) -> list[DevicePoseFrame]:
        """Frames that name a still, in capture order (one per photo, last sample wins)."""
        by_name: dict[str, DevicePoseFrame] = {}
        for f in sorted(self.frames, key=lambda f: f.timestamp):
            if f.image:
                by_name[f.image] = f
        return sorted(by_name.values(), key=lambda f: f.timestamp)


class Survey:
    """Paths for one survey folder."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.model = self.root / "model"
        self.sparse = self.model / "sparse"
        self.database = self.model / "database.db"
        self.poses_path = self.root / "poses.json"

    def images(self) -> list[str]:
        exts = {".jpg", ".jpeg", ".png"}
        return sorted(p.name for p in self.raw.iterdir() if p.suffix.lower() in exts)

    def poses(self) -> DevicePoseTrack | None:
        if not self.poses_path.exists():
            return None
        return DevicePoseTrack.model_validate_json(self.poses_path.read_text())

    def write_json(self, name: str, data: dict | list) -> Path:
        self.model.mkdir(parents=True, exist_ok=True)
        path = self.model / name
        path.write_text(json.dumps(data, indent=2, default=float) + "\n")
        return path
