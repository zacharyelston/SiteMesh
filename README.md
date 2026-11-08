# SiteMesh

Phone photos + ARKit poses → a metric, georegistered point cloud and mesh.

This is a rebuild of the hydra-vision survey pipeline, following its rebuild guide
(AnceraInc/hydra-vision `docs/rebuild-guide.md`). The capture protocol stays the same. What
changes is that the ARKit data the phone records is used *inside* the reconstruction, not only
fitted afterwards.

## Capture: the standing triangle

Stand still, keep the subject framed, and take shots from a small triangle around your head:
**left, right, above**. Move your body, not your wrist.

**Baseline ≈ subject distance ÷ 6.**

| subject at | move between shots |
|---|---|
| 1 m | 17 cm (a hand-length) |
| 2 m | 33 cm (shoulder width) |
| 5 m | 83 cm (full two-hand sweep) |

Shoot a 0.5× pass for connectivity, then a 1.0× pass for detail. Use stills, not video, when you
want a mesh.

## Input folder

The input is the same as a hydra-vision / HailHydra capture:

```
<survey>/raw/IMG_*.jpg      stills
<survey>/poses.json         ARKit DevicePoseTrack (per-shot pose, intrinsics, GPS track)
```

## Run

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'
```

Check the capture on site. This needs only `poses.json` and takes seconds:

```bash
sitemesh qc data/my-scene
```

Build the sparse model, georegister it, run the scale check and the acceptance gate:

```bash
sitemesh sparse data/my-scene
```

Dense stereo and the Poisson mesh need a CUDA host:

```bash
sitemesh dense data/my-scene
```

Outputs go to `<survey>/model/`:

| file | contents |
|---|---|
| `qc.json` | stance-card check per shot: D/B ratio, wrist pivots, cross-parallax |
| `sparse/` | COLMAP model, already in the output frame (metres, Z up, ENU when GPS allows) |
| `model.ply`, `cameras.json` | sparse cloud and camera centres in that frame |
| `stance_vs_registration.json` | capture geometry joined to what SfM registered, used to re-fit the D/B envelope from data |
| `report.json` | everything above, plus the acceptance gate verdict |
| `dense/fused.ply`, `dense/meshed-poisson.ply` | written by the dense stage |

## What is different from hydra-vision

| | hydra-vision | SiteMesh |
|---|---|---|
| intrinsics | COLMAP `AUTO`, self-calibrated (one camera per frame on video) | one camera per lens, seeded from ARKit with a prior focal |
| VIO poses | one similarity fit after SfM | COLMAP pose priors, `use_prior_position` in the mapper |
| matching | exhaustive O(n²) | spatial k-nearest over the priors |
| output frame | dense refit after the fact | similarity applied to the model before dense, so every artifact starts in ENU |
| ARKit axis convention | unchecked | `device_rotation_rms_deg` in the report catches a wrong convention |
| capture QC | none | `sitemesh qc` from `poses.json` before upload |
| accuracy | internal consistency only | AprilTag scale check (room-kit sizes) and a gate that never reports unmeasured as passed |

## Status

- Sparse, georegistration, QC, scale check and gate run on Apple Silicon (CPU pycolmap 4.1.1).
- The dense stage is written against pycolmap's API but **has not been run**: it needs a CUDA
  build of pycolmap.
- The scale check is tested on synthetic renders only. No real capture with printed tags has
  been run through it yet.
