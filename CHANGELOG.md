# Changelog

## 0.1.2: merge disconnected sub-models (2026-11-08)

### Added
- `merge_submodels` (default on, `--no-merge` to disable): every COLMAP sub-model is aligned to
  VIO, combined, re-triangulated, and refined with a pose-prior BA. Pieces whose own VIO RMS
  exceeds 0.25 m are dropped, with the reason in `report.json` → `merge`.

### Verified
- d4737a5d 53 → **80/86** (seeds 0, 1; seed 2: 66/86, one degenerate piece dropped at 0.32 m).
  b8b8ebd7 22 → **47/77** on all seeds. Orbits unchanged.
- Merged models stay Z-up (plane tilt 0.08° / 0.12°) and in `enu`. VIO RMS 19 mm / 90 mm =
  0.09% / 0.20% of the scene diagonal. Reprojection under 1.0 px. Merge cost about 1 s.

### Not verified
- No feature track crosses any join on these captures (`n_joint_points` = 0), so the joins are
  only as good as ARKit tracking. Not independently checked against a known distance.
- The 0.25 m rejection threshold is set from 21 observed pieces, not fitted.
- Merged registration on d4737a5d varies with seed (80 / 80 / 66).

## 0.1.1: takeover, first real-data runs (2026-11-08)

All measurements are on one Apple M6 laptop (12 cores, 32 GB, CPU pycolmap 4.1.1), the
field-agent spec. Details and every number: `docs/results.md`.

### Fixed
- Images ARKit marked `limited` got no matching pairs and never registered (object-orbit-2 was
  83/85).
- `pycolmap.match_spatial` aborted the process (`std::out_of_range`, SIGABRT) in about half of
  runs. It's replaced by numpy k-nearest-prior pairs plus `match_image_pairs`.
- `georef._gps_at` crashed georegistration when a shot and the first GPS fix share a timestamp.
- The dense docstring and skip message pointed to a `tools/objectcapture` that doesn't exist.

### Changed
- **Default:** affine-shape SIFT (`sift_affine=True`, `--no-affine` to undo). object-orbit 40/44
  → 44/44 on 3 seeds. The sweep that chose it is in the commit message.
- `stance_vs_registration.json` gains `neighbour_d_over_b` (defined for unregistered shots too),
  `qc_d_over_b`, `qc_flags` and `tracking`.
- Dense skip result includes `depth_bounds_m_planned`.
- New: `scripts/bench.py` (reproducible config × seed runner with timing, peak RSS and sleep
  detection), `scripts/frame_check.py`, `scripts/stance_fit.py`. New CLI flag `--guided`.
- Tests: 9 → 18 default, plus 1 `slow` integration test, excluded by default.

### Verified (measured, reproducible with the commands in `docs/results.md`)
- `sitemesh sparse` runs end to end from a clean clone on all four HANDOFF captures.
- Registration: object-orbit-2 **85/85**, object-orbit **44/44** (baseline 40/44), on every seed.
- Pose priors are consumed by the 4.1.1 mapper (`Loading pose priors... 83`), and the model
  comes out at device scale (1.000, against 0.36 / 0.16 blind).
- Priors do **not** improve registration, points, reprojection or VIO alignment against blind
  matching. They make the output metric by construction, and with spatial matching cut an
  85-shot run from 144 s to 105 s.
- VIO alignment equals the hydra-vision baselines (20.2 mm / 5.2 mm), not better. A 2 mm prior
  σ still leaves 18.9 mm, so the residual is a real VIO vs image disagreement.
- ARKit→COLMAP axis convention `diag(1,-1,-1)`: rotation RMS 0.43–0.76° on every run.
- Refined focal is 1–2% longer than ARKit's, in every config.
- Output frame: metric and Z-up (object-orbit-2 floor level to 0.09°, cameras 0.7–1.6 m above
  it). `enu_position_only` with an arbitrary heading reported honestly for walks under 3 m.
  Synthetic unit test covers scale, Z-up and "GPS cannot tilt".
- Stance envelope: registration falls off below D/B ≈ 2 (64% → 45% → 8%) across 292 shots.
  `StanceLimits` defaults are unchanged because 2.0 is at the knee.
- M6 budget: 85 shots in ~105 s at ~9 GB peak RSS; 44 shots in ~19 s.

### Not verified
- **Dense and mesh: skipped.** No CUDA on this host (`pycolmap.has_cuda` False). No
  `fused.ply` or `meshed-poisson.ply` was produced; fused point count, mesh quality and dense
  timings are not measured.
- Absolute scale against a known length. No AprilTag or tape measurement exists for these
  captures, so the scale check reports `not_measured` and the gate verdict is never a full `pass`.
- The gate's own `device_alignment_rms_m ≤ 10 mm` target fails on object-orbit-2 (20 mm). The
  cause (pose-sample timing against the shutter, or VIO drift) is not established.
- object-orbit's dominant (desk) plane reads 1.57° from level; whether that's the desk or ARKit
  gravity wasn't checked.
- A fully blind comparison (hydra-vision `AUTO` self-calibrated cameras): `--no-priors` still
  seeds ARKit intrinsics.
- Hard captures: d4737a5d keeps 53/86 and b8b8ebd7 22/77 in the output model. Most misses are
  disconnected sub-models (80/86 and 47/77 reconstructed somewhere). Merging them via the shared
  ARKit frame is proposed, not built.
- One seed only for the two hard captures, and one machine for everything.
- Peak RSS is per subprocess (`ru_maxrss`), not a system-wide memory profile.
