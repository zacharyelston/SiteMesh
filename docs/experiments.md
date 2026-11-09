# Experiments with the existing captures (issue #18)

What more can the four reference captures give a site survey, with no printed tags, no tape and
no CUDA? Every number below was measured on the M6, and every command is
`python scripts/experiments.py ...` or `python scripts/dense_cpu.py ...` against the bench
models (`bench/` = pre-merge, `bench-gate/` = current pipeline). Negative results are kept.

## Summary

| # | question | answer |
|---|---|---|
| 1 | Is the ~20 mm VIO disagreement a pose-timing error? | **No.** The best time offset is 0 s on every capture. |
| 2 | Is it device motion at the shutter? | **No.** Speed vs residual correlation flips sign between captures (−0.27 to +0.26). |
| 3 | Is it slow drift? | **Yes, mostly.** Residuals are smooth in capture order, and removing a smooth trend halves them. |
| 4 | Can CPU stereo densify without CUDA? | **Somewhat.** 1.1–2.7× the sparse points where shot spacing allows, at 0.3–1.4% depth error, in ≤ 15 s. |
| 5 | Can we read bulk measures? | **Yes.** Corridor ceiling height 2.78 m, consistent to ±2 cm along 14 m (not tape-checked). |
| 6 | Is there free ground truth? | **Weak.** The pool waterline is level within its ±0.6° uncertainty. |
| 7 | Does GPS confirm scale independently? | **Outdoors, to ±5%:** GPS/VIO scale 1.03 [0.98, 1.08] on the backyard. Useless indoors. |
| 8 | Can never-reconstructed shots be recovered? | **Rarely:** 4 of 36. Most share no verified feature with the model. |

## 1–3. Where the ~20 mm VIO residual comes from

`experiments.py timing|drift data/<c> bench/<c>__blind_exhaustive__s0/model/sparse`. Blind models
are used because their geometry comes from the images only, so the priors can't bias them.

`poses.json` holds the ARKit track at 4 Hz (0.25 s); each still has exactly one sample.

| capture | speed at shutter (median / p90 m/s) | RMS at Δt = 0 | best Δt | lag-1 cosine of residuals (shuffled) | RMS after removing smooth drift | speed↔residual Spearman |
|---|---|---|---|---|---|---|
| object-orbit-2 | 0.41 / 0.52 | 20.1 mm | 0.00 s | **+0.75** (0.00 ± 0.08) | **9.2 mm** | −0.27 |
| object-orbit | 0.21 / 0.31 | 5.1 mm | 0.00 s | **+0.69** (−0.01 ± 0.09) | 2.9 mm | +0.03 |
| d4737a5d | 0.52 / 0.86 | 17.7 mm | 0.00 s | **+0.51** (−0.01 ± 0.08) | 13.8 mm | +0.21 |
| b8b8ebd7 | 0.82 / 1.06 | 136.1 mm | −0.05 s | **+0.67** (−0.02 ± 0.16) | 86.9 mm | +0.26 |

- **Timing is right.** Shifting the VIO sample by ±50 ms makes object-orbit-2 worse (23–26 mm);
  the minimum sits at 0 s.
- **Motion isn't the driver**, despite the phone moving at 0.2–0.8 m/s at the shutter (the protocol
  says stand still; the operator didn't).
- **Drift is.** Neighbouring shots' residuals point the same way (cosine +0.5 to +0.75, against 0
  shuffled), and a 7-shot moving average explains half the RMS. Whether ARKit or the SfM chain is
  the one drifting can't be told without ground truth; ARKit drift is the usual suspect.
- **Consequence:** don't pin SfM to VIO. The default prior σ (5 cm) already lets the images set the
  local shape; tightening it to 2 mm only moved object-orbit-2 from 20.2 to 18.9 mm (§2 of
  results.md).

## 4. CPU dense-lite (`scripts/dense_cpu.py`)

Semi-global matching (OpenCV SGBM) on rectified pairs with the known poses, half resolution, up to
3 sideways partners per image (baseline 3–15% of depth, ≤ 25° apart). Pairs whose depth disagrees
with the sparse points they see by more than 3% are dropped. Depths are clipped to the sparse depth
range ±20%, and only 2 cm voxels seen from ≥ 2 distinct reference views are kept.

| capture | pairs kept / tried | median pair depth error vs sparse | fused points | sparse points | time |
|---|---|---|---|---|---|
| object-orbit-2 | 45 / 73 | 0.96% | 15,405 | 14,407 | 11 s |
| object-orbit | 6 / 8 | 0.35% | 182 | 5,017 | 1.5 s |
| d4737a5d | 19 / 52 | 1.35% | 14,390 | 5,397 | 4.5 s |
| b8b8ebd7 | 19 / 23 | 0.70% | 18,375 | 10,512 | 3.6 s |

- It runs anywhere and is fast. The gain is modest because **the stance protocol spaces shots
  wide** (D/B ≈ 3–4), which is good for SfM but leaves few short-baseline pairs for stereo.
  object-orbit has almost none.
- Getting it right took two fixes, both kept: without depth clipping, untextured pixels formed
  streaks along view rays that the sparse-point check couldn't see; and support has to count
  distinct reference views.
- Full resolution (`--scale 1`) on object-orbit-2: 22,562 points in 31 s, but only 20 pairs pass.
- **Not a mesh**, and not a substitute for patch-match. It's a denser, validated point set for
  bulk measures.

## 5. Bulk measures: the corridor (d4737a5d)

`experiments.py corridor bench-gate/d4737a5d__priors_spatial__s0/model/sparse [--extra-ply .../dense_cpu/fused.ply]`.
2 m slices perpendicular to the camera track (14.1 m long):

| points used | floor-to-ceiling height (median, IQR) | wall-to-wall width (median, IQR) |
|---|---|---|
| sparse (5,397) | **2.78 m** (2.77–2.81) | 1.92 m (1.83–4.16) |
| sparse + dense-lite (19,787) | **2.78 m** (2.73–2.79) | 2.04 m (1.99–3.21) |

Height is consistent to ±2 cm along the corridor, about a 9 ft commercial ceiling. Width is noisier
because doorways and side openings widen some slices; the dense points tighten its IQR. **Neither
is tape-verified**: one tape measurement of this ceiling would turn it into a scale check.

## 6. Free ground truth: the pool waterline (b8b8ebd7)

Water is level by physics. The pool's cyan waterline tiles are the only pool points in the sparse
model (the water itself has no features). They form a line, not a plane, so this checks tilt about
one axis only. `experiments.py waterline <model>`:

| model | points | length | slope | ±1σ | vertical scatter |
|---|---|---|---|---|---|
| merged (current) | 32 | 8.3 m | −0.50° | 0.65° | 141 mm |
| largest piece only | 39 | 8.6 m | +0.10° | 0.58° | 140 mm |

Consistent with level, but too scattered (tile band plus coping, mostly 3-view points) to be a
strong check. Compare §3 of results.md: floors and ceilings are level to 0.08–0.12°.

## 7. GPS as an independent scale check

`experiments.py gps-scale data/<c>`: 2D similarity from the VIO track (all 4 Hz samples,
interpolated to GPS times) to GPS ENU, weighted by reported accuracy. The CI is a block bootstrap
(blocks of 10 fixes, since GPS errors are correlated in time).

| capture | fixes | GPS accuracy (median) | VIO extent | GPS/VIO scale | 95% CI |
|---|---|---|---|---|---|
| **b8b8ebd7 (outdoor)** | 256 | 4.7 m | 26 × 24 m | **1.033** | **0.977–1.077** |
| d4737a5d (indoor) | 83 | 15.6 m | 14 × 5 m | 1.287 | 1.019–1.477 |
| object-orbit-2 | 70 | 7.8 m | 2.5 × 3.3 m | 0.211 | 0.138–0.916 |
| object-orbit | 7 | 11.7 m | 1.1 × 1.1 m | — | too few fixes |

Outdoors over ~25 m, GPS independently confirms the metric scale to about ±5%. Indoors, GPS is
biased (multipath) and the result is meaningless. Small walks are dominated by GPS noise.

## 8. Recovering never-reconstructed shots

`experiments.py recover bench-gate/<c>__priors_spatial__s0`: for each missing shot, collect its
verified database matches to registered images, solve PnP + RANSAC against those 3D points, and
accept if ≥ 15 inliers and within 0.3 m / 5° of the VIO pose.

| capture | missing | 0 correspondences | ≥ 6 correspondences | accepted |
|---|---|---|---|---|
| b8b8ebd7 | 30 | 20 | 8 | **3** (IMG_0025, 0029, 0052: 21–27 inliers, 0.07–0.20 m from VIO) |
| d4737a5d | 6 | 1 | 3 | **1** (IMG_0061: 25 inliers, 18 mm from VIO) |

Most missing shots share **no verified feature** with the model: foliage close-ups and sky on the
backyard. VIO could still place them as camera positions, but they'd add no 3D points, so they're
useless for measurement. Not built into the pipeline.

## What this says about the next captures

- **One known length per site** (a tape-measured ceiling, door or marked stick) turns §5 into a scale
  verification. The data already supports the measurement; only the reference is missing.
- **Overlap beats coverage:** the never-recovered shots (§8) and the split sub-models are shots with
  no shared texture with their neighbours. Point at textured, static surfaces (patio, fence, walls),
  not moving foliage or sky.
- **For CPU dense, add short-baseline pairs:** a second shot about 10 cm sideways at some stances
  would feed §4 directly.
- **Outdoors, GPS gives a free ±5% scale check** over ~25 m of walking (§7). It's worth keeping in the
  report for outdoor sites.
