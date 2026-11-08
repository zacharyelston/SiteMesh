# SiteMesh results

Every number here was measured on this host from the commits on this branch, and can be
reproduced with the commands given. Nothing was copied from a docstring or estimated. Anything
not measured says so.

**Host:** Apple M6 laptop (12 cores, 32 GB RAM), macOS on AC power, CPU-only pycolmap 4.1.1,
Python 3.12. This laptop is the spec for the field agent. There is no CUDA here (see §4).

**Reproduce:**

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'
.venv/bin/python scripts/bench.py data/object-orbit-2 data/object-orbit --seeds 0 1 2
.venv/bin/python scripts/bench.py data/d4737a5d data/b8b8ebd7 --configs priors_spatial blind_exhaustive --seeds 0
.venv/bin/python scripts/bench.py --table
```

`scripts/bench.py` runs the real `sitemesh sparse` CLI per (capture, config, seed) in a throw-away
survey folder under `bench/`, under `caffeinate -i -s`, and records wall time and peak RSS per
run. It also checks `pmset -g log` for sleep during each run.

> **Timing caveat:** this laptop idle-sleeps after 1 minute (`pmset sleep 1`). In the first pass,
> three runs had a single 2–9 minute gap in their log that matched a sleep entry in `pmset -g log`
> exactly (e.g. 17:22:45 → 17:31:43 inside one bundle adjustment). Those runs were re-run awake,
> and no row below spans a sleep. A field agent on this hardware needs the same sleep assertion.

## 1. It runs (issue #1)

Seed 0, default config (priors + spatial matching), from a clean clone:

| capture | registered | target | hydra-vision baseline |
|---|---|---|---|
| object-orbit-2 | **85/85** | ≥ 85 | 85/85 |
| object-orbit | **44/44** | ≥ 41 | 40/44 |

Three failures were fixed, one commit each (see `git log`):

1. ARKit-`limited` shots had no prior, so spatial matching gave them no pairs (83/85, mapper log
   `connected 83`).
2. `pycolmap.match_spatial` aborted with `std::out_of_range` (SIGABRT) in 3 of 6 identical runs.
   Replaced with numpy k-nearest pairs plus `match_image_pairs`: 0 crashes in 8 repeat runs, and
   none in the ~40 pipeline runs since (bench plus experiments).
3. object-orbit's blurred, oblique IMG_0027–0030 didn't register: 40/44. Affine-shape SIFT fixes
   it (44/44 on every seed). The full option sweep is in that commit message.

## 2. Do ARKit priors beat blind COLMAP? (issue #2)

Configs: `priors_spatial` = default. `blind_exhaustive` = `--no-priors --matcher exhaustive`.
`priors_exhaustive` = `--matcher exhaustive`.

**"Blind" here keeps ARKit intrinsics.** `--no-priors` drops the pose priors only. Every config
starts from the ARKit focal with a prior-focal flag, so this isolates pose priors and matching.
It isn't a full re-creation of hydra-vision's `AUTO` self-calibrated cameras.

### Summary (mean over seeds; peak RSS is the max)

| capture | config | seeds | registered | points (min–max) | reproj px | align RMS mm | rot RMS ° | SfM→device scale | extract s | match s | map s | total s | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object-orbit-2 | priors_spatial | 3 | 85/85 | 14,410 (14,408–14,412) | 0.821 | 20.2 | 0.43 | 1.000–1.000 | 47 | 49 | 7.0 | 105 | 8.9 |
| object-orbit-2 | blind_exhaustive | 3 | 85/85 | 14,542 (14,528–14,561) | 0.823 | 20.2 | 0.43 | 0.360–0.360 | 47 | 88 | 6.6 | 144 | 8.9 |
| object-orbit-2 | priors_exhaustive | 3 | 85/85 | 14,546 (14,530–14,573) | 0.822 | 20.1 | 0.44 | 1.000–1.000 | 50 | 91 | 8.1 | 152 | 8.9 |
| object-orbit | priors_spatial | 3 | 44/44 | 5,011 (4,992–5,026) | 0.938 | 5.2 | 0.63 | 1.000–1.000 | 11 | 6 | 2.0 | 19 | 8.8 |
| object-orbit | blind_exhaustive | 3 | 44/44 | 5,023 (5,012–5,034) | 0.939 | 5.1 | 0.63 | 0.159–0.159 | 11 | 6 | 1.8 | 20 | 8.5 |
| object-orbit | priors_exhaustive | 3 | 44/44 | 5,025 (5,004–5,041) | 0.939 | 5.1 | 0.63 | 1.000–1.000 | 12 | 6 | 2.1 | 20 | 8.8 |
| d4737a5d | priors_spatial | 1 | 53/86 | 4,000 | 0.938 | 15.7 | 0.74 | 1.000 | 15 | 4 | 2.8 | 22 | 8.8 |
| d4737a5d | blind_exhaustive | 1 | 53/86 | 4,076 | 0.933 | 17.7 | 0.76 | 0.998 | 15 | 7 | 2.6 | 25 | 8.2 |
| b8b8ebd7 | priors_spatial | 1 | 22/77 | 4,902 | 0.766 | 133.1 | 0.59 | 1.000 | 25 | 26 | 5.4 | 57 | 8.8 |
| b8b8ebd7 | blind_exhaustive | 1 | 22/77 | 4,901 | 0.766 | 136.1 | 0.57 | 0.939 | 26 | 43 | 3.5 | 73 | 8.8 |

Hydra-vision baselines (blind COLMAP + post-hoc fit): object-orbit-2 85/85, 12,799 pts, 0.786 px,
20 mm. object-orbit 40/44, 3,336 pts, 0.884 px, 5 mm.

### What the numbers say

**Priors are consumed.** The mapper log reads `Loading pose priors... 83` (object-orbit-2) and
`... 42` (object-orbit), the number `write_priors` wrote. The reconstruction comes out of SfM
already at device scale: `sfm_to_device_scale` is 1.000 ± 0.00005 with priors, against 0.360
(object-orbit-2) and 0.159 (object-orbit) blind. Blind runs also log `Loading pose priors... 85`:
those are EXIF-GPS priors pycolmap 4.1.1 writes during extraction (cleared in priors mode), and
they go unused because `use_prior_position` is False. Blind scale on d4737a5d (0.998) and
b8b8ebd7 (0.939) happens to land near 1, most likely because COLMAP sets the initial pair's
baseline to one unit and those pairs were about 1 m apart. Not investigated further.

**Priors don't improve accuracy or registration.** Within each capture, all configs register
the same images. Points differ by under 1% and reprojection by under 0.005 px. VIO alignment is
within 3% on three captures; d4737a5d is the exception at 15.7 mm with priors against 17.7 mm
blind, one seed each. Said plainly: **feeding ARKit pose priors into SfM
did not make these reconstructions better. It made them metric by construction, and with
spatial matching, faster.** The registration gains over hydra-vision (40 → 44 on object-orbit)
come from affine-shape SIFT, and the blind config gets them too.

**Speed:** spatial matching over the priors cuts object-orbit-2's matching from 88–91 s to 49 s,
and its total from 144 s to 105 s (−27%). On 44 images there's no difference, since 40 neighbours
is already nearly exhaustive. b8b8ebd7: matching 43 → 26 s.

**`device_alignment_rms_m` against the hydra-vision baseline:** object-orbit-2 20.2 mm (baseline
20 mm), object-orbit 5.2 mm (baseline 5 mm). Equal, not better. Tightening the prior can't
explain the gap away:

| object-orbit-2, seed 0 | prior σ | align RMS |
|---|---|---|
| default | 5 cm | 20.2 mm |
| `--prior-sigma-m 0.01` | 1 cm | 20.2 mm |
| `--prior-sigma-m 0.002` | 2 mm | 18.9 mm |

With no robust loss on the prior (`use_robust_loss_on_prior_position` is False), a 2 mm σ should
pin the cameras if VIO and images agreed. The ~19–20 mm residual is a real disagreement between
the per-shot ARKit positions and the image geometry, and it's the same 20 mm the post-hoc fit
found. Candidate causes, **not yet tested**: the ARKit pose is the last sample near the shutter,
not at it (hand motion), and VIO drift over the orbit. The gate's own target (10 mm) fails on
object-orbit-2 for this reason. The default σ is unchanged. On b8b8ebd7 a 1 cm σ moves alignment
from 133 to 108 mm with registration unchanged (22/77).

**ARKit axis convention (`diag(1,-1,-1)`) confirmed:** `device_rotation_rms_deg` is 0.43–0.76° on
every run (< 3°). A wrong convention shows up as a constant angle of tens of degrees.

**Refined focal against ARKit focal** (px, per camera group; ARKit fx varies with focus, so
images group into 1–3 cameras):

| capture | refined vs ARKit |
|---|---|
| object-orbit-2 | 1455 vs 1432 (+1.6%), 1460 vs 1439 (+1.5%) |
| object-orbit | 1468 vs 1443 (+1.7%), 1472–1474 vs 1459 (+0.9–1.1%), 1500–1522 vs 1473 (+1.9–3.3%) |
| d4737a5d | 1432–1433 vs 1429 (+0.2%) |
| b8b8ebd7 | 1454 vs 1432 (+1.5%), 1447–1448 vs 1430 (+1.2%) |

BA consistently lengthens the ARKit focal by 1–2%, the same in every config. The third
object-orbit group (5 images) varies most across seeds.

**Run-to-run spread:** identical runs aren't bit-identical (threaded RANSAC). Across seeds,
registration was identical, points varied ≤ 0.8%, reprojection ≤ 0.005 px, and alignment ≤ 0.2 mm.

**M6 field budget:** an 85-shot capture takes 105 s end to end with a ~9 GB peak RSS, and a 44-shot
capture 19 s. Extraction and matching dominate; mapping is under 10 s.

### Every run

| capture | config | seed | registered | points | reproj px | align RMS mm | rot RMS ° | SfM→device scale | refined vs ARKit focal (px) | extract s | match s | map s | total s | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| object-orbit-2 | priors_spatial | 0 | 85/85 | 14,412 | 0.822 | 20.2 | 0.43 | 1.00000 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 47 | 49 | 6.9 | 105 | 8.9 |
| object-orbit-2 | priors_spatial | 1 | 85/85 | 14,411 | 0.821 | 20.2 | 0.44 | 0.99999 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 45 | 48 | 7.1 | 102 | 8.9 |
| object-orbit-2 | priors_spatial | 2 | 85/85 | 14,408 | 0.821 | 20.2 | 0.43 | 1.00000 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 49 | 49 | 7.1 | 107 | 8.9 |
| object-orbit-2 | blind_exhaustive | 0 | 85/85 | 14,537 | 0.823 | 20.1 | 0.44 | 0.35994 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 47 | 87 | 6.6 | 143 | 8.9 |
| object-orbit-2 | blind_exhaustive | 1 | 85/85 | 14,528 | 0.823 | 20.2 | 0.44 | 0.35988 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 46 | 88 | 6.5 | 142 | 8.9 |
| object-orbit-2 | blind_exhaustive | 2 | 85/85 | 14,561 | 0.822 | 20.2 | 0.43 | 0.35998 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 48 | 90 | 6.6 | 147 | 8.9 |
| object-orbit-2 | priors_exhaustive | 0 | 85/85 | 14,530 | 0.823 | 20.2 | 0.44 | 1.00000 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 51 | 93 | 7.9 | 154 | 8.9 |
| object-orbit-2 | priors_exhaustive | 1 | 85/85 | 14,573 | 0.823 | 20.1 | 0.44 | 1.00000 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 50 | 90 | 8.1 | 151 | 8.9 |
| object-orbit-2 | priors_exhaustive | 2 | 85/85 | 14,534 | 0.820 | 20.0 | 0.44 | 0.99999 | 1455 vs 1432 (+1.6%) / 1460 vs 1439 (+1.5%) | 50 | 91 | 8.1 | 152 | 8.9 |
| object-orbit | priors_spatial | 0 | 44/44 | 5,016 | 0.940 | 5.2 | 0.63 | 1.00000 | 1468 vs 1443 (+1.7%) / 1474 vs 1459 (+1.1%) / 1510 vs 1473 (+2.6%) | 10 | 5 | 2.0 | 18 | 8.5 |
| object-orbit | priors_spatial | 1 | 44/44 | 4,992 | 0.937 | 5.2 | 0.63 | 0.99999 | 1468 vs 1443 (+1.7%) / 1474 vs 1459 (+1.0%) / 1520 vs 1473 (+3.2%) | 11 | 6 | 2.0 | 20 | 8.8 |
| object-orbit | priors_spatial | 2 | 44/44 | 5,026 | 0.937 | 5.2 | 0.63 | 0.99996 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1500 vs 1473 (+1.9%) | 11 | 6 | 2.1 | 20 | 8.8 |
| object-orbit | blind_exhaustive | 0 | 44/44 | 5,034 | 0.939 | 5.1 | 0.63 | 0.15928 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1505 vs 1473 (+2.2%) | 11 | 6 | 1.7 | 20 | 8.5 |
| object-orbit | blind_exhaustive | 1 | 44/44 | 5,023 | 0.936 | 5.1 | 0.63 | 0.15942 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1506 vs 1473 (+2.3%) | 11 | 6 | 1.7 | 20 | 8.5 |
| object-orbit | blind_exhaustive | 2 | 44/44 | 5,012 | 0.941 | 5.2 | 0.63 | 0.15929 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1522 vs 1473 (+3.3%) | 12 | 6 | 1.9 | 20 | 8.5 |
| object-orbit | priors_exhaustive | 0 | 44/44 | 5,031 | 0.939 | 5.1 | 0.63 | 0.99998 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1504 vs 1473 (+2.2%) | 12 | 6 | 2.0 | 20 | 8.8 |
| object-orbit | priors_exhaustive | 1 | 44/44 | 5,004 | 0.938 | 5.2 | 0.63 | 0.99998 | 1468 vs 1443 (+1.7%) / 1473 vs 1459 (+1.0%) / 1515 vs 1473 (+2.9%) | 11 | 6 | 2.0 | 20 | 8.8 |
| object-orbit | priors_exhaustive | 2 | 44/44 | 5,041 | 0.938 | 5.1 | 0.62 | 0.99996 | 1468 vs 1443 (+1.7%) / 1472 vs 1459 (+0.9%) / 1506 vs 1473 (+2.2%) | 11 | 6 | 2.2 | 20 | 8.8 |
| d4737a5d | priors_spatial | 0 | 53/86 | 4,000 | 0.938 | 15.7 | 0.74 | 1.00000 | 1433 vs 1429 (+0.2%) | 15 | 4 | 2.8 | 22 | 8.8 |
| d4737a5d | blind_exhaustive | 0 | 53/86 | 4,076 | 0.933 | 17.7 | 0.76 | 0.99843 | 1432 vs 1429 (+0.2%) | 15 | 7 | 2.6 | 25 | 8.2 |
| b8b8ebd7 | priors_spatial | 0 | 22/77 | 4,902 | 0.766 | 133.1 | 0.59 | 1.00000 | 1454 vs 1432 (+1.5%) / 1448 vs 1430 (+1.2%) | 25 | 26 | 5.4 | 57 | 8.8 |
| b8b8ebd7 | blind_exhaustive | 0 | 22/77 | 4,901 | 0.766 | 136.1 | 0.57 | 0.93864 | 1454 vs 1432 (+1.5%) / 1447 vs 1430 (+1.2%) | 26 | 43 | 3.5 | 73 | 8.8 |

### Not yet done

- A fully blind config (hydra-vision's `AUTO` self-calibrated cameras, no ARKit intrinsics).
  There's no flag for it yet.
- Reproducibility on another machine: every number here is from one M6.

## 3. Output frame (issue #3)

`python scripts/frame_check.py <survey>` on the seed-0 default models (`bench/<capture>__priors_spatial__s0`):

| check | object-orbit-2 | object-orbit |
|---|---|---|
| `output_frame` | `enu_position_only` | `enu_position_only` |
| `heading` | `arbitrary (walked 1.7 m < 3.0 m)` | `arbitrary (walked 0.9 m < 3.0 m)` |
| GPS yaw-fit RMS (reported, not applied) | 2.34 m | 0.59 m |
| `sfm_to_device_scale` | 1.000 | 1.000 |
| dominant horizontal plane | carpet floor: 8,775 of 14,412 points within ±3 cm | desk top: 3,166 of 5,016 points |
| plane tilt from +Z | **0.09°** | **1.57°** |
| camera height above that plane | 0.74 / 1.11 / 1.55 m (min / median / max) | 0.14 / 0.42 / 0.64 m |
| mean camera pitch | −32° (looking down) | −44° |

- **Metric and Z-up.** The floor in object-orbit-2 is level to 0.09°, and hand-held camera
  heights of 0.7–1.6 m above it are plausible. object-orbit's dominant plane is the desk the object
  sits on; cameras 0.1–0.6 m above it match leaning over a desk. Its 1.57° tilt is measured as-is;
  a real desk or ARKit's gravity estimate could account for it, and neither was checked.
- **Upright.** A side view of object-orbit-2's `model.ply` (X across, Z up, `--png`) shows a flat
  floor with the ottoman, bottle and crate standing on it. Colour-selected yellow bottle points sit
  0.40–0.59 m above the floor, consistent with a bottle on a ~0.45 m pouf. Nothing was
  tape-measured, so this is a sanity check, not a scale verification.
- **`output_frame` is honest for short walks.** Both walked under the 3 m threshold, so GPS sets
  position only and the heading is reported as arbitrary. A GPS heading fit would have used fixes
  whose RMS error (2.3 m) exceeds the walk.
- `model.ply`, `cameras.json` and `sparse/` are all written after `georegister` transforms the
  reconstruction (`pipeline.run_sparse`), so they share this frame.

**Unit test** `tests/test_georef.py`: a synthetic reconstruction pushed into a 50° tilted, 0.37×
scaled gauge comes back on the truth to 1e-6 m with +Z up. A 1.5 m walk gives
`enu_position_only` with no rotation applied. A 20 m walk with 8 m GPS altitude noise gets the
GPS heading but no tilt (heights off by one constant only). Writing it exposed a crash in
`georef._gps_at` when a shot and the first GPS fix share a timestamp, now fixed in its own commit.

## 4. Dense and mesh (issue #4): **skipped, no CUDA**

COLMAP patch-match stereo is CUDA-only. This host is an Apple M6: `pycolmap.has_cuda` is `False`
and there is no NVIDIA GPU, so no CUDA build of pycolmap 4.1.1 can be installed here, and calling
the `colmap` CUDA binary isn't possible either. `sitemesh dense data/object-orbit-2` returns:

```json
{"status": "skipped",
 "reason": "patch-match stereo needs CUDA (pycolmap.has_cuda is False on this host). Run `sitemesh dense <survey>` on a CUDA host.",
 "depth_bounds_m_planned": [0.429, 12.146]}
```

- No `dense/fused.ply` or `dense/meshed-poisson.ply` was produced, and none is claimed.
- **Not measured:** fused point count, mesh, dense timings, and whether the dense outputs share
  `model.ply`'s frame (by construction they should: dense reads the georegistered `sparse/`).
- `depth_bounds_m_planned` is computed on CPU from the sparse model (1st–99th percentile depth,
  ±50%). These are the bounds a CUDA run would pass to patch-match.
- `tests/test_dense.py` checks the no-CUDA path: status `skipped`, sane bounds, and no `dense/`
  directory written.

For the field agent: dense and mesh on this hardware need either a GPU host or a non-COLMAP
Metal/CPU MVS. Neither was tried.

## 5. Stance envelope from data (issue #5)

`python scripts/stance_fit.py bench/<capture>__priors_spatial__s0 ...` over all four captures
(292 shots, default config, seed 0). "In output model" means the shot is in the model sitemesh
writes (the largest). "In any sub-model" means COLMAP reconstructed it in some disconnected
piece.

| capture | in output model | in any sub-model | QC verdict | QC median D/B | cross-parallax |
|---|---|---|---|---|---|
| object-orbit-2 | 85/85 | 85/85 | ok | 3.63 | 99% |
| object-orbit | 44/44 | 44/44 | ok | 4.13 | 100% |
| d4737a5d | 53/86 | 80/86 | marginal | 0.96 | 97% |
| b8b8ebd7 | 22/77 | 47/77 | marginal | 3.53 | 48% |

### D/B against registration

D/B as the QC computes it from `poses.json` before SfM: subject distance by ray convergence ÷
baseline to the previous shot. This is what the on-site check and `StanceLimits` use. Available
for 185 of 292 shots.

| QC D/B | shots | in output model | rate | in any sub-model |
|---|---|---|---|---|
| 0–1 | 13 | 1 | 8% | 4 (31%) |
| 1–1.5 | 11 | 5 | 45% | 8 (73%) |
| 1.5–2 | 11 | 7 | 64% | 8 (73%) |
| 2–3 | 34 | 30 | 88% | 32 (94%) |
| 3–4 | 37 | 34 | 92% | 36 (97%) |
| 4–6 | 54 | 46 | 85% | 51 (94%) |
| 6–10 | 17 | 14 | 82% | 15 (88%) |
| 10–15 | 6 | 1 | 17% | 5 (83%) |
| 15+ | 2 | 1 | 50% | 1 |

Cross-check with D/B from the reconstruction: median triangulated depth of registered shots
within ±2 in capture order ÷ VIO baseline. Defined for registered and unregistered shots alike;
available for 217 shots:

| measured D/B | shots | in output model | rate |
|---|---|---|---|
| < 1.5 | 5 | 1 | 20% |
| 1.5–2 | 5 | 4 | 80% |
| 2–3 | 27 | 23 | 85% |
| 3–6 | 96 | 93 | 97% |
| 6–15 | 61 | 56 | 92% |
| 15+ | 23 | 21 | 91% |

**Where registration falls off:** below **D/B ≈ 2**, and steeply below 1.5. From 2 to 10,
registration is flat at 82–92% (88–97% counting sub-models), so the 2–3 band does as well as the
card's ÷6 target. The card's "closer than ÷3 breaks matching" bound is too strict by this data.
Below 2 it drops to 64%, then 45%, then 8%. The hedge `StanceLimits.min_d_over_b = 2.0` sits at
the measured knee.

**`StanceLimits` defaults are unchanged.** `min_d_over_b = 2.0` matches the data. The upper bound
`max_d_over_b = 15` can't be re-fit: only 8 shots fall above 10, and their misses are
connectivity (5 of 6 in the 10–15 band reconstructed in a sub-model).

Caveats: the D/B < 2 bins hold 35 shots, mostly from d4737a5d and b8b8ebd7. The baseline is to
the previous shot in capture order, not to the best-overlapping shot. One seed per capture.

### Did the QC predict the outcome?

- **Verdict:** ranks correctly. `ok` gave 100% and 100%; `marginal` gave 62% (d4737a5d) and 29%
  (b8b8ebd7). But "marginal: expect some unregistered shots" badly undersells b8b8ebd7, where 71%
  of shots missed the output model.
- **Per-shot flags:**

| QC flag | shots | in output model | in any sub-model |
|---|---|---|---|
| (none) | 215 | 179 (83%) | 208 (97%) |
| `too_wide` (D/B < 2) | 35 | 13 (37%) | 20 (57%) |
| `no_cross_parallax` | 44 | 7 (16%) | 24 (55%) |
| `tracking_limited` | 6 | 5 | 5 |
| `wrist_pivot` | 2 | 2 | 2 |
| `too_tight` | 2 | 1 | 1 |

  `no_cross_parallax` is the strongest per-shot predictor of missing the output model (16%
  against 83% unflagged), but `_verdict` ignores it ("informational"). On b8b8ebd7 it flags 40 of
  77 shots. Counting it in the verdict would have called b8b8ebd7 a re-shoot. **Not changed
  here**: it rests on one capture.
- **Coverage gap:** the QC estimated subject distance for only **11 of 86** d4737a5d shots (ray
  convergence fails on near-parallel views), so its "median D/B ≈ 1" headline rests on 13% of the
  capture.

### What the hard captures need (not implemented)

Most "failures" on the hard captures are **connectivity, not per-shot geometry**. d4737a5d
reconstructed 80/86 shots in three pieces (shots 2–15, 17–71, 73–86); b8b8ebd7 reconstructed
47/77 in four. With pose priors every piece is solved in the same metric ARKit frame
(`sfm_to_device_scale` = 1.000), so the pieces could be merged by transform with no shared
features: 53 → 80 on d4737a5d. That's the strongest argument found for the priors, and the
recommended next step.

## 6. Merging disconnected sub-models (issue #14)

For site surveys the deliverable is bulk measures over the whole site, so coverage matters more
than per-point precision. COLMAP splits a capture wherever consecutive shots share too few
features, and sitemesh used to keep only the largest piece. Every piece is solved against the
same metric VIO priors, so `merge_submodels` aligns each piece to VIO on its own, combines them,
re-triangulates from the database, and runs a pose-prior bundle adjustment. Default on;
`--no-merge` restores keep-the-largest.

```bash
SITEMESH_BENCH_DIR=bench-merge .venv/bin/python scripts/bench.py data/d4737a5d data/b8b8ebd7 --configs priors_spatial --seeds 0 1 2
SITEMESH_BENCH_DIR=bench-merge .venv/bin/python scripts/bench.py data/object-orbit-2 data/object-orbit --configs priors_spatial --seeds 0
```

| capture | seed | before (largest piece) | **after merge** | points | reproj px | VIO RMS mm | scene diagonal | RMS / diagonal | merge s |
|---|---|---|---|---|---|---|---|---|---|
| d4737a5d | 0 | 53/86 | **80/86** | 4,000 → 5,435 | 0.938 → 0.972 | 15.7 → 19.3 | 21.6 m | 0.09% | 0.5 |
| d4737a5d | 1 | — | **80/86** | 5,274 | 0.967 | 21.0 | | | 0.8 |
| d4737a5d | 2 | — | **66/86** | 4,501 | 0.953 | 21.0 | | | 0.5 |
| b8b8ebd7 | 0 | 22/77 | **47/77** | 4,902 → 10,534 | 0.766 → 0.846 | 133.1 → 90.2 | 45.8 m | 0.20% | 1.0 |
| b8b8ebd7 | 1 | — | **47/77** | 10,503 | 0.844 | 90.1 | | | 0.9 |
| b8b8ebd7 | 2 | — | **47/77** | 10,453 | 0.843 | 90.2 | | | 1.1 |
| object-orbit-2 | 0 | 85/85 | 85/85 (one piece, untouched) | 14,424 | 0.822 | 20.1 | 7.7 m | 0.26% | 0.0 |
| object-orbit | 0 | 44/44 | 44/44 (one piece, untouched) | 5,009 | 0.936 | 5.1 | 3.0 m | 0.17% | 0.0 |

"Before" is the seed-0 default run from §2. "Scene diagonal" is the horizontal 2nd–98th
percentile extent of the merged point cloud.

### Per-piece VIO fit (seed 0)

| capture | piece | images | own VIO RMS | kept |
|---|---|---|---|---|
| d4737a5d | largest | 53 | 16 mm | yes |
| | | 14 | 12 mm | yes |
| | | 13 | 31 mm | yes |
| b8b8ebd7 | largest | 22 | 134 mm | yes |
| | | 15 | 131 mm | yes |
| | | 14 | 13 mm | yes |
| | | 11 | 9 mm | yes |

### What it means

- **Coverage:** +27 shots on d4737a5d and +25 on b8b8ebd7 land in one metric model. The merge
  costs about 1 s.
- **The joins rest on VIO alone.** `n_joint_points` is **0** on every real capture: no
  feature track crosses a join, so each piece's placement relative to the others comes from ARKit
  tracking, accurate to roughly that piece's own VIO RMS (9–134 mm here). That's adequate for
  bulk measures (0.09–0.20% of the scene); it's not a photogrammetric tie.
- **Still in `enu`, and still Z-up.** Both hard captures were already `enu` before merging (the
  largest piece walked over 3 m). Merging lengthens the walk the GPS heading is fitted over: 7.3
  → 8.2 m (d4737a5d) and 7.2 → 12.5 m (b8b8ebd7). `frame_check.py` on the merged models: dominant
  plane tilt 0.08° (d4737a5d, a corridor ceiling with cameras about 1 m below) and 0.12°
  (b8b8ebd7, ground with cameras 1.1–2.1 m above). Rotation RMS 0.55–1.0°.
- **Reprojection rises** slightly (+0.03–0.08 px) because the pieces are held by VIO rather than
  shared features. It stays under 1.0 px.
- **Rejection works, but the outcome varies with seed.** On d4737a5d seed 2 the mapper split
  differently and produced a 6-image piece that disagrees with VIO by **0.32 m**. It was dropped
  (`max_segment_rms_m = 0.25`), leaving 66/86 instead of 80/86. The threshold keeps every piece
  seen so far (≤ 134 mm) and drops that degenerate one. Its value is a judgement from these 21
  pieces, not a fit.
- **Gate:** the merged hard captures still `fail` the gate. They're under 98% registration, and
  their VIO RMS is over the 10 mm target. The target is unchanged (see Not yet done).

### Not yet done

- Shots that never reconstructed (d4737a5d 6, b8b8ebd7 30) are still missing. With priors they
  could be placed from VIO and triangulated against neighbours. Not tried.
- The gate's 10 mm VIO target suits object scans, not 20–50 m sites. A relative target
  (RMS / scene size) would match how bulk measures are used. Not changed without a decision.
