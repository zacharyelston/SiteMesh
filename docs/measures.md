# Bulk measures (issue #20)

`sitemesh measure <survey> [--dense]` writes `model/measures.json` and a readable
`model/measures.md`. It also runs at the end of every `sitemesh sparse`. Everything is computed
from the georegistered output model (metres, Z up), so the numbers are metric only as far as the
VIO scale is right. Each report says what its scale rests on. **No measure is tape-verified.**

## What it reports

| measure | how |
|---|---|
| site length × width, diagonal, main-axis heading | oriented extent of occupied 25 cm voxels (≥ 2 points), 1% trimmed. Covers everything seen, including background through doors and windows. ±1 voxel resolution. |
| occupied footprint (m²) | occupied 25 cm XY cells |
| horizontal levels | peaks in the height histogram holding ≥ 3% of points: height above ground, tilt, footprint. *Ground* = the lowest level. |
| ceiling | highest level more than 0.3 m above the median camera height |
| tallest point above ground | 99th percentile of occupied voxels |
| camera heights, camera path | from the registered shots, in capture order |
| height along the main axis | 2 m slices perpendicular to the camera track; median and IQR of the 2nd–98th percentile height |
| clear width along the main axis | first *vertical* wall each side (a 10 cm distance bin covering ≥ 40% of the slice height). **Reported only if found in ≥ 3 slices and at least half of them, with IQR ≤ 25% of the median; otherwise "not measured", with the reason.** |
| scale basis + GPS scale check | `VIO only`, `VIO, GPS-consistent` or `VIO, GPS-inconsistent` (`sitemesh/gpscheck.py`) |

### GPS scale check

A 2D similarity from the 4 Hz VIO track to GPS ENU, weighted by reported accuracy, with a 95%
block-bootstrap CI. It runs only with ≥ 30 fixes, a walk extent ≥ 10 m and median accuracy ≤ 10 m;
otherwise it's `not_applicable` with the reason. "Consistent" means 1 is inside the CI or within
±5%. It's stored in `report.json` → `gps_scale_check` as evidence. **It isn't a gate check and never
changes the model.**

## Results on the four captures

Full pipeline (`sitemesh sparse`, default config), seeds 0/1/2,
`SITEMESH_BENCH_DIR=bench-measure scripts/bench.py ... --configs priors_spatial --seeds 0 1 2`.
Ranges are over the three seeds.

| capture | scale basis | length × width | ground levels above ground | ceiling | height along axis | clear width |
|---|---|---|---|---|---|---|
| object-orbit-2 | VIO only (walk 3.3 m < 10 m) | 11.1–11.3 × 4.7–5.1 m | 0, +0.56 m | — | 1.24–1.25 m | not measured |
| object-orbit | VIO only (7 fixes < 30) | 7.2–8.1 × 4.1 m | 0, +0.26 m | — | 0.31–0.32 m | not measured |
| d4737a5d | VIO only (GPS accuracy 15.6 m > 10 m) | 20.1–22.1 × 13.1–13.2 m (14.8 × 3.8 m on the run that kept 66/86) | 0, +0.86, +1.32–1.34, +2.73–2.74 m | **2.73–2.74 m** | **2.77–2.81 m** | not measured |
| b8b8ebd7 | **VIO, GPS-consistent: 1.033 [0.977, 1.077]** | 46.1–46.5 × 21.9–22.4 m | 0, +0.50, +0.75, +0.94, +1.24–1.25 m | — | 5.59–5.64 m | not measured |

### How it did

- **Heights and levels are the robust measures.** They repeat to within 1–2 cm across seeds. The
  corridor ceiling (2.73–2.74 m as a level, 2.77–2.81 m as slice height) agrees with
  experiments.md §5 (2.78 m). Level tilts: object-orbit-2 0.07–0.31°, b8b8ebd7 0.01–0.15°,
  d4737a5d 0.05–1.43°, object-orbit 0.41–2.3° (its small desk-top levels hold few points).
- **The GPS check behaves as designed.** It's consistent on the one outdoor site, and refuses the
  other three, each with a specific reason.
- **Extents follow coverage.** They're stable across seeds when registration is (b8b8ebd7 ±0.4 m on
  46 m). They include background seen through doors and windows, which is why object-orbit, a desk
  scan, reads about 8 × 4 m (the room behind it), with ±0.5 m seed spread from sparse background
  points. On d4737a5d, the run where a degenerate piece was dropped (66/86) covers less corridor,
  and its extent shrinks accordingly.
- **Clear width is never reported on these captures.** The walls are too sparse in the
  reconstruction (white corridor walls, foliage), so the vertical-wall finder succeeds in at most 2
  of 8–11 slices and the report says "not measured". On a synthetic room with solid walls it
  recovers 3.0 m (unit test). Adding the CPU dense-lite points (`--dense`) doesn't change this.
  That's the clearest case for denser reconstruction: it's what width needs.
- **Unexplained:** a level at +0.86 m on d4737a5d holding 16% of the points (a handrail line or
  wainscot band?). Not identified.

### Density matters for extents only through coverage

Extents were first computed from raw point percentiles. Adding the dense-lite points then shrank
b8b8ebd7's length from 44 to 23 m, because percentiles weight dense areas. Switching to occupied
voxels made sparse and sparse + dense agree (46.5 vs 45.8 m).
