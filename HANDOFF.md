# Handoff prompt: finish and prove SiteMesh

Paste everything below the line into the new agent. Grading criteria are in `GRADING.md`. The
agent may read that file; it is not a secret.

---

You are taking over **SiteMesh** (this repo). SiteMesh turns phone photos plus the ARKit pose
track the phone records into a **metric, georegistered sparse model, dense cloud and mesh**. It
rebuilds an earlier pipeline (hydra-vision). Read `README.md` first, then the rebuild guide it
follows: `docs/rebuild-guide.md` in AnceraInc/hydra-vision (PR #201). If you can't reach it, the
README's "What is different" table is the summary.

## What exists (commit 1, scaffold)

| module | does | state |
|---|---|---|
| `sitemesh/contracts.py` | survey folder + `poses.json` models (hydra-vision compatible) | done |
| `sitemesh/geometry.py` | ARKit→COLMAP axes, Umeyama, yaw-only fit, ray intersection, DLT, WGS84→ENU | unit-tested |
| `sitemesh/qc.py` | stance-card check from `poses.json` alone; post-SfM stance vs registration table | runs on real captures |
| `sitemesh/sfm.py` | ARKit intrinsics → one camera per lens with a prior focal; VIO centres → COLMAP pose priors; spatial matching; mapper with `use_prior_position` | **never completed a real run** (see below) |
| `sitemesh/georef.py` | Umeyama to VIO → level → yaw-only GPS heading → applied to the model before writing | untested on real data |
| `sitemesh/scalecheck.py` | AprilTag (36h11, room-kit sizes) corner triangulation → edge length vs printed size | tested on synthetic renders |
| `sitemesh/dense.py` | undistort → patch-match (explicit depth bounds) → geometric fusion → Poisson | **never run**: needs CUDA |
| `sitemesh/report.py` | acceptance gate; unmeasured never counts as passed | unit-tested |
| `sitemesh/cli.py` | `sitemesh qc / sparse / dense / run / scalecheck <survey>` | imports only |

Test status: 9 tests pass, ruff is clean, pycolmap is pinned to 4.1.1.

## Known facts from the previous machine

- `match_spatial` with `max_distance=1e6` aborts with `std::out_of_range`. It is now set to 500.
  `ignore_z` must be False because ARKit's vertical is Y, not Z.
- The real-data SfM run was stopped during feature extraction because the old machine was too
  slow. **Nothing past extraction has been observed on real data.** Treat every SfM, georef and
  dense claim in the docstrings as a hypothesis.
- ARKit `capturedImage` is 1920×1440 landscape, and per-shot `intrinsics` match that buffer. The
  ARKit→COLMAP camera convention used is `diag(1,-1,-1)`. `device_rotation_rms_deg` in the report
  is the check: a wrong convention shows up as a large constant angle.
- Pose priors are attached with `prior.corr_data_id = image.data_id`. It is unverified whether the
  4.1.1 mapper actually consumes them; check the log and the result.

## Test data

Get these from the owner. Locally they are in
`hydra-vision/data/device-rescue-final/<id>/{raw/,poses.json}`; in the cloud, the `surveys` blob
container. Put each one at `data/<name>/` (gitignored).

| name | id | shots | hydra-vision baseline (blind COLMAP + post-hoc fit) |
|---|---|---|---|
| object-orbit-2 | AB814BD8-424F-460C-AAF7-B57F9C9724F4 | 85 | 85/85 registered, 12,799 pts, 0.786 px, VIO RMS **20 mm** |
| object-orbit | 905068D5-366E-4530-BF39-903A777C95F7 | 44 | 40/44, 3,336 pts, 0.884 px, VIO RMS **5 mm** |
| (harder) | D4737A5D-DD13-4B47-8148-5CEECB0DEFBE | 86 | QC says marginal (median D/B ≈ 1) |
| (large, flat) | B8B8EBD7-00EE-4A1F-A828-3A96F61C2700 | 77 | QC: 52% lack cross-parallax |

## Your tasks, in order

1. **Make it run.** Run `sitemesh sparse` end to end on object-orbit-2, then on object-orbit, and
   fix whatever breaks. Commit each fix with the failure it fixes.
2. **Prove the priors matter.** For both orbit captures, run three configurations with
   `--seed 0..2`:
   - priors + spatial (default)
   - `--no-priors --matcher exhaustive` (the hydra-vision equivalent)
   - priors + exhaustive

   Put a results table in `docs/results.md`: registered, points, reprojection,
   `device_alignment_rms_m`, `device_rotation_rms_deg`, refined focal against the ARKit focal, and
   wall time. Say plainly if priors don't help.
3. **Verify the frame.** Confirm the written `sparse/` and `model.ply` are metric and Z-up: camera
   heights should be about 1–2 m above the lowest points, and the object should be upright.
   Confirm `output_frame` is reported honestly. object-orbit-2 walked only a few metres, so expect
   `enu_position_only`.
4. **Dense.** If this machine has CUDA, build or install a CUDA pycolmap 4.1.1 (or call the
   `colmap` CUDA binary from `dense.py` behind the same interface). Run `sitemesh dense` on
   object-orbit-2 and report the fused point count and mesh. If there is no CUDA, say so and skip;
   do not fake it.
5. **Re-fit the stance envelope.** Use `stance_vs_registration.json` across all four captures:
   - Report the measured D/B at which shots stop registering.
   - Change `StanceLimits` defaults only if the data supports it, and cite the numbers.
6. **Tests.**
   - Add a small integration test that runs `sitemesh sparse` on a ≤15-image subset with priors.
     Mark it `@pytest.mark.slow` and keep it out of default CI.
   - Add a unit test proving `georegister` produces Z-up on a synthetic tilted model.

## Rules

- Never report an unmeasured number as measured. The report gate already encodes this; keep it.
- Keep `poses.json` compatibility with hydra-vision captures.
- Keep commits small, with messages that say *why*. Don't commit `data/`, `.ply` or `.db` files.
- If you change a default, the commit or `docs/results.md` must show the measurement behind it.
- Finish with: all tests passing, ruff clean, `docs/results.md`, and a short `CHANGELOG.md` entry
  listing what you verified and what you could not.
