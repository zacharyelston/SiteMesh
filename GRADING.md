# Grading the SiteMesh handoff

100 points, plus a list of disqualifiers. Every item can be checked from the repo, the commit
history and a re-run. Re-run, don't trust the write-up.

## How to grade (≈30 min on a fast machine)

```bash
git log --oneline <handoff-base>..HEAD
```

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'
```

```bash
.venv/bin/ruff check sitemesh tests && .venv/bin/python -m pytest -q
```

```bash
sitemesh qc data/object-orbit-2
```

```bash
sitemesh sparse data/object-orbit-2 --seed 0
```

Then compare `data/object-orbit-2/model/report.json` against `docs/results.md`.

## Disqualifiers (cap the score at 40)

- A number in `docs/results.md` that a re-run can't reproduce within a reasonable tolerance:
  registration ±1 image, reprojection ±0.05 px, alignment ±30%.
- A dense result or mesh claimed without CUDA, or a mocked or stubbed stage presented as working.
- The gate changed so that `not_measured` counts as a pass.
- Tests deleted or skipped to get green, or real captures and `.ply`/`.db` files committed.

## Rubric

### A. It runs (25)

| pts | check |
|---|---|
| 10 | `sitemesh sparse` completes on object-orbit-2 and object-orbit from a clean clone |
| 5 | registration on object-orbit-2 ≥ 85/85, matching the hydra-vision baseline |
| 5 | registration on object-orbit ≥ 41/44, beating the 40/44 baseline |
| 5 | each fix is its own commit naming the failure it fixed |

### B. Priors are real and measured (25)

| pts | check |
|---|---|
| 8 | `docs/results.md` has the 3-config × 2-capture × 3-seed table, with every listed column |
| 6 | evidence the mapper consumed the priors: log lines, or results that differ from `--no-priors` |
| 5 | `device_alignment_rms_m` with priors ≤ the hydra-vision baseline (20 mm on orbit-2, 5 mm on orbit), or an honest explanation of why not |
| 3 | `device_rotation_rms_deg` reported and < 3°, which confirms the ARKit axis convention |
| 3 | refined focal against the ARKit focal reported per capture |

### C. Output frame is correct (15)

| pts | check |
|---|---|
| 6 | `model.ply` is metric and Z-up: grader opens it, camera heights are plausible, object upright |
| 4 | `output_frame` / `heading` honest for short walks (`enu_position_only`, not a confident `enu`) |
| 5 | new unit test: synthetic tilted model comes out Z-up through `georegister` |

### D. Dense (10)

| pts | check |
|---|---|
| 10 | CUDA present: `sitemesh dense` produces `fused.ply` and `meshed-poisson.ply` in the output frame, with point count reported |
| 10 | No CUDA: clearly states it was skipped, and why, in results and changelog. Full marks; honesty is the test |

### E. Stance envelope from data (10)

| pts | check |
|---|---|
| 6 | measured D/B vs registered, across ≥ 3 captures, with the numbers or plot in results |
| 4 | `StanceLimits` defaults left alone, or changed with the numbers cited in the commit |

### F. Engineering hygiene (15)

| pts | check |
|---|---|
| 5 | all tests pass, ruff clean, CI workflow still valid |
| 4 | slow integration test on a ≤ 15-image subset, marked `slow`, excluded from default run |
| 3 | `CHANGELOG.md` separates *verified* from *not verified* |
| 3 | no hard-coded absolute paths, and no new dependency without a reason in the commit |

## Score bands

| score | meaning |
|---|---|
| 85–100 | ship it; next step is the phone-side live stance guidance |
| 65–84 | works, but some claims are thin; ask for the missing measurements |
| 40–64 | partial; re-scope before giving it more time |
| < 40 | disqualified, or didn't run |
