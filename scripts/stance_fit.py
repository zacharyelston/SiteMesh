"""Relate capture D/B to registration across surveys (issue #5).

Recomputes stance_vs_registration from each survey's saved sparse model with the current
`post_sfm_stance` (so models built before it gained neighbour D/B still work), then prints:
per-survey QC verdict vs outcome, registration rate per D/B bin, and flag vs outcome counts.

    python scripts/stance_fit.py bench/object-orbit-2__priors_spatial__s0 bench/d4737a5d__...
    python scripts/stance_fit.py --json out.json <surveys...>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

from sitemesh.contracts import Survey
from sitemesh.qc import post_sfm_stance, stance_qc

BINS = [0, 1, 1.5, 2, 3, 4, 6, 10, 15, np.inf]


def rows_for(path: Path) -> tuple[dict, list[dict]]:
    s = Survey(path)
    track = s.poses()
    qc = stance_qc(track)
    rec = pycolmap.Reconstruction(s.sparse)
    rows = post_sfm_stance(track, rec, qc)
    for r in rows:
        r["survey"] = path.name
    reg = sum(r["registered"] for r in rows)
    summary = {
        "survey": path.name, "n": len(rows), "registered": reg,
        "qc_verdict": qc["summary"]["verdict"],
        "qc_median_d_over_b": qc["summary"]["median_d_over_b"],
        "frac_cross_parallax": qc["summary"]["frac_cross_parallax"],
        "flag_counts": qc["summary"]["flag_counts"],
    }
    return summary, rows


def binned(rows: list[dict], key: str) -> list[dict]:
    out = []
    for lo, hi in itertools.pairwise(BINS):
        sel = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
        if sel:
            reg = sum(r["registered"] for r in sel)
            out.append({"bin": f"{lo:g}–{hi:g}", "n": len(sel), "registered": reg,
                        "rate": reg / len(sel)})
    return out


def flags(rows: list[dict]) -> dict:
    counts: dict[str, list[int]] = {}
    for r in rows:
        for fl in r.get("qc_flags") or ["(none)"]:
            c = counts.setdefault(fl, [0, 0])
            c[0] += 1
            c[1] += r["registered"]
    return {k: {"n": n, "registered": g} for k, (n, g) in sorted(counts.items())}


def md_bins(title: str, b: list[dict]) -> str:
    lines = [f"**{title}**", "", "| D/B | shots | registered | rate |", "|---|---|---|---|"]
    lines += [f"| {x['bin']} | {x['n']} | {x['registered']} | {x['rate']:.0%} |" for x in b]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("surveys", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    summaries, rows = [], []
    for p in args.surveys:
        s, r = rows_for(p)
        summaries.append(s)
        rows += r
    print("| survey | registered | QC verdict | QC median D/B | cross-parallax | flags |")
    print("|---|---|---|---|---|---|")
    for s in summaries:
        print(f"| {s['survey']} | {s['registered']}/{s['n']} | {s['qc_verdict']} "
              f"| {s['qc_median_d_over_b']:.2f} | {s['frac_cross_parallax']:.0%} "
              f"| {s['flag_counts']} |")
    print()
    print(md_bins("QC D/B (poses.json only, pre-SfM)", binned(rows, "qc_d_over_b")))
    print()
    print(md_bins("Neighbour-measured D/B (SfM depth of nearby registered shots / VIO "
                  "baseline)", binned(rows, "neighbour_d_over_b")))
    print()
    print(md_bins("Own measured D/B (registered shots only)", binned(rows, "measured_d_over_b")))
    print()
    print("| QC flag | shots | registered |\n|---|---|---|")
    for k, v in flags(rows).items():
        print(f"| {k} | {v['n']} | {v['registered']} |")
    unreg = [r for r in rows if not r["registered"]]
    print(f"\nunregistered shots: {len(unreg)}")
    for r in unreg:
        print(f"  {r['survey']} {r['image']} tracking={r['tracking']} "
              f"qc_dob={r.get('qc_d_over_b')} nbr_dob={r.get('neighbour_d_over_b')} "
              f"flags={r.get('qc_flags')}")
    if args.json:
        args.json.write_text(json.dumps({"summaries": summaries, "rows": rows}, indent=1,
                                        default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
