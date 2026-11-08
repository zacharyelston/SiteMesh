"""Benchmark `sitemesh sparse` configurations across captures and seeds.

Each run gets its own throw-away survey folder (raw/ symlinked, poses.json copied) under
$SITEMESH_BENCH_DIR (default ./bench), runs the real CLI in a subprocess, and records the report
plus wall time and peak RSS of that subprocess. Results append to <bench>/results.jsonl; rerunning
skips (capture, config, seed) triples already there unless --force.

    python scripts/bench.py data/object-orbit-2 data/object-orbit --seeds 0 1 2
    python scripts/bench.py data/* --configs priors_spatial --seeds 0
    python scripts/bench.py --table          # print the markdown table from results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "priors_spatial": [],
    "blind_exhaustive": ["--no-priors", "--matcher", "exhaustive"],
    "priors_exhaustive": ["--matcher", "exhaustive"],
}


def bench_dir() -> Path:
    return Path(os.environ.get("SITEMESH_BENCH_DIR", "bench")).resolve()


def run_one(capture: Path, config: str, seed: int, extra: list[str]) -> dict:
    work = bench_dir() / f"{capture.name}__{config}__s{seed}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "raw").symlink_to(capture.resolve() / "raw")
    if (capture / "poses.json").exists():
        shutil.copy(capture / "poses.json", work / "poses.json")
    exe = Path(sys.executable).with_name("sitemesh")
    cmd = [str(exe), "sparse", str(work), "--seed", str(seed), *CONFIGS[config], *extra]
    t0 = time.time()
    with open(work / "run.log", "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.time() - t0
    exit_code = os.waitstatus_to_exitcode(status)
    # ru_maxrss: bytes on macOS, KiB on Linux.
    rss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
    row = {
        "capture": capture.name, "config": config, "seed": seed, "cmd": cmd[1:],
        "exit": exit_code, "wall_s": wall, "peak_rss_gb": rss / 1e9,
        "host": f"{platform.machine()} {platform.processor() or ''}".strip(),
    }
    report_path = work / "model" / "report.json"
    if report_path.exists():
        r = json.loads(report_path.read_text())
        log = (work / "run.log").read_text()
        row.update(
            n_photos=r.get("n_photos"), n_registered=r.get("n_registered"),
            n_points3d=r.get("n_points3d"), reproj_px=r.get("mean_reprojection_px"),
            align_rms_m=r.get("device_alignment_rms_m"),
            rot_rms_deg=r.get("device_rotation_rms_deg"),
            sfm_to_device_scale=r.get("sfm_to_device_scale"),
            n_priors=r.get("n_priors"), matcher=r.get("matcher"),
            arkit_cameras=r.get("extract", {}).get("cameras"),
            refined_cameras=r.get("refined_cameras"), seconds=r.get("seconds"),
            seconds_total=r.get("seconds_total"), output_frame=r.get("output_frame"),
            gate=r.get("gate", {}).get("verdict"),
            mapper_loaded_priors=_mapper_priors(log),
        )
    return row


def _mapper_priors(log: str) -> int | None:
    """The mapper's own count from 'Loading pose priors...' followed by ' <n> in <t>s'."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if "Loading pose priors" in line and i + 1 < len(lines):
            tail = lines[i + 1].split("]")[-1].split()
            if tail and tail[0].isdigit():
                return int(tail[0])
    return None


def load() -> list[dict]:
    path = bench_dir() / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def focal_pairs(row: dict) -> str:
    ark = [c["intrinsics"][0] for c in row.get("arkit_cameras") or [] if c.get("intrinsics")]
    ref = [c["params"][0] for c in row.get("refined_cameras") or []]
    # cameras are created in the same order they are extracted
    return " / ".join(f"{r:.0f} vs {a:.0f} ({(r / a - 1) * 100:+.1f}%)" for r, a in zip(ref, ark))


def table(rows: list[dict]) -> str:
    head = ("| capture | config | seed | registered | points | reproj px | align RMS mm "
            "| rot RMS ° | SfM→device scale | refined vs ARKit focal (px) "
            "| extract s | match s | map s | total s | peak RSS GB |")
    out = [head, "|" + "|".join("---" for _ in range(head.count("|") - 1)) + "|"]
    for r in rows:
        if r.get("n_registered") is None:
            out.append(f"| {r['capture']} | {r['config']} | {r['seed']} | **crashed (exit "
                       f"{r['exit']})** |" + " |" * 11)
            continue
        s = r["seconds"]
        out.append(
            f"| {r['capture']} | {r['config']} | {r['seed']} "
            f"| {r['n_registered']}/{r['n_photos']} | {r['n_points3d']:,} | {r['reproj_px']:.3f} "
            f"| {r['align_rms_m'] * 1000:.1f} | {r['rot_rms_deg']:.2f} "
            f"| {r['sfm_to_device_scale']:.5f} | {focal_pairs(r)} "
            f"| {s['extract']:.0f} | {s['match']:.0f} | {s['map']:.1f} | {r['seconds_total']:.0f} "
            f"| {r['peak_rss_gb']:.1f} |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="*", type=Path)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--extra", default="", help="extra flags for sitemesh sparse, one string")
    args = ap.parse_args()
    if args.table:
        order = {c: i for i, c in enumerate(CONFIGS)}
        rows = sorted(load(), key=lambda r: (r["capture"], order[r["config"]], r["seed"]))
        print(table(rows))
        return 0
    extra = args.extra.split()
    bench_dir().mkdir(parents=True, exist_ok=True)
    done = {(r["capture"], r["config"], r["seed"]) for r in load()}
    for cap in args.captures:
        for cfg in args.configs:
            for seed in args.seeds:
                if (cap.name, cfg, seed) in done and not args.force:
                    continue
                row = run_one(cap, cfg, seed, extra)
                with open(bench_dir() / "results.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{cap.name} {cfg} s{seed}: exit {row['exit']} "
                      f"reg {row.get('n_registered')}/{row.get('n_photos')} "
                      f"{row['wall_s']:.0f}s {row['peak_rss_gb']:.1f}GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
