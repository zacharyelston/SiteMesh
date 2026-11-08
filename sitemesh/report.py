"""Acceptance gate: fail loudly instead of publishing a bad model.

Targets come from the hydra-vision rebuild guide §5. A metric that could not be measured is
reported as `not_measured`, never as a pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Gate:
    min_registration: float = 0.98
    max_reprojection_px: float = 1.0
    max_device_alignment_rms_m: float = 0.01
    max_device_rotation_rms_deg: float = 2.0
    max_scale_error_pct: float = 0.5


def evaluate(report: dict, gate: Gate | None = None) -> dict:
    gate = gate or Gate()
    checks = []

    def check(name, value, ok_fn, target):
        if value is None:
            checks.append({"check": name, "status": "not_measured", "target": target})
        else:
            checks.append(
                {"check": name, "value": value, "target": target,
                 "status": "pass" if ok_fn(value) else "fail"}
            )

    n, reg = report.get("n_photos"), report.get("n_registered")
    check("registration", reg / n if n else None, lambda v: v >= gate.min_registration,
          f">= {gate.min_registration:.0%}")
    check("mean_reprojection_px", report.get("mean_reprojection_px"),
          lambda v: v <= gate.max_reprojection_px, f"<= {gate.max_reprojection_px}")
    check("device_alignment_rms_m", report.get("device_alignment_rms_m"),
          lambda v: v <= gate.max_device_alignment_rms_m, f"<= {gate.max_device_alignment_rms_m}")
    check("device_rotation_rms_deg", report.get("device_rotation_rms_deg"),
          lambda v: v <= gate.max_device_rotation_rms_deg, f"<= {gate.max_device_rotation_rms_deg}")
    sc = report.get("scale_check", {}).get("summary", {}).get("scale_check_error_pct")
    check("scale_check_error_pct", abs(sc) if sc is not None else None,
          lambda v: v <= gate.max_scale_error_pct, f"<= {gate.max_scale_error_pct}")

    statuses = {c["status"] for c in checks}
    verdict = "fail" if "fail" in statuses else ("pass" if statuses == {"pass"} else "pass_unverified")
    return {"verdict": verdict, "gate": asdict(gate), "checks": checks}
