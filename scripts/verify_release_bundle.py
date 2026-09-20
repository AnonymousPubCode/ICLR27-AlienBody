#!/usr/bin/env python3
"""Self-check for this release bundle.

Offline only: no network, no GPU, no base weights, no checkpoints.  It answers
"is what shipped here internally consistent and runnable?" — not "does the
paper reproduce?".

    python scripts/verify_release_bundle.py            # everything
    python scripts/verify_release_bundle.py --quick    # skip the stub regressions

Checks
  1  every shipped ``alienbody`` module imports
  2  the F4 trajectory files have the expected line counts
  3  the frozen suites' recorded digests recompute from the shipped files
  4  hygiene: no absolute internal paths, personal or vendor names in shipped text
  5  the offline stub regressions (tests/test_nextstep_stubs.py) pass

Exit status 0 means every check passed (skips inside check 5 are allowed and
listed).  Anything else prints the failing check with its detail.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAJECTORIES = {
    "f4_scale_200.jsonl": 200,
    "f4_scale_500.jsonl": 500,
    "f4_scale_1000.jsonl": 1000,
    "f4_scale_2000.jsonl": 2000,
    "f4_cot_2000.jsonl": 2000,
}

HYGIENE = {
    "personal name": r"hengchang|holden",
    "home path": r"C:\\{1,2}Users|C:/Users|/home/[a-z]",
    "cluster path": r"/project/",
    "vendor name": r"(?i)fuxi|leihuo|danlu",
    "hardcoded device": r"CUDA_VISIBLE_DEVICES=[0-9]",
}

# This file names the patterns it searches for, so it excludes itself.
SELF = Path(__file__).resolve()

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".gif", ".ico", ".npy",
                 ".npz", ".pt", ".bin", ".safetensors"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "results", "models"}


def check_imports() -> list[str]:
    """Import every module under alienbody/ (the shippable library)."""
    failures = []
    modules = set()
    for f in sorted((ROOT / "alienbody").rglob("*.py")):
        parts = [p for p in f.relative_to(ROOT).with_suffix("").parts
                 if p != "__pycache__"]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join(parts))
    for m in sorted(modules):
        try:
            importlib.import_module(m)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{m}: {type(exc).__name__}: {exc}")
    print(f"[1] imports: {len(modules) - len(failures)}/{len(modules)} modules")
    return failures


def check_trajectories() -> list[str]:
    failures = []
    for name, expected in TRAJECTORIES.items():
        path = ROOT / "data" / "fmb_trajectories" / name
        if not path.exists():
            failures.append(f"{name}: missing")
            continue
        n = sum(1 for _ in path.open(encoding="utf-8"))
        if n != expected:
            failures.append(f"{name}: {n} lines, expected {expected}")
    print(f"[2] trajectories: {len(TRAJECTORIES) - len(failures)}"
          f"/{len(TRAJECTORIES)} files at the recorded sizes")
    return failures


def check_suites() -> list[str]:
    """Recompute each suite's result_sha256 from the shipped files."""
    failures = []
    from alienbody.nextstep.records import dataset_sha256

    f4 = ROOT / "data" / "frozen_suites" / "nextstep_f4_formal"
    manifest = json.loads((f4 / "manifest.json").read_text(encoding="utf-8"))
    names = [e["file"] for e in manifest["environments"]]
    present = sorted(p.name for p in f4.glob("env_*.json"))
    if present != sorted(names):
        failures.append("nextstep_f4_formal: env files do not match the manifest")
    elif dataset_sha256(f4 / n for n in names) != manifest["result_sha256"]:
        failures.append("nextstep_f4_formal: result_sha256 mismatch")
    else:
        print(f"[3] nextstep_f4_formal: {len(names)} envs, digest OK")

    wf = ROOT / "data" / "frozen_suites" / "nextstep_workflow_formal"
    manifest = json.loads((wf / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((wf / "dev_instances.json").read_bytes()).hexdigest()
    if digest != manifest["result_sha256"]:
        failures.append("nextstep_workflow_formal: result_sha256 mismatch")
    else:
        print(f"[3] nextstep_workflow_formal: {len(manifest['instances'])}"
              " instances, digest OK")
    return failures


def check_hygiene() -> list[str]:
    patterns = {label: re.compile(pat) for label, pat in HYGIENE.items()}
    hits = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.resolve() == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for label, pattern in patterns.items():
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{path.relative_to(ROOT)}:{line}: {label} "
                            f"({m.group(0)})")
    print(f"[4] hygiene: {scanned} text files scanned, {len(hits)} hits")
    return hits


def check_stub_regressions() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_nextstep_stubs"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    summary = " | ".join(line for line in tail[-4:] if line.strip())
    if proc.returncode != 0:
        return [f"stub regressions failed: {summary}"]
    print(f"[5] stub regressions: {summary}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the stub regressions (check 5)")
    args = parser.parse_args()

    failures = []
    failures += check_imports()
    failures += check_trajectories()
    failures += check_suites()
    failures += check_hygiene()
    if not args.quick:
        failures += check_stub_regressions()

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
