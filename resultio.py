"""
Result persistence for the experiment scripts.
==============================================

WHY THIS EXISTS. Before this module, every headline number in the paper reached
LaTeX by hand: run a diagnose script -> read stdout -> paste into the script's
docstring -> retype into cnsns_v1.tex and into figures_paper.py as a literal
array. That path produced several manuscript/code mismatches that the CNSNS
referees (rightly) probed. Numbers now go to results/<name>.json, and the
figure/table generators read those files.

Usage, at the end of an experiment script:

    from resultio import dump
    dump("mc_quality", dict(config=..., rows=[...]))

and to consume:

    from resultio import load
    res = load("mc_quality")
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _git_sha():
    """Short SHA of HEAD, or None outside a repo / without git on PATH."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty():
    """True if the working tree has uncommitted changes (so the SHA is not the
    whole story). Recorded because a number produced from a dirty tree is not
    reproducible from the SHA alone."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays so json.dump accepts them."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        # NaN/inf are not valid JSON; keep them readable rather than silently
        # emitting a token that json.load will reject.
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def dump(name, payload, echo=True):
    """
    Write results/<name>.json with provenance. Returns the path.

    payload : any JSON-able structure (numpy types are converted).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = dict(
        name=name,
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        argv=sys.argv,
        payload=_jsonable(payload),
    )
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    if echo:
        print(f"\n[resultio] wrote {path}")
    return path


def load(name):
    """Read results/<name>.json and return the payload. Raises if absent --
    a missing result should fail loudly, not silently fall back to a literal."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the experiment that produces '{name}' first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["payload"]


def provenance(name):
    """Read the provenance header (sha, timestamp, platform) without the payload."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    return {k: v for k, v in rec.items() if k != "payload"}


if __name__ == "__main__":
    p = dump("_selftest", dict(a=np.float64(1.5), b=np.arange(3),
                               c=dict(nan=np.nan, ok=True)))
    back = load("_selftest")
    assert back["a"] == 1.5 and back["b"] == [0, 1, 2]
    assert back["c"]["nan"] is None and back["c"]["ok"] is True
    print("resultio self-test OK:", provenance("_selftest"))
    os.remove(p)
