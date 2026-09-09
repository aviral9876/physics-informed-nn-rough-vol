r"""
Pick the reference parameter vector for the revision, on the corrected objective.
=================================================================================

Four candidate fits now exist for the same headline surface:

  calib_real.json             the submitted fit (buggy objective, buggy pricer)
  calib_real_fixed.json       corrected objective, ORIGINAL box, unbounded polish
  calib_real_wide_fixed.json  corrected objective, WIDE box, bounded polish
  calib_real_wider.json       corrected objective, WIDER box (lambda unpinned)

They are not comparable as reported, because the first was scored by an
objective that dropped the points it could not price. This scores every one of
them under the SAME corrected objective on the same 104 points, reports the
spread -- which is itself the identifiability result -- and writes the best as
calib_reference.json.

The submitted fit is preserved twice over -- in git, and as
calib_real.submitted.json -- and the chosen vector is then written to
calib_real.json so that the twenty-six scripts which read that path pick up the
corrected basis without twenty-six edits. Pass --no-swap to score the candidates
and write calib_reference.json without touching calib_real.json.

Why the swap is the right default: the submitted vector was produced by an
objective that dropped the points it could not price, and by a Fourier inversion
truncated at u_max = 100 which mis-prices maturities under two weeks by up to
0.33 in implied volatility. It is not a fit that should seed new experiments.
"""
import json, os, shutil, sys
import numpy as np, pandas as pd

from surface import build_surface
from generate_calib import cap_points_per_maturity
from calibrate import PARAM_NAMES
from diagnose_calib_uncertainty import make_losses

CANDIDATES = [
    ("submitted",        "calib_real.json"),
    ("corrected, original box", "calib_real_fixed.json"),
    ("corrected, wide box",     "calib_real_wide_fixed.json"),
    ("corrected, wider box",    "calib_real_wider.json"),
]
OUT = "calib_reference.json"


def load_reference(default="calib_real.json"):
    """Every downstream script should call this rather than opening a path."""
    return json.load(open(OUT if os.path.exists(OUT) else default))


if __name__ == "__main__":
    surf = cap_points_per_maturity(build_surface(pd.read_csv("data/deribit_chain.csv"), r=0.0))
    both = make_losses(surf)
    print("scoring every candidate on the corrected objective, %d points\n" % len(surf))
    print("%-26s %-22s %9s %11s  %s" % ("fit", "file", "weighted", "unweighted", "H / nu / rho"))
    rows = []
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print("%-26s %-22s  (not present)" % (label, path))
            continue
        d = json.load(open(path))
        v = [float(d[k]) for k in PARAM_NAMES]
        w, u = both(v)
        rows.append(dict(label=label, path=path, weighted_vol_bp=1e4 * w,
                         unweighted_vol_bp=1e4 * u,
                         params={k: float(d[k]) for k in PARAM_NAMES}))
        print("%-26s %-22s %9.1f %11.1f  %.4f / %.3f / %+.3f"
              % (label, path, 1e4 * w, 1e4 * u, d["H"], d["nu"], d["rho"]))

    best = min(rows, key=lambda q: q["weighted_vol_bp"])
    Hs = [q["params"]["H"] for q in rows if q["label"] != "submitted"]
    print("\nbest: %s (%s)" % (best["label"], best["path"]))
    if len(Hs) > 1:
        print("H across the corrected fits: %.4f to %.4f -- a factor of %.1f at fits within %.0f vol bp"
              % (min(Hs), max(Hs), max(Hs) / max(min(Hs), 1e-9),
                 max(q["weighted_vol_bp"] for q in rows) - min(q["weighted_vol_bp"] for q in rows)))

    src = json.load(open(best["path"]))
    out = dict(src)
    out.update(reference_of=best["path"], reference_label=best["label"],
               weighted_vol_bp=best["weighted_vol_bp"],
               unweighted_vol_bp=best["unweighted_vol_bp"],
               note="chosen by select_reference.py on the corrected objective; "
                    "calib_real.json is left as the submitted record",
               all_candidates=rows)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\nsaved -> %s" % OUT)

    if "--no-swap" in sys.argv:
        print("--no-swap: calib_real.json left as the submitted fit")
    else:
        if not os.path.exists("calib_real.submitted.json"):
            shutil.copy("calib_real.json", "calib_real.submitted.json")
            print("preserved the submitted fit -> calib_real.submitted.json")
        json.dump(out, open("calib_real.json", "w"), indent=2)
        print("calib_real.json now holds the %s fit (H=%.4f, nu=%.3f, rho=%+.3f)"
              % (best["label"], out["H"], out["nu"], out["rho"]))
        print("revert with:  copy calib_real.submitted.json calib_real.json")

    from resultio import dump
    dump("reference_selection", dict(candidates=rows, chosen=best, n_points=int(len(surf)),
                                     swapped=("--no-swap" not in sys.argv)))
