r"""
J2: retrain the parametric PINN on both boxes, at the architecture the paper
states. (Reviewer 2, item 3; correction C1)
==============================================================================

The manuscript says width 96 / depth 5; the committed nets are 80 / 4. And the
original box (DEFAULT_BOX) does not contain the T = 0.15-slice optimum
(nu ~ 0.87, theta ~ 0.39). Both are fixed here, and both boxes are trained so the
box failure becomes a controlled ablation rather than a defect: the "default"
nets reproduce the manuscript's original setting, the "wide" nets contain the
optimum. Three seeds each, 20k iterations, same collocation budget as the
committed run (diagnose_task5_parametric.py).

Outputs: outputs/param_<box>_w96d5_seed<s>.pt and results/train_parametric_boxes.json
Each finished net is recorded immediately, so a crash loses at most one net.
"""
import json, os, sys, time
import numpy as np
import torch

from parametric_pinn import ParametricPINN, DEFAULT_BOX
from bench_surrogate import BOX as WIDE_BOX

WIDTH, DEPTH, ITERS = 96, 5, 20000
SEEDS = (0, 1, 2)
BOXES = dict(default=dict(DEFAULT_BOX), wide=dict(WIDE_BOX))
OUT = "results/train_parametric_boxes.json"

if __name__ == "__main__":
    which = sys.argv[1:] or list(BOXES)
    cal = json.load(open("calib_real.json")); V0 = float(cal["V0"])
    os.makedirs("outputs", exist_ok=True); os.makedirs("results", exist_ok=True)
    log = json.load(open(OUT)) if os.path.exists(OUT) else dict(runs=[])

    for name in which:
        box = BOXES[name]
        for s in SEEDS:
            tag = "param_%s_w%dd%d" % (name, WIDTH, DEPTH)
            path = "outputs/%s_seed%d.pt" % (tag, s)
            if os.path.exists(path):
                print("skip %s (exists)" % path, flush=True); continue
            torch.manual_seed(s); np.random.seed(s)
            m = ParametricPINN(box=box, V0=V0, T=0.15, width=WIDTH, depth=DEPTH, n=10)
            print("\n[%s box, seed %d] training %d iters at w%d/d%d ..." % (name, s, ITERS, WIDTH, DEPTH), flush=True)
            t0 = time.time()
            m.train(iters=ITERS, n_col=2500, n_bc=500, lr=1e-3, log_every=5000)
            secs = time.time() - t0
            torch.save(m.net.state_dict(), path)
            log["runs"].append(dict(box=name, seed=s, width=WIDTH, depth=DEPTH, iters=ITERS,
                                    n_col=2500, seconds=secs, path=path,
                                    box_values=box, torch_threads=torch.get_num_threads()))
            json.dump(log, open(OUT, "w"), indent=1)
            print("  saved %s  (%.1f min)" % (path, secs / 60), flush=True)

    from resultio import dump
    dump("train_parametric_boxes", log)
    print("\ndone; results/train_parametric_boxes.json")
