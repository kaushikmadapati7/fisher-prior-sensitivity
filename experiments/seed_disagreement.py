r"""Are self-supervised qMRI maps reproducible? (#83)

    python experiments/seed_disagreement.py --cache-dir CACHE --manifest-dir MANIFESTS --seeds 5

THE QUESTION. Physics-informed networks are trained to recover tissue parameters that are supposed
to be properties of the patient. If two training runs on the same images produce different maps,
those maps are partly a property of the run. This measures whether they do, and where.

THE PREDICTION. Our analysis says the data constrains some parameter directions and not others,
and that the unconstrained direction `v` -- the near-null eigenvector of the Fisher matrix -- is
where the prior, and anything else not pinned by the data, decides the answer. A self-supervised
network is trained on an objective, and an objective cannot pin what the data does not constrain.
So disagreement between runs should concentrate along `v`.

WHAT MAKES THIS RUNNABLE ON REAL IMAGES. Nothing here needs ground-truth parameter maps, and
nothing here shifts a prior. Train k networks that differ only in seed, predict on the same exams,
and decompose their pairwise disagreement into the component along `v` and the component
orthogonal to it. That sidesteps the instability that makes the prior-shift version of this arm
unmeasurable (run log, phase 30): we are measuring the non-uniqueness itself rather than trying to
measure through it.

Reported per exam, so the result is a distribution over the cohort rather than one number.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from physharm.analysis.crlb_table import acquisition, canonical_scanner   # noqa: E402
from physharm.analysis.selfsup_arm import acq_features                    # noqa: E402
from physharm.analysis.synthetic_arm import (PARAMS, TISSUE, _jacobian,   # noqa: E402
                                             map_estimate)
from physharm.analysis.traveling_arm import _rows_and_signal, load_case   # noqa: E402
from physharm.data.manifests import read_manifests                        # noqa: E402

import importlib.util                                                     # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "sst", __file__.rsplit("/", 1)[0] + "/selfsup_torch.py")
_sst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sst)

SIGMA, PRIOR_SD, ARM = 0.05, 10.0, "A1"

#: A network that outputs the prior everywhere scores a few thousand on the held-out set, and a
#: well-trained one scores tens. Anything above this is not a worse fit, it is a broken one.
EVAL_LOSS_CAP = 1e5


def sloppy_direction(x, rows, w):
    """Near-null eigenvector of F = J'WJ at the fitted point, per exam."""
    J = _jacobian(x, rows)
    F = np.einsum("mkn,mln->kln", J, J).mean(2) * w
    ev, V = np.linalg.eigh(F)
    return V[:, 0], ev


def disagreement(nets, eval_exams, x0, w, base_mean):
    """Pairwise disagreement between trained nets on a FIXED evaluation set.

    Returns per-exam means of the component along the unconstrained direction and the component
    orthogonal to it. Per-exam values are aggregated by the caller and never printed.
    """
    al, orth = [], []
    for rows, Y, af in eval_exams:
        preds = [_sst.predict(n, Y, af, sc, x0) for n, sc in nets]
        v, _ = sloppy_direction(map_estimate(Y, rows, base_mean, prior_sd=PRIOR_SD,
                                             sigma=SIGMA), rows, w)
        a_, o_ = [], []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                d = preds[i] - preds[j]
                p = v @ d
                a_.append(np.median(np.abs(p)))
                o_.append(np.median(np.linalg.norm(d - np.outer(v, p), axis=0)))
        al.append(np.mean(a_))
        orth.append(np.mean(o_))
    return np.array(al), np.array(orth)


def cohort_loss(net, scale, exams, x0, w, lam):
    from physharm.analysis.selfsup_arm import physics_loss
    n = sum(Y.shape[1] for _, Y, _ in exams)
    return sum(physics_loss(_sst.predict(net, Y, af, scale, x0), Y, rows, w, lam, x0) * Y.shape[1]
               for rows, Y, af in exams) / n


def run_config(train_exams, eval_exams, x0, w, lam, base_mean, seeds, steps, width, lr, clip):
    """Train `seeds` networks and summarise their disagreement.

    A seed that diverges is DROPPED, loudly, rather than averaged in. Mixing a diverged network
    into a median produces a plausible-looking number from garbage, which is how a previous run
    reported a training-set-size sweep in which every seed at the smallest size had diverged.
    """
    nets, losses, dropped = [], [], 0
    for s in range(seeds):
        try:
            net, scale = _sst.train(train_exams, x0, w, lam, steps=steps, seed=s,
                                    width=width, lr=lr, clip=clip)
        except SystemExit as e:
            print(f"      seed {s} dropped: {e}", flush=True)
            dropped += 1
            continue
        l_eval = cohort_loss(net, scale, eval_exams, x0, w, lam)
        # The training guard watches the BATCH loss. A network can satisfy it and still be
        # hopeless on held-out exams -- at n_train = 50 every seed passed training and evaluated
        # at 1e72, which is generalisation failure rather than divergence. Either way its
        # predictions are meaningless and must not enter a median.
        if not np.isfinite(l_eval) or l_eval > EVAL_LOSS_CAP:
            print(f"      seed {s} dropped: evaluation loss {l_eval:.4g} exceeds the cap "
                  f"{EVAL_LOSS_CAP:g}; trained but does not generalise", flush=True)
            dropped += 1
            continue
        nets.append((net, scale))
        losses.append(l_eval)
    if len(nets) < 2:
        return {"along": float("nan"), "orth": float("nan"), "ratio": float("nan"),
                "share": float("nan"), "loss_median": float("nan"), "loss_spread": float("nan"),
                "losses": [], "dropped": dropped, "n_seeds": len(nets)}
    A, O = disagreement(nets, eval_exams, x0, w, base_mean)
    # PER-EXAM values, not just their medians. The medians alone cannot be drawn, and the
    # figure for this arm needs one point per exam. Kept alongside the medians so nothing
    # already reported changes.
    return {"per_exam": {"along": [float(a) for a in np.ravel(A)],
                         "orth": [float(o) for o in np.ravel(O)]},
            "along": float(np.median(A)), "orth": float(np.median(O)),
            "ratio": float(np.median(A) / np.median(O)),
            "share": float(np.mean(A > O)),
            "loss_median": float(np.median(losses)),
            "loss_spread": float(max(losses) - min(losses)),
            "losses": [float(x) for x in losses],
            "dropped": dropped, "n_seeds": len(nets)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=300, help="exams to read; 0 means all")
    ap.add_argument("--n-vox", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient-norm clip. Must match selfsup_torch.train's default; a CLI "
                         "default left behind silently overrode a tightening once already.")
    ap.add_argument("--n-eval", type=int, default=80,
                    help="exams held out for evaluation, fixed across every configuration so "
                         "that a larger training set is not also a larger test set")
    ap.add_argument("--exam-sweep", type=int, nargs="*", default=[],
                    help="training-set sizes. Does more data reduce the disagreement? If it does "
                         "not, this is unidentifiability rather than estimation error.")
    ap.add_argument("--width-sweep", type=int, nargs="*", default=[],
                    help="network widths. If the concentration is architecture-independent it is "
                         "a property of the problem, which is what licenses predicting it from F "
                         "before training anything.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w, lam = 1.0 / SIGMA ** 2, 1.0 / PRIOR_SD ** 2
    base_mean = {k: v[0] for k, v in TISSUE.items()}
    x0 = np.array([[np.log(base_mean[k])] for k in PARAMS])

    mans = read_manifests(args.manifest_dir)
    if args.limit:
        mans = mans[:args.limit]
    exams = []
    for m in mans:
        acq, _ = acquisition(m)
        if acq is None:
            continue
        cached = load_case(args.cache_dir, m.get("case_id"))
        if cached is None:
            continue
        got = _rows_and_signal(cached, acq, arm=ARM, n_vox=args.n_vox,
                               rng=np.random.default_rng(0))
        if got is None:
            continue
        rows, y = got
        exams.append((rows, y, acq_features(acq).T))
    if len(exams) <= args.n_eval:
        raise SystemExit(f"only {len(exams)} usable exams; need more than --n-eval")
    _sst.check_forward_model(exams[0][0])

    ev, pool = exams[:args.n_eval], exams[args.n_eval:]
    print(f"{len(exams)} exams: {len(pool)} available for training, {len(ev)} held out for "
          f"evaluation throughout", flush=True)

    report = {"n_eval": len(ev), "n_pool": len(pool), "seeds": args.seeds, "steps": args.steps}

    sizes = args.exam_sweep or [len(pool)]
    print(f"\n=== TRAINING-SET SIZE ===", flush=True)
    print(f"{'n_train':>8} {'seeds':>7} {'loss':>10} {'along v':>9} {'orth':>9} {'ratio':>7} {'share':>7}", flush=True)
    report["by_size"] = {}
    for n in sizes:
        if n > len(pool):
            continue
        r = run_config(pool[:n], ev, x0, w, lam, base_mean, args.seeds, args.steps,
                       256, args.lr, args.clip)
        report["by_size"][str(n)] = r
        print(f"{n:>8} {r['n_seeds']:>3}/{args.seeds:<3} {r['loss_median']:>10.4g} "
              f"{r['along']:>9.4f} {r['orth']:>9.4f} {r['ratio']:>7.2f} "
              f"{100*r['share']:>6.0f}%", flush=True)

    if args.width_sweep:
        print(f"\n=== ARCHITECTURE ===", flush=True)
        print(f"{'width':>8} {'seeds':>7} {'loss':>10} {'along v':>9} {'orth':>9} {'ratio':>7} {'share':>7}", flush=True)
        report["by_width"] = {}
        for wd in args.width_sweep:
            r = run_config(pool, ev, x0, w, lam, base_mean, args.seeds, args.steps,
                           wd, args.lr, args.clip)
            report["by_width"][str(wd)] = r
            print(f"{wd:>8} {r['n_seeds']:>3}/{args.seeds:<3} {r['loss_median']:>10.4g} "
                  f"{r['along']:>9.4f} {r['orth']:>9.4f} {r['ratio']:>7.2f} "
                  f"{100*r['share']:>6.0f}%", flush=True)

    # Does a lower loss mean a more trustworthy map? Pool every configuration we trained.
    L, R = [], []
    for grp in ("by_size", "by_width"):
        for r in report.get(grp, {}).values():
            if r.get("dropped", 0) == 0 and np.isfinite(r["ratio"]):
                L.append(r["loss_median"]); R.append(r["ratio"])
    if len(L) >= 4:
        c = float(np.corrcoef(L, R)[0, 1])
        report["loss_vs_ratio_corr"] = c
        print(f"\ncorrelation between cohort loss and the along/orthogonal ratio, over "
              f"{len(L)} configurations: {c:+.3f}", flush=True)
        print("  (near zero means the training loss carries no information about how much of the "
              "map\n   is decided by the seed)", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out} (aggregates only)", flush=True)


if __name__ == "__main__":
    main()
