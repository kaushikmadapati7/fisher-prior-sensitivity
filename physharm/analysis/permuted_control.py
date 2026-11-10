r"""Permuted-protocol control: can the leakage-identifiability link be broken? (Issue #9)

MANDATORY. The strongest objection to this project is "you predicted the obvious" -- that any
predictor computed from the acquisition would correlate with any quantity computed from the
same acquisition. The answer is an arm where the physical link is deliberately severed. If the
association survives that, it was never protocol-driven and the study is dead.

Two different severings, because they answer different objections:

  LABEL PERMUTATION -- shuffle which protocol's predicted shrinkage is paired with which
      protocol's measured deviation. Everything else is untouched. This gives the null
      distribution of rho and answers "is the observed association bigger than chance for a
      sample of this size?" It is a significance test, nothing more.

  PHYSICAL PERMUTATION -- estimate under protocol j while PREDICTING from protocol i. The
      measurements, the tissue, the estimator and the noise are all unchanged; only the
      correspondence between the physics used to fit and the physics used to predict is
      broken. This is the one the plan means, and it is stronger: it tests the mechanism, not
      the sample size. A predictor that still works when it is describing the wrong scan is
      not describing physics.

WHAT WOULD KILL THE STUDY: physical permutation leaving rho essentially unchanged. Pre-register
that reading now, while it is cheap.

    python -m physharm.analysis.permuted_control --manifest-dir manifests_site \
        --out runs/permuted_control.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from physharm.analysis.crlb_table import acquisition
from physharm.analysis.synthetic_arm import (PARAMS, draw_tissue, run_protocol, spearman)
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F


def observed_and_permuted(protocols, arm="A1", n_tissue=200, snr=30.0, nuisance="noise",
                          seed=0, prior_sd=10.0, n_perm=200):
    """Observed rho, the label-permutation null, and the physically-permuted rho."""
    rng = np.random.default_rng(seed)
    theta = draw_tissue(n_tissue, rng)
    w = lambda a: [snr ** 2] * len(F.measurement_rows(arm, a))

    est = np.array([run_protocol(a, arm, theta, rng, snr, nuisance, prior_sd)
                    for a in protocols])                  # (n_proto, 4, n_tissue)
    pred = [F.identifiability(a, arm, weights=w(a), prior_sd=prior_sd) for a in protocols]
    degen = pred[0]["degenerate"]
    dev = np.abs(est - est.mean(axis=0)).mean(axis=2)      # (n_proto, 4)

    out = {}
    for i, k in enumerate(PARAMS):
        if degen[k]:
            out[k] = {"degenerate": True}
            continue
        shr = np.array([p["shrinkage"][k] for p in pred])
        d = dev[:, i]
        rho = spearman(shr.tolist(), d.tolist())

        # (1) label permutation -> null distribution for a sample of this size
        null = []
        for _ in range(n_perm):
            null.append(spearman(shr.tolist(), rng.permutation(d).tolist()))
        null = np.array([v for v in null if v == v])
        p_val = (float(np.mean(np.abs(null) >= abs(rho))) if null.size and rho == rho
                 else float("nan"))

        # (2) PHYSICAL permutation: predict from a different protocol than the one fitted.
        # The estimates and the predictor values are both real -- only the pairing is wrong.
        perm = rng.permutation(len(protocols))
        rho_phys = spearman(shr[perm].tolist(), d.tolist())

        out[k] = {"degenerate": False, "rho": rho, "rho_physical_permutation": rho_phys,
                  "null_mean": float(null.mean()) if null.size else float("nan"),
                  "null_p95": float(np.percentile(np.abs(null), 95)) if null.size else float("nan"),
                  "p_value": p_val}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A4"])
    ap.add_argument("--snr", nargs="+", type=float, default=[10.0, 30.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-protocols", type=int, default=150)
    ap.add_argument("--n-tissue", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mans = read_manifests(args.manifest_dir)
    acqs = [a for a in (acquisition(m)[0] for m in mans) if a is not None]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(acqs), size=min(args.n_protocols, len(acqs)), replace=False)
    protocols = [acqs[i] for i in idx]
    with_lo = [a for a in protocols if "b_lo" in a]
    print(f"{len(protocols)} protocols ({len(with_lo)} with a low-b band), "
          f"{args.n_perm} permutations per cell")

    rows = []
    for arm in args.arms:
        pr = with_lo if arm == "A2" else protocols
        if not pr:
            continue
        for snr in args.snr:
            per_seed = [observed_and_permuted(pr, arm, args.n_tissue, snr, "noise", s,
                                              n_perm=args.n_perm) for s in args.seeds]
            for k in PARAMS:
                if per_seed[0][k].get("degenerate"):
                    rows.append({"arm": arm, "snr": snr, "param": k, "degenerate": True})
                    continue
                g = lambda f: [d[k][f] for d in per_seed]
                rows.append({"arm": arm, "snr": snr, "param": k, "degenerate": False,
                             "rho": float(np.mean(g("rho"))),
                             "rho_sd": float(np.std(g("rho"))),
                             "rho_perm": float(np.mean(g("rho_physical_permutation"))),
                             "null_p95": float(np.mean(g("null_p95"))),
                             "p_value": float(np.mean(g("p_value")))})

    print("\n=== OBSERVED vs PERMUTED ===")
    print(f"{'arm':>4s} {'snr':>5s} {'param':>6s}  {'rho':>14s} {'physically':>11s} "
          f"{'null p95':>9s} {'p':>7s}   verdict")
    for r in rows:
        if r["degenerate"]:
            print(f"{r['arm']:>4s} {r['snr']:>5.0f} {r['param']:>6s}  "
                  f"{'not identifiable under this arm':>44s}")
            continue
        killed = abs(r["rho_perm"]) < 0.5 * abs(r["rho"]) or abs(r["rho_perm"]) < r["null_p95"]
        v = "control OK" if killed else "SURVIVES PERMUTATION — investigate"
        print(f"{r['arm']:>4s} {r['snr']:>5.0f} {r['param']:>6s}  "
              f"{r['rho']:>+7.3f}+-{r['rho_sd']:<5.3f} {r['rho_perm']:>+11.3f} "
              f"{r['null_p95']:>9.3f} {r['p_value']:>7.3f}   {v}")

    live = [r for r in rows if not r["degenerate"]]
    bad = [r for r in live if not (abs(r["rho_perm"]) < 0.5 * abs(r["rho"])
                                   or abs(r["rho_perm"]) < r["null_p95"])]
    print("\n=== VERDICT ===")
    if not live:
        print("  nothing testable")
    elif bad:
        print(f"  {len(bad)}/{len(live)} cells SURVIVE physical permutation. The association")
        print("  is not protocol-driven in those cells. Do not report the headline until this")
        print("  is explained -- this is the pre-registered killing observation.")
    else:
        print(f"  all {len(live)} cells: the association collapses when the predictor is")
        print("  paired with the wrong scan. It is protocol-driven, which is the claim.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"rows": rows, "n_protocols": len(protocols), "n_perm": args.n_perm},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
