"""Do smaller CENTRAL prior shifts drive the clinical ratios toward one?

    PYTHONPATH=. python -u experiments/cohort_shift_limit.py \
        --cache-dir "$CACHE" --manifest-dir "$MANIFESTS" --out runs/cohort_shift_limit.json

The full-Hessian correction leaves 1.061, 0.963 and 1.018 on the cohort. Is that residue a
finite-shift artifact? The theory predicts an infinitesimal derivative while the main
experiment uses a 25% shift (Delta = log 1.25 = 0.2231). On the network
arm a 32x central sweep leaves the ratio at 1.03-1.05 with no trend toward one, so the residue
there is real; this runs the same test on the cohort, which is the arm the question is about.

`prior_shift.collect` already accepts a delta list and `case_shifts` applies the shift as
base_mean * exp(d), so a NEGATIVE delta is the opposite shift and needs no change to the
validated module. For each |d| we form

    one-sided  obs(+d) / d                       -> O(d) error
    central   (obs(+d) - obs(-d)) / (2 d)        -> O(d^2) error

and compare both against the full-Hessian prediction at that |d|. If the residue is a
finite-shift artifact the central sequence converges to one as |d| shrinks; if it is real the
central ratios sit at the same offset the 25% shift reports.
"""
import argparse, json, os
import numpy as np

from physharm.analysis import prior_shift as PS
from physharm.data.manifests import read_manifests
from physharm.analysis.synthetic_arm import PARAMS

#: log-space shifts. 0.2231 is log 1.25, the shift the paper's main experiment uses.
HALVINGS = (0.2231, 0.1116, 0.0558, 0.0279, 0.0139)


def merge(paths, out_path):
    """Combine shard outputs. Medians are taken over the pooled per-case ratios."""
    import collections
    pooled = collections.defaultdict(lambda: {"one_h": [], "cen_h": [], "one_f": [], "cen_f": []})
    n_cases = 0
    halvings = None
    for p in paths:
        d = json.load(open(p))
        n_cases += d.get("n_cases", 0)
        halvings = halvings or d.get("halvings")
        for r in d["per_delta"]:
            tgt = pooled[(r["abs_delta"], r["param"])]
            for k in ("one_h", "cen_h", "one_f", "cen_f"):
                tgt[k].extend(r.get("raw", {}).get(k, []))
    out = {"n_cases": n_cases, "halvings": halvings, "merged_from": list(paths),
           "per_delta": []}
    print(f"merged {len(paths)} shards, {n_cases} cases\n")
    print(f"{'|d|':>8} {'param':>5} {'one-sided/H':>12} {'central/H':>11} "
          f"{'one-sided/F':>12} {'central/F':>11} {'n':>6}")
    for (h, p), v in sorted(pooled.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        if not v["one_h"]:
            continue
        rec = {"abs_delta": h, "param": p, "n": len(v["one_h"]),
               "one_sided_over_hessian": float(np.median(v["one_h"])),
               "central_over_hessian": float(np.median(v["cen_h"])),
               "one_sided_over_fisher": float(np.median(v["one_f"])) if v["one_f"] else None,
               "central_over_fisher": float(np.median(v["cen_f"])) if v["cen_f"] else None}
        out["per_delta"].append(rec)
        print(f"{h:8.4f} {p:>5} {rec['one_sided_over_hessian']:12.4f} "
              f"{rec['central_over_hessian']:11.4f} "
              f"{(rec['one_sided_over_fisher'] or float('nan')):12.4f} "
              f"{(rec['central_over_fisher'] or float('nan')):11.4f} {rec['n']:6d}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    print(f"\n-> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-vox", type=int, default=4000)
    ap.add_argument("--n-iter", type=int, default=25)
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--shard", default=None, metavar="N/K",
                    help="stride the manifest list; each shard writes its own --out")
    ap.add_argument("--merge", nargs="+", default=None, metavar="SHARD_JSON",
                    help="combine shard outputs and print the table; ignores --cache-dir")
    ap.add_argument("--out", default="runs/cohort_shift_limit.json")
    a = ap.parse_args()

    if a.merge:
        return merge(a.merge, a.out)

    deltas = tuple(sorted({d for h in HALVINGS for d in (+h, -h)}, reverse=True))
    print(f"deltas ({len(deltas)}): {['%+.4f' % d for d in deltas]}", flush=True)

    manifests = read_manifests(a.manifest_dir)
    if a.shard:
        n, k = map(int, a.shard.split("/"))
        manifests = [m for i, m in enumerate(manifests) if i % k == n]
        print(f"shard {n}/{k}: {len(manifests)} manifests", flush=True)
    rows, why = PS.collect(a.cache_dir, manifests, prior_sd=a.prior_sd, n_vox=a.n_vox,
                           limit=a.limit, deltas=deltas, n_iter=a.n_iter, arm=a.arm,
                           n_perm=0)          # nulls are not what this run is for
    print(f"kept {len(rows)} cases; skips: {dict(why)}", flush=True)

    out = {"n_cases": len(rows), "prior_sd": a.prior_sd, "arm": a.arm,
           "halvings": list(HALVINGS), "per_delta": []}
    print(f"\n{'|d|':>8} {'param':>5} {'one-sided/H':>12} {'central/H':>11} "
          f"{'one-sided/F':>12} {'central/F':>11} {'n':>6}")
    for h in HALVINGS:
        kp, km = f"{+h:.4f}", f"{-h:.4f}"
        for p in PARAMS:
            o1, oc, f1l, fcl = [], [], [], []
            for r in rows:
                sh = r["shifts"]
                if kp not in sh or km not in sh:
                    continue
                try:
                    op = sh[kp][p]["observed"][p]
                    om = sh[km][p]["observed"][p]
                    ph = sh[kp][p]["predicted_hessian"][p]
                    pf = sh[kp][p]["predicted_voxel"][p]
                except KeyError:
                    continue
                # `observed` is a raw displacement; `predicted_*` is M*d. So the central
                # estimate of that same displacement is the half-difference, and dividing by
                # the prediction at +d gives a ratio comparable to the one-sided one.
                cen = (op - om) / 2.0
                if np.isfinite(ph) and ph != 0:
                    o1.append(op / ph); oc.append(cen / ph)
                if np.isfinite(pf) and pf != 0:
                    f1l.append(op / pf); fcl.append(cen / pf)
            if not o1:
                continue
            # RAW ratios, not just the median: a merge over shards must take the median of
            # all cases, and a median of per-shard medians is not that.
            rec = {"abs_delta": h, "param": p, "n": len(o1),
                   "one_sided_over_hessian": float(np.median(o1)),
                   "central_over_hessian": float(np.median(oc)),
                   "one_sided_over_fisher": float(np.median(f1l)) if f1l else None,
                   "central_over_fisher": float(np.median(fcl)) if fcl else None,
                   "raw": {"one_h": o1, "cen_h": oc, "one_f": f1l, "cen_f": fcl}}
            out["per_delta"].append(rec)
            print(f"{h:8.4f} {p:>5} {rec['one_sided_over_hessian']:12.4f} "
                  f"{rec['central_over_hessian']:11.4f} "
                  f"{(rec['one_sided_over_fisher'] or float('nan')):12.4f} "
                  f"{(rec['central_over_fisher'] or float('nan')):11.4f} {len(o1):6d}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
