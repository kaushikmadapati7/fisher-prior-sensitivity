r"""Real arm: does the shrinkage mechanism show up in estimates from real images? (Issue #10)

WHAT THIS MEASURES, AND WHY NOT SEPARABILITY ALONE. The issue asks for per-parameter scanner
separability -- can a classifier read scanner identity off a parameter map. That is what the
harmonization literature reports, and it is included here as a secondary. But it collapses to
ONE number per parameter, i.e. an n=4 comparison, and this project has already been misled
twice by n=4 rank statistics (the saturated rho_crlb = -1.00 in #24, and #7's per-parameter
correlation).

The high-powered test uses the property that makes this cohort unusual: protocol varies WITHIN
a scanner (ICC(CRLB T2 | scanner) = 0.224), so shrinkage varies case to case with TR and TE
while the scanner is held fixed. Shrinkage pulls an estimate toward the prior. So the mechanism
makes a per-case prediction:

    | theta_hat - theta_prior |   DECREASES as shrinkage RISES

n = the cohort, not the parameter count, and scanner enters as a COVARIATE rather than as the
unit of analysis. It tests the mechanism directly instead of a proxy for it.

WHY A MAP FIT AND NOT A TRAINED ENCODER. Same reason as the synthetic arm: a learned encoder
adds a learned prior, so its leakage mixes identifiability with whatever the network absorbed
from its training distribution. That is a different and worthwhile question -- a stretch goal,
not this one. The MAP fit is the estimator the theory describes and is already validated
against its own sampling covariance.

WHAT WOULD FALSIFY IT: a flat or positive slope, i.e. cases whose protocol determines a
parameter less do NOT sit closer to the prior. That would mean the shrinkage decomposition
does not describe what the estimator does on real data.

    python -m physharm.analysis.real_arm --cache-dir cache_shared \
        --manifest-dir manifests_site --out runs/real_arm.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from physharm.analysis.crlb_table import acquisition, canonical_scanner
from physharm.analysis.synthetic_arm import PARAMS, TISSUE, spearman
from physharm.analysis.traveling_arm import estimate_case, load_case
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

#: below this, (1-s) is effectively constant across cases and the regression has no regressor
MIN_LEVERAGE = 1.05


def collect(cache_dir, manifests, arm="A1", prior_sd=10.0, sigma=0.05, n_vox=4000,
            limit=0, log=print):
    """Per case: ROI-median theta, its distance from the prior, and its predicted shrinkage."""
    rows, why = [], Counter()
    for i, m in enumerate(manifests):
        if limit and len(rows) >= limit:
            break
        cid = m.get("case_id")
        acq, bad = acquisition(m)
        if acq is None:
            why[bad or "no acquisition params"] += 1
            continue
        cached = load_case(cache_dir, cid)
        if cached is None:
            why["not in cache"] += 1
            continue
        th = estimate_case(cached, acq, prior_sd=prior_sd, sigma=sigma, n_vox=n_vox, arm=arm)
        if th is None:
            why["fit failed"] += 1
            continue
        # WEIGHTS. The fit runs at W = 1/sigma^2; fisher_matrix defaults to unit weight. A
        # global noise scale cancels out of the CRLB ordering -- which is why this survived
        # #24 and #38 -- but shrinkage is an absolute fraction and was inflated 400x by it.
        # Rank correlations below are unaffected (W is the same for every case); the
        # shrinkage column and the leverage ratios are not.
        # One weight per measurement THIS arm actually has. Hardcoding 3 made every arm but
        # A1 die in identifiability with a broadcast error (A4 has 2 rows, A3 has 1), which
        # is why the arm sweep only ever returned one of three.
        n_rows = len(F.measurement_rows(arm, acq, {k: v[0] for k, v in TISSUE.items()}))
        pred = F.identifiability(acq, arm, weights=[1.0 / sigma ** 2] * n_rows,
                                 prior_sd=prior_sd)
        rows.append({
            "case_id": cid, "patient_id": m.get("patient_id"),
            "scanner": canonical_scanner(m.get("scanner_model")),
            # distance from the prior mean, in log units -- the quantity shrinkage acts on
            "dist": {k: abs(th[k] - float(np.log(TISSUE[k][0]))) for k in PARAMS},
            "theta": {k: float(th[k]) for k in PARAMS},
            "shrinkage": {k: float(pred["shrinkage"][k]) for k in PARAMS},
            "crlb": {k: float(pred["crlb"][k]) for k in PARAMS},
        })
        if log and (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(manifests)}  kept {len(rows)}")
    return rows, why


def leverage(rows, key="shrinkage"):
    """How much room does the regressor have to move? No spread, no test.

    The mechanism predicts |theta_hat - theta_0| proportional to (1 - s). So the swing in
    (1 - s) across the cohort IS the test's leverage: the ratio of the predicted signal at
    the 5th percentile of shrinkage to its value at the 95th. A ratio of 1.0 means every case
    is predicted to sit the same distance from the prior and there is nothing to correlate
    against -- which is NOT the same finding as a flat slope, though the two print alike.

    Measured on the site cohort: T1 10.5x, PD 1.21x, T2 1.09x, D 1.0009x. Reporting D as
    "flat -- no mechanism" claimed a null result where no test had been performed.
    """
    out = {}
    for p in PARAMS:
        s = np.array([r[key][p] for r in rows], dtype=float)
        lo, hi = float(np.percentile(s, 5)), float(np.percentile(s, 95))
        out[p] = {"p05": lo, "p95": hi, "median": float(np.median(s)),
                  "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
                  # predicted-signal swing; guard the degenerate s -> 1 end
                  "ratio": float((1.0 - lo) / (1.0 - hi)) if hi < 1.0 else float("inf")}
    return out


def sign_test(vals):
    """One-sided binomial p that this many of n per-scanner correlations came out negative.

    Each instrument gets one vote. A median over 5 scanners weights an n=85 scanner the same
    as an n=2075 one, so on its own it can invert: PD's median is +0.113 while its largest
    scanner sits at -0.413. The sign count is what survives that.
    """
    v = [x for x in vals if x == x]
    if len(v) < 3:
        return {"n": len(v), "n_neg": sum(x < 0 for x in v), "p": float("nan")}
    n, k = len(v), sum(x < 0 for x in v)
    p = sum(math.comb(n, j) for j in range(k, n + 1)) / (2.0 ** n)
    return {"n": n, "n_neg": k, "p": float(p)}


def within_scanner_regression(rows, key="shrinkage"):
    """rho(distance-from-prior, shrinkage), pooled and WITHIN each scanner.

    The within-scanner version is the one that matters: it holds the instrument fixed and
    varies only the protocol, so a correlation there cannot be a scanner effect wearing a
    protocol costume.
    """
    out = {}
    for p in PARAMS:
        x = [r[key][p] for r in rows]
        y = [r["dist"][p] for r in rows]
        pooled = spearman(x, y)
        per = {}
        by = defaultdict(list)
        for r in rows:
            by[r["scanner"]].append(r)
        for s, rs in by.items():
            if len(rs) < 50:
                continue
            per[s] = spearman([r[key][p] for r in rs], [r["dist"][p] for r in rs])
        vals = [v for v in per.values() if v == v]
        out[p] = {"pooled": pooled, "per_scanner": per, "sign": sign_test(vals),
                  # a median over fewer than 3 scanners is not a summary; report it as
                  # undefined rather than let a 2-point median read as a result
                  "median_within": (float(np.median(vals)) if len(vals) >= 3
                                    else float("nan")),
                  "median_within_raw": float(np.median(vals)) if vals else float("nan"),
                  "n_scanners": len(vals)}
    return out


def _balanced_acc(rows, param, fold, labels):
    """Nearest-centroid balanced accuracy for one parameter under a given label assignment."""
    accs, n_eff = [], []
    for f in sorted(set(fold.values())):
        tr = [r for r in rows if fold[r["patient_id"] or r["case_id"]] != f]
        te = [r for r in rows if fold[r["patient_id"] or r["case_id"]] == f]
        by = defaultdict(list)
        for r in tr:
            by[labels[r["case_id"]]].append(r["theta"][param])
        cent = {s: float(np.median(v)) for s, v in by.items() if len(v) >= 20}
        if len(cent) < 2 or not te:
            continue
        per_class = defaultdict(lambda: [0, 0])
        for r in te:
            lab = labels[r["case_id"]]
            if lab not in cent:
                continue
            pred = min(cent, key=lambda s: abs(r["theta"][param] - cent[s]))
            per_class[lab][1] += 1
            per_class[lab][0] += int(pred == lab)
        recalls = [c / n for c, n in per_class.values() if n >= 5]
        if recalls:
            accs.append(float(np.mean(recalls)))
            n_eff.append(len(recalls))
    return (float(np.mean(accs)) if accs else float("nan"),
            float(np.std(accs)) if accs else float("nan"),
            float(np.mean(n_eff)) if n_eff else float("nan"))


def separability(rows, seed=0, n_folds=5, n_perm=200):
    """Secondary: can scanner identity be read off each parameter alone? Patient-level folds.

    Deliberately simple -- nearest-centroid on the single ROI median -- so it measures what the
    PARAMETER carries rather than what a flexible model can mine from a long feature vector.
    The archived vendor_separability.py notes the latter detects any systematic difference,
    task-relevant or not.

    SIGNIFICANCE COMES FROM A PERMUTATION NULL, not from a standard error. Two earlier
    attempts were wrong: `acc > 1/n_scanners_present` understated chance because most scanners
    never enter the classifier (measured: 0.20 reported where the effective chance was 0.50),
    and a 2-standard-error margin over 5 folds is itself too noisy -- uninformative features
    cleared it in 2 of 12 trials. Shuffling the scanner labels and taking the 95th percentile
    of the resulting accuracy assumes nothing and cannot be gamed by fold count.
    """
    rng = np.random.default_rng(seed)
    pats = sorted({r["patient_id"] or r["case_id"] for r in rows})
    rng.shuffle(pats)
    fold = {p: i % n_folds for i, p in enumerate(pats)}
    true_labels = {r["case_id"]: r["scanner"] for r in rows}

    out = {}
    for p in PARAMS:
        acc, sd, eff = _balanced_acc(rows, p, fold, true_labels)
        # permute scanner labels ACROSS PATIENTS, keeping the label distribution intact
        null = []
        by_pat = defaultdict(list)
        for r in rows:
            by_pat[r["patient_id"] or r["case_id"]].append(r["case_id"])
        pat_label = {q: true_labels[cs[0]] for q, cs in by_pat.items()}
        keys = list(pat_label)
        for _ in range(n_perm):
            vals = list(pat_label.values())
            rng.shuffle(vals)
            perm = {}
            for q, v in zip(keys, vals):
                for c in by_pat[q]:
                    perm[c] = v
            a, _, _ = _balanced_acc(rows, p, fold, perm)
            if a == a:
                null.append(a)
        thr = float(np.percentile(null, 95)) if null else float("nan")
        out[p] = {"balanced_acc": acc, "sd": sd, "n_effective_classes": eff,
                  "chance": 1.0 / eff if eff and eff == eff and eff >= 1 else float("nan"),
                  "null_p95": thr, "n_perm": len(null),
                  "above_chance": bool(acc == acc and thr == thr and acc > thr)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-vox", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mans = read_manifests(args.manifest_dir)
    print(f"{len(mans)} manifests; fitting cached cases (MAP, not a trained encoder)")
    rows, why = collect(args.cache_dir, mans, args.arm, args.prior_sd, args.sigma,
                        args.n_vox, args.limit)
    print(f"\n{len(rows)} cases fitted")
    for k, v in why.most_common():
        print(f"  skipped {v:5d}  {k}")
    if len(rows) < 100:
        raise SystemExit("too few cases fitted to say anything")

    print("\n=== PRIMARY: does the estimate sit CLOSER to the prior when shrinkage is higher? ===")
    print("  Prediction from the shrinkage decomposition: NEGATIVE rho.")
    print("  The within-scanner column holds the instrument fixed and varies only protocol.\n")
    reg = within_scanner_regression(rows)
    lev = leverage(rows)
    print(f"  {'param':>5s} {'shrinkage':>10s} {'leverage':>9s} {'rho within':>11s} "
          f"{'neg/n':>7s} {'sign p':>7s}   verdict")
    shr = {p: lev[p]["median"] for p in PARAMS}
    for p in PARAMS:
        d, L, sg = reg[p], lev[p], reg[p]["sign"]
        mw = d["median_within"]
        if d["n_scanners"] < 3:
            v = f"UNDERPOWERED — only {d['n_scanners']} scanner(s) above the case threshold"
        elif L["ratio"] < MIN_LEVERAGE:
            v = (f"NO LEVERAGE — (1-s) moves {100*(L['ratio']-1):.2f}% p05→p95; "
                 "untested, not falsified")
        elif sg["n_neg"] >= sg["n"] - 1 and mw < -0.1:
            v = "supports"
        elif sg["n_neg"] <= 1 and mw > 0.1:
            v = "CONTRADICTS"
        else:
            v = "no coherent effect — scanners disagree on sign"
        mws = "      n/a" if mw != mw else f"{mw:>+11.3f}"
        ps = "    n/a" if sg["p"] != sg["p"] else f"{sg['p']:>7.3f}"
        print(f"  {p:>5s} {shr[p]:>10.3f} {L['ratio']:>8.2f}x {mws} "
              f"{sg['n_neg']:>3d}/{sg['n']:<3d} {ps}   {v}")
    print("  leverage = predicted signal (1-s) at p05 shrinkage over its value at p95. At 1.00x")
    print("  every case is predicted to sit the same distance from the prior, so a flat slope")
    print("  is the absence of a test rather than the result of one.")
    print("  A median over 5 scanners weights n=85 like n=2075; the sign count does not.")

    print("\n=== SECONDARY: scanner separability per parameter (what the field reports) ===")
    sep = separability(rows)
    print(f"  {'param':>5s} {'bal. acc':>9s} {'sd':>7s} {'classes':>8s} {'chance':>8s}   "
          f"verdict (vs a permutation null)")
    for p in PARAMS:
        d = sep[p]
        v = "above chance" if d["above_chance"] else "at or below chance"
        print(f"  {p:>5s} {d['balanced_acc']:>9.3f} {d['sd']:>7.3f} "
              f"{d['n_effective_classes']:>8.1f} {d['chance']:>8.3f}   {v}"
              f"  (permutation p95 {d['null_p95']:.3f}, n={d['n_perm']})")
    print("  Nearest-centroid on the single ROI median, patient-level folds -- it measures what")
    print("  the PARAMETER carries, not what a flexible model can mine from a long feature")
    print("  vector. The archived vendor_separability.py notes that the latter detects any")
    print("  systematic difference, task-relevant or not.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"n_cases": len(rows), "arm": args.arm, "prior_sd": args.prior_sd,
                   "regression": reg, "separability": sep, "leverage": lev,
                   "median_shrinkage": shr, "skipped": dict(why), "cases": rows},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
