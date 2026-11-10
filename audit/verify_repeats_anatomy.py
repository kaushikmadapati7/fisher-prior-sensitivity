r"""Settle the repeat question with ANATOMY, since de-identification killed the tags.

WHERE WE ARE. `verify_repeats.py` returned an ambiguous answer:

  * null test: 152 same-unit observed vs 106.9 expected-at-random -> z = 4.66, so pure
    PatientID collisions are REJECTED (p < 1e-5). There is real structure.
  * but 1.42x enrichment is far below the 2.44x a fully-genuine cohort produced in
    simulation, so this looks like a MIXTURE.
  * and the corroborating tags are gone: PatientName / PatientBirthDate / PatientSex are
    all 0% populated. Test 2 cannot run.

The mixture fraction is underdetermined by scanner pairing alone: solving
f*s + (1-f)*0.122 = 0.1735 gives anywhere from ~78 to ~580 genuine repeats depending on
s = P(same scanner | genuine), which is unknown because the site has 40 scanner units and a
fleet that changes over years.

THE WAY THROUGH. Identity is still recoverable from the IMAGES, and we do not even need
to load them: `pilot_log.json` records the prostate segmentation per case --
`seg_strategy = full(vol=30.4,dist=0.06,svr=0.25)` and `n_mask_vox`. Prostate volume is a
per-person biometric: roughly 20-120 cc across men, but stable within one man over months
(BPH growth is a few percent per year). So:

    two exams of the SAME man     -> similar prostate volume
    two exams of DIFFERENT men    -> volumes as unrelated as any random pair

This is completely independent of de-identification, costs no DICOM reads and no image
loads, and yields not just a yes/no but an estimate of the genuine fraction AND a
high-confidence subset we can actually use.

CONFOUNDS, stated up front:
  * volume genuinely drifts with time (BPH) and with treatment -- real pairs separated by
    years will differ more, so this test UNDERCOUNTS genuine repeats. It is a lower bound.
  * segmentation error adds noise; `svr` and `dist` are recorded per case and a failed
    segmentation should be excluded rather than treated as biology.
  * volume is one number, so some unrelated men match by chance -- which is exactly what
    the null pairing quantifies, and why the estimate is a mass-excess, not a count of
    pairs below a threshold.

WHICH DECISION THIS CHANGES:
  strong separation  -> genuine repeats exist; use the high-confidence subset written to
                        --out as the traveling-subject cohort, and report ITS size, not 607.
  no separation      -> the repeats are collisions after all, the earlier physharm zero was
                        right, and the traveling-subject arm is dead.

    python verify_repeats_anatomy.py --cache runs/header_cache.json \
        --reg-root <image-root> --out runs/anatomy_repeats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict


def _vol(cdir):
    """(volume_cc, n_mask_vox, svr) from pilot_log.json, or (None, None, None)."""
    try:
        d = json.load(open(os.path.join(cdir, "pilot_log.json")))
    except (OSError, json.JSONDecodeError):
        return None, None, None
    st = d.get("stats", {})
    vol = svr = None
    m = re.search(r"vol=([\d.]+)", str(st.get("seg_strategy", "")))
    if m:
        vol = float(m.group(1))
    m = re.search(r"svr=([\d.]+)", str(st.get("seg_strategy", "")))
    if m:
        svr = float(m.group(1))
    n = st.get("n_mask_vox")
    return vol, (float(n) if isinstance(n, (int, float)) else None), svr


def _rel(a, b):
    return abs(a - b) / max(1e-9, 0.5 * (a + b))


def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="runs/header_cache.json")
    ap.add_argument("--reg-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class-tolerance", action="store_true",
                    help="DIAGNOSTIC, not a fix. Calibrates the tolerance separately for "
                         "cross-scanner and same-scanner pairs so both carry the same "
                         "false-match rate, and prints the gap between them as evidence that "
                         "volume-based identity is scanner-confounded. It does NOT rescue the "
                         "cross-scanner cohort: simulation shows precision moving only "
                         "71.3%%->72.1%%, because scanner-dependent segmentation noise widens "
                         "the genuine and false distributions TOGETHER. See the run log "
                         "Phase 3c.")
    ap.add_argument("--tol", type=float, default=None,
                    help="relative-volume tolerance defining a high-confidence pair; "
                         "default = the null distribution's 10th percentile, so the "
                         "false-match rate is 10%% by construction")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    by_case = json.load(open(args.cache))
    vols = {}
    for c in by_case:
        v, n, svr = _vol(os.path.join(args.reg_root, c))
        if v and v > 0:
            vols[c] = (v, n, svr)
    print(f"{len(by_case)} cached exams | {len(vols)} with a prostate volume in pilot_log")
    if len(vols) < 100:
        raise SystemExit("too few volumes parsed -- check pilot_log seg_strategy format")

    allv = [v for v, _, _ in vols.values()]
    print(f"  prostate volume cc: p10={_q(allv,.1):.1f} p50={_q(allv,.5):.1f} "
          f"p90={_q(allv,.9):.1f}  (across-men spread is what gives the test power)")

    per_patient = defaultdict(list)
    for c, r in by_case.items():
        if r.get("patient_id") and c in vols:
            per_patient[r["patient_id"]].append(c)
    repeats = {p: cs for p, cs in per_patient.items() if len(cs) > 1}

    within = []
    for p, cs in repeats.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                within.append((p, cs[i], cs[j],
                               _rel(vols[cs[i]][0], vols[cs[j]][0])))
    print(f"  {len(repeats)} putative repeat patients -> {len(within)} within-pairs")

    # null: pairs of exams from DIFFERENT PatientIDs, same count
    cases = [c for c in vols if by_case[c].get("patient_id")]
    null, null_pairs_ids = [], []
    guard = 0
    while len(null) < max(len(within), 2000) and guard < 10 ** 7:
        guard += 1
        a, b = rng.choice(cases), rng.choice(cases)
        if a == b or by_case[a]["patient_id"] == by_case[b]["patient_id"]:
            continue
        null.append(_rel(vols[a][0], vols[b][0]))
        null_pairs_ids.append((a, b))

    print("\n=== RELATIVE PROSTATE-VOLUME DIFFERENCE ===")
    print(f"{'':10s} {'p10':>8s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'p90':>8s}")
    wv = [w[3] for w in within]
    for nm, v in (("within", wv), ("null", null)):
        print(f"{nm:10s} " + "".join(f"{_q(v,p):>8.3f}" for p in (.1, .25, .5, .75, .9)))

    # AUC that a random within-pair is more similar than a random null pair
    ns = sorted(null)
    import bisect
    auc = sum(bisect.bisect_left(ns, w) for w in wv) / (len(wv) * len(ns))
    auc = 1 - auc
    print(f"\n  AUC(within more similar than null) = {auc:.3f}   (0.5 = indistinguishable)")

    # A pooled tolerance is confounded. Identity is verified from segmented prostate volume,
    # which is measured from the images and therefore varies with the scanner -- the very
    # effect the traveling-subject arm exists to measure. Within one scanner the measurement
    # is consistent and a 9% window genuinely selects the same man; across scanners the window
    # admits many more unrelated men. Measured consequence (runs/traveling_arm.json): the
    # cross-scanner cohort was INDISTINGUISHABLE from random pairs on all four parameters,
    # while same-scanner pairs were clearly more similar. Calibrate each class separately.
    def _is_cross(a, b):
        ma, mb = by_case.get(a, {}).get("model"), by_case.get(b, {}).get("model")
        return bool(ma and mb and ma != mb)

    tol = args.tol if args.tol is not None else _q(null, .10)
    tol_cross = tol_same = tol
    if args.per_class_tolerance:
        # Split BOTH the null and the within-pairs by class, and set each tolerance at the
        # same false-match rate. Without this the cross-scanner cohort is mostly false
        # matches: measured in runs/traveling_arm.json, cross-scanner pairs were
        # indistinguishable from unrelated men on all four parameters (cross/null 0.87-1.10)
        # while same-scanner pairs were clearly more similar (same/null 0.44-0.75).
        null_c = [_rel(vols[a][0], vols[b][0]) for a, b in null_pairs_ids if _is_cross(a, b)]
        null_s = [_rel(vols[a][0], vols[b][0]) for a, b in null_pairs_ids if not _is_cross(a, b)]
        if len(null_c) > 50 and len(null_s) > 50:
            tol_cross, tol_same = _q(null_c, .10), _q(null_s, .10)
            print(f"\n  per-class tolerances at a matched 10% false-match rate:")
            print(f"    cross-scanner {tol_cross:.4f}   (n_null {len(null_c)})")
            print(f"    same-scanner  {tol_same:.4f}   (n_null {len(null_s)})")
            print(f"    pooled would have been {tol:.4f} for both")
            if tol_cross > tol_same:
                print("    -> the cross-scanner null is WIDER, exactly as expected: prostate")
                print("       volume is measured from the images, so it varies with the")
                print("       scanner. A pooled window admits more unrelated men across")
                print("       scanners than within one.")
        else:
            print("\n  per-class tolerance requested but too few null pairs per class; "
                  "falling back to pooled")
    alpha = sum(1 for x in null if x <= tol) / len(null)
    p_w = sum(1 for x in wv if x <= tol) / len(wv)
    f_est = max(0.0, (p_w - alpha) / max(1e-9, 1 - alpha))
    print(f"\n  tolerance {tol:.3f} (null 10th pct; false-match rate {alpha:.1%})")
    print(f"  within-pairs under tolerance: {p_w:.1%}")
    print(f"  -> excess-mass estimate of the GENUINE fraction: {f_est:.1%} "
          f"(~{f_est*len(repeats):.0f} of {len(repeats)} patients)")
    print("     lower bound: volume drifts with time, so long-interval genuine repeats")
    print("     fall outside the tolerance and are not counted.")

    hi = [w for w in within
          if w[3] <= (tol_cross if _is_cross(w[1], w[2]) else tol_same)]
    hi_pat = {w[0] for w in hi}
    cross = {w[0] for w in hi
             if by_case[w[1]]["model"] != by_case[w[2]]["model"]}
    same = {w[0] for w in hi if by_case[w[1]]["unit"] == by_case[w[2]]["unit"]}
    print(f"\n=== HIGH-CONFIDENCE COHORT (cross <= {tol_cross:.3f}, same <= {tol_same:.3f}) ===")
    print(f"  {len(hi_pat)} patients, {len(hi)} pairs")
    print(f"    cross-MODEL (candidate traveling subjects): {len(cross)}")
    print(f"    same-UNIT   (test-retest noise floor):      {len(same)}")
    print(f"  expect ~{alpha*len(within):.0f} false matches at this tolerance.")

    # Significance of the excess mass under the pure-collision null: if every putative
    # repeat were a collision, the fraction under tolerance would equal `alpha`.
    # AUC alone is the WRONG gate here -- in a genuine 30%% mixture the AUC is diluted to
    # ~0.63 by the 70%% of pairs that are collisions, so a high-AUC threshold would call a
    # real cohort "weak". Test the excess directly instead.
    n_w = len(wv)
    se = (alpha * (1 - alpha) / max(1, n_w)) ** 0.5
    z = (p_w - alpha) / max(1e-9, se)
    print(f"\n  excess over the collision null: z = {z:.2f} "
          f"({p_w:.1%} vs {alpha:.1%} on n={n_w} pairs)")

    # Precision/size trade-off. The default tolerance (null 10th pct) admits a 10%
    # false-match rate; a tighter one buys purity at the cost of cohort size. Which
    # point to use is a design decision, so show the curve rather than one number.
    print("\n=== TOLERANCE SWEEP (choose the operating point) ===")
    print(f"{'tol':>7s} {'false-match':>12s} {'pairs':>7s} {'est. false':>11s} "
          f"{'precision':>10s} {'cross-model':>12s} {'same-unit':>10s}")
    for pct in (0.01, 0.02, 0.05, 0.10, 0.20):
        t = _q(null, pct)
        a_ = sum(1 for x in null if x <= t) / len(null)
        sel = [w for w in within if w[3] <= t]
        if not sel:
            continue
        fp = a_ * len(within)
        prec = max(0.0, 1 - fp / len(sel))
        xm = len({w[0] for w in sel
                  if by_case[w[1]]["model"] != by_case[w[2]]["model"]})
        su = len({w[0] for w in sel if by_case[w[1]]["unit"] == by_case[w[2]]["unit"]})
        mark = "  <- default" if abs(pct - 0.10) < 1e-9 else ""
        print(f"{t:>7.3f} {a_:>11.1%} {len(sel):>7d} {fp:>11.0f} {prec:>9.0%} "
              f"{xm:>12d} {su:>10d}{mark}")
    print("  'cross-model' and 'same-unit' are PATIENT counts before subtracting false")
    print("  matches; multiply by precision for the defensible figure.")

    print("\n=== VERDICT ===")
    if z >= 4 and f_est >= 0.05:
        print(f"  ANATOMY CONFIRMS genuine repeats (z = {z:.1f}, AUC {auc:.3f}).")
        print(f"  But the usable cohort is the HIGH-CONFIDENCE SUBSET: {len(cross)} cross-model")
        print(f"  patients and {len(same)} same-unit controls -- NOT {len(repeats)}.")
        print("  Report that number with the tolerance and the false-match rate attached.")
        print("  Both earlier results were partly right: genuine repeats exist, and the raw")
        print("  count was inflated by PatientID collisions.")
        if len(cross) < 30:
            print(f"  WARNING: {len(cross)} cross-model patients is below the ~30 needed for a")
            print("  benchmark. Treat as a case study and keep the ADC-matched proxy primary.")
    elif z >= 2:
        print(f"  MARGINAL (z = {z:.1f}). Some genuine repeats, but the excess is weak.")
        print("  Keep the ADC-matched proxy primary and report these as supporting evidence.")
    else:
        print(f"  NO EXCESS over the collision null (z = {z:.1f}, AUC {auc:.3f}). The putative")
        print("  repeats look like unrelated men. The earlier physharm zero was right --")
        print("  drop the traveling-subject arm.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(auc=auc, tol=tol, false_match_rate=alpha, excess_z=z,
                       genuine_fraction_estimate=f_est,
                       n_putative_repeat_patients=len(repeats),
                       n_high_conf_patients=len(hi_pat),
                       n_high_conf_cross_model=len(cross),
                       n_high_conf_same_unit=len(same),
                       high_conf_pairs=[{"patient": p, "a": a, "b": b, "rel_vol_diff": d}
                                        for p, a, b, d in hi]),
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
