r"""Traveling-subject arm: the gold-standard invariance measurement. (Issue #11, the project plan 3.6)

The same person, scanned on two different machines. Biology is (nearly) constant, so the
difference between the two estimates is scanner. Every other arm in this project is a proxy
for this one, and the harmonization literature does not have it at this scale: PhyCHarm used
9 traveling subjects, Borges et al. used none and never measured invariance at all, and the
2025 survey names validation-without-ground-truth as the field's open problem.

WHAT MAKES IT INTERPRETABLE IS THE CONTROL, NOT THE COHORT. Study dates were destroyed by
de-identification, so we cannot filter to short intervals and cannot claim biology held still.
We do not need to. The same-unit repeats come from the same clinical process, so they carry
the same unknown interval distribution and therefore the same biological drift. The reported
quantity is the EXCESS:

    excess = |dtheta| (cross-scanner)  -  |dtheta| (same-scanner)

Anything at or below the same-unit floor is not a scanner effect. Without those controls the
cross-scanner number is uninterpretable, which is why splits.py holds them out too.

ESTIMATOR. The MAP fit from the synthetic arm, applied per voxel to cached images -- not a
trained encoder. Two reasons: it is the estimator the Cramer-Rao/shrinkage theory actually
describes, and it makes this arm independent of #10 rather than downstream of it. The encoder
version is a second measurement that belongs inside the real arm.

THE PREDICTION IS NOT "SLOPPIER LEAKS MORE". Part II was corrected by #7: leakage is an
inverted U in identifiability. Pre-registered expectation, so it cannot be fitted after the
fact:

    D   anchored by ADC, shrinkage ~0.01   excess ~ 0   (data pins it)
    T2  intermediate,    shrinkage ~0.09   excess > 0
    PD  intermediate,    shrinkage ~0.19   excess > 0, the largest
    T1  prior-pinned,    shrinkage ~0.89   excess ~ 0   (the prior pins it, identically
                                                         on every scanner)

A monotone result would CONTRADICT the corrected thesis, and should be reported as such.

    python -m physharm.analysis.traveling_arm --cache-dir cache_shared \
        --manifest-dir manifests_site --traveling runs/traveling_usable.json \
        --anatomy runs/anatomy_repeats.json --out runs/traveling_arm.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from physharm.analysis.crlb_table import acquisition, canonical_scanner
from physharm.analysis.synthetic_arm import PARAMS, TISSUE, map_estimate
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

CONTRAST_ROWS = {"T2W": "t2w", "DWI": "dwi", "S0": "s0"}


def load_case(cache_dir, case_id):
    """Cached arrays for one case, or None. Keys follow dataset_physics.build_cache."""
    p = os.path.join(cache_dir, f"{case_id.replace('/', '__')}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: np.asarray(z[k], dtype=np.float32) for k in z.files}


def _rows_and_signal(cached, acq, arm="A1", n_vox=4000, rng=None):
    """(measurement descriptors, log-signal matrix) for the ROI voxels of one case.

    Returns None when the case lacks a needed contrast. The signal is taken in log space, so
    the per-case normalization scale becomes an additive constant that PD absorbs -- which is
    exactly why the cache must be built with `shared` normalization (#28): `ref_median`
    forces each contrast's ROI median to 1 and destroys the inter-contrast ratio where T2
    lives.

    ARMS. The rows returned here must match `fisher.measurement_rows(arm, acq)` exactly, in
    count and in kind, or the predicted leak describes a different experiment from the one the
    estimator ran. `tests/test_prior_shift.py` asserts that correspondence for every arm rather
    than trusting these two lists to be edited together.

    This argument existed and was ignored until #48 -- every caller got A1 whatever it asked
    for. A3 (T2W alone) and A4 (no ADC anchor) are subsets of what A1 already reads from the
    cache. A2 needs the low-b band, which the cache does not carry.
    """
    mask = cached.get("__mask")
    if mask is None:
        return None
    idx = np.flatnonzero(mask.ravel() > 0.5)
    if idx.size < 100:
        return None
    rng = rng or np.random.default_rng(0)
    if idx.size > n_vox:
        idx = rng.choice(idx, size=n_vox, replace=False)

    if arm == "A2":
        return None                       # low-b band is not in the cache; needs a rebuild
    if arm == "A5":
        return None                       # counterfactual: no short-TR series is registered
    if arm not in ("A1", "A3", "A4"):
        raise ValueError(f"unknown arm {arm!r}")

    want = [("T2W", acq["TR_T2W"], acq["TE_T2W"], 0.0)]
    if arm in ("A1", "A4"):
        want.append(("DWI", acq["TR_DWI"], acq["TE_DWI"], acq["b_hi"]))

    rows, sig = [], []
    for name, TR, TE, b in want:
        a = cached.get(name)
        if a is None:
            return None
        rows.append((TR, TE, b))
        sig.append(a.ravel()[idx])
    if arm == "A1":
        adc = cached.get("__adc")
        if adc is None:
            return None
        rows.append("anchor")
        sig.append(adc.ravel()[idx])

    S = np.stack(sig)
    ok = np.all(S > 0, axis=0)                 # log space: drop non-positive voxels
    if ok.sum() < 50:
        return None
    return rows, np.log(S[:, ok])


def estimate_case(cached, acq, prior_sd=10.0, sigma=0.05, n_vox=4000, seed=0, arm="A1"):
    """ROI-median theta for one exam, as a dict of the four parameters in log units.

    `arm` selects which measurements the fit sees. It used to be fixed at A1 here while
    real_arm's --arm changed only the predicted column, so an --arm sweep compared a
    prediction for one measurement set against a fit of another.
    """
    got = _rows_and_signal(cached, acq, arm=arm, n_vox=n_vox, rng=np.random.default_rng(seed))
    if got is None:
        return None
    rows, y = got
    x = map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()},
                     prior_sd=prior_sd, sigma=sigma)
    return {k: float(np.median(x[i])) for i, k in enumerate(PARAMS)}


def pair_deltas(pairs, cache_dir, acqs, **kw):
    """|delta theta| per pair, plus WHY each dropped pair dropped.

    Half the cross-scanner pairs failed on the first real run and the script reported only an
    aggregate. Selective attrition would bias the surviving subset, so the reasons are counted.
    """
    out, why = [], Counter()
    for a, b in pairs:
        ca, cb = load_case(cache_dir, a), load_case(cache_dir, b)
        if ca is None or cb is None:
            why["exam not in cache (excluded, or a different cohort)"] += 1
            continue
        if a not in acqs or b not in acqs:
            why["no acquisition params in the manifest"] += 1
            continue
        ta, tb = estimate_case(ca, acqs[a], **kw), estimate_case(cb, acqs[b], **kw)
        if ta is None or tb is None:
            why["fit failed (missing contrast or ROI too small)"] += 1
            continue
        out.append({"a": a, "b": b,
                    "delta": {k: abs(ta[k] - tb[k]) for k in PARAMS}})
    return out, why


def classify_pairs(traveling_json, anatomy_json, by_case, exclude=()):
    """Anatomy-verified pairs, split into cross-scanner and the same-unit noise floor.

    `anatomy_repeats.json` carries the verified pairs with their relative volume difference;
    `traveling_usable.json` carries which of those are usable. Only pairs whose BOTH exams
    have a known scanner are classified -- a missing model tag is not a scanner change.

    `exclude` drops pairs where either exam is on a named scanner. This exists because of #56:
    a per-scanner SCALE fault adds a constant to |dtheta| for every cross-scanner pair that
    scanner appears in, and nothing at all to same-unit pairs -- the exact signature Phase 3c
    attributed to a circular identity verifier. MAGNETOM Vida carries a 2.4x cached-ADC error
    and appears in 23% of the cross pairs, so the two explanations are not separable until it
    is dropped. Excluding it is the cheap half of the answer; rebuilding the cache is the
    other half.
    """
    anat = json.load(open(anatomy_json))
    tol, fmr = anat.get("tol"), anat.get("false_match_rate")
    # Canonicalize before comparing. the project plan ground rule 5: the raw ManufacturerModelName
    # splits one physical model across spelling variants, which would file same-machine
    # repeats as cross-scanner AND remove them from the same-unit floor -- biasing the
    # excess toward zero from both ends at once. --exclude-scanners matches canonically too,
    # which is what its help text already promises.
    ex = {canonical_scanner(e) for e in exclude}
    cross, same, dropped = [], [], 0
    for d in anat.get("high_conf_pairs", []):
        a, b = d.get("a"), d.get("b")
        ra, rb = by_case.get(a), by_case.get(b)
        if not ra or not rb or not ra.get("model") or not rb.get("model"):
            continue
        ma, mb = canonical_scanner(ra["model"]), canonical_scanner(rb["model"])
        if ex and ({ma, mb} & ex):
            dropped += 1
            continue
        (cross if ma != mb else same).append((a, b))
    return {"cross": cross, "same": same, "tolerance": tol, "false_match_rate": fmr,
            "excluded_scanners": sorted(ex), "n_pairs_dropped": dropped}


def null_pairs(all_cases, by_case, n, rng):
    """Pairs of exams from DIFFERENT patients, matched to the observed cross-scanner design.

    The same-unit group controls for SCANNER but not for IDENTITY. If two unrelated men show
    the same |dtheta| as one man on two machines, the within-patient signal is measuring
    nothing -- and on the first real run the same-unit floor was already ~0.5 log units (a 65%
    within-patient T2 difference on ONE scanner), which is not plausible test-retest noise.
    This is the arm that tells us whether any of it is identity at all.
    """
    ids = [c for c in all_cases if by_case.get(c, {}).get("patient_id")]
    out, guard = [], 0
    while len(out) < n and guard < 50 * max(n, 1):
        guard += 1
        a, b = rng.choice(ids), rng.choice(ids)
        if a == b or by_case[a]["patient_id"] == by_case[b]["patient_id"]:
            continue
        if by_case[a].get("model") == by_case[b].get("model"):
            continue                       # match the cross-SCANNER design
        out.append((a, b))
    return out


def summarize(cross, same):
    """Excess over the same-unit floor, with bootstrap CIs over PAIRS."""
    rng = np.random.default_rng(0)
    out = {}
    for k in PARAMS:
        c = np.array([d["delta"][k] for d in cross], dtype=float)
        s = np.array([d["delta"][k] for d in same], dtype=float)
        if c.size < 3 or s.size < 3:
            out[k] = {"n_cross": int(c.size), "n_same": int(s.size), "excess": None}
            continue
        boot = []
        for _ in range(2000):
            boot.append(np.median(rng.choice(c, c.size)) - np.median(rng.choice(s, s.size)))
        boot = np.array(boot)
        out[k] = {"n_cross": int(c.size), "n_same": int(s.size),
                  "median_cross": float(np.median(c)), "median_same": float(np.median(s)),
                  "excess": float(np.median(c) - np.median(s)),
                  "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--anatomy", required=True, help="runs/anatomy_repeats.json")
    ap.add_argument("--header-cache", default=None,
                    help="runs/header_cache.json, for each exam's scanner model")
    ap.add_argument("--exclude-scanners", nargs="*", default=[],
                    help="drop pairs involving these scanner models (canonicalized, "
                         "case-insensitive). Use for MAGNETOM VIDA and VERIO while the "
                         "cache ADC scale fault of #56 is open: Vida carries a 2.4x scale "
                         "error that adds a fixed offset to every cross-scanner pair it "
                         "appears in, which is 23%% of them.")
    ap.add_argument("--n-vox", type=int, default=4000)
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    acqs = {}
    for m in read_manifests(args.manifest_dir):
        a, _ = acquisition(m)
        if a is not None:
            acqs[m.get("case_id")] = a
    by_case = json.load(open(args.header_cache)) if args.header_cache else {}
    if not by_case:
        raise SystemExit("--header-cache is required: pair classification needs each exam's "
                         "scanner model, and a missing model tag must not count as a change")

    cls = classify_pairs(None, args.anatomy, by_case, exclude=args.exclude_scanners)
    if cls["excluded_scanners"]:
        print(f"  excluding {', '.join(cls['excluded_scanners'])}: "
              f"{cls['n_pairs_dropped']} pairs dropped (#56 ADC scale fault)")
    print(f"{len(acqs)} manifests with acquisition params")
    print(f"anatomy-verified pairs: {len(cls['cross'])} cross-scanner, "
          f"{len(cls['same'])} same-unit  "
          f"(tolerance {cls['tolerance']}, false-match rate {cls['false_match_rate']})")

    kw = dict(prior_sd=args.prior_sd, sigma=args.sigma, n_vox=args.n_vox)
    cross, why_c = pair_deltas(cls["cross"], args.cache_dir, acqs, **kw)
    same, why_s = pair_deltas(cls["same"], args.cache_dir, acqs, **kw)
    print(f"estimated: {len(cross)}/{len(cls['cross'])} cross-scanner, "
          f"{len(same)}/{len(cls['same'])} same-unit pairs")
    for label, why in (("cross", why_c), ("same", why_s)):
        for reason, n in why.most_common():
            print(f"    {label} dropped {n:4d}: {reason}")

    # THE identity control. Without it, "within-patient" is an assumption.
    rng = np.random.default_rng(0)
    null, why_n = pair_deltas(
        null_pairs(sorted(by_case), by_case, len(cross), rng),
        args.cache_dir, acqs, **kw)
    print(f"    null (different patients, different scanners): {len(null)} pairs")

    res = summarize(cross, same)
    res_null = summarize(cross, null)          # cross vs UNRELATED men
    print("\n=== WITHIN-PATIENT |dtheta| (log units) ===")
    print(f"{'param':>5s} {'n_cross':>8s} {'n_same':>7s} {'cross':>9s} {'same':>9s} "
          f"{'EXCESS':>9s} {'95% CI':>20s}")
    for k in PARAMS:
        r = res[k]
        if r.get("excess") is None:
            print(f"{k:>5s} {r['n_cross']:>8d} {r['n_same']:>7d}   too few pairs to summarize")
            continue
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        print(f"{k:>5s} {r['n_cross']:>8d} {r['n_same']:>7d} {r['median_cross']:>9.4f} "
              f"{r['median_same']:>9.4f} {r['excess']:>+9.4f} {ci:>20s}")

    print("\n=== IDENTITY CONTROL: is any of this within-patient at all? ===")
    print("  If unrelated men differ by as much as one man on two scanners, the pairing")
    print("  carries no identity information and nothing above is interpretable.")
    print(f"  {'param':>5s} {'same-unit':>10s} {'cross':>9s} {'null':>9s}   reading")
    identity_ok = True
    for k in PARAMS:
        r, rn = res[k], res_null[k]
        if r.get("excess") is None or rn.get("excess") is None:
            print(f"  {k:>5s}   too few pairs")
            continue
        nullmed = rn["median_same"]            # summarize(cross, null) -> null is 'same' slot
        ratio = nullmed / max(r["median_cross"], 1e-9)
        ok = ratio > 1.15
        identity_ok = identity_ok and ok
        tag = "ok — unrelated men differ MORE" if ok else "!! null ~ cross: NO identity signal"
        print(f"  {k:>5s} {r['median_same']:>10.4f} {r['median_cross']:>9.4f} "
              f"{nullmed:>9.4f}   {tag}")
    if not identity_ok:
        print("\n  ** DO NOT INTERPRET THE EXCESS TABLE. At least one parameter shows")
        print("     unrelated men differing no more than the same patient on two scanners.")
        print("     That is the pre-registered killing observation for this arm.")

    print("\n=== AGAINST THE PRE-REGISTERED PREDICTION (the project plan Part II, inverted U) ===")
    # COHORT MEDIAN, not whichever manifest read_manifests globbed first. The pre-registered
    # `expected` label below is a threshold on this number, and over a 140-case cohort PD
    # ranges 0.0003-0.535 and T2 0.0001-0.356 -- both straddle the 0.05 cut, so anchoring on
    # one arbitrary exam let directory ordering decide which parameters get flagged as
    # CONTRADICTS. sigma comes from the run rather than a hardcoded 0.05, which was the only
    # identifiability call in the repo ignoring its own --sigma.
    if not acqs:
        raise SystemExit("no acquisitions: cannot score the pre-registered prediction")
    per_case = []
    for a in acqs.values():
        n_rows = len(F.measurement_rows("A1", a, {k: v[0] for k, v in TISSUE.items()}))
        per_case.append(F.identifiability(a, "A1", weights=[1.0 / args.sigma ** 2] * n_rows,
                                          prior_sd=args.prior_sd)["shrinkage"])
    shr = {k: float(np.median([c[k] for c in per_case])) for k in PARAMS}
    print(f"  {'param':>5s} {'shrinkage':>10s} {'expected':>10s} {'observed':>10s}")
    for k in PARAMS:
        exp = "~0" if shr[k] < 0.05 or shr[k] > 0.7 else ">0"
        r = res[k]
        obs = ("n/a" if r.get("excess") is None else
               ">0" if r["ci95"][0] > 0 else "~0")
        flag = "" if obs == exp or obs == "n/a" else "   <-- CONTRADICTS"
        print(f"  {k:>5s} {shr[k]:>10.3f} {exp:>10s} {obs:>10s}{flag}")
    print("\n  A monotone 'sloppier leaks more' pattern would contradict the correction in")
    print("  Part II and must be reported as such, not fitted after the fact.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"summary": res, "summary_vs_null": res_null,
                   "n_cross_pairs": len(cross), "n_same_pairs": len(same),
                   "n_null_pairs": len(null),
                   "dropped_cross": dict(why_c), "dropped_same": dict(why_s),
                   "tolerance": cls["tolerance"], "false_match_rate": cls["false_match_rate"],
                   "shrinkage": shr, "cross": cross, "same": same},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
