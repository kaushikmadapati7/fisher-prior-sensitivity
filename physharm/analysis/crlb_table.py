r"""The study's predictor: a per-case CRLB table over the whole cohort. (Issue #6, the project plan 3.3/3.5)

Every downstream arm regresses scanner leakage against this table, so it is computed ONCE,
from measured per-case acquisition parameters, and reused. It needs no images, no GPU and no
labels -- only the manifests -- so it runs on a login node in seconds.

For each case and each measurement set A1-A5 it records the regularized CRLB for
{PD, T1, T2, D}, the Fisher eigenvalues, the condition number and the null-space overlap. It
then reports the two things that decide whether the study is sound:

  * ICC(CRLB | scanner) per arm -- is predicted identifiability separable from scanner
    identity, or merely a relabelling of it?
  * whether the parameter ORDERING survives the assumptions it rests on (the prior, and the
    unmeasurable relative per-contrast noise), cohort-wide rather than at one protocol.

    python -m physharm.analysis.crlb_table --manifest-dir manifests_site \
        --out runs/crlb_table.json [--csv runs/crlb_table.csv]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import Counter, defaultdict

import numpy as np

import re

from physharm.analysis.stats import icc, pcts
from physharm.data import exclusions as EX
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

ARMS = ("A1", "A2", "A3", "A4", "A5")


def canonical_scanner(s):
    """Collapse spelling variants so ONE physical scanner model is ONE group.

    the project plan ground rule 5. DICOM ManufacturerModelName is inconsistent in case and
    punctuation -- this cohort carries 'SIGNA Premier' next to 'Signa HDxt', and station
    names appear as SITECBMR6 / CBMR6 / SITE-CBMR6. Grouping on the raw string splits one
    model across several groups, which biases ICC DOWNWARD: the extra groups are drawn from
    the same protocol distribution, so they shrink the between-group share of variance.

    This was a real defect. The first cohort run of this script reported ICC(CRLB T2) = 0.204
    against 0.266 from `audit/crlb_spread.py`, which does canonicalize; part of that gap was
    exactly this. The verdict was unaffected (both far below 0.3), but the number quoted in
    the paper must come from canonicalized groups.
    """
    t = re.sub(r"[^A-Za-z0-9]+", " ", str(s or "")).strip().upper()
    t = re.sub(r"\s+", " ", t)
    # Vendor prefixes only. A cohort whose station names carry an institution prefix
    # should add it to this alternation; it is site-specific and none is assumed here.
    t = re.sub(r"^(GE MEDICAL SYSTEMS|GE|SIEMENS HEALTHINEERS|SIEMENS|PHILIPS"
               r"|PHILIPS MEDICAL SYSTEMS)\s+(?=\S)", "", t)
    return t or "UNKNOWN"

#: A5 adds a short-TR acquisition. the site registered folder has no such series, so on real
#: data A5 is a COUNTERFACTUAL protocol: it quantifies what an acquisition would have to look
#: like to make T1 identifiable, and is not a measurement. Reported, never regressed against
#: real leakage. (the site does hold an `AX T1 PROSTATE` GRE at TR~4.9 ms in raw_dicom -- a real
#: short-TR lever if it were ever registered. Worth revisiting; out of scope for #6.)
A5_IS_COUNTERFACTUAL = True
DEFAULT_TR_SHORT_MS = 600.0


def acquisition(man):
    """Pull the acquisition parameters out of one manifest, or return None with a reason.

    Schema is `audit/build_manifest_site.py`'s: per-contrast TR_ms/TE_ms under `contrasts`,
    b-values as a list, `S0` present only when a low-b band was registered.
    """
    con = man.get("contrasts") or {}
    t2w, dwi = con.get("T2W"), con.get("DWI")
    if not t2w or not dwi:
        return None, "missing T2W or DWI contrast"
    try:
        acq = {"TR_T2W": float(t2w["TR_ms"]), "TE_T2W": float(t2w["TE_ms"]),
               "TR_DWI": float(dwi["TR_ms"]), "TE_DWI": float(dwi["TE_ms"])}
    except (KeyError, TypeError, ValueError):
        return None, "non-numeric TR/TE"
    # The HIGHEST b, not b_values[0]. audit/build_manifest_site.py writes a single-element
    # list so [0] happened to be right for the published cohort, but physharm's own
    # build_manifest writes the full sorted-ASCENDING set -- where [0] is b=0, the D column
    # of the DWI Jacobian collapses to -0.0, and the forward model silently drops an
    # exp(-b*D) factor that PD then absorbs. Under A4 (no anchor) D goes fully degenerate.
    bvals = dwi.get("b_values") or [man.get("b_hi")]
    bvals = [b for b in bvals if b is not None]
    b_hi = max(bvals) if bvals else None
    if b_hi is None:
        return None, "no b-value"
    acq["b_hi"] = float(b_hi)
    s0 = con.get("S0")
    if s0 and (s0.get("b_values") or man.get("b_lo") is not None):
        acq["b_lo"] = float((s0.get("b_values") or [man["b_lo"]])[0])
    acq["TR_SHORT"] = DEFAULT_TR_SHORT_MS
    if not all(np.isfinite(v) and v > 0 for k, v in acq.items() if k.startswith("T")):
        return None, "non-positive TR/TE"
    return acq, None


#: The onboarding's known trap: series labelled T2W with TR < 2000 are a DIFFERENT
#: SEQUENCE. `audit/crlb_spread.py` gates on this and drops 176 of 4,849 cases; their
#: CRLBs are extreme outliers, which inflate WITHIN-scanner variance and push ICC down.
#: The full exclusion list is issue #3; this flag exists so the gate's effect on the
#: headline number can be measured now rather than assumed.
T2W_TR_RANGE_MS = (2000.0, 12000.0)


def rows_for_case(man, weights=None, prior_sd=10.0, t2w_tr_range=None):
    """CRLB + spectrum for one case under every arm it supports."""
    acq, why = acquisition(man)
    if acq is None:
        return None, why
    if t2w_tr_range is not None:
        lo, hi = t2w_tr_range
        if not (lo <= acq["TR_T2W"] <= hi):
            return None, f"T2W TR outside [{lo:g}, {hi:g}] (different sequence)"
    out = {"case_id": man.get("case_id"), "patient_id": man.get("patient_id"),
           "scanner_model": canonical_scanner(man.get("scanner_model")),
           "scanner_model_raw": man.get("scanner_model"), "vendor": man.get("vendor"),
           "scanner_unit": man.get("scanner_unit"), "acq": acq, "arms": {}}
    for arm in ARMS:
        if arm == "A2" and "b_lo" not in acq:
            continue                      # no low-b band registered for this case
        w = weights
        if w is not None:
            n = len(F.measurement_rows(arm, acq))
            w = list(w)[:n] + [1.0] * max(0, n - len(w))
        out["arms"][arm] = F.identifiability(acq, arm, weights=w, prior_sd=prior_sd)
    return out, None


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


#: Operating point for the predictor. Chosen, not swept -- see `predictor_stability`.
PRIOR_SD = 10.0


def predictor_stability(cases, n_sample=300, seed=0, prior_sd=PRIOR_SD,
                        weight_grid=((0.04, 0.1), (0.2, 1.0), (1.0, 1.0), (5.0, 1.0), (25.0, 10.0)),
                        prior_curve=(1.0, 3.0, 10.0, 30.0, 100.0)):
    """Is the predictor stable under the thing we could not measure?

    Two different questions, and conflating them nearly cost this study its sign:

    (1) WITHIN-CASE ordering of the four parameters. Cosmetic -- it is how the result is
        described, not what is regressed.
    (2) ACROSS-CASE ranking within one parameter. **This is the regression predictor.**

    Sweeping prior_sd from 1 to 100 leaves the within-case ordering at `D < T2 < PD < T1`
    for a typical protocol while INVERTING the across-case ranking (PD Spearman -0.94).
    Checking only (1) would have certified a predictor whose regression sign flips.

    WHY prior_sd IS FIXED, NOT SWEPT. The usable prior is a BAND, not a floor:

      * too strong (small prior_sd, or ordinary prior_sd with heavily down-weighted
        contrasts) -- the bound saturates, every case returns ~prior_sd, and the residual
        ranking is noise on a flat function. Measured worst case: PD Spearman **-0.936** at
        prior_sd = 3 with w_DWI = 0.04, w_ADC = 0.1.
      * too weak (prior_sd = 100) -- the near-null direction is barely regularized and the
        ranking is driven by differences between tiny eigenvalues. T2 Spearman falls to 0.724.

    So prior_sd is an ANALYSIS CHOICE fixed at :data:`PRIOR_SD`, justified by
    :func:`physharm.physics.fisher.prior_dominance` reporting nothing prior-dominated there,
    and the sensitivity is REPORTED as a curve rather than hidden inside a sweep the
    predictor was never stable across.

    What IS swept is the relative per-contrast noise weighting -- the quantity we genuinely
    could not measure (DWI SNR, the run log Phase 2c). At the operating point the across-case
    ranking survives it at Spearman >= 0.9.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cases), size=min(n_sample, len(cases)), replace=False)
    acqs = [a for a in (acquisition(cases[i])[0] for i in idx) if a is not None]
    if len(acqs) < 10:
        return {"n": len(acqs)}

    def crlbs(w_dwi, w_adc, ps):
        return [F.identifiability(a, "A1", weights=[1.0, w_dwi, w_adc], prior_sd=ps)["crlb"]
                for a in acqs]

    ref = crlbs(1.0, 1.0, prior_sd)
    orderings = Counter()
    worst = {p: 1.0 for p in F.PARAMS}
    # Seeded with the reference setting: a perfectly stable parameter (Spearman 1.0 for
    # every weighting -- D is one) never enters the update branch below, and a None here
    # crashes the reporter. "Worst = the reference" is the correct reading in that case.
    worst_at = {p: (1.0, 1.0) for p in F.PARAMS}
    for w_dwi, w_adc in weight_grid:
        cur = crlbs(w_dwi, w_adc, prior_sd)
        for d in cur:
            orderings[tuple(sorted(d, key=d.get))] += 1
        for p in F.PARAMS:
            r = spearman([d[p] for d in ref], [d[p] for d in cur])
            if r < worst[p]:
                worst[p], worst_at[p] = r, (w_dwi, w_adc)

    curve = {}
    for ps in prior_curve:
        cur = crlbs(1.0, 1.0, ps)
        curve[ps] = {p: spearman([d[p] for d in ref], [d[p] for d in cur]) for p in F.PARAMS}

    dom = [F.prior_dominance(F.fisher_matrix(F.measurement_rows("A1", a), [1.0, 1.0, 1.0]),
                             prior_sd)["fraction_prior_dominated"] for a in acqs]
    return {"n": len(acqs), "prior_sd": prior_sd, "orderings": orderings,
            "worst_spearman": worst, "worst_at": worst_at, "prior_curve": curve,
            "median_fraction_prior_dominated": float(np.median(dom))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--weights", type=float, nargs="*", default=None,
                    help="relative 1/sigma^2 per measurement row, in row order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", action="store_true",
                    help="apply the full exclusion list (physharm.data.exclusions) and print "
                         "the per-scanner gate table. Off by default so the ungated number "
                         "stays reproducible. NOTE: this supersedes the old --t2w-tr-range "
                         "flag; `--exclude` reproduces `--t2w-tr-range 2000 12000` plus the "
                         "b_hi and missing-contrast gates.")
    ap.add_argument("--physics-sanity", default=None, metavar="JSON",
                    help="runs/physics_sanity.json, for the mislabelled-b-band case IDs")
    args = ap.parse_args()

    mans = read_manifests(args.manifest_dir)
    if args.limit:
        mans = mans[:args.limit]
    print(f"{len(mans)} manifests read from {args.manifest_dir}")

    if args.exclude:
        bad = EX.load_bad_bands(args.physics_sanity)
        if args.physics_sanity and not bad:
            print(f"  WARNING: no bad_bands read from {args.physics_sanity}")
        mans, ex_report = EX.apply(mans, bad_bands=bad)
        print("\n=== EXCLUSIONS (ground rule 6: report every gate per scanner) ===")
        print(EX.format_report(ex_report))
        print()
    else:
        ex_report = None

    rows, why = [], Counter()
    for m in mans:
        r, w = rows_for_case(m, args.weights, args.prior_sd)
        (rows.append(r) if r else why.update([w]))
    print(f"{len(rows)} cases with a usable CRLB"
          + ("  (exclusion list applied)" if args.exclude else "  (UNGATED — see --exclude)"))
    for k, v in why.most_common():
        print(f"  skipped {v:5d}  {k}")
    if not rows:
        raise SystemExit("no usable cases")

    variants = defaultdict(set)
    for r in rows:
        variants[r["scanner_model"]].add(r["scanner_model_raw"])
    collapsed = {k: sorted(v) for k, v in variants.items() if len(v) > 1}
    print(f"  {len(variants)} scanner models after canonicalization (ground rule 5)")
    if collapsed:
        print("  collapsed spelling variants — grouping on the raw string would have split")
        print("  these and biased ICC downward:")
        for k, v in sorted(collapsed.items()):
            print(f"    {k}  <-  {v}")
    else:
        print("  no spelling variants found; canonicalization was a no-op here")

    n_a2 = sum(1 for r in rows if "A2" in r["arms"])
    print(f"  A2 (low-b band) available for {n_a2}/{len(rows)} ({n_a2/len(rows):.1%})")
    if A5_IS_COUNTERFACTUAL:
        print(f"  A5 is COUNTERFACTUAL (TR={DEFAULT_TR_SHORT_MS:g} ms): no short-TR series is")
        print("     registered, so it quantifies what would be needed, and must not be")
        print("     regressed against real leakage.")

    print("\n=== CRLB by arm (cohort p10/50/90) ===")
    print(f"{'arm':>4s} {'n':>6s}  " + "  ".join(f"{p:>22s}" for p in F.PARAMS))
    for arm in ARMS:
        sel = [r for r in rows if arm in r["arms"]]
        if not sel:
            continue
        cells = []
        for p in F.PARAMS:
            q = pcts([r["arms"][arm]["crlb"][p] for r in sel])
            cells.append(f"{q.get(10,float('nan')):6.2f}/{q.get(50,float('nan')):6.2f}/{q.get(90,float('nan')):6.2f}")
        print(f"{arm:>4s} {len(sel):>6d}  " + "  ".join(f"{c:>22s}" for c in cells))

    print("\n=== ICC(CRLB | scanner model) — is the predictor separable from the scanner? ===")
    print(f"{'arm':>4s}  " + "  ".join(f"{p:>8s}" for p in F.PARAMS) + "   reading (on T2)")
    iccs = {}
    for arm in ARMS:
        sel = [r for r in rows if arm in r["arms"]]
        if len(sel) < 20:
            continue
        vals = {p: icc([r["arms"][arm]["crlb"][p] for r in sel],
                       [r["scanner_model"] for r in sel]) for p in F.PARAMS}
        iccs[arm] = vals
        t2 = vals["T2"]
        tag = ("n/a" if t2 is None else
               "CONFOUNDED" if t2 > 0.9 else
               "testable with scanner as covariate" if t2 > 0.3 else
               "STRONG — match across scanners at equal CRLB")
        print(f"{arm:>4s}  " + "  ".join(f"{(vals[p] if vals[p] is not None else float('nan')):>8.3f}"
                                         for p in F.PARAMS) + f"   {tag}")

    print("\n=== predictor stability, cohort-wide ===")
    st = predictor_stability(mans)
    total = sum(st["orderings"].values())
    print(f"  {st['n']} sampled cases; median fraction of parameters that are")
    print(f"  prior-dominated at prior_sd=10: {st['median_fraction_prior_dominated']:.2f}"
          "   (high = the bound has saturated and the predictor is not defined)")
    print("\n  (1) within-case parameter ordering — cosmetic, not what is regressed:")
    for o, n in st["orderings"].most_common():
        print(f"      {n:6d} ({n/total:5.1%})  " + " < ".join(o))
    print("\n  (2) ACROSS-case ranking per parameter — this IS the regression predictor.")
    print(f"      Worst Spearman over the relative-noise sweep, at prior_sd={st['prior_sd']:g}:")
    for p in F.PARAMS:
        at = st["worst_at"][p]
        flag = "  <-- UNSTABLE" if st["worst_spearman"][p] < 0.8 else ""
        print(f"      {p:>4s}  {st['worst_spearman'][p]:>7.3f}   at w_DWI={at[0]:g}"
              f" w_ADC={at[1]:g}{flag}")
    if min(st["worst_spearman"].values()) >= 0.8:
        print("\n      -> the unmeasurable DWI SNR does not threaten the regression.")
    else:
        print("\n      -> NOT SAFE. Do not regress leakage against this predictor yet.")

    print("\n  (3) prior sensitivity — REPORTED, not swept over. The usable prior is a BAND:")
    print(f"      {'prior_sd':>9s}  " + "  ".join(f"{p:>7s}" for p in F.PARAMS))
    for ps, r in sorted(st["prior_curve"].items()):
        mark = "  <- operating point" if ps == st["prior_sd"] else (
            "  saturated" if min(r.values()) < 0 else "")
        print(f"      {ps:>9g}  " + "  ".join(f"{r[p]:>7.3f}" for p in F.PARAMS) + mark)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"n_cases": len(rows), "prior_sd": args.prior_sd,
                   "excluded": bool(args.exclude),
                   "exclusion_report": (None if ex_report is None else
                                        {k: v for k, v in ex_report.items()
                                         if k != "excluded"}),
                   "weights": args.weights, "a5_counterfactual": A5_IS_COUNTERFACTUAL,
                   "tr_short_ms": DEFAULT_TR_SHORT_MS,
                   "icc": {a: {p: v[p] for p in F.PARAMS} for a, v in iccs.items()},
                   "ordering_counts": {" < ".join(k): v for k, v in st["orderings"].items()},
                   "worst_spearman": st["worst_spearman"],
                   "prior_sd": st["prior_sd"],
                   "prior_curve": {str(k): v for k, v in st["prior_curve"].items()},
                   "median_fraction_prior_dominated": st["median_fraction_prior_dominated"],
                   "skipped": dict(why), "cases": rows}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            cols = (["case_id", "patient_id", "scanner_model", "vendor", "scanner_unit",
                     "scanner_model_raw", "TR_T2W", "TE_T2W", "TR_DWI", "TE_DWI",
                     "b_hi", "b_lo"] +
                    [f"{a}_crlb_{p}" for a in ARMS for p in F.PARAMS] +
                    [f"{a}_cond" for a in ARMS] + [f"{a}_rank" for a in ARMS])
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = {k: r.get(k) for k in ("case_id", "patient_id", "scanner_model",
                                             "scanner_model_raw", "vendor", "scanner_unit")}
                row.update({k: r["acq"].get(k) for k in
                            ("TR_T2W", "TE_T2W", "TR_DWI", "TE_DWI", "b_hi", "b_lo")})
                for a, d in r["arms"].items():
                    row[f"{a}_cond"], row[f"{a}_rank"] = d["condition_number"], d["rank"]
                    for p in F.PARAMS:
                        row[f"{a}_crlb_{p}"] = d["crlb"][p]
                w.writerow(row)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
