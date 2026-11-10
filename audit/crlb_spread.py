r"""Is Candidate A a study, or a confound? — manifest-only feasibility audit.

Candidate A claims: the Fisher information of the MR signal equations, evaluated at a
case's ACTUAL acquisition parameters, predicts which estimated tissue-parameter maps
stay scanner-separable.

That claim is only testable if predicted identifiability can be **decoupled from scanner
identity**. If every GE case has the same CRLB and every Siemens case has a different
same CRLB, then "CRLB predicts leakage" and "vendor predicts leakage" are the same
sentence, and no experiment can tell them apart. The study would be unidentifiable --
which would be an unfortunate way to fail a paper about identifiability.

WHICH DECISION THIS CHANGES (standing rule: say it before running anything):

  ICC(CRLB by scanner) high (>~0.9)  -> protocol is collinear with scanner at the site.
                                        Candidate A cannot be tested observationally.
                                        Fall back to the synthetic-protocol arm as the
                                        PRIMARY result and drop the real-data regression.
  ICC moderate (~0.3-0.9)            -> testable with scanner as a covariate. Design the
                                        real arm as within-scanner regression of leakage
                                        on CRLB. This is the GO case.
  ICC low (<~0.3)                    -> protocol varies mostly within scanner. Strongest
                                        case: match cases ACROSS scanners on CRLB and
                                        compare leakage at matched identifiability.

Reads manifests only -- no images, no GPU, no labels. Runs on the login node.
Manifest schema follows Project 7 `xscan/cohort.py`: one JSON per case with per-contrast
TE_ms / TR_ms and a b-value, plus a scanner/vendor field.

Usage
-----
    python crlb_spread.py --manifest-dir <DIR> --out runs/crlb_audit.json
    python crlb_spread.py --manifest-dir <DIR> --dump-schema 3   # if fields aren't found

Nothing here is hardcoded silently: every physical constant is a flag with a stated
source, and every manifest field name is discovered and reported.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Physical assumptions. Each is a flag, each has a source. None is a bare literal.
# ---------------------------------------------------------------------------
# Prostate tissue values at 3T used as the LINEARIZATION POINT for the Fisher matrix.
# The FIM is local, so these set where the sensitivities are evaluated. They are not
# fitted quantities and the audit reports how much the conclusion moves under them.
#   T1 ~1400 ms, T2 ~80 ms, D ~1.3e-3 mm^2/s for peripheral-zone prostate at 3T.
#   Sources to attach before this leaves the lab: de Bazelaire 2004 (Radiology) for
#   3T abdominal/pelvic T1/T2; the ADC value is the median measured in our own cohort
#   (onboarding 3.2 records GE 1.30 / Siemens 1.31 x1e-3), so it is a MEASUREMENT.
DEFAULTS = dict(T1_ms=1400.0, T2_ms=80.0, D_mm2s=1.3e-3, PD=1.0)

CONTRAST_ALIASES = {
    "T2W": ("T2W", "T2w", "t2w", "T2", "AX T2", "T2_TSE"),
    "DWI": ("DWI", "dwi", "DWI_high", "DW", "DIFF"),
    "ADC": ("ADC", "adc", "ADC_map"),
    "S0":  ("S0", "DWI_low", "b0", "B0"),
}
TR_KEYS = ("TR_ms", "TR", "RepetitionTime", "repetition_time_ms", "tr")
TE_KEYS = ("TE_ms", "TE", "EchoTime", "echo_time_ms", "te")
B_KEYS = ("b_values", "b_value", "b", "b_hi", "bvalue", "diffusion_bvalue")
SCANNER_KEYS = ("scanner_model", "ManufacturerModelName", "manufacturer_model",
                "model", "station", "StationName", "vendor", "Manufacturer")
PATIENT_KEYS = ("patient_id", "PatientID", "encrypted_patient", "subject_id")


def canonical_domain(s: str) -> str:
    """Collapse the spelling variants recorded in onboarding 4 / xscan.cohort.

    'SIEMENS' vs 'Siemens Healthineers'; 'SITECBMR6' vs 'CBMR6' vs 'SITE-CBMR6'.
    Uncanonicalized, scanner identity leaks through spelling alone.
    """
    s = str(s).strip().upper().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(SIEMENS HEALTHINEERS|SIEMENS|GE MEDICAL SYSTEMS|GE)\s+", "", s)
    return s.strip()


def _walk(obj, prefix=""):
    """Yield (dotted_path, value) for every leaf in a nested manifest."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        yield prefix, obj
    else:
        yield prefix, obj


def _first_numeric(man, contrast, key_names):
    """Find a numeric field for `contrast` by searching aliases then bare key names.

    Returns (value, path_it_came_from) so the audit can report exactly which manifest
    field was used -- guessing silently is how a lookup table gets mistaken for data.
    """
    leaves = list(_walk(man))
    aliases = CONTRAST_ALIASES.get(contrast, (contrast,))
    for want_contrast in (True, False):
        for path, val in leaves:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            parts = path.split(".")
            if parts[-1] not in key_names:
                continue
            in_contrast = any(a in path for a in aliases)
            if in_contrast == want_contrast:
                return float(val), path
    return None, None


def _bvalues(man, contrast="DWI"):
    """All b-values available for the DWI series, as a sorted list."""
    out, paths = set(), []
    for path, val in _walk(man):
        parts = path.split(".")
        if parts[-1] not in B_KEYS:
            continue
        if isinstance(val, list):
            for v in val:
                if isinstance(v, (int, float)):
                    out.add(float(v))
            paths.append(path)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            out.add(float(val))
            paths.append(path)
        # `bands` dicts key on the b-value itself: {"100": "...", "1500": "..."}
    for path, val in _walk(man):
        if path.split(".")[-1] == "bands" and isinstance(val, dict):
            for k in val:
                try:
                    out.add(float(k))
                except (TypeError, ValueError):
                    pass
            paths.append(path)
    return sorted(out), paths


# ---------------------------------------------------------------------------
# Fisher information for the bi-parametric clinical protocol
# ---------------------------------------------------------------------------
def log_sensitivities(TR, TE, b, theta):
    r"""Dimensionless log-sensitivity row d log S / d log theta for one measurement.

    Signal (Project 7 methodology, eqs. 4 and 6; spin-echo with Stejskal-Tanner):

        S = g * PD * (1 - exp(-TR/T1)) * exp(-TE/T2) * exp(-b*D)

    Derivatives, with E1 = exp(-TR/T1):

        dlogS/dlogPD =  1
        dlogS/dlogT1 = -(TR/T1) * E1 / (1 - E1)      <- carries the E1 factor that
                                                        vanishes at clinical TR
        dlogS/dlogT2 =  TE / T2
        dlogS/dlogD  = -b * D

    Order of theta is (PD, T1, T2, D). b in s/mm^2, D in mm^2/s so b*D is dimensionless;
    TR/TE/T1/T2 all in ms.
    """
    T1, T2, D = theta["T1_ms"], theta["T2_ms"], theta["D_mm2s"]
    E1 = math.exp(-TR / T1)
    dT1 = -(TR / T1) * E1 / (1.0 - E1) if (1.0 - E1) > 1e-12 else 0.0
    return np.array([1.0, dT1, TE / T2, -b * D], dtype=float)


def crlb(rows, weights, prior_sd=10.0, theta_names=("PD", "T1", "T2", "D")):
    r"""Per-parameter uncertainty from stacked log-sensitivity rows.

        F = J^T W J        (Sigma = diag(1/w); measurements are in log space)

    THE TRAP THIS FUNCTION EXISTS TO AVOID. For clinical bpMRI there are fewer
    independent measurements than parameters, so F is singular -- for the bi-parametric
    protocol it has a genuinely zero eigenvalue whose eigenvector is T1-dominated. The
    true Cramer-Rao bound along that direction is +infinity: T1 is unrecoverable.

    A pseudo-inverse does NOT report that. `pinv` sets the null-space contribution to
    ZERO, which amounts to asserting the unidentifiable direction is known exactly, and
    returns a *small* variance for the very parameter that is least determined. Verified
    numerically: with pinv, T1 comes out at 2.59 against PD 5.07 and T2 4.64 -- an
    ordering that inverts the physics. Any regression built on that number would be
    fitting an artifact of the inverse, not a property of the protocol.

    So the bound is regularized explicitly instead, as a Bayesian CRLB:

        Cov >= (F + Sigma_prior^-1)^-1,     Sigma_prior^-1 = I / prior_sd^2

    with `prior_sd` in log units (prior_sd=10 -> a deliberately weak prior; a sloppy
    direction returns ~prior_sd and is thereby visibly prior-dominated rather than
    silently well-determined). This is an ASSUMPTION, it is a flag, and it belongs in
    the sensitivity analysis -- the ORDERING of parameters must be checked to be stable
    across prior_sd, since only the ordering is used downstream.

    Also returns `null_overlap`: the squared projection of each parameter's unit vector
    onto the numerically-null eigenvectors of F. This is prior-free -- overlap ~1 means
    the parameter lies in the unidentifiable subspace no matter what prior is chosen.

    DO NOT USE `null_overlap` AS THE REGRESSION PREDICTOR. It is a thresholded quantity
    and therefore discontinuous: a case with three measurements has an exactly-zero
    eigenvalue and returns overlap(T1)~0.87, while the same case with a low-b band added
    is full-rank and returns exactly 0.0 -- a jump driven by rank, not by physics. Since
    b-band availability is vendor-linked (GE 100/1500, FOCUS 0/1500, MUSE 0/1400 --
    Project 7 docs/the multi-b notes), regressing on it would manufacture precisely the
    scanner confound this study has to avoid. Use the regularized `crlb`, which is
    continuous in the acquisition parameters, and keep `null_overlap` as a diagnostic
    for reporting which cases are exactly singular.
    """
    J = np.asarray(rows, dtype=float)
    W = np.diag(np.asarray(weights, dtype=float))
    F = J.T @ W @ J
    evals, evecs = np.linalg.eigh(F)
    evals = np.clip(evals, 0.0, None)

    n = F.shape[0]
    F_reg = F + np.eye(n) / (prior_sd ** 2)
    sd = np.sqrt(np.clip(np.diag(np.linalg.inv(F_reg)), 0.0, None))

    # Null directions: eigenvalues negligible against the stiffest one.
    tol = max(evals.max(), 1e-30) * 1e-8
    null = evecs[:, evals <= tol]
    overlap = (null ** 2).sum(axis=1) if null.size else np.zeros(n)

    nz = evals[evals > tol]
    cond = float(nz.max() / nz.min()) if nz.size >= 2 else float("inf")
    return (dict(zip(theta_names, sd.tolist())), evals.tolist(), cond,
            dict(zip(theta_names, overlap.tolist())))


def icc(values, groups):
    r"""One-way ICC: fraction of variance that is BETWEEN groups.

    ICC = var_between / (var_between + var_within), on group means weighted by size.
    1.0 -> the quantity is a function of the group label (here: protocol is collinear
    with scanner, and Candidate A is not observationally testable).
    0.0 -> the quantity varies entirely within groups (the strongest case for A).
    """
    by = defaultdict(list)
    for v, g in zip(values, groups):
        if v is not None and np.isfinite(v):
            by[g].append(float(v))
    by = {g: np.array(v) for g, v in by.items() if len(v) >= 2}
    if len(by) < 2:
        return None, by
    grand = np.concatenate(list(by.values())).mean()
    n_tot = sum(len(v) for v in by.values())
    k = len(by)
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for v in by.values())
    ss_w = sum(((v - v.mean()) ** 2).sum() for v in by.values())
    df_b, df_w = k - 1, n_tot - k
    if df_b <= 0 or df_w <= 0:
        return None, by
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    n0 = (n_tot - sum(len(v) ** 2 for v in by.values()) / n_tot) / df_b
    var_b = max(0.0, (ms_b - ms_w) / n0)
    denom = var_b + ms_w
    return (var_b / denom if denom > 0 else None), by


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True,
                    help="directory of per-case manifest JSONs (xscan.cohort layout)")
    ap.add_argument("--out", default=None, help="write the audit JSON here")
    ap.add_argument("--dump-schema", type=int, default=0, metavar="N",
                    help="print the flattened key paths of N manifests and exit; "
                         "use this first if field discovery reports misses")
    ap.add_argument("--t2w-tr-min", type=float, default=2000.0,
                    help="onboarding 4: ~49%% of GE series labelled T2W have TR<2000 and "
                         "are a different sequence. Gate, and report the rate per scanner.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v,
                        help=f"linearization point for the Fisher matrix (default {v})")
    ap.add_argument("--w-t2w", type=float, default=1.0, help="relative weight (1/var) T2W")
    ap.add_argument("--w-dwi", type=float, default=1.0, help="relative weight (1/var) DWI")
    ap.add_argument("--prior-sd", type=float, default=10.0,
                    help="log-space prior SD regularizing the singular Fisher matrix; "
                         "an ASSUMPTION -- check the parameter ORDERING is stable in it")
    ap.add_argument("--w-adc", type=float, default=1.0,
                    help="relative weight of the ADC clamp on D; ADC is a fitted map, so "
                         "this is an ASSUMPTION -- carry it into the sensitivity analysis")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.manifest_dir, "*.json"))
                   + glob.glob(os.path.join(args.manifest_dir, "*", "*.json")))
    files = [f for f in files if os.path.basename(f) != "index.json"]
    if not files:
        raise SystemExit(f"no manifests under {args.manifest_dir}")
    print(f"found {len(files)} manifest files")

    if args.dump_schema:
        for f in files[:args.dump_schema]:
            print(f"\n--- {f}")
            with open(f) as fh:
                for path, val in _walk(json.load(fh)):
                    shown = val if not isinstance(val, list) else f"[list len {len(val)}]"
                    print(f"  {path} = {shown}")
        return

    theta = {k: getattr(args, k) for k in DEFAULTS}
    rows, field_paths, misses = [], defaultdict(set), defaultdict(int)

    for f in files:
        try:
            with open(f) as fh:
                man = json.load(fh)
        except (json.JSONDecodeError, OSError):
            misses["unreadable manifest"] += 1
            continue

        case = man.get("case_id") or os.path.splitext(os.path.basename(f))[0]
        patient = next((str(man[k]) for k in PATIENT_KEYS if man.get(k)), case)
        scanner = next((canonical_domain(man[k]) for k in SCANNER_KEYS if man.get(k)), None)
        if not scanner:
            misses["no scanner field"] += 1
            continue

        tr_t2, p1 = _first_numeric(man, "T2W", TR_KEYS)
        te_t2, p2 = _first_numeric(man, "T2W", TE_KEYS)
        tr_dw, p3 = _first_numeric(man, "DWI", TR_KEYS)
        te_dw, p4 = _first_numeric(man, "DWI", TE_KEYS)
        bvals, p5 = _bvalues(man)
        for p in (p1, p2, p3, p4):
            if p:
                field_paths[p.split(".")[-1]].add(p)
        for p in p5:
            field_paths["b"].add(p)

        if None in (tr_t2, te_t2, tr_dw, te_dw) or not bvals:
            misses["incomplete acquisition parameters"] += 1
            continue
        if not (args.t2w_tr_min <= tr_t2 <= 12000.0):
            misses[f"T2W TR outside [{args.t2w_tr_min:.0f}, 12000]"] += 1
            misses[f"__gate__{scanner}"] += 1
            continue

        b_hi, b_lo = max(bvals), min(bvals)
        # Measurement set: T2W, high-b DWI, and the ADC clamp on D. A low-b band, when
        # present, is a diffusion-free second echo at the DWI TE -- the clean TE-ratio
        # anchor for T2 (Project 7 docs/the multi-b notes), so it enters as its own row.
        J = [log_sensitivities(tr_t2, te_t2, 0.0, theta),
             log_sensitivities(tr_dw, te_dw, b_hi, theta),
             np.array([0.0, 0.0, 0.0, 1.0])]           # ADC measures D directly
        w = [args.w_t2w, args.w_dwi, args.w_adc]
        multib = len(bvals) >= 2 and b_lo < b_hi
        if multib:
            J.insert(2, log_sensitivities(tr_dw, te_dw, b_lo, theta))
            w.insert(2, args.w_dwi)

        sd, evals, cond, overlap = crlb(J, w, prior_sd=args.prior_sd)
        rows.append(dict(case_id=case, patient_id=patient, scanner=scanner,
                         TR_T2W=tr_t2, TE_T2W=te_t2, TR_DWI=tr_dw, TE_DWI=te_dw,
                         b_values=bvals, b_hi=b_hi, b_lo=b_lo, multib=multib,
                         crlb=sd, null_overlap=overlap,
                         eigenvalues=evals, condition_number=cond))

    if not rows:
        raise SystemExit("no usable cases -- rerun with --dump-schema 3 and check field names")

    print("\n=== FIELDS DISCOVERED (verify these are real DICOM reads, not defaults) ===")
    for k, paths in sorted(field_paths.items()):
        for p in sorted(paths):
            print(f"  {k:4s} <- {p}")
    print("\n=== CASES DROPPED ===")
    for k, v in sorted(misses.items()):
        if not k.startswith("__gate__"):
            print(f"  {v:6d}  {k}")

    scanners = sorted({r["scanner"] for r in rows})
    n_pat = len({r["patient_id"] for r in rows})
    print(f"\nusable cases {len(rows)} | patients {n_pat} | scanners {len(scanners)}")
    if n_pat < len(rows):
        print(f"  NOTE: {len(rows) - n_pat} repeat exams -> splits must be patient-level")

    print("\n=== 1. PROTOCOL SPREAD PER SCANNER (p10/50/90) ===")
    print(f"{'scanner':26s} {'n':>5s} {'multi-b':>8s} {'TR_T2W':>22s} {'TE_T2W':>22s} {'b_hi':>16s}")
    per = defaultdict(list)
    for r in rows:
        per[r["scanner"]].append(r)
    for s in scanners:
        rs = per[s]
        def pct(key):
            v = np.array([x[key] for x in rs], dtype=float)
            return f"{np.percentile(v,10):6.0f}/{np.percentile(v,50):6.0f}/{np.percentile(v,90):6.0f}"
        mb = sum(x["multib"] for x in rs)
        print(f"{s[:26]:26s} {len(rs):>5d} {mb:>8d} {pct('TR_T2W'):>22s} "
              f"{pct('TE_T2W'):>22s} {pct('b_hi'):>16s}")

    print("\n=== 2. THE DECISIVE NUMBER: is predicted identifiability separable "
          "from scanner identity? ===")
    print("ICC = fraction of variance that is BETWEEN scanners. "
          "High -> confounded, low -> testable.\n")
    print(f"{'quantity':22s} {'ICC':>8s}  {'reading':s}")
    verdicts = {}
    for name, get in [("TR_T2W", lambda r: r["TR_T2W"]),
                      ("TE_T2W", lambda r: r["TE_T2W"]),
                      ("b_hi", lambda r: r["b_hi"]),
                      ("CRLB(T2)", lambda r: r["crlb"]["T2"]),
                      ("CRLB(PD)", lambda r: r["crlb"]["PD"]),
                      ("CRLB(T1)", lambda r: r["crlb"]["T1"]),
                      ("nullOverlap(T1)", lambda r: r["null_overlap"]["T1"]),
                      ("nullOverlap(T2)", lambda r: r["null_overlap"]["T2"]),
                      ("log10 cond", lambda r: math.log10(r["condition_number"])
                       if np.isfinite(r["condition_number"]) else None)]:
        val, _ = icc([get(r) for r in rows], [r["scanner"] for r in rows])
        verdicts[name] = val
        if val is None:
            print(f"{name:22s} {'n/a':>8s}  too few groups or no within-group variance")
        else:
            tag = ("CONFOUNDED - protocol is a scanner label" if val > 0.9 else
                   "TESTABLE with scanner as covariate" if val > 0.3 else
                   "STRONG - varies mostly within scanner")
            print(f"{name:22s} {val:8.3f}  {tag}")

    key = verdicts.get("CRLB(T2)")
    print("\n=== VERDICT FOR CANDIDATE A ===")
    if key is None:
        print("  INCONCLUSIVE - CRLB(T2) had no within-scanner variance to decompose.")
    elif key > 0.9:
        print(f"  ICC(CRLB T2) = {key:.3f} > 0.9. Protocol is collinear with scanner.")
        print("  -> A is NOT observationally testable on this cohort. Make the synthetic")
        print("     protocol sweep the primary result; drop the real-data regression.")
    elif key > 0.3:
        print(f"  ICC(CRLB T2) = {key:.3f}. GO, with scanner as a covariate.")
        print("  -> Design the real arm as a within-scanner regression of leakage on CRLB.")
    else:
        print(f"  ICC(CRLB T2) = {key:.3f} < 0.3. Protocol varies mostly WITHIN scanner.")
        print("  -> Strongest design available: match cases across scanners on CRLB and")
        print("     compare leakage at matched identifiability.")

    n_mb = sum(r["multib"] for r in rows)
    print(f"\n  multi-b cases: {n_mb}/{len(rows)} ({100*n_mb/len(rows):.1f}%) -- these carry a")
    print("  diffusion-free second echo (the TE-ratio anchor for T2), so the contrast-subset")
    print("  ablation is only available on this subset.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(theta=theta, prior_sd=args.prior_sd,
                           weights=dict(T2W=args.w_t2w, DWI=args.w_dwi, ADC=args.w_adc),
                           t2w_tr_min=args.t2w_tr_min,
                           n_cases=len(rows), n_patients=n_pat, scanners=scanners,
                           icc=verdicts, dropped=dict(misses), cases=rows), fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
