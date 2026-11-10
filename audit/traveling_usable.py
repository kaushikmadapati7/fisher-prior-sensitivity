r"""How many of the 607 traveling subjects can we actually USE, and are the intervals real?

`traveling_subjects.py` counted repeat structure over all 6,044 case dirs. Two things it
did not answer, both decisive, both computable from the cached headers with no tar reads.

(1) INTERVALS. Absolute dates are corrupt (year range 1001..9975) and the year filter
    rejected every gap, so we currently have no interval information at all. But
    de-identification date-shifting is conventionally a CONSTANT OFFSET PER PATIENT --
    which destroys absolute dates while preserving intra-patient intervals exactly. If
    that holds here, the gaps are recoverable and the earlier filter was throwing away
    the one thing still valid. Test: are a patient's exam dates all in the same era?

(2) USABILITY. The traveling-subject count is over case DIRS. Only 4,849 have manifests
    and the Siemens ones were largely gated out (23 usable of ~1,021). A cross-vendor
    traveling subject whose Siemens exam has no manifest is not usable today. The number
    that matters is: patients with >=2 exams that BOTH have manifests.

WHICH DECISION THIS CHANGES:
  usable cross-model >= ~50   -> the traveling-subject arm is the paper's primary control.
  usable is small but recoverable-after-re-registration is large
                              -> run the multi-b re-registration; it is now on the
                                 critical path rather than being upside.
  both small                  -> keep the ADC-matched proxy primary, report these as a
                                 case study, and say plainly why.

    python traveling_usable.py --cache runs/header_cache.json \
        --manifest-dir manifests_site --reg-root <image-root> \
        --out runs/traveling_usable.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date


def _d(s):
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, IndexError, TypeError):
        return None


def _bands(cdir):
    try:
        names = os.listdir(cdir)
    except OSError:
        return []
    return sorted(int(m.group(1)) for m in
                  (re.search(r"DWI_b(\d+)_to_T2W\.nii\.gz$", n) for n in names) if m)


def _pcts(v, ps=(10, 25, 50, 75, 90)):
    v = sorted(v)
    return {p: v[min(len(v) - 1, int(p / 100 * len(v)))] for p in ps} if v else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="runs/header_cache.json")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--reg-root", default=None,
                    help="registered root; enables the 'recoverable after re-registration' count")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-patients", default=None, metavar="ANATOMY_JSON",
                    help="restrict to the anatomy-verified high-confidence patients in "
                         "runs/anatomy_repeats.json. Without this the counts include "
                         "PatientID collisions -- only ~38%% of putative repeats are "
                         "genuine, so the unrestricted numbers are inflated ~2-3x.")
    args = ap.parse_args()

    by_case = json.load(open(args.cache))
    have = {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(args.manifest_dir, "*.json"))
            if not os.path.basename(f).startswith("index")}
    print(f"{len(by_case)} cached exams | {len(have)} with manifests")

    per_patient = defaultdict(list)
    for c, r in by_case.items():
        if r.get("patient_id") and r.get("identified"):
            per_patient[r["patient_id"]].append(c)
    repeats = {p: cs for p, cs in per_patient.items() if len(cs) > 1}
    if args.only_patients:
        a = json.load(open(args.only_patients))
        keep = {d["patient"] for d in a.get("high_conf_pairs", [])}
        before = len(repeats)
        repeats = {p: cs for p, cs in repeats.items() if p in keep}
        print(f"  restricted to anatomy-verified patients: {before} -> {len(repeats)}"
              f"  (false-match rate {a.get('false_match_rate', float('nan')):.1%})")

    # ---- (1) is the date shift per-patient constant? -------------------------
    print("\n=== 1. ARE INTRA-PATIENT INTERVALS RECOVERABLE? ===")
    same_era = n_pairs = 0
    gaps, cross_era = [], 0
    for p, cs in repeats.items():
        ds = [_d(by_case[c].get("study_date", "")) for c in cs]
        ds = sorted(d for d in ds if d)
        if len(ds) < 2:
            continue
        n_pairs += 1
        if ds[-1].year - ds[0].year <= 20:
            same_era += 1
            for a, b in zip(ds, ds[1:]):
                gaps.append((b - a).days)
        else:
            cross_era += 1
    if n_pairs < 20:
        print(f"  ONLY {n_pairs} patients have >=2 PARSEABLE dates -- the test cannot run.")
        print("  This is not evidence either way about per-patient offsets; it means the")
        print("  dates themselves are destroyed (month/day fields invalid, not just the")
        print("  year shifted). Practical consequence: INTERVALS ARE UNAVAILABLE. Drop")
        print("  every interval-based claim and state it as a limitation.")
    elif n_pairs:
        frac = same_era / n_pairs
        print(f"  {same_era}/{n_pairs} patients ({frac:.1%}) have all exam dates within a")
        print("  20-year window of each other, despite absolute years being nonsense.")
        if frac > 0.9:
            print("  -> CONSISTENT with a CONSTANT PER-PATIENT date offset. Intra-patient")
            print("     intervals ARE meaningful; only absolute dates were destroyed. The")
            print("     year-range filter in traveling_subjects.py was discarding valid data.")
        elif frac > 0.5:
            print(f"  -> MIXED. {cross_era} patients have exams in different eras; their")
            print("     intervals are not trustworthy. Use the same-era subset only.")
        else:
            print("  -> NOT a per-patient offset. Intervals are unrecoverable; drop every")
            print("     interval-based claim and say so.")
    if gaps:
        q = _pcts([abs(g) for g in gaps])
        print(f"\n  intra-patient interval (days), same-era pairs, n={len(gaps)}:")
        print("    " + "  ".join(f"p{k}={v}" for k, v in q.items()))
        for lo, hi, lab in [(0, 60, "<= 2 months  (good traveling subject)"),
                            (60, 365, "2-12 months  (usable, note progression)"),
                            (365, 10**9, "> 1 year     (biology likely changed)")]:
            n = sum(1 for g in gaps if lo <= abs(g) < hi)
            print(f"    {lab:42s} {n:5d} ({n/len(gaps):5.1%})")

    # ---- (2) how many traveling subjects are actually usable? ----------------
    print("\n=== 2. USABLE TRAVELING SUBJECTS (>=2 exams WITH manifests) ===")
    rows = {"cross-vendor": [], "cross-model": [], "cross-unit": [], "same-unit": []}
    usable = {k: [] for k in rows}
    recoverable = {k: [] for k in rows}
    for p, cs in repeats.items():
        v = {by_case[c]["vendor"] for c in cs}
        m = {by_case[c]["model"] for c in cs}
        u = {by_case[c]["unit"] for c in cs}
        cls = ("cross-vendor" if len(v) > 1 else "cross-model" if len(m) > 1
               else "cross-unit" if len(u) > 1 else "same-unit")
        rows[cls].append(p)
        if sum(1 for c in cs if c in have) >= 2:
            usable[cls].append(p)
        elif args.reg_root:
            # would a low-b-only exam become usable after multi-b re-registration?
            n_ok = 0
            for c in cs:
                if c in have:
                    n_ok += 1
                else:
                    b = _bands(os.path.join(args.reg_root, c))
                    if b and max(b) < 500:
                        n_ok += 1
            if n_ok >= 2:
                recoverable[cls].append(p)

    print(f"{'class':16s} {'total':>8s} {'usable now':>12s} {'+ after re-reg':>16s}")
    for k in ("cross-vendor", "cross-model", "cross-unit", "same-unit"):
        print(f"{k:16s} {len(rows[k]):>8d} {len(usable[k]):>12d} {len(recoverable[k]):>16d}")

    n_now = len(usable["cross-vendor"]) + len(usable["cross-model"])
    n_rec = len(recoverable["cross-vendor"]) + len(recoverable["cross-model"])
    print(f"\n  cross-scanner traveling subjects usable NOW: {n_now}")
    print(f"  additional if the multi-b re-registration runs:  +{n_rec}")
    print(f"  same-unit controls usable now: {len(usable['same-unit'])} "
          f"(the noise floor -- without these the cross-scanner number means nothing)")

    print("\n=== VERDICT ===")
    if n_now >= 50:
        print(f"  {n_now} usable cross-scanner traveling subjects. This is the paper's")
        print("  primary invariance control; the ADC-matched proxy becomes a secondary check.")
        if len(usable["same-unit"]) < 20:
            print(f"  WARNING: only {len(usable['same-unit'])} same-unit controls. The noise")
            print("  floor will be poorly estimated -- prioritise recovering more.")
    elif n_rec >= 50:
        print(f"  Only {n_now} usable now, but +{n_rec} after the multi-b re-registration.")
        print("  That moves the re-registration ONTO THE CRITICAL PATH -- it is no longer")
        print("  upside, it is what makes the traveling-subject arm exist.")
    else:
        print(f"  {n_now} now, +{n_rec} recoverable -- too few for a benchmark. Keep the")
        print("  ADC-matched proxy primary and report these as a case study.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(n_same_era=same_era, n_pairs=n_pairs,
                       interval_pcts={str(k): v for k, v in (_pcts([abs(g) for g in gaps]) or {}).items()},
                       totals={k: len(v) for k, v in rows.items()},
                       usable={k: v for k, v in usable.items()},
                       recoverable={k: v for k, v in recoverable.items()}),
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
