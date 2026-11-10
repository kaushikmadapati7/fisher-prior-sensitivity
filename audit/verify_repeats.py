r"""Are the 876 "repeat patients" real, or are PatientIDs colliding?

CONFLICT TO RESOLVE. `traveling_subjects.py` reports 876 patients with >1 exam and 607
scanned on >1 scanner model. An earlier physharm run of `xscan/find_repeat_scans.py`
reported **zero**. One of them is wrong.

Sampling math explains part of it: at the documented fast look (`--limit 300`) the
expected yield is only ~2.9 repeat pairs and ~2.0 cross-model patients, so a zero has
~5% / ~13% probability. Possible -- but not comfortable enough to declare the earlier
run wrong, and there is an independent red flag in the NEW result:

    only 152/876 (17.4%) of repeat patients returned to the SAME scanner unit.

Real patients tend to come back to the same imaging center. An 83% cross-scanner rate is
what you would see if the second exam's scanner were drawn at RANDOM -- i.e. if
de-identification produced PatientID COLLISIONS across unrelated patients, manufacturing
phantom repeats. That would inflate the traveling-subject count from ~0 to ~600 and would
neatly explain both results at once.

TWO INDEPENDENT TESTS:

  (1) NULL TEST (instant, cache only). Compare the observed same-unit rate against the
      rate expected if each exam's scanner were an independent draw from the cohort's
      marginal scanner distribution. Collisions -> observed ~= null. Real repeats ->
      observed MUCH higher than null (patients return to their own site).

  (2) CORROBORATION (targeted, ~1.9k tars). Two exams sharing a PatientID should also
      agree on PatientBirthDate and PatientSex. Disagreement proves a collision. This is
      the same tie-break `xscan/find_repeat_scans.py:identity_audit` was written for,
      applied to the repeat clusters specifically rather than to a blind sample.

WHICH DECISION THIS CHANGES:
  corroborated + null rejected  -> 607 stands; the earlier zero was a sampling artifact.
  collisions found              -> my number is wrong, discard the traveling-subject arm,
                                   and the ADC-matched proxy stays primary.
  identity tags all scrubbed    -> UNVERIFIABLE. Do not build a primary control on it.

    python verify_repeats.py --cache runs/header_cache.json \
        --site-root <image-root> --out runs/verify_repeats.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from collections import Counter, defaultdict

MAX_MEMBERS = 20000
ID_TAGS = ("PatientID", "PatientName", "PatientBirthDate", "PatientSex",
           "AccessionNumber", "StudyInstanceUID", "PatientIdentityRemoved",
           "DeidentificationMethod")


def identity(tar_path):
    """Identity tags from the first MR object. require_te is NOT applied: identity
    tags do not need an acquisition, and secondary captures carry them fine."""
    import pydicom
    with tarfile.open(tar_path) as tf:
        seen = set()
        for i, m in enumerate(tf):
            if i > MAX_MEMBERS:
                break
            if not m.isfile():
                continue
            key = m.name.rsplit("/", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            try:
                d = pydicom.dcmread(io.BytesIO(tf.extractfile(m).read()),
                                    stop_before_pixels=True, force=True)
            except Exception:
                continue
            if str(getattr(d, "Modality", "")) != "MR":
                continue
            return {t: str(getattr(d, t, "")).strip() for t in ID_TAGS}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="runs/header_cache.json")
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-corroboration", action="store_true",
                    help="run only the instant null test")
    args = ap.parse_args()

    by_case = json.load(open(args.cache))
    ok = {c: r for c, r in by_case.items() if r.get("patient_id") and r.get("identified")}
    per_patient = defaultdict(list)
    for c, r in ok.items():
        per_patient[r["patient_id"]].append(c)
    repeats = {p: cs for p, cs in per_patient.items() if len(cs) > 1}
    print(f"{len(ok)} identified exams | {len(per_patient)} PatientIDs | "
          f"{len(repeats)} with >1 exam")

    # ---------- TEST 1: null model ------------------------------------------
    print("\n=== TEST 1: is the scanner pairing better than random? ===")
    units = Counter(r["unit"] for r in ok.values())
    tot = sum(units.values())
    share = {u: n / tot for u, n in units.items()}
    print(f"  {len(units)} scanner units; top shares: " +
          ", ".join(f"{u.split('#')[0][:18]}={share[u]:.1%}"
                    for u, _ in units.most_common(4)))

    exp_same = 0.0
    for p, cs in repeats.items():
        k = len(cs)
        exp_same += sum(s ** k for s in share.values())   # all k exams on one unit
    obs_same = sum(1 for cs in repeats.values()
                   if len({ok[c]["unit"] for c in cs}) == 1)
    print(f"\n  same-unit repeat patients   observed {obs_same:5d}"
          f"   expected-if-random {exp_same:8.1f}")
    if exp_same > 0:
        ratio = obs_same / exp_same
        print(f"  enrichment over random = {ratio:.2f}x")
        if ratio > 2.0:
            print("  -> REAL. Patients return to their own scanner far more often than")
            print("     chance. PatientID is grouping genuinely related exams.")
        elif ratio > 1.2:
            print("  -> WEAK enrichment. Some real structure, but collisions may be mixed in.")
        else:
            print("  -> CONSISTENT WITH RANDOM. This is the collision signature: the")
            print("     'repeats' pair scanners no better than chance, which is what")
            print("     unrelated patients sharing a hashed ID would look like.")

    # ---------- TEST 2: corroborating identity tags -------------------------
    result = dict(n_repeat_patients=len(repeats), obs_same_unit=obs_same,
                  exp_same_unit_random=exp_same)
    if args.skip_corroboration:
        print("\n(skipping corroboration)")
    else:
        tar_root = os.path.join(args.site_root, "raw_dicom")
        cases = sorted({c for cs in repeats.values() for c in cs})
        print(f"\n=== TEST 2: do repeat exams corroborate on birth date / sex? ===")
        print(f"  reading identity tags for {len(cases)} exams in repeat clusters")
        ident = {}
        for i, c in enumerate(cases):
            tp = os.path.join(tar_root, f"{c}.tar")
            if not os.path.exists(tp):
                continue
            try:
                r = identity(tp)
            except Exception:
                r = None
            if r:
                ident[c] = r
            if (i + 1) % 250 == 0:
                print(f"    {i+1}/{len(cases)}", flush=True)

        cov = {t: sum(1 for r in ident.values() if r.get(t)) for t in ID_TAGS}
        print(f"\n  tag coverage across {len(ident)} exams:")
        for t in ID_TAGS:
            print(f"    {t:24s} {cov[t]:5d} ({cov[t]/max(1,len(ident)):5.1%})")

        usable = [t for t in ("PatientBirthDate", "PatientSex") if cov[t] > 0.5 * len(ident)]
        if not usable:
            print("\n  -> NO corroborating tag survived de-identification.")
            print("     The repeat structure is UNVERIFIABLE from DICOM alone. Do not")
            print("     make it the primary control; treat 607 as an upper bound.")
            result["verdict"] = "unverifiable"
        else:
            agree = disagree = 0
            bad = []
            for p, cs in repeats.items():
                vals = {t: {ident[c][t] for c in cs if c in ident and ident[c].get(t)}
                        for t in usable}
                if any(len(v) > 1 for v in vals.values()):
                    disagree += 1
                    if len(bad) < 5:
                        bad.append((p, {t: sorted(v) for t, v in vals.items()}))
                elif any(vals.values()):
                    agree += 1
            n = agree + disagree
            print(f"\n  corroborated on {', '.join(usable)}:")
            print(f"    agree    {agree:5d}")
            print(f"    DISAGREE {disagree:5d}  ({disagree/max(1,n):.1%}) <- collisions")
            for p, v in bad:
                print(f"      e.g. PatientID {p[:16]!r}: {v}")
            result.update(n_agree=agree, n_disagree=disagree, usable_tags=usable)
            frac = disagree / max(1, n)
            if frac < 0.02:
                print("\n  -> CORROBORATED. Repeat clusters are internally consistent;")
                print("     these are the same patient. The earlier zero was a sampling")
                print("     artifact of the 300-case fast look, not a real disagreement.")
                result["verdict"] = "corroborated"
            elif frac < 0.2:
                print(f"\n  -> MOSTLY REAL with {disagree} collisions. Drop the disagreeing")
                print("     clusters and recount before using the cohort.")
                result["verdict"] = "mostly_real"
            else:
                print("\n  -> COLLISIONS DOMINATE. My 607 is wrong; the earlier zero was")
                print("     closer to the truth. Discard the traveling-subject arm.")
                result["verdict"] = "collisions"

    print("\n=== HOW TO SETTLE THE ORIGINAL CONFLICT DIRECTLY ===")
    print("  Re-run the physharm test on the FULL cohort, not the fast look:")
    print("    python -m xscan.find_repeat_scans --raw-root <RAW> --out runs/site_patients.csv")
    print("  At --limit 300 it expects ~2.9 repeat pairs; at full size it expects ~1186.")
    print("  If the full run still reports zero, the disagreement is real and mine is wrong.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
