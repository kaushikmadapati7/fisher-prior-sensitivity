r"""Do the site 1,031 repeat exams cross scanner models? And fix the split blocker.

`skip_audit.py` established that 6,044 exams come from 5,013 distinct PatientIDs -- 1,031
repeats across 876 patients. It did NOT record which scanner each exam was on, so the
question that matters is still open:

    of the 876 patients with repeats, how many were scanned on MORE THAN ONE scanner?

Those are **traveling subjects**: the same biology measured under different acquisition,
which is the gold-standard control the entire physics-harmonization literature lacks.
PhyCHarm used 9. Borges et al. used none and never measured invariance at all. The 2025
harmonization survey names validation-without-ground-truth as the field's open problem.

WHICH DECISION THIS CHANGES:

  >= ~30 cross-scanner patients   -> Candidate C is ALIVE. The real arm gets a gold-standard
                                     invariance measurement instead of the ADC-matched proxy,
                                     and that becomes the paper's strongest figure.
  a handful                       -> a case-study figure, not a benchmark. Proxy stays primary.
  none                            -> confirms the original assumption; proxy stays primary and
                                     we say so explicitly rather than leaving it unexamined.

Second job, unconditional: manifests carry **no patient field**, so a patient-level split
cannot currently be constructed and any training run right now would leak 1,031 exams across
the train/test boundary. `--patch-manifests` writes `patient_id` into each manifest.

Reads DICOM headers once and caches them, so this never needs re-running.

    python traveling_subjects.py --site-root <image-root> \
        --out runs/traveling.json [--patch-manifests manifests_site]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile
from collections import Counter, defaultdict

import build_manifest_site as bm

MAX_MEMBERS = 20000


def probe(tar_path):
    """Identity + scanner from the tar, mirroring the builder's series filter.

    CRITICAL: `Modality == "MR"` alone is NOT enough. PACS secondary captures and
    derived maps (ADC / DCE parameter maps) satisfy it while carrying stripped
    headers -- no ManufacturerModelName, TE = -1. `build_manifest_site._read_tar_series`
    additionally requires a real EchoTime, and this must match it, or the two disagree
    about what a case's scanner is.

    An earlier version returned the FIRST Modality=="MR" series, which produced empty
    model names on ~366 patients. Treating "" as a distinct model then counted a
    missing DICOM tag as a scanner change, inflating the traveling-subject count by
    an order of magnitude. Hence `identified`: a record whose scanner is unknown is
    marked, never silently used as a domain label.
    """
    import pydicom
    best = fallback = any_mr = None
    with tarfile.open(tar_path) as tf:
        seen = set()
        for i, m in enumerate(tf):
            if i > MAX_MEMBERS or best is not None:
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
            model = str(getattr(d, "ManufacturerModelName", "")).strip()
            station = bm.canonical_station(str(getattr(d, "StationName", "")))
            date = ""
            for tag in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
                v = str(getattr(d, tag, "")).strip()
                if len(v) == 8 and v.isdigit():
                    date = v
                    break
            rec = {"patient_id": str(getattr(d, "PatientID", "")).strip(),
                   "study_date": date,
                   "study_uid": str(getattr(d, "StudyInstanceUID", "")).strip(),
                   "vendor": bm.canonical_vendor(str(getattr(d, "Manufacturer", ""))),
                   "model": model, "station": station,
                   "unit": f"{model}#{station}" if station and model else model,
                   "identified": bool(model)}
            te = getattr(d, "EchoTime", None)
            try:
                te = float(te) if te is not None else None
            except (TypeError, ValueError):
                te = None
            if model and te is not None and te >= 0:
                best = rec                      # same filter the builder applies
            elif model and fallback is None:
                fallback = rec
            elif any_mr is None:
                any_mr = rec
    return best or fallback or any_mr


def _days(a: str, b: str):
    """Absolute gap in days between two DICOM YYYYMMDD dates, or None."""
    from datetime import date
    try:
        da = date(int(a[:4]), int(a[4:6]), int(a[6:8]))
        db = date(int(b[:4]), int(b[4:6]), int(b[6:8]))
    except (ValueError, IndexError):
        return None
    gap = abs((da - db).days)
    # A previous run reported a median interval of 476,781 days (~1,306 years).
    # De-identification date-shifting can emit sentinel years, so an out-of-range
    # gap is a corrupt date, not a long follow-up. Reject rather than report.
    if gap > 20 * 365 or not (1990 <= da.year <= 2035) or not (1990 <= db.year <= 2035):
        return None
    return gap


def _summarize(name, groups, by_case):
    """Report one class of repeat: how many patients, and the interval distribution."""
    if not groups:
        print(f"  {name:38s} 0 patients")
        return []
    gaps = []
    for cases in groups.values():
        dates = sorted(by_case[c]["study_date"] for c in cases if by_case[c]["study_date"])
        for x, y in zip(dates, dates[1:]):
            g = _days(x, y)
            if g is not None:
                gaps.append(g)
    n_ex = sum(len(v) for v in groups.values())
    line = f"  {name:38s} {len(groups):5d} patients, {n_ex:5d} exams"
    if gaps:
        gaps.sort()
        q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
        line += f" | interval days p10/50/90 = {q(.1)}/{q(.5)}/{q(.9)}"
    print(line)
    return gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--reg-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default="runs/header_cache.json",
                    help="probed headers are cached here so tars are read only once")
    ap.add_argument("--patch-manifests", default=None, metavar="DIR",
                    help="write patient_id into each manifest in DIR (required before "
                         "any training: without it a patient-level split is impossible)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    reg_root = args.reg_root or os.path.join(args.site_root, "registered")
    tar_root = os.path.join(args.site_root, "raw_dicom")

    by_case = {}
    if args.cache and os.path.exists(args.cache):
        by_case = json.load(open(args.cache))
        # Entries written before the probe fix lack `identified` and may carry a blank
        # model from a secondary-capture series. Re-probe those; keep the rest.
        stale = [c for c, r in by_case.items()
                 if "identified" not in r or not r.get("model")]
        print(f"loaded {len(by_case)} cached headers from {args.cache}"
              + (f" ({len(stale)} stale -> will re-probe)" if stale else ""))
        for c in stale:
            by_case.pop(c, None)

    cases = sorted(d for d in os.listdir(reg_root)
                   if os.path.isdir(os.path.join(reg_root, d)))
    if args.limit:
        cases = cases[:args.limit]
    todo = [c for c in cases if c not in by_case]
    print(f"{len(cases)} case dirs | {len(todo)} to probe")
    for i, case in enumerate(todo):
        tp = os.path.join(tar_root, f"{case}.tar")
        if not os.path.exists(tp):
            continue
        try:
            rec = probe(tp)
        except Exception:
            rec = None
        if rec:
            by_case[case] = rec
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    if args.cache and (todo or True):
        os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
        json.dump(by_case, open(args.cache, "w"))
        print(f"cached {len(by_case)} headers -> {args.cache}")

    have_pid = {c: r for c, r in by_case.items() if r.get("patient_id")}
    print(f"\n{len(by_case)} exams probed | {len(have_pid)} with a PatientID")
    if not have_pid:
        print("PatientID stripped by de-identification -- patient-level splits impossible.")
        return

    # An exam whose ManufacturerModelName is missing has an UNKNOWN scanner, not a
    # distinct one. Treating "" as its own model previously inflated the
    # traveling-subject count ~11x (366 vs 33). Such exams are excluded from every
    # cross-scanner determination and reported separately.
    unident = {c for c, r in have_pid.items() if not r.get("identified")}
    if unident:
        print(f"  !! {len(unident)} exams have no ManufacturerModelName -- scanner UNKNOWN.")
        print("     Excluded from cross-scanner determination (a missing tag is not a")
        print("     scanner change). Their patients are reported as ambiguous below.")

    per_patient = defaultdict(list)
    for c, r in have_pid.items():
        per_patient[r["patient_id"]].append(c)

    repeats = {p: cs for p, cs in per_patient.items() if len(cs) > 1}
    # A patient is DECIDABLE only if every one of their exams has a known scanner.
    decidable = {p: cs for p, cs in repeats.items() if not any(c in unident for c in cs)}
    ambiguous = {p: cs for p, cs in repeats.items() if p not in decidable}

    dates = [r["study_date"] for r in have_pid.values() if r.get("study_date")]
    years = Counter(d[:4] for d in dates)
    print(f"\n  StudyDate present on {len(dates)}/{len(have_pid)} exams; "
          f"year range {min(years) if years else '-'}..{max(years) if years else '-'}")
    odd = {y: n for y, n in years.items() if not ("1990" <= y <= "2035")}
    if odd:
        print(f"  !! implausible years present (de-id date shifting?): {dict(sorted(odd.items())[:8])}")
        print("     Intervals from these are rejected, so some groups may report no interval.")

    # Classes are computed over `decidable` only: an exam with no model tag has an
    # UNKNOWN scanner, and must never act as a domain label.
    multi_unit = {p: cs for p, cs in decidable.items()
                  if len({have_pid[c]["unit"] for c in cs}) > 1}
    multi_model = {p: cs for p, cs in decidable.items()
                   if len({have_pid[c]["model"] for c in cs}) > 1}
    multi_vendor = {p: cs for p, cs in decidable.items()
                    if len({have_pid[c]["vendor"] for c in cs}) > 1}
    same_unit = {p: cs for p, cs in decidable.items()
                 if len({have_pid[c]["unit"] for c in cs}) == 1}

    print(f"\n=== REPEAT STRUCTURE ===")
    print(f"  {len(per_patient)} patients over {len(have_pid)} exams "
          f"({len(repeats)} patients with >1 exam)")
    print(f"  {len(decidable)} decidable (every exam's scanner known) | "
          f"{len(ambiguous)} ambiguous (>=1 exam with no model tag)")
    _summarize("same scanner unit (NOISE FLOOR)", same_unit, have_pid)
    _summarize("different scanner UNIT", multi_unit, have_pid)
    _summarize("different scanner MODEL", multi_model, have_pid)
    _summarize("different VENDOR", multi_vendor, have_pid)

    print("\n  the same-unit group is the mandatory control: it bounds the test-retest")
    print("  variation any cross-scanner difference must exceed to mean anything.")

    if multi_model:
        pairs = Counter()
        for cs in multi_model.values():
            ms = sorted({have_pid[c]["model"] for c in cs})
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    pairs[(ms[i], ms[j])] += 1
        print("\n=== CROSS-MODEL PAIRS (patients scanned on both) ===")
        for (a, b), n in pairs.most_common(15):
            print(f"  {n:5d}  {a}  <->  {b}")

    print("\n=== VERDICT: does the site have traveling subjects? ===")
    n = len(multi_model)
    print(f"  {n} patients have >1 scanner MODEL among exams whose scanner is known.")
    if ambiguous:
        print(f"  {len(ambiguous)} further patients are UNDECIDABLE (an exam lacks a model")
        print(f"  tag), so the true count lies in [{n}, {n + len(ambiguous)}]. Report the")
        print("  floor, not the upper bound.")
    if n >= 30:
        print("\n  -> Candidate C is ALIVE. PhyCHarm used 9 traveling subjects; Borges et al.")
        print("     used none and never measured invariance. This replaces the ADC-matched")
        print("     proxy as the primary invariance control.")
        print("  CAVEATS, both load-bearing:")
        print("   1. These are clinical repeats, not a traveling-subject study. Check the")
        print("      interval distribution -- prostate active-surveillance rescans run months")
        print("      to years, over which biology changes. Prefer short-interval pairs.")
        print("   2. Every cross-scanner difference must be read against the same-unit")
        print("      noise floor above. A difference below it is not a scanner effect.")
    elif n > 0:
        print("\n  -> MARGINAL: a case-study figure, not a benchmark. Keep the ADC-matched")
        print("     proxy primary and report these alongside it.")
    else:
        print("\n  -> NO: repeats never cross a scanner model. Proxy stays primary; say so.")

    if args.patch_manifests:
        n_ok = 0
        for f in glob.glob(os.path.join(args.patch_manifests, "*.json")):
            if os.path.basename(f).startswith("index"):
                continue
            case = os.path.splitext(os.path.basename(f))[0]
            rec = by_case.get(case)
            if not rec or not rec.get("patient_id"):
                continue
            man = json.load(open(f))
            man["patient_id"] = rec["patient_id"]
            man["study_date"] = rec["study_date"]
            man["scanner_unit_dicom"] = rec["unit"]
            json.dump(man, open(f, "w"), indent=2)
            n_ok += 1
        print(f"\npatched patient_id into {n_ok} manifests in {args.patch_manifests}")
        print("  -> patient-level splits are now constructible. Before this, any split")
        print("     would have leaked repeat exams across the train/test boundary.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(n_exams=len(have_pid), n_patients=len(per_patient),
                       n_repeat_patients=len(repeats),
                       n_multi_unit=len(multi_unit), n_multi_model=len(multi_model),
                       n_multi_vendor=len(multi_vendor),
                       n_decidable=len(decidable), n_ambiguous=len(ambiguous),
                       n_unidentified_exams=len(unident),
                       ambiguous_patients=list(ambiguous),
                       multi_model_patients={p: cs for p, cs in multi_model.items()},
                       same_unit_patients={p: cs for p, cs in same_unit.items()},
                       by_case=have_pid), open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
