r"""Why did 1,372 cases drop out, and is the loss vendor-selective?

The first audit run produced a usable cohort that is **99.5% GE** (4,649 GE / 23 Siemens /
1 Philips), against onboarding 3.1's stated the site split of GE ~83% / Siemens ~17%. Only ~2.3%
of the expected Siemens cases survived. The prime suspect is the builder's own gate:

    b_hi < 500  ->  "not diffusion-weighted"        614 cases (b=50) + 69 (b=0) = 683

whose source comment reads "Siemens cases registered a b=50 band as their DWI". If that gate is
firing overwhelmingly on one vendor, the cohort is not GE-dominated because the site is -- it is
GE-dominated because of a preprocessing choice, and the loss is **recoverable** by running the
multi-b re-registration in `Project 7/docs/the multi-b notes`, which restores the true high-b band.

WHICH DECISION THIS CHANGES:

  b_hi gate is >70% Siemens   -> the cross-VENDOR arm is recoverable. Run the multi-b
                                 re-registration before designing that arm. Until then, treat
                                 every cross-vendor number as unavailable, not as null.
  b_hi gate is vendor-spread  -> the vendor imbalance has another cause; the cross-vendor arm
                                 is not recoverable cheaply. Build the study on the 5 GE
                                 scanner models instead and say so explicitly.

Second question, same pass: **do repeat exams exist?** The manifests carry no patient field, so
the first audit reported "patients 4673" purely because `patient_id` fell back to `case_id`. That
number is an artifact and must not be read as evidence that repeats are absent -- if a patient
contributed two exams, a patient-level split is required and we currently cannot construct one.

Reads DICOM headers for the DROPPED cases only (~1.4k tars), plus the existing manifests.
No images, no GPU. Login/CPU node.

    python skip_audit.py --site-root <image-root> \
        --manifest-dir manifests_site --out runs/skip_audit.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import tarfile
from collections import Counter, defaultdict

import build_manifest_site as bm

MAX_MEMBERS = 20000


def _header_probe(tar_path, prefer_series=None):
    """First usable MR header in the tar: vendor, model, PatientID, StudyDate.

    Prefers `prefer_series` (the picked T2W) so vendor matches what the manifest would
    have recorded, but falls back to any MR series -- a case dropped for "missing
    T2W/DWI/ADC file" may have no usable T2W at all, and we still want its vendor.
    """
    import pydicom
    best, fallback = None, None
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
            rec = {"vendor": bm.canonical_vendor(str(getattr(d, "Manufacturer", ""))),
                   "model": str(getattr(d, "ManufacturerModelName", "")).strip(),
                   "patient_id": str(getattr(d, "PatientID", "")).strip(),
                   "study_date": str(getattr(d, "StudyDate", "")).strip(),
                   "study_uid": str(getattr(d, "StudyInstanceUID", "")).strip()}
            sn = getattr(d, "SeriesNumber", None)
            if prefer_series is not None and sn is not None and int(sn) == prefer_series:
                best = rec
            elif fallback is None:
                fallback = rec
    return best or fallback


def _bands_from_dir(cdir):
    """DWI b-bands present as registered files -- no DICOM needed."""
    out = []
    try:
        names = os.listdir(cdir)
    except OSError:
        return out
    for n in names:
        m = re.search(r"DWI_b(\d+)_to_T2W\.nii\.gz$", n)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--reg-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    reg_root = args.reg_root or os.path.join(args.site_root, "registered")
    tar_root = os.path.join(args.site_root, "raw_dicom")
    cases = sorted(d for d in os.listdir(reg_root)
                   if os.path.isdir(os.path.join(reg_root, d)))
    if args.limit:
        cases = cases[:args.limit]

    have = {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(args.manifest_dir, "*.json"))
            if not os.path.basename(f).startswith("index")}
    print(f"{len(cases)} case dirs | {len(have)} with manifests | "
          f"{len(cases)-len(have)} to probe")

    tab = defaultdict(Counter)          # reason -> vendor -> n
    band_tab = defaultdict(Counter)     # vendor -> highest-band bucket -> n
    ident, rows = [], []

    for i, case in enumerate(cases):
        cdir = os.path.join(reg_root, case)
        tar_path = os.path.join(tar_root, f"{case}.tar")
        kept = case in have
        if kept:
            try:
                man = json.load(open(os.path.join(args.manifest_dir, f"{case}.json")))
            except (OSError, json.JSONDecodeError):
                continue
            vendor, reason = man.get("vendor", "unknown"), "KEPT"
        else:
            _, reason = bm.build_case(case, reg_root, tar_root)
            reason = re.sub(r"b_hi=\d+", "b_hi=<low>", str(reason))
            reason = re.sub(r"series \S+ not in tar", "series not in tar", reason)
            vendor = "unknown"
        # vendor + identity always come from the DICOM (manifests carry no patient field)
        if os.path.exists(tar_path):
            picked = {}
            try:
                picked = json.load(open(os.path.join(cdir, "pilot_log.json"))
                                   ).get("stats", {}).get("picked", {})
            except (OSError, json.JSONDecodeError):
                pass
            try:
                rec = _header_probe(tar_path, bm._series_number(picked.get("t2")))
            except Exception:
                rec = None
            if rec:
                if not kept:
                    vendor = rec["vendor"] or "unknown"
                ident.append((case, rec["patient_id"], rec["study_date"], rec["study_uid"]))
        tab[reason][vendor] += 1
        bands = _bands_from_dir(cdir)
        hi = max(bands) if bands else None
        bucket = ("none" if hi is None else "b_hi<500 (recoverable)" if hi < 500
                  else "b_hi>=500 (usable)")
        band_tab[vendor][bucket] += 1
        rows.append(dict(case_id=case, vendor=vendor, reason=reason, bands=bands))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(cases)}", flush=True)

    vendors = sorted({v for c in tab.values() for v in c})
    print("\n=== OUTCOME x VENDOR ===")
    print(f"{'reason':44s} " + "".join(f"{v:>10s}" for v in vendors) + f"{'total':>9s}")
    for reason in sorted(tab, key=lambda r: -sum(tab[r].values())):
        c = tab[reason]
        print(f"{reason[:44]:44s} " + "".join(f"{c.get(v,0):>10d}" for v in vendors)
              + f"{sum(c.values()):>9d}")

    print("\n=== HIGH-B BAND AVAILABILITY x VENDOR (from filenames; no DICOM) ===")
    buckets = ["b_hi>=500 (usable)", "b_hi<500 (recoverable)", "none"]
    print(f"{'vendor':12s} " + "".join(f"{b:>26s}" for b in buckets))
    for v in sorted(band_tab, key=lambda v: -sum(band_tab[v].values())):
        print(f"{v:12s} " + "".join(f"{band_tab[v].get(b,0):>26d}" for b in buckets))

    print("\n=== VERDICT: is the b_hi gate vendor-selective? ===")
    low = Counter()
    for v, c in band_tab.items():
        low[v] += c.get("b_hi<500 (recoverable)", 0)
    tot_low = sum(low.values())
    if tot_low == 0:
        print("  no low-b-only cases found -- the gate is not the cause.")
    else:
        top, n = low.most_common(1)[0]
        frac = n / tot_low
        print(f"  {tot_low} cases have only a low-b band; {n} ({frac:.1%}) are {top}.")
        if frac > 0.7:
            print(f"  -> VENDOR-SELECTIVE. The cross-vendor arm is RECOVERABLE: run the")
            print(f"     multi-b re-registration (Project 7 docs/the multi-b notes) to restore")
            print(f"     the true high-b band for these {tot_low} cases, then rebuild manifests.")
        else:
            print("  -> spread across vendors. The imbalance has another cause; build the")
            print("     study on the GE scanner models and state the limitation explicitly.")

    print("\n=== REPEAT EXAMS (the 'patients 4673' number was an artifact) ===")
    pids = [p for _, p, _, _ in ident if p]
    if not pids:
        print("  PatientID is empty in every probed header -- de-identification stripped it.")
        print("  Repeat exams CANNOT be detected from these tars. A patient-level split is")
        print("  not constructible; say so rather than assuming one exam per patient.")
    else:
        n_pat, n_exam = len(set(pids)), len(pids)
        print(f"  {n_exam} exams over {n_pat} distinct PatientIDs")
        if n_pat < n_exam:
            dup = Counter(pids)
            multi = sum(1 for v in dup.values() if v > 1)
            print(f"  -> {n_exam-n_pat} repeat exams across {multi} patients. "
                  f"SPLITS MUST BE PATIENT-LEVEL, and manifests need a patient_id field.")
        else:
            print("  -> one exam per patient; case-level splits are safe.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(dict(outcome_by_vendor={k: dict(v) for k, v in tab.items()},
                           bands_by_vendor={k: dict(v) for k, v in band_tab.items()},
                           n_exams=len(ident),
                           n_patients=len({p for _, p, _, _ in ident if p}),
                           cases=rows), fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
