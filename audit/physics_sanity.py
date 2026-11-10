r"""The last data unknowns before any physics runs: ADC units, per-scanner SNR, b-band sanity.

Three things gate every downstream experiment, and all three are MEASURED here rather
than assumed.

(1) ADC UNITS -- and we measure them, we do not read them off a filename.
    the site ADC files are named `ADC_(10^-6_mm^2_s)`, implying values ~1300 rather than
    ~1.3e-3. `physics/decoders.py` requires `b*D` to be dimensionless with D in mm^2/s, and
    its `assert_units()` checks TR, TE, FA and b but says NOTHING about D. A factor-1e6
    error would therefore pass silently and make every diffusion term meaningless.

    The measurement: with two b-bands we can recover D ourselves from Stejskal-Tanner,

        D_est = ln(S_lo / S_hi) / (b_hi - b_lo)          [mm^2/s by construction]

    and compare against the stored ADC map. The ratio stored/D_est is the unit factor --
    1.0 or ~1e6 -- read off the data. This simultaneously validates that the b-band LABELS
    are right, since a mislabelled band would break the identity too.

(2) PER-SCANNER SNR. A Cramer-Rao bound is a VARIANCE bound: the synthetic arm's noise
    level determines the entire predicted-leakage curve. `eval/synthetic_validation.py`
    currently hardcodes `strength=0.2`, an arbitrary constant. With no noise the CRLB
    prediction is vacuous; with the wrong noise it is wrong. Two independent estimators
    are reported and disagreement is flagged rather than averaged away.

(3) B-BAND ORDERING. The high-b volume must actually be darker than the low-b one inside
    the prostate. A case where it is not has mislabelled bands, and would silently corrupt
    both the ADC identity and the encoder inputs.

Needs nibabel + numpy. Reads images, so it is the first script here that is not
manifest-only.

RUNNING IT OVER THE WHOLE COHORT. At ~3 s/case this is hours, and an interactive session that
drops takes the whole run with it -- there is no partial credit, because the output is written
once at the end. So:

  --shard N/K   partitions the cohort for a SLURM array; each shard writes its own JSON
  --merge       combines shard JSONs into one file with the same schema as a single run
  --bands-only  skips the ADC-ratio and SNR work and checks ONLY the b-band ordering

--bands-only is usually what you want for a full-cohort run. The ADC factor is already settled
(1.037e6 over 149 cases, confirmed against real volumes by the loader guard) and the SNR
numbers are stable and swept downstream anyway -- but `bad_bands` is a PER-CASE exclusion
list, so a sample genuinely cannot substitute for it. Bands-only loads 3 volumes instead of 5
and skips the percentile sorts, which is the cheap fraction of the work carrying all of the
remaining value.

    python physics_sanity.py --manifest-dir manifests_site \
        --image-root <image-root> --limit 150 --out runs/physics_sanity.json

    # full cohort (0 = no limit) -- required before the cache build, since the offenders it
    # finds change the exclusion list and therefore the cohort_id
    python physics_sanity.py --manifest-dir manifests_site \
        --image-root <image-root> --limit 0 --out runs/physics_sanity_full.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

_EPS = 1e-9


def estimate_D(s_lo, s_hi, b_lo, b_hi, mask):
    """Stejskal-Tanner two-point diffusion coefficient, in mm^2/s.

        S(b) = S0 exp(-b D)  =>  D = ln(S_lo/S_hi) / (b_hi - b_lo)

    Returns the ROI median. Voxels with non-positive signal are dropped rather than
    clamped: a clamp would bias D toward zero exactly where SNR is worst.
    """
    if b_hi <= b_lo:
        return None
    ok = mask & (s_lo > _EPS) & (s_hi > _EPS)
    if ok.sum() < 50:
        return None
    d = np.log(s_lo[ok] / s_hi[ok]) / (b_hi - b_lo)
    return float(np.median(d))


def snr_background(img, mask):
    """SNR from the air/background, using the Rayleigh MEDIAN, not its std.

    In signal-free regions magnitude MR noise is Rayleigh with median = sigma*sqrt(2 ln 2)
    = 1.1774*sigma, so sigma = median(air) / 1.1774.

    A previous version selected the lowest DECILE of out-of-ROI voxels and took its std.
    That is wrong twice over: the decile is a truncated tail whose std badly
    under-estimates sigma (verified: it returned SNR 122 against a planted 20, a 6x
    over-estimate), and std is the wrong statistic for a Rayleigh anyway. The median of a
    threshold-selected air region is robust to the truncation, because thresholding
    removes the UPPER tail while the median tracks the bulk.

    Air is selected as voxels below 20%% of the ROI median -- well under any real tissue,
    well above the noise floor. Returns None when too little air survives, which happens
    when the registered volume is cropped tightly around the pelvis.
    """
    out = ~mask
    if out.sum() < 500 or mask.sum() < 50:
        return None
    sig = float(np.median(img[mask]))
    if not np.isfinite(sig) or sig <= _EPS:
        return None
    air = img[out]
    air = air[air < 0.20 * sig]
    if air.size < 200:
        return None
    sigma = float(np.median(air)) / 1.1774
    if sigma <= _EPS:
        return None
    return sig / sigma


def snr_highfreq(img, mask):
    """SNR from in-ROI high-frequency content: sigma ~ MAD of the slice-to-slice difference.

    Independent of `snr_background` -- it does not need any air in the field of view, which
    matters because these volumes are registered and may be cropped. Assumes tissue is
    smooth relative to noise between adjacent slices, which over-estimates sigma wherever
    anatomy changes fast; hence it is reported ALONGSIDE the background estimate, never
    averaged with it.
    """
    if img.ndim != 3 or img.shape[0] < 3 or mask.sum() < 50:
        return None
    d = np.diff(img, axis=0)
    m = mask[:-1] & mask[1:]
    if m.sum() < 50:
        return None
    diff = d[m]
    sigma = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0)
    sig = float(np.median(img[mask]))
    return sig / max(sigma, _EPS)


def _load(path):
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--shard", default=None, metavar="N/K",
                    help="process shard N of K (0-indexed) and write <out>.shard<N>.json. "
                         "For a SLURM array over a multi-hour cohort.")
    ap.add_argument("--merge", nargs="+", default=None, metavar="JSON",
                    help="merge shard outputs into --out and exit")
    ap.add_argument("--bands-only", action="store_true",
                    help="check ONLY the b-band ordering: skips ADC-ratio and SNR, loads 3 "
                         "volumes instead of 5, no percentile sorts. The fast path for a "
                         "full-cohort bad_bands list.")
    ap.add_argument("--limit", type=int, default=150,
                    help="cases to examine; 0 = NO LIMIT (matches every other script here). "
                         "This slice used to be unconditional, so --limit 0 read zero cases "
                         "and still wrote an output file -- silently disabling the "
                         "mislabelled-b-band gate downstream.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.merge:
        merged, seen = {"bad_bands": [], "n_cases": 0, "adc_ratio_by_scanner": {},
                        "shards_merged": len(args.merge)}, set()
        ratios = defaultdict(list)
        for f in args.merge:
            d = json.load(open(f))
            for c in d.get("bad_bands", []):
                if c not in seen:
                    seen.add(c)
                    merged["bad_bands"].append(c)
            merged["n_cases"] += int(d.get("n_cases") or 0)
            for k, v in (d.get("adc_ratio_by_scanner") or {}).items():
                ratios[k].append(v)
        merged["adc_ratio_by_scanner"] = {k: float(np.median(v)) for k, v in ratios.items()}
        if ratios:
            merged["adc_ratio_median"] = float(np.median(
                [x for v in ratios.values() for x in v]))
        print(f"merged {len(args.merge)} shards: {merged['n_cases']} cases, "
              f"{len(merged['bad_bands'])} mislabelled-band cases")
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            json.dump(merged, open(args.out, "w"), indent=2)
            print(f"wrote {args.out}")
        return

    files = sorted(f for f in glob.glob(os.path.join(args.manifest_dir, "*.json"))
                   if not os.path.basename(f).startswith("index"))
    if args.shard:
        n, k = map(int, args.shard.split("/"))
        files = [f for i, f in enumerate(files) if i % k == n]
        if args.out:
            base, ext = os.path.splitext(args.out)
            args.out = f"{base}.shard{n}{ext}"
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} manifests to check"
          + ("" if args.limit else "  (no limit)"))
    if not files:
        raise SystemExit(
            f"no manifests found under {args.manifest_dir} -- refusing to run. An empty "
            f"result here is not 'no offenders', it is 'nothing was examined', and writing "
            f"it would disable the mislabelled-b-band gate while looking like a clean pass.")

    ratios, snr = defaultdict(list), defaultdict(lambda: defaultdict(list))
    band_bad, n_ok, why = [], 0, defaultdict(int)

    for f in files:
        man = json.load(open(f))
        case = man.get("case_id") or os.path.splitext(os.path.basename(f))[0]
        cdir = os.path.join(args.image_root, case)
        con = man.get("contrasts", {})
        scanner = man.get("scanner_model") or man.get("vendor") or "unknown"
        try:
            mask = _load(os.path.join(cdir, "prostate_mask.nii.gz")) > 0.5
            if mask.sum() < 50:
                why["empty mask"] += 1
                continue
            adc = t2w = None
            if not args.bands_only:
                adc = _load(os.path.join(cdir, con["ADC"]["file_stem"] + ".nii.gz"))
                t2w = _load(os.path.join(cdir, con["T2W"]["file_stem"] + ".nii.gz"))
            hi_stem = con["DWI"]["file_stem"]
            b_hi = float(max(con["DWI"].get("b_values") or [0]))
            s_hi = _load(os.path.join(cdir, hi_stem + ".nii.gz"))
            s0 = con.get("S0")
        except (OSError, KeyError) as e:
            why[f"load: {type(e).__name__}"] += 1
            continue

        if args.bands_only:
            s0 = con.get("S0")
            if s0:
                try:
                    s_lo = _load(os.path.join(cdir, s0["file_stem"] + ".nii.gz"))
                except OSError:
                    s_lo = None
                if s_lo is not None:
                    if np.median(s_hi[mask]) >= np.median(s_lo[mask]):
                        band_bad.append(case)
                    n_ok += 1
            continue

        for name, img in (("T2W", t2w), ("DWI", s_hi)):
            for est, fn in (("bg", snr_background), ("hf", snr_highfreq)):
                v = fn(img, mask)
                if v and np.isfinite(v):
                    snr[name][est].append((scanner, v))

        if s0:
            b_lo = float(min(s0.get("b_values") or [0]))
            try:
                s_lo = _load(os.path.join(cdir, s0["file_stem"] + ".nii.gz"))
            except OSError:
                s_lo = None
            if s_lo is not None:
                if np.median(s_hi[mask]) >= np.median(s_lo[mask]):
                    band_bad.append(case)          # high-b must be darker
                d_est = estimate_D(s_lo, s_hi, b_lo, b_hi, mask)
                a_med = float(np.median(adc[mask]))
                if d_est and d_est > 0 and np.isfinite(a_med):
                    ratios[scanner].append(a_med / d_est)
                    n_ok += 1

    what = "band-checked" if args.bands_only else "contributing an ADC-unit measurement"
    print(f"\ncases {what}: {n_ok}")
    for k, v in sorted(why.items()):
        print(f"  skipped {v:4d}  {k}")

    if args.bands_only:
        # Do NOT print empty ADC/SNR tables. An empty table reads as "measured, found
        # nothing" when it means "skipped" -- the same defect as the hardcoded b_hi
        # footer fixed in #22. Name what was skipped, and why it is safe to skip.
        print("\n=== 1-2. ADC UNITS and SNR: SKIPPED (--bands-only) ===")
        print("  Not measured in this mode -- not measured-as-empty. The ADC factor is")
        print("  settled at 1.037e6 over 149 cases and was confirmed on real volumes by")
        print("  the loader guard; SNR is stable and swept downstream regardless.")
        allr = []
    else:
        print("\n=== 1. ADC UNITS (measured, not assumed) ===")
        print(f"{'scanner':24s} {'n':>5s} {'median stored/D_est':>21s}  reading")
        allr = []
        for s, v in sorted(ratios.items(), key=lambda kv: -len(kv[1])):
            med = float(np.median(v))
            allr += v
            tag = ("mm^2/s (factor 1)" if 0.2 < med < 5 else
                   "10^-6 mm^2/s (factor 1e6)" if 2e5 < med < 5e6 else
                   f"UNEXPECTED ({med:.3g})")
            print(f"{s[:24]:24s} {len(v):>5d} {med:>21.4g}  {tag}")
        if allr:
            g = float(np.median(allr))
            print(f"\n  cohort median ratio = {g:.4g}")
            if 2e5 < g < 5e6:
                print("  -> ADC is stored in 10^-6 mm^2/s. DIVIDE BY 1e6 before it reaches the")
                print("     physics decoder, and add the assertion to decoders.assert_units().")
            elif 0.2 < g < 5:
                print("  -> ADC is already mm^2/s; no rescale needed. Still add the assertion.")
            else:
                print("  -> NEITHER. Do not proceed: the stored ADC and the two-band estimate")
                print("     disagree by a factor that is not a unit convention, which means the")
                print("     b-band labels or the ADC map itself are wrong.")
            spread = max(np.median(v) for v in ratios.values()) / max(1e-9, min(np.median(v) for v in ratios.values()))
            if spread > 3:
                print(f"  !! per-scanner ratios differ by {spread:.1f}x -- a VENDOR-DEPENDENT ADC")
                print("     scaling, which is a confound in its own right (the archived")
                print("     vendor_separability.py excludes D for exactly this reason).")

        print("\n=== 2. PER-SCANNER SNR (two independent estimators) ===")
        for c in ("T2W", "DWI"):
            print(f"\n  {c}:")
            print(f"    {'scanner':24s} {'n':>5s} {'SNR bg':>9s} {'SNR hf':>9s}  agreement")
            per = defaultdict(lambda: {"bg": [], "hf": []})
            for est in ("bg", "hf"):
                for s, v in snr[c][est]:
                    per[s][est].append(v)
            for s, d in sorted(per.items(), key=lambda kv: -len(kv[1]["bg"]) - len(kv[1]["hf"])):
                bg = float(np.median(d["bg"])) if d["bg"] else float("nan")
                hf = float(np.median(d["hf"])) if d["hf"] else float("nan")
                r = bg / hf if (hf == hf and hf > 0 and bg == bg) else float("nan")
                tag = ("ok" if 0.5 < r < 2 else "DISAGREE -- do not average, investigate")
                print(f"    {s[:24]:24s} {max(len(d['bg']),len(d['hf'])):>5d} "
                      f"{bg:>9.1f} {hf:>9.1f}  {tag}")
        print("\n  Use these to replace the hardcoded strength=0.2 in synthetic_validation.py.")
        print("  If the two estimators disagree, the volumes are probably cropped (no air for")
        print("  the background estimate) -- prefer the high-frequency one and say so.")

    print(f"\n=== 3. B-BAND SANITY ===")
    print(f"  cases where high-b is NOT darker than low-b: {len(band_bad)}")
    if band_bad:
        print(f"    e.g. {band_bad[:5]} -- mislabelled bands; exclude before training.")
    if args.bands_only and n_ok:
        print(f"  rate this shard: {len(band_bad)}/{n_ok} = "
              f"{len(band_bad)/n_ok:.2%}")

    if args.bands_only:
        print(f"\nbands-only: {n_ok} cases checked, {len(band_bad)} with mislabelled bands")
    if args.out and n_ok == 0:
        raise SystemExit(
            f"{len(files)} manifests examined but ZERO produced an ADC measurement -- "
            f"refusing to write {args.out}. Downstream this file becomes the "
            f"mislabelled-b-band exclusion list, and an empty one silently disables that "
            f"gate. Check --image-root and the skip reasons above.")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(adc_ratio_median=(float(np.median(allr)) if allr else None),
                       adc_ratio_by_scanner={k: float(np.median(v)) for k, v in ratios.items()},
                       snr={c: {e: {s: float(np.median([x for ss, x in snr[c][e] if ss == s]))
                                    for s in {ss for ss, _ in snr[c][e]}}
                                for e in ("bg", "hf")} for c in ("T2W", "DWI")},
                       bad_bands=band_bad, n_cases=n_ok), open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
