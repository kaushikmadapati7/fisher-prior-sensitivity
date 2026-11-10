"""Build per-case acquisition-parameter manifests for Plan A.

The physics decoder needs per-contrast TR/TE/FA/b for every case. The processed
NIfTIs carry no such metadata and (in this sample) cannot be linked back to the
raw DICOM patients, so v1 assigns parameters from per-vendor defaults keyed by
``center -> vendor`` (:mod:`physharm.manifest.contrast_rules`). The DICOM-parsing
utilities here also (a) let you derive/validate those defaults and (b) provide
the real PatientName/StudyUID linkage path used at full cohort scale.

Outputs:
    <out>/<center>/<case>.json   per-case manifest
    <out>/index.json             list of cases + linkage_coverage summary

Usage:
    python -m physharm.manifest.build_manifest \
        --processed-root ".../processed_registered_samples/Image_volumes" \
        --dicom-root ".../newbatch_raw_DICOM_samples" \
        --out physharm/manifests
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from physharm.manifest.contrast_rules import (
    CENTER_VENDOR,
    DEFAULT_VENDOR,
    VENDOR_DEFAULTS,
    classify_series,
    decode_ge_bvalue,
    parse_bvalues_from_desc,
)
from physharm.physics.decoders import assert_units

REQUIRED_CONTRASTS = ("T2W", "DWI", "ADC", "DCE_pre")

# Processed-file stems we look for, to record which contrasts a case has.
_CONTRAST_FILE_HINTS = {
    "T2W": ["T2W", "T2WI"],
    "DWI": ["DWI_to_T2W", "DWI_to_T2WI"],
    "ADC": ["ADC_to_T2W", "ADC_to_T2WI"],
    "DCE_pre": ["DCE_pre_to_T2W", "DCE_to_T2WI", "DCE_to_T2W"],
}


# ---------------------------------------------------------------------------
# DICOM parsing (used for default derivation / full-scale linkage)
# ---------------------------------------------------------------------------
def _first_dicom(series_dir: Path):
    import pydicom
    for f in sorted(series_dir.iterdir()):
        if f.is_dir():
            continue
        try:
            return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
    return None


def parse_series_params(series_dir: Path) -> dict | None:
    """Read one DICOM and return {contrast, TR_ms, TE_ms, FA_deg, b_values, ...}."""
    ds = _first_dicom(series_dir)
    if ds is None:
        return None
    desc = str(getattr(ds, "SeriesDescription", ""))
    tr = _num(getattr(ds, "RepetitionTime", None))
    te = _num(getattr(ds, "EchoTime", None))
    fa = _num(getattr(ds, "FlipAngle", None))
    vendor = str(getattr(ds, "Manufacturer", "")).upper()

    bvals = parse_bvalues_from_desc(desc)
    if (0x0043, 0x1039) in ds:  # GE private diffusion b
        b = decode_ge_bvalue(ds[(0x0043, 0x1039)].value)
        if b:
            bvals = sorted(set(bvals) | {b})
    if hasattr(ds, "DiffusionBValue"):  # Philips clean tag
        b = _num(ds.DiffusionBValue)
        if b:
            bvals = sorted(set(bvals) | {b})

    label = classify_series(desc, tr, te, fa, has_bvalue=bool(bvals))
    return {
        "contrast": label, "desc": desc, "vendor": vendor,
        "TR_ms": tr, "TE_ms": te, "FA_deg": fa, "b_values": bvals,
        "series_dir": str(series_dir),
    }


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------
def _present_contrasts(case_dir: Path) -> dict:
    names = {p.name for p in case_dir.glob("*.nii*")}
    present = {}
    for contrast, hints in _CONTRAST_FILE_HINTS.items():
        for h in hints:
            if any(n.startswith(h + ".") or n == h + ".nii.gz" for n in names):
                present[contrast] = h
                break
    return present


def build_case_manifest(case_id: str, center: str, case_dir: Path) -> dict:
    vendor = CENTER_VENDOR.get(center, DEFAULT_VENDOR)
    guessed = center not in CENTER_VENDOR
    defaults = VENDOR_DEFAULTS[vendor]
    present = _present_contrasts(case_dir)

    contrasts = {}
    for contrast in REQUIRED_CONTRASTS:
        if contrast in present and contrast in defaults:
            entry = dict(defaults[contrast])
            entry["file_stem"] = present[contrast]
            contrasts[contrast] = entry
    return {
        "case_id": case_id, "center": center, "vendor": vendor,
        "vendor_guessed": guessed, "linkage": "fallback", "contrasts": contrasts,
    }


def validate_manifest(manifest: dict) -> None:
    """Assert units and required contrasts on a per-case manifest."""
    cs = manifest["contrasts"]
    for needed in ("T2W", "DWI", "ADC", "DCE_pre"):
        assert needed in cs, f"{manifest['case_id']}: missing contrast {needed}"
    for name, params in cs.items():
        if not params.get("is_anchor"):
            assert_units(params, f"{manifest['case_id']}:{name}")


def load_manifest(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(processed_root: Path, out_dir: Path, dicom_root: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"cases": [], "linkage_coverage": {}}
    n_fallback = n_matched = n_skipped = 0
    guessed_centers: set[str] = set()
    skipped_examples: list[str] = []

    for center_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        center = center_dir.name
        for case_dir in sorted(p for p in center_dir.iterdir() if p.is_dir()):
            case_id = f"{center}/{case_dir.name}"
            manifest = build_case_manifest(case_id, center, case_dir)
            missing = [c for c in REQUIRED_CONTRASTS if c not in manifest["contrasts"]]
            if missing:  # skip incomplete cases rather than crash
                n_skipped += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(f"{case_id} (missing {','.join(missing)})")
                continue
            validate_manifest(manifest)
            if manifest.get("vendor_guessed"):
                guessed_centers.add(center)
            n_fallback += manifest["linkage"] == "fallback"
            n_matched += manifest["linkage"] == "matched"

            dst = out_dir / center / f"{case_dir.name}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w") as fh:
                json.dump(manifest, fh, indent=2)
            index["cases"].append({"case_id": case_id, "path": str(dst),
                                   "linkage": manifest["linkage"]})

    total = max(1, len(index["cases"]))
    index["linkage_coverage"] = {
        "total": len(index["cases"]),
        "fallback": n_fallback, "matched": n_matched,
        "fallback_pct": round(100 * n_fallback / total, 1),
        "skipped_incomplete": n_skipped,
        "skipped_examples": skipped_examples,
        "vendor_guessed_centers": sorted(guessed_centers),
    }
    with open(out_dir / "index.json", "w") as fh:
        json.dump(index, fh, indent=2)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", required=True,
                    help="path to processed_registered_samples/Image_volumes")
    ap.add_argument("--dicom-root", default=None,
                    help="raw DICOM root (optional; for linkage/validation)")
    ap.add_argument("--out", default="physharm/manifests")
    args = ap.parse_args()

    index = build(Path(args.processed_root), Path(args.out),
                  Path(args.dicom_root) if args.dicom_root else None)
    cov = index["linkage_coverage"]
    print(f"Wrote {cov['total']} case manifests to {args.out}")
    print(f"  linkage: {cov['matched']} matched, {cov['fallback']} fallback "
          f"({cov['fallback_pct']}% fallback)")
    if cov["skipped_incomplete"]:
        print(f"  skipped {cov['skipped_incomplete']} incomplete cases "
              f"(missing a contrast); e.g. {cov['skipped_examples'][:3]}")
    if cov["vendor_guessed_centers"]:
        print(f"  WARNING: vendor defaulted to {DEFAULT_VENDOR} for unmapped "
              f"centers {cov['vendor_guessed_centers']} -- edit CENTER_VENDOR "
              f"if any are GE/Philips (wrong params bias those maps)")


if __name__ == "__main__":
    main()
