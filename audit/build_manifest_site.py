"""Build REAL per-case acquisition manifests for the site prostate cohort.

Links the registered NIfTIs to the raw DICOM headers *inside the per-case tars*:

    registered/<case>/{T2W, DWI_b<N>_to_T2W, ADC_to_T2W}.nii.gz + prostate_mask
                      + pilot_log.json  (records the PICKED source series, e.g.
                                         "t2": "4_AX_T2_PROSTATE.nii.gz")
    raw_dicom/<case>.tar                (the exam's DICOMs)
    -> per-case manifest with REAL TR/TE/vendor, linkage="matched", source="dicom"

The leading integer of each picked filename is its SeriesNumber, which is how we
find the matching DICOM in the tar (same trick as build_manifest_real, but the
series live in a tar instead of a parquet-indexed tree).

Why this cohort: the site T2W TE~124 vs DWI TE~65 => dTE ~60 ms, so T2 is genuinely
identifiable (contrast with Philips dTE~1-4 ms, where it rails). Multi-b includes
a real b=0, which is emitted as an ``S0`` contrast -- a diffusion-free anchor at
the DWI echo time (enables the biparametric_s0 modes).

NOTE: PACS secondary-capture objects (Manufacturer "AGFA"/IMPAX) and derived maps
(ADC / DCE parameter maps, which carry TE=-1) are skipped -- only Modality==MR
series with a real EchoTime are used for acquisition params.

    python build_manifest_site.py \
      --site-root <image-root> --out manifests_site [--shard 0/20]

PROVENANCE: lifted verbatim from
  Project 7/archive/2026-08_pre-reset/physharm/manifest/build_manifest_site.py
with `canonical_vendor` inlined and the module-run invocation changed. No logic
was altered. It is ARCHIVED (pre-reset) code and therefore [WARN] -- its skip
counts and vendor split have not been re-verified against the current data.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import tarfile



# ---------------------------------------------------------------------------
# Vendored from physharm/manifest/contrast_rules.py (archive/2026-08_pre-reset).
# Inlined so this script runs standalone on the cluster: the `physharm` package
# did not survive the Project 7 reset and only `xscan/` is in the live tree.
# Logic is byte-for-byte the original; only the import was removed.
# ---------------------------------------------------------------------------
def canonical_vendor(raw: str) -> str:
    """Map a DICOM Manufacturer string to a canonical vendor label.

    DICOM Manufacturer is inconsistent ('GE MEDICAL SYSTEMS', 'SIEMENS',
    'Siemens', 'Philips Medical Systems', 'UIH', ...). Phase-B invariance groups
    by scanner, so collapse to one label per vendor.
    """
    s = (raw or "").strip().upper()
    if not s:
        return "unknown"
    if s.startswith("GE") or "GENERAL ELECTRIC" in s:
        return "GE"
    if "SIEMENS" in s:
        return "Siemens"
    if "PHILIPS" in s:
        return "Philips"
    if s.startswith("UIH") or "UNITED IMAGING" in s:
        return "UIH"
    if "TOSHIBA" in s or "CANON" in s:
        return "Canon"
    if "HITACHI" in s:
        return "Hitachi"
    return raw.strip()  # keep original if unrecognized


MAX_MEMBERS = 20000          # safety cap when scanning a tar
RENDER = {"T2W": "tse_t2", "DWI": "se_epi", "S0": "se_epi"}


def canonical_station(raw: str) -> str:
    """Normalize StationName so ONE physical scanner is ONE domain.

    the site headers spell the same machine several ways -- "SITECBMR6", "CBMR6",
    "the site-CBMR6" -- so using the raw value would split one scanner into multiple
    domains AND let a leave-one-out split put the same machine in train and test
    (domain leakage, which inflates the held-out result). Strip punctuation and the
    institution prefix.
    """
    s = re.sub(r"[^A-Za-z0-9]", "", (raw or "")).upper()
    for pref in ("the site", "UC"):
        if s.startswith(pref) and len(s) > len(pref) + 2:
            s = s[len(pref):]
            break
    return s


def _series_number(picked) -> int | None:
    m = re.match(r"\s*(\d+)_", str(picked or ""))
    return int(m.group(1)) if m else None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve_stems(cdir) -> dict:
    """File stems present in the registered case folder; DWI bands keyed by b."""
    names = {n for n in os.listdir(cdir) if n.endswith(".nii.gz")}
    stems, bands = {}, {}
    if "T2W.nii.gz" in names:
        stems["T2W"] = "T2W"
    for n in names:
        if n.startswith("DWI") and n.endswith("_to_T2W.nii.gz"):
            stem = n[:-len(".nii.gz")]
            m = re.search(r"DWI_b(\d+)_to_T2W$", stem)
            if m:
                bands[int(m.group(1))] = stem     # b0000 -> 0, b1000 -> 1000
        elif n.startswith("ADC") and n.endswith("_to_T2W.nii.gz"):
            stems["ADC"] = n[:-len(".nii.gz")]
    if bands:
        stems["DWI_bands"] = bands
    return stems


def _read_tar_series(tar_path, wanted: set) -> dict:
    """SeriesNumber -> header dict, for the wanted series numbers.

    Reads ONE dicom per series directory and stops once every wanted series is
    found, so a big exam tar is not fully decompressed.
    """
    import pydicom
    out, seen = {}, set()
    with tarfile.open(tar_path) as tf:
        for i, m in enumerate(tf):
            if i > MAX_MEMBERS or wanted.issubset(out):
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
                continue                     # skip PACS secondary captures
            sn = getattr(d, "SeriesNumber", None)
            te = _num(getattr(d, "EchoTime", None))
            if sn is None or te is None or te < 0:
                continue                     # skip derived maps (TE=-1)
            out[int(sn)] = {
                "TR_ms": _num(getattr(d, "RepetitionTime", None)),
                "TE_ms": te,
                "vendor": str(getattr(d, "Manufacturer", "")).strip(),
                "model": str(getattr(d, "ManufacturerModelName", "")).strip(),
                "station": str(getattr(d, "StationName", "")).strip(),
                "serial": str(getattr(d, "DeviceSerialNumber", "")).strip(),
                "sw": str(getattr(d, "SoftwareVersions", "")).strip(),
                "field_T": _num(getattr(d, "MagneticFieldStrength", None)),
                "desc": str(getattr(d, "SeriesDescription", "")).strip(),
            }
    return out


def build_case(case, reg_root, tar_root, min_b_hi=500.0, max_b_lo=100.0):
    cdir = os.path.join(reg_root, case)
    plog = os.path.join(cdir, "pilot_log.json")
    tar_path = os.path.join(tar_root, f"{case}.tar")
    if not os.path.exists(plog):
        return None, "no pilot_log"
    if not os.path.exists(tar_path):
        return None, "no tar"
    try:
        picked = json.load(open(plog)).get("stats", {}).get("picked", {})
    except Exception as e:
        return None, f"pilot_log: {e}"
    stems = _resolve_stems(cdir)
    if "T2W" not in stems or "DWI_bands" not in stems or "ADC" not in stems:
        return None, "missing T2W/DWI/ADC file"
    if not os.path.exists(os.path.join(cdir, "prostate_mask.nii.gz")):
        return None, "no prostate_mask"

    sn_t2, sn_dwi = _series_number(picked.get("t2")), _series_number(picked.get("dwi"))
    if sn_t2 is None or sn_dwi is None:
        return None, "no series number in pilot_log"
    try:
        hdr = _read_tar_series(tar_path, {sn_t2, sn_dwi})
    except Exception as e:
        return None, f"tar: {e}"
    if sn_t2 not in hdr or sn_dwi not in hdr:
        return None, f"series {sn_t2}/{sn_dwi} not in tar"

    h2, hd = hdr[sn_t2], hdr[sn_dwi]
    bands = stems["DWI_bands"]
    hi = max(bands)
    # GUARD: the "DWI" channel must actually be diffusion-weighted. Siemens cases
    # registered a b=50 band as their DWI (exp(-50*D)~0.94, i.e. no diffusion
    # weighting), which silently made GE (b=1000) and Siemens (b=50) incomparable
    # inputs and confounded every cross-vendor result. Reject rather than model it.
    if hi < min_b_hi:
        return None, f"b_hi={int(hi)} < {int(min_b_hi)} (not diffusion-weighted)"
    contrasts = {
        "T2W": {"render": "tse_t2", "TR_ms": h2["TR_ms"], "TE_ms": h2["TE_ms"],
                "source": "dicom", "series_number": sn_t2, "file_stem": stems["T2W"]},
        "DWI": {"render": "se_epi", "TR_ms": hd["TR_ms"], "TE_ms": hd["TE_ms"],
                "b_values": [float(hi)], "source": "dicom",
                "series_number": sn_dwi, "file_stem": bands[hi]},
        "ADC": {"render": "anchor", "is_anchor": True, "source": "dicom",
                "file_stem": stems["ADC"]},
    }
    # Low band -> near-diffusion-free anchor at the DWI echo time. Accept any
    # b <= max_b_lo (not just exactly 0) and record the TRUE b: GE ships b=0 while
    # Siemens' lowest band is b=50 (exp(-50*D) ~ 0.94, i.e. ~6% attenuated). Keying
    # on b==0 would emit S0 for GE ONLY, which silently makes every S0-mode run
    # single-vendor -- the same structural confound as the b=50 DWI bug. The render
    # uses the recorded b, so the small attenuation is modeled rather than ignored.
    lo = min(bands)
    if lo <= max_b_lo and len(bands) >= 2 and lo != hi:
        contrasts["S0"] = {"render": "se_epi", "TR_ms": hd["TR_ms"], "TE_ms": hd["TE_ms"],
                           "b_values": [float(lo)], "source": "dicom",
                           "series_number": sn_dwi, "file_stem": bands[lo]}
    # canonicalize: the site headers carry both "SIEMENS" and "Siemens Healthineers",
    # which would otherwise split one vendor into two domains downstream
    return {"case_id": case, "center": "the site", "b_hi": float(hi),
            "b_lo": float(min(bands)),
            "vendor": canonical_vendor(h2["vendor"]), "vendor_raw": h2["vendor"],
            "scanner_model": h2["model"], "field_strength_T": h2["field_T"],
            # scanner_unit = the individual machine (finest domain); falls back to
            # model when the de-id stripped the identifiers
            "station_name": h2.get("station", ""), "device_serial": h2.get("serial", ""),
            "software_version": h2.get("sw", ""),
            "station_canon": canonical_station(h2.get("station", "")),
            # model#station: StationName alone is not unique (e.g. "MBIOMB" appears
            # for two different models), and DeviceSerialNumber is stripped by de-id
            "scanner_unit": (f"{h2['model']}#{canonical_station(h2.get('station',''))}"
                             if canonical_station(h2.get("station", "")) else h2["model"]),
            "linkage": "matched", "contrasts": contrasts}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site-root", required=True)
    ap.add_argument("--out", default="physharm/manifests_site")
    ap.add_argument("--reg-root", default=None,
                    help="registered-image root; point at the multi-b overlay to pick "
                         "up the recovered high-b bands (default <site-root>/registered)")
    ap.add_argument("--shard", default=None, help="N/K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-b-hi", type=float, default=500.0,
                    help="reject cases whose highest DWI band is below this "
                         "(a b=50 band is not diffusion-weighted and makes "
                         "vendors incomparable)")
    ap.add_argument("--max-b-lo", type=float, default=100.0,
                    help="a band at or below this becomes the S0 anchor (GE ships "
                         "b=0, Siemens b=50; keying on ==0 makes S0 GE-only)")
    args = ap.parse_args()

    # --reg-root lets us point at the multi-b OVERLAY (scratch) while still reading
    # DICOM headers from the original raw_dicom tars. Defaults to <site-root>/registered.
    reg_root = args.reg_root or os.path.join(args.site_root, "registered")
    tar_root = os.path.join(args.site_root, "raw_dicom")
    cases = sorted(d for d in os.listdir(reg_root)
                   if os.path.isdir(os.path.join(reg_root, d)))
    if args.limit:
        cases = cases[:args.limit]
    if args.shard:
        n, k = map(int, args.shard.split("/"))
        cases = [c for i, c in enumerate(cases) if i % k == n]
    os.makedirs(args.out, exist_ok=True)

    n_ok = n_skip = 0
    why_counts, index = {}, {"cases": []}
    for i, case in enumerate(cases):
        outp = os.path.join(args.out, f"{case}.json")
        if os.path.exists(outp):
            index["cases"].append({"case_id": case, "linkage": "matched"})
            continue
        man, why = build_case(case, reg_root, tar_root, args.min_b_hi, args.max_b_lo)
        if man is None:
            n_skip += 1
            why_counts[why] = why_counts.get(why, 0) + 1
            continue
        with open(outp, "w") as fh:
            json.dump(man, fh, indent=2)
        index["cases"].append({"case_id": case, "linkage": "matched"})
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  {i+1}/{len(cases)} written={n_ok} skipped={n_skip}", flush=True)

    idx = "index.json" if not args.shard else f"index_s{args.shard.split('/')[0]}.json"
    with open(os.path.join(args.out, idx), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"done: wrote {n_ok} manifests ({n_skip} skipped) -> {args.out}")
    if why_counts:
        print("  skip reasons:", dict(sorted(why_counts.items(),
                                             key=lambda kv: -kv[1])[:8]))


if __name__ == "__main__":
    main()
