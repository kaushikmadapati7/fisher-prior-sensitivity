r"""Where does the per-scanner ADC scale come from? Source vs manifest vs cache. (#56)

MAGNETOM Vida reached `cache_shared` at an ROI-median ADC of 0.00056 mm^2/s against a GE cohort
at ~0.00146, and its source data is normal (1355 x10^-6 read straight off the registered
volumes). Something between the two applies a per-case factor. This prints all three numbers
side by side so the transform is visible instead of inferred.

WHAT WE ALREADY KNOW FROM THE CODE, and what this is meant to settle:

  * `dataset_physics._preprocess` scales ADC by `manifest["adc_rescale"]` (default 1.0), a
    per-case factor documented as a Philips fix worth ~0.46. Vida's cache/source ratio is
    0.413. That is close enough that a Siemens scanner receiving the Philips correction is the
    leading hypothesis, but NOTHING in this repository ever writes `adc_rescale` -- so if the
    cluster manifests carry it, they were built by something outside version control, and this
    script is how we find out.
  * `to_physical_adc` also clips raw ADC at 3000 before scaling. That explains a compressed
    Verio (source ~11980) but not a low Vida (source ~1355, well under the clip).

Reads the cache, the manifests and the source; prints per scanner. Nothing leaves the cluster.

    python -m audit.adc_scale_provenance --cache-dir cache_shared \
        --manifest-dir manifests_site \
        --image-root <image-root>
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from physharm.analysis.crlb_table import canonical_scanner
from physharm.data.manifests import read_manifests
from physharm.data.units import ADC_STORED_TO_MM2S, roi_scale_problem, scanner_scale_report


_FIND_CACHE = {}


def _find_source(image_root, stem, case_id=""):
    """Locate `<stem>.nii.gz` under the image root. Returns a path or None.

    The first version guessed at manifest keys that do not exist and silently produced `nan`
    for every scanner, which is why two rounds of this diagnostic could not compare source
    against cache at all. It now tries the likely layouts and then falls back to one cached
    recursive walk, because the alternative is another round trip to find out.
    """
    if not (image_root and stem):
        return None
    for cand in (os.path.join(image_root, case_id, f"{stem}.nii.gz"),
                 os.path.join(image_root, f"{stem}.nii.gz")):
        if os.path.exists(cand):
            return cand
    if "index" not in _FIND_CACHE:
        idx = {}
        for root, _dirs, files in os.walk(image_root):
            for fn in files:
                if fn.endswith(".nii.gz"):
                    idx.setdefault(fn, os.path.join(root, fn))
        _FIND_CACHE["index"] = idx
    return _FIND_CACHE["index"].get(f"{stem}.nii.gz")


def _source_medians(path):
    """(whole-volume median, in-mask median, background fraction) from the REGISTERED ADC.

    THE MASK HAS TO COME FROM THE SAME GRID. `ADC_to_T2W.nii.gz` is already resampled onto the
    T2W grid, and so is `prostate_mask.nii.gz`, which sits beside it in the case folder -- so an
    in-mask source measurement needs no geometry matching at all. The previous version indexed
    the registered ADC with the CACHED mask, which lives on the resampled cache grid, so the
    shapes never matched and every scanner's source in-mask column read `nan`. The ADC file was
    right; the mask was not.

    This is what accounts for the disagreement in Phase 13. A whole-volume median takes in
    background, pelvic muscle, fat and bowel; an in-mask median takes prostate. Measured on
    SIGNA Premier those are 702 and about 1460 -- the same volume, two statistics, a factor of
    two apart. The whole-volume number is not a proxy for prostate ADC and should never have
    been read as one.

    `bg_frac` is the share of positive voxels lying OUTSIDE the mask. If one scanner carries a
    much larger positive background -- bigger field of view, different padding, a different zero
    convention -- its whole-volume median is dragged down for reasons that have nothing to do
    with diffusion, which is the alternative to the source being genuinely anomalous.
    """
    try:
        import nibabel as nib
    except ImportError:
        return None, None, None
    try:
        a = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    except Exception:
        return None, None, None
    pos = a > 0
    whole = float(np.median(a[pos])) if pos.any() else None

    mpath = os.path.join(os.path.dirname(path), "prostate_mask.nii.gz")
    if not os.path.exists(mpath):
        return whole, None, None
    try:
        m = np.asarray(nib.load(mpath).dataobj, dtype=np.float32)
    except Exception:
        return whole, None, None
    if m.shape != a.shape:
        return whole, "shape", None
    inside = m > 0.5
    v = a[inside & pos]
    inm = float(np.median(v)) if v.size >= 50 else None
    bg = float((pos & ~inside).sum() / max(pos.sum(), 1))
    return whole, inm, bg


def _check_image_root(root):
    """Refuse to run with an --image-root that will silently yield nothing.

    Two rounds of this diagnostic reported `nan` for every scanner's source column and neither
    said why -- the first because it looked for manifest keys that do not exist, the second
    because the path was a guess. A third silent round is not acceptable, so a root that is
    missing, or that holds no NIfTI within a few levels, stops the run and says so.
    """
    if not os.path.isdir(root):
        raise SystemExit(
            f"--image-root {root!r} is not a directory.\n"
            f"  Point it at the directory holding the registered volumes for your cohort.\n"
            f"  The '<image-root>' in this file's docstring is a placeholder, not a path.")
    for depth, (cur, _d, files) in enumerate(os.walk(root)):
        if any(f.endswith(".nii.gz") for f in files):
            return
        if depth > 5000:
            break
    raise SystemExit(
        f"--image-root {root!r} exists but contains no .nii.gz. Point it at the directory "
        f"holding the registered volumes, not its parent.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--image-root", default=None,
                    help="registered volumes; omit to skip the source column")
    ap.add_argument("--limit-per-scanner", type=int, default=8,
                    help="source volumes to read per scanner (they are slow)")
    args = ap.parse_args()

    if args.image_root:
        _check_image_root(args.image_root)
    mans = read_manifests(args.manifest_dir)
    cache, rescale, src, seen = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
    outside_med, src_in = defaultdict(list), defaultdict(list)
    cache_whole, bg_frac = defaultdict(list), defaultdict(list)
    shape_mismatch = [0]
    missing = 0
    for m in mans:
        sc = canonical_scanner(m.get("scanner_model"))
        rescale[sc].append(float(m.get("adc_rescale", 1.0)))
        f = os.path.join(args.cache_dir, f"{m.get('case_id')}.npz")
        if not os.path.exists(f):
            missing += 1
            continue
        try:
            with np.load(f) as z:
                adc, mask = z["__adc"], z["__mask"]
        except (OSError, KeyError):
            continue
        whole_c = adc[adc > 0]
        if whole_c.size >= 50:
            cache_whole[sc].append(float(np.median(whole_c)))
        inside = adc[mask > 0.5]
        inside = inside[inside > 0]
        outside = adc[mask <= 0.5]
        outside = outside[outside > 0]
        if inside.size >= 50:
            cache[sc].append(float(np.median(inside)))
        if outside.size >= 50:
            outside_med[sc].append(float(np.median(outside)))
        if args.image_root and seen[sc] < args.limit_per_scanner:
            # manifest schema: contrasts.ADC.file_stem, resolved under the image root
            stem = ((m.get("contrasts") or {}).get("ADC") or {}).get("file_stem")
            if stem:
                fp = _find_source(args.image_root, stem, m.get("case_id", ""))
                if fp:
                    whole, inm, bg = _source_medians(fp)
                    if whole is not None:
                        src[sc].append(whole)
                        seen[sc] += 1
                    if inm == "shape":
                        shape_mismatch[0] += 1
                    elif inm is not None:
                        src_in[sc].append(inm)
                    if bg is not None:
                        bg_frac[sc].append(bg)

    print(f"{len(mans)} manifests, {missing} without a cached file\n")
    print(f"{'scanner':<22} {'n':>5} {'cache in':>10} {'SRC IN':>10} {'cache/src':>10} "
          f"{'src whole':>10} {'bg frac':>8} {'vs cohort':>10}")
    rows, bad = scanner_scale_report(cache)
    order = [r["scanner"] for r in rows]
    for sc in order:
        r = next(x for x in rows if x["scanner"] == sc)
        rs = sorted(set(round(v, 4) for v in rescale.get(sc, [1.0])))
        s = float(np.median(src[sc])) if src.get(sc) else float("nan")
        ratio = r["median"] / (s * ADC_STORED_TO_MM2S) if s == s and s > 0 else float("nan")
        out = float(np.median(outside_med[sc])) if outside_med.get(sc) else float("nan")
        si = float(np.median(src_in[sc])) * ADC_STORED_TO_MM2S if src_in.get(sc) else float("nan")
        sw = s * ADC_STORED_TO_MM2S if s == s else float("nan")
        cs = r["median"] / si if si == si and si > 0 else float("nan")
        bf = float(np.median(bg_frac[sc])) if bg_frac.get(sc) else float("nan")
        print(f"{r['scanner']:<22} {r['n']:>5} {r['median']:>10.5f} {si:>10.5f} {cs:>10.3f} "
              f"{sw:>10.5f} {bf:>8.3f} {r['ratio']:>9.3f}x")
    print("\n  READ cache/src: cache in-mask over SOURCE in-mask, both prostate, both on the")
    print("  T2W grid. ~1.0 means the pipeline preserved prostate ADC and any oddity is in the")
    print("  source; well below 1.0 means the pipeline changed it.")
    print("  src whole is the WHOLE-VOLUME median and is NOT a proxy for prostate ADC -- it")
    print("  takes in background, muscle, fat and bowel. On Premier the two differ by about 2x")
    print("  (702 vs ~1460), which is the whole of the Phase 13 disagreement.")
    print("  bg frac is the share of positive voxels outside the mask. A scanner with much more")
    print("  positive background has its whole-volume median dragged down for reasons that have")
    print("  nothing to do with diffusion.")
    if shape_mismatch[0]:
        print(f"\n  source mask shape mismatched the ADC for {shape_mismatch[0]} cases.")
    print("\n  * settled: adc_rescale is [1.0] on every scanner; the 3000 clip cannot apply;")
    print("    and the earlier in/out column was never a control.")
    if bad:
        print(f"\n  FLAGGED (>{1.35}x from cohort): {', '.join(bad)}")
    for sc in order:
        r = next(x for x in rows if x["scanner"] == sc)
        p = roi_scale_problem(r["median"])
        if p:
            print(f"    {r['scanner']}: {p}")


if __name__ == "__main__":
    main()
