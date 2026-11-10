r"""Is the traveling-arm result a scanner effect, or a per-scanner scale offset?

WHY THIS EXISTS. Phase 3c reported that cross-scanner pairs are indistinguishable from
unrelated men (cross/null 0.87-1.10) and concluded the identity verifier is circular, then
withdrew the traveling-subject framing. There is a second explanation with the same
signature, and it has to be excluded before the withdrawal stands.

    A FIXED PER-SCANNER OFFSET in any theta component adds a constant to |dtheta| for every
    cross-scanner pair involving that scanner, and adds NOTHING to same-unit pairs. That is
    exactly the observed pattern -- and unlike contamination, it can push cross ABOVE the
    null, which contamination provably cannot (contamination only pulls toward the null).

Note that cross > null is already reported for PD (0.7221 vs 0.6535). Contamination cannot
produce that, so something systematic is present. Whether it is physics or arithmetic is
what this script decides.

PRECEDENT, IN THIS DATASET. `physics_sanity.py` verified ADC units on DISCOVERY MR750w /
MR750 / SIGNA Premier -- all GE -- and found a 4.6% scanner-dependent spread even there. The
Siemens units were never checked. An independent pass over the raw registered ADC found
VERIO at ~1.2e4 against ~1.4e3 for every other model: a 10x offset, which is ln(10) = 2.3
in log units and would dwarf every effect in the traveling table.

WHAT IT REPORTS.
  1. per-scanner median of each RAW cached channel   -- where a units fault originates
  2. per-scanner median theta from THEIR estimator   -- where it lands
  3. the implied floor: for each scanner pair, the |dtheta| forced by the offset alone,
     with no biology, compared against the reported cross-scanner medians

WHICH DECISION THIS CHANGES:
  offset ~ 0            -> Phase 3c stands; the effect is physical, and cross > null means
                           the scanner effect exceeds between-patient variation. That is the
                           finding, and the withdrawal was too pessimistic.
  offset large          -> the traveling table is arithmetic. Fix units, re-run, and the
                           "not viable as constructed" conclusion is void.

    python audit/param_scale_by_scanner.py --cache-dir cache_shared \
        --manifest-dir manifests_site --header-cache runs/header_cache.json \
        --out runs/param_scale.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import sys

import numpy as np

# This script deliberately imports physharm rather than reimplementing the estimator: the
# whole point is to audit the code that actually produced the traveling-arm table. The other
# audit/ scripts are standalone, so the repo root is not on sys.path when run as
# `python audit/...`; put it there.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physharm.analysis.crlb_table import acquisition
from physharm.analysis.synthetic_arm import PARAMS
from physharm.analysis.traveling_arm import estimate_case, load_case
from physharm.data.manifests import read_manifests

RAW_KEYS = ("T2W", "DWI", "__adc")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--header-cache", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-vox", type=int, default=2000)
    ap.add_argument("--min-n", type=int, default=20,
                    help="scanners below this many exams are excluded from the reference "
                         "and from the pair table; their 'offset' is sampling noise")
    ap.add_argument("--anatomy", default=None,
                    help="runs/anatomy_repeats.json -- if given, reports which scanner "
                         "pairs actually populate the traveling cohort and the |dtheta| "
                         "floor those pairs carry")
    ap.add_argument("--flag-log", type=float, default=0.4,
                    help="flag a scanner whose median theta is this far (log units) from "
                         "the cohort median; 0.4 ~ a 50%% offset, 2.3 ~ 10x")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.header_cache):
        # Self-diagnose rather than send the user hunting: say what IS present and what
        # builds the missing file. traveling_subjects.py writes it (json.dump at ~line 199).
        print(f"header cache not found: {args.header_cache}\n")
        for d in ("runs", ".", os.path.dirname(args.header_cache) or "."):
            if os.path.isdir(d):
                names = sorted(os.listdir(d))
                js = [n for n in names if n.endswith(".json")]
                print(f"  {d}/  -- {len(names)} entries"
                      + (f", json: {', '.join(js[:12])}" if js else ", no .json"))
        cands = sorted(glob.glob("**/*header*.json", recursive=True)
                       + glob.glob("**/*by_case*.json", recursive=True))
        if cands:
            print(f"\n  candidates found: {cands[:5]}")
            print("  -> re-run with --header-cache <that path>")
        else:
            print("\n  nothing header-cache-like found. It is built by:")
            print("     python audit/traveling_subjects.py --raw-root <RAW> \\")
            print("         --cache runs/header_cache.json")
            print("  (that pass reads the tars, so it is the slow one)")
        raise SystemExit(1)
    by_case = json.load(open(args.header_cache))
    if not by_case:
        raise SystemExit("header cache is empty")
    n_model = sum(1 for v in by_case.values() if isinstance(v, dict) and v.get("model"))
    print(f"header cache: {len(by_case)} exams, {n_model} with a scanner model "
          f"({n_model/max(1,len(by_case)):.0%})")
    if not n_model:
        raise SystemExit("no 'model' field in the header cache -- wrong file?")
    acqs = {}
    for m in read_manifests(args.manifest_dir):
        a, _ = acquisition(m)
        if a is not None:
            acqs[m.get("case_id")] = a

    ids = sorted(os.path.basename(p)[:-4].replace("__", "/")
                 for p in glob.glob(os.path.join(args.cache_dir, "*.npz")))
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} cached cases; {len(acqs)} with acquisition params; "
          f"{len(by_case)} in header cache\n", flush=True)

    raw = defaultdict(lambda: defaultdict(list))
    th = defaultdict(lambda: defaultdict(list))
    n_done = skipped = 0
    for i, cid in enumerate(ids):
        model = (by_case.get(cid) or {}).get("model")
        acq = acqs.get(cid)
        if not model or acq is None:
            skipped += 1
            continue
        cached = load_case(args.cache_dir, cid)
        if cached is None:
            skipped += 1
            continue
        mask = cached.get("__mask")
        if mask is not None and (mask > 0.5).any():
            m = mask > 0.5
            for k in RAW_KEYS:
                a = cached.get(k)
                if a is not None and a.shape == mask.shape:
                    v = a[m]
                    v = v[np.isfinite(v) & (v > 0)]
                    if v.size:
                        raw[model][k].append(float(np.median(v)))
        t = estimate_case(cached, acq, n_vox=args.n_vox)
        if t:
            for k, v in t.items():
                th[model][k].append(v)
            n_done += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ids)}  fitted={n_done} skipped={skipped}", flush=True)
    print(f"\nfitted {n_done}, skipped {skipped}\n")
    if not th:
        raise SystemExit("nothing fitted -- check --cache-dir / --header-cache keys")

    order = sorted(th, key=lambda m: -len(th[m][PARAMS[0]]))
    solid = [m for m in order if len(th[m][PARAMS[0]]) >= args.min_n]
    if len(solid) < 2:
        print(f"** fewer than 2 scanners reach --min-n {args.min_n}; offsets will be noise")
        solid = order[:2]

    print("=== 1. RAW cached channels, median over ROI, per scanner ===")
    print("    (a units fault originates here; the cache is shared-normalised so T2W/DWI")
    print("     should sit near 1, while __adc is NOT normalised)")
    print(f"  {'scanner':24s} {'n':>5s} " + " ".join(f"{k:>13s}" for k in RAW_KEYS))
    for m in order:
        cells = []
        for k in RAW_KEYS:
            v = raw[m].get(k) or []
            cells.append(f"{np.median(v):13.4g}" if v else f"{'-':>13s}")
        print(f"  {m[:24]:24s} {len(th[m][PARAMS[0]]):>5d} " + " ".join(cells))
    # Only __adc is in physical units, so only __adc can have a units fault. T2W is forced
    # to 1 by the shared normalisation, and the DWI/T2W ratio is exp(-dTE/T2)*exp(-b*D):
    # it varies legitimately with b-value and TE across protocols, so a spread there is
    # physics. Flagging it was a false positive on the first run.
    med = [np.median(raw[m]["__adc"]) for m in order if raw[m].get("__adc")]
    if len(med) > 1:
        span = max(med) / max(1e-12, min(med))
        verdict = ("UNITS FAULT" if span > 3 else
                   "wide but plausible" if span > 1.5 else "consistent")
        print(f"  __adc spans {min(med):.4g} .. {max(med):.4g} ({span:.2f}x) -- {verdict}")
        if 1e-4 < np.median(med) < 1e-2:
            print("  __adc is in mm^2/s and physiological; the physics_sanity unit fix is applied.")
    print("  (DWI is NOT flagged: DWI/T2W = exp(-dTE/T2)exp(-b*D) varies with protocol by")
    print("   design, so a spread across scanners there is physics, not arithmetic.)")

    print("\n=== 2. THETA per scanner (log units, from their estimator) ===")
    print(f"  {'scanner':24s} {'n':>5s} " + " ".join(f"{p:>10s}" for p in PARAMS))
    # cohort reference from well-sampled scanners only: a 1-exam scanner must not move it
    cohort = {p: np.median([v for m in solid for v in th[m][p]]) for p in PARAMS}
    for m in order:
        print(f"  {m[:24]:24s} {len(th[m][PARAMS[0]]):>5d} "
              + " ".join(f"{np.median(th[m][p]):10.3f}" for p in PARAMS))
    print(f"  {'cohort median':24s} {'':>5s} "
          + " ".join(f"{cohort[p]:10.3f}" for p in PARAMS))

    print(f"\n=== 3. OFFSET from cohort median (log units) ===")
    print(f"    reference computed from the {len(solid)} scanners with n >= {args.min_n}: "
          f"{', '.join(solid)}")
    print(f"    rows below that threshold are shown but marked (n<{args.min_n}) -- with a")
    print("    handful of exams the 'offset' is sampling noise, not a scanner property.")
    print(f"  {'scanner':24s} " + " ".join(f"{p:>10s}" for p in PARAMS))
    off = {}
    for m in order:
        off[m] = {p: float(np.median(th[m][p]) - cohort[p]) for p in PARAMS}
        small = len(th[m][PARAMS[0]]) < args.min_n
        flags = "".join(" **" if (abs(off[m][p]) > args.flag_log and not small) else "   "
                        for p in PARAMS)
        tag = f"  (n<{args.min_n})" if small else ""
        print(f"  {m[:24]:24s} " + " ".join(f"{off[m][p]:10.3f}" for p in PARAMS) + flags + tag)
    print(f"  (** = |offset| > {args.flag_log} log units; ln(10)=2.303 would be a 10x fault)")

    print("\n=== 4. |dtheta| FORCED BY THE OFFSET ALONE, per scanner pair ===")
    print("    This is the floor a cross-scanner pair cannot go below, with zero biology.")
    print("    Compare against the reported cross-scanner medians; if the floor is of the")
    print("    same order, the traveling table is measuring arithmetic.")
    big = solid
    print(f"  {'pair':44s} " + " ".join(f"{p:>9s}" for p in PARAMS))
    shown = 0
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            d = {p: abs(off[a][p] - off[b][p]) for p in PARAMS}
            if max(d.values()) < 0.1:
                continue
            print(f"  {(a[:20]+' vs '+b[:20]):44s} "
                  + " ".join(f"{d[p]:9.3f}" for p in PARAMS))
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break
    if not shown:
        print("  (no scanner pair differs by >0.1 log units in any parameter)")
        print("  -> no per-scanner offset to explain the traveling result; Phase 3c stands.")

    # ---- 5. what the traveling cohort is actually MADE OF ------------------
    # The traveling table's medians depend entirely on which scanner pairs populate it,
    # and that composition is not reported anywhere. A cohort dominated by an
    # offset-free pair measures almost no scanner difference; one dominated by an
    # offset-carrying pair measures mostly the offset. This distinguishes them.
    if args.anatomy and os.path.exists(args.anatomy):
        from physharm.analysis.traveling_arm import classify_pairs
        cls = classify_pairs(None, args.anatomy, by_case)
        print(f"\n=== 5. TRAVELING COHORT COMPOSITION ===")
        print(f"    {len(cls['cross'])} cross-scanner pairs, {len(cls['same'])} same-unit "
              f"(tolerance {cls.get('tolerance')}, false-match {cls.get('false_match_rate')})")
        comp = defaultdict(int)
        for a, b in cls["cross"]:
            ma = (by_case.get(a) or {}).get("model")
            mb = (by_case.get(b) or {}).get("model")
            if ma and mb:
                comp[tuple(sorted([ma, mb]))] += 1
        if not comp:
            print("    (no cross pairs with both models known)")
        else:
            tot = sum(comp.values())
            print(f"\n  {'scanner pair':46s} {'pairs':>6s} {'share':>6s}  "
                  + " ".join(f"{p:>8s}" for p in PARAMS))
            floors = {p: [] for p in PARAMS}
            for (a, b), n in sorted(comp.items(), key=lambda kv: -kv[1])[:12]:
                if a in off and b in off:
                    d = {p: abs(off[a][p] - off[b][p]) for p in PARAMS}
                    for p in PARAMS:
                        floors[p] += [d[p]] * n
                    cells = " ".join(f"{d[p]:8.3f}" for p in PARAMS)
                else:
                    cells = " ".join(f"{'  n/a':>8s}" for _ in PARAMS)
                print(f"  {(a[:21]+' vs '+b[:21]):46s} {n:>6d} {n/tot:>5.0%}  {cells}")
            print("\n  IMPLIED FLOOR -- median |dtheta| the cohort carries from offsets alone,")
            print("  with zero biology, weighted by the actual pair composition:")
            print(f"  {'':46s} {'':>6s} {'':>6s}  "
                  + " ".join(f"{p:>8s}" for p in PARAMS))
            print(f"  {'implied floor':46s} {'':>6s} {'':>6s}  "
                  + " ".join(f"{np.median(floors[p]) if floors[p] else float('nan'):8.3f}"
                             for p in PARAMS))
            print(f"  {'reported cross median (Phase 3c)':46s} {'':>6s} {'':>6s}  "
                  + f"{0.7221:8.3f} {0.5091:8.3f} {0.6458:8.3f} {0.1129:8.3f}")
            print("\n  If the implied floor approaches the reported median for a parameter,")
            print("  that column is measuring the offset rather than a physical difference.")
            print("  If it is far below, the reported value is genuine and the offset is not")
            print("  the explanation for that parameter.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"theta_median": {m: {p: float(np.median(th[m][p])) for p in PARAMS}
                                    for m in order},
                   "offset": off, "cohort_median": {p: float(v) for p, v in cohort.items()},
                   "raw_median": {m: {k: float(np.median(v)) for k, v in raw[m].items() if v}
                                  for m in order},
                   "n": {m: len(th[m][PARAMS[0]]) for m in order}},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
