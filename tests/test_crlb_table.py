"""Tests for the per-case CRLB table (issue #6).

Run with pytest, or standalone:  python tests/test_crlb_table.py
Numpy only. Builds synthetic manifests with KNOWN structure so the ICC and the ordering
sweep are checked against ground truth, not just executed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from physharm.analysis.crlb_table import (acquisition, canonical_scanner,
                                          predictor_stability, rows_for_case,
                                          spearman)
from physharm.analysis.stats import icc
from physharm.physics import fisher as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _man(case, model, tr_t2, te_t2, b_lo=0.0, vendor="GE", patient=None):
    con = {"T2W": {"render": "tse_t2", "TR_ms": tr_t2, "TE_ms": te_t2},
           "DWI": {"render": "se_epi", "TR_ms": 4000.0, "TE_ms": 70.0, "b_values": [1000.0]},
           "ADC": {"render": "anchor", "is_anchor": True}}
    if b_lo is not None:
        con["S0"] = {"render": "se_epi", "TR_ms": 4000.0, "TE_ms": 70.0, "b_values": [b_lo]}
    return {"case_id": case, "patient_id": patient or case, "scanner_model": model,
            "vendor": vendor, "scanner_unit": f"{model}#S1", "b_hi": 1000.0,
            "b_lo": b_lo, "contrasts": con}


def test_acquisition_parses_the_builder_schema():
    acq, why = acquisition(_man("c", "SIGNA Premier", 8136.0, 153.0))
    assert why is None
    assert acq["TR_T2W"] == 8136.0 and acq["TE_T2W"] == 153.0
    assert acq["TR_DWI"] == 4000.0 and acq["TE_DWI"] == 70.0
    assert acq["b_hi"] == 1000.0 and acq["b_lo"] == 0.0


def test_missing_contrast_is_reported_not_guessed():
    m = _man("c", "M", 8000.0, 150.0)
    del m["contrasts"]["DWI"]
    acq, why = acquisition(m)
    assert acq is None and "DWI" in why


def test_a2_absent_when_no_low_b_band():
    """A2 needs a registered low-b band; ~0.5% of cases lack one and must be skipped,
    not silently given a fabricated b_lo."""
    m = _man("c", "M", 8000.0, 150.0, b_lo=None)
    m.pop("b_lo", None)
    r, why = rows_for_case(m)
    assert why is None
    assert "A2" not in r["arms"] and "A1" in r["arms"]


def test_every_arm_produces_the_expected_structure():
    r, _ = rows_for_case(_man("c", "M", 8136.0, 153.0))
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        d = r["arms"][arm]
        assert set(d["crlb"]) == set(F.PARAMS)
        assert d["rank"] >= 1 and len(d["eigenvalues"]) == 4
        assert all(np.isfinite(v) and v > 0 for v in d["crlb"].values())


def test_each_arm_uses_a_consistent_measurement_set():
    """No arm may silently change its row count between cases. THE invariant here.

    `audit/crlb_spread.py` built rows as `[t2w, dwi_hi, adc]` plus a low-b row *if present*,
    so it used 3 rows for the 23 cases lacking a low-b band and 4 rows for the other 4,650.
    Multi-b availability is vendor-linked (MAGNETOM Vida n=20 and Verio n=3 have none), so
    that assigned a different estimator to a scanner-correlated subset and manufactured
    between-scanner variance out of the analysis itself -- inflating ICC. Reproduced
    synthetically at +0.047, and it is why the audit reported 0.266 against 0.224/0.179 here.

    This design avoids it by construction: an arm either applies to a case or is omitted for
    it, never silently degraded.
    """
    with_s0 = _man("a", "M", 8000.0, 150.0)
    no_s0 = _man("b", "M", 8000.0, 150.0, b_lo=None)
    no_s0.pop("b_lo", None)
    ra, rb = rows_for_case(with_s0)[0], rows_for_case(no_s0)[0]
    for arm in ("A1", "A3", "A4", "A5"):
        assert ra["arms"][arm]["rank"] == rb["arms"][arm]["rank"], \
            f"{arm} must not depend on low-b availability"
    assert "A2" in ra["arms"] and "A2" not in rb["arms"], \
        "A2 must be OMITTED, never silently degraded to A1"


def test_mixing_measurement_sets_biases_icc_in_either_direction():
    """The artifact does NOT always inflate -- and that is the point.

    Giving a scanner-correlated subset a different measurement set injects a scanner-aligned
    offset into the predictor. Whether it raises or lowers ICC depends on whether that offset
    reinforces or opposes the real group separation; both signs are reachable (measured:
    -0.48 to +0.31 across small-group protocol means). The resulting number is therefore
    partly an artifact of the ANALYSIS, which is why an arm must never silently degrade.

    An earlier version of this test asserted "mixing inflates", generalizing from a single
    configuration. It does not.
    """
    def build(tr_small):
        rng = np.random.default_rng(5)
        mans = []
        for model, n, tr0 in (("BIG", 400, 8000.0), ("SMALL", 60, tr_small)):
            for i in range(n):
                has_s0 = (model == "BIG") or (i % 3 != 0)   # SMALL keeps partial coverage
                m = _man(f"{model}{i}", model,
                         float(np.clip(rng.normal(tr0, 1500), 2500, 11000)), 150.0,
                         b_lo=0.0 if has_s0 else None)
                if not has_s0:
                    m.pop("b_lo", None)
                mans.append(m)
        rs = [rows_for_case(m)[0] for m in mans]
        sel = [r for r in rs if "A2" in r["arms"]]
        consistent = icc([r["arms"]["A2"]["crlb"]["T2"] for r in sel],
                         [r["scanner_model"] for r in sel])
        mixed = icc([r["arms"]["A2" if "A2" in r["arms"] else "A1"]["crlb"]["T2"] for r in rs],
                    [r["scanner_model"] for r in rs])
        return consistent, mixed

    deltas = [m - c for c, m in (build(tr) for tr in (4000.0, 6000.0, 8000.0, 10000.0))]
    assert max(abs(d) for d in deltas) > 0.1, "the artifact must be material"
    assert min(deltas) < 0 < max(deltas), f"both signs must be reachable: {deltas}"


def test_outliers_depress_icc():
    """Direction check for the reconciliation: wrong-sequence cases inflate within-scanner
    variance, so gating them out RAISES ICC. Measured, not assumed.

    Gating itself now lives in physharm.data.exclusions (issue #3) so there is exactly one
    implementation; this only pins the ICC consequence.
    """
    rng = np.random.default_rng(11)
    mans = []
    for i in range(400):
        is_a = bool(i % 2)
        tr = float(np.clip(rng.normal(7000.0 if is_a else 4300.0, 700.0), 2500.0, 11000.0))
        mans.append(_man(f"c{i}", "A" if is_a else "B", tr, 130.0))
    for i in range(40):                       # 10% wrong-sequence contamination
        mans.append(_man(f"x{i}", "A" if i % 2 else "B", float(rng.uniform(600, 1500)), 130.0))

    from physharm.data import exclusions as EX

    def measure(gate):
        sel = EX.apply(mans)[0] if gate else mans
        rs = [r for r in (rows_for_case(m)[0] for m in sel) if r]
        return icc([r["arms"]["A1"]["crlb"]["T2"] for r in rs],
                   [r["scanner_model"] for r in rs])
    ungated, gated = measure(None), measure((2000.0, 12000.0))
    assert gated > ungated, f"gating should raise ICC: {ungated:.3f} -> {gated:.3f}"


def test_scanner_names_are_canonicalized():
    """Ground rule 5. Grouping on the raw string splits one model into several groups and
    biases ICC downward -- the defect behind the 0.204-vs-0.266 gap on the first cohort run."""
    assert canonical_scanner("SIGNA Premier") == canonical_scanner("Signa premier")
    assert canonical_scanner("DISCOVERY MR750w") == canonical_scanner("Discovery MR750W")
    assert canonical_scanner("GE MEDICAL SYSTEMS SIGNA Premier") == canonical_scanner("SIGNA Premier")
    assert canonical_scanner("MAGNETOM Vida") != canonical_scanner("MAGNETOM Sola")
    assert canonical_scanner(None) == "UNKNOWN"
    assert canonical_scanner("") == "UNKNOWN"


def test_splitting_a_model_by_spelling_lowers_icc():
    """Direction of the bias, so the fix is not taken on faith.

    Requires WITHIN-group protocol variance to show up at all -- with a deterministic
    protocol, splitting a group changes nothing and ICC stays 1.0. Within-scanner protocol
    variation is exactly the site situation and the premise of the whole study, so the bias
    is live here: measured +0.03 at realistic spread, growing to +0.06 at 2x.
    """
    rng = np.random.default_rng(7)
    mans, raw_groups = [], []
    for i in range(400):
        is_signa = bool(i % 2)
        # protocol tracks the model but VARIES within it
        tr = float(rng.normal(7000.0 if is_signa else 4200.0, 800.0))
        tr = float(np.clip(tr, 2500.0, 11000.0))
        spelling = "Signa premier" if (is_signa and i % 4 == 1) else (
            "SIGNA Premier" if is_signa else "DISCOVERY MR750")
        mans.append(_man(f"c{i}", spelling, tr, 130.0))
        raw_groups.append(spelling)

    vals = [rows_for_case(m)[0]["arms"]["A1"]["crlb"]["T2"] for m in mans]
    canon = [rows_for_case(m)[0]["scanner_model"] for m in mans]   # canonicalized by the code
    icc_raw, icc_canon = icc(vals, raw_groups), icc(vals, canon)
    assert len({canonical_scanner(g) for g in raw_groups}) == 2, "two real machines"
    assert len(set(raw_groups)) == 3, "three spellings"
    assert icc_canon > icc_raw, f"canonicalized {icc_canon:.4f} !> raw {icc_raw:.4f}"


def test_icc_recovers_known_structure():
    """The number the whole study rests on. Check both extremes against ground truth."""
    # protocol a pure function of scanner -> ICC 1 (the Bao lookup-table case)
    conf = [_man(f"c{i}", "A" if i % 2 else "B", 7300.0 if i % 2 else 3540.0, 135.0)
            for i in range(80)]
    v = [rows_for_case(m)[0]["arms"]["A1"]["crlb"]["T2"] for m in conf]
    g = [m["scanner_model"] for m in conf]
    assert icc(v, g) > 0.95, "a lookup-table cohort must read as CONFOUNDED"

    # protocol drawn independently of scanner -> ICC ~ 0 (the site case)
    rng = np.random.default_rng(0)
    free = [_man(f"c{i}", "A" if i % 2 else "B",
                 float(rng.uniform(4000, 9000)), float(rng.uniform(100, 170)))
            for i in range(200)]
    v = [rows_for_case(m)[0]["arms"]["A1"]["crlb"]["T2"] for m in free]
    g = [m["scanner_model"] for m in free]
    assert icc(v, g) < 0.15, "independently drawn protocols must read as separable"


def _cohort(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return [_man(f"c{i}", "M", float(rng.uniform(4000, 9000)), float(rng.uniform(100, 170)))
            for i in range(n)]


def test_predictor_survives_the_unmeasurable_noise_weighting():
    """At the operating point, the quantity we could NOT measure does not threaten the
    regression. This is the load-bearing stability claim.

    Note this is NOT the within-case four-parameter ordering. Sweeping prior_sd from 1 to
    100 leaves that ordering at `D < T2 < PD < T1` for a typical protocol while INVERTING
    the across-case ranking, so checking only the ordering would certify a predictor whose
    regression sign flips. See test_small_prior_inverts_the_predictor.
    """
    st = predictor_stability(_cohort())
    assert min(st["worst_spearman"].values()) >= 0.85, st["worst_spearman"]


def test_operating_point_is_not_prior_dominated():
    """The justification for fixing prior_sd rather than sweeping it."""
    st = predictor_stability(_cohort())
    assert st["median_fraction_prior_dominated"] == 0.0


def test_usable_prior_is_a_band_not_a_floor():
    """Too strong saturates the bound; too weak leaves the near-null direction unregularized.
    Both ends degrade, which is why prior_sd is a fixed analysis choice with a reported
    sensitivity curve rather than a swept nuisance parameter."""
    st = predictor_stability(_cohort())
    c = st["prior_curve"]
    assert min(c[1.0].values()) < 0, "prior_sd=1 must show saturation/inversion"
    assert min(c[100.0].values()) < min(c[10.0].values()), "prior_sd=100 must degrade too"
    assert min(c[10.0].values()) > 0.99, "the operating point is the reference"


def test_small_prior_inverts_the_predictor():
    """REGRESSION TEST. prior_sd is not a free knob; below the Fisher scale the bound
    saturates and the across-case ranking inverts. Documents WHY prior_grid starts at 3."""
    acqs = [acquisition(m)[0] for m in _cohort()]  # noqa: E501
    ref = [F.identifiability(a, "A1", prior_sd=10.0)["crlb"] for a in acqs]
    tiny = [F.identifiability(a, "A1", prior_sd=1.0)["crlb"] for a in acqs]
    r_pd = spearman([d["PD"] for d in ref], [d["PD"] for d in tiny])
    assert r_pd < 0, f"expected inversion at prior_sd=1, got Spearman {r_pd:.3f}"
    dom = F.prior_dominance(F.fisher_matrix(F.measurement_rows("A1", acqs[0])), 1.0)
    assert dom["fraction_prior_dominated"] > F.prior_dominance(
        F.fisher_matrix(F.measurement_rows("A1", acqs[0])), 100.0)["fraction_prior_dominated"]


def test_within_case_ordering_always_ends_in_T1():
    """What IS invariant: T1 is the sloppiest parameter under every setting tried."""
    st = predictor_stability(_cohort())
    assert all(o[-1] == "T1" for o in st["orderings"]), list(st["orderings"])


def test_crlb_varies_with_te_more_than_tr():
    """TE is the strong lever (~111% vs ~13% over the real the site ranges), which is why the
    design keys on TE heterogeneity. Guards against a sign/derivation slip."""
    lo = rows_for_case(_man("a", "M", 8136.0, 120.0))[0]["arms"]["A1"]["crlb"]["T2"]
    hi = rows_for_case(_man("b", "M", 8136.0, 170.0))[0]["arms"]["A1"]["crlb"]["T2"]
    tr_lo = rows_for_case(_man("c", "M", 4000.0, 153.0))[0]["arms"]["A1"]["crlb"]["T2"]
    tr_hi = rows_for_case(_man("d", "M", 7986.0, 153.0))[0]["arms"]["A1"]["crlb"]["T2"]
    assert abs(hi - lo) / lo > abs(tr_hi - tr_lo) / tr_lo


def test_end_to_end_cli_writes_json_and_csv():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        md = os.path.join(d, "manifests")
        os.makedirs(md)
        for i in range(60):
            m = _man(f"c{i:03d}", ["SIGNA Premier", "DISCOVERY MR750"][i % 2],
                     float(rng.uniform(4000, 9000)), float(rng.uniform(100, 170)),
                     patient=f"p{i//2:03d}")
            json.dump(m, open(os.path.join(md, f"{m['case_id']}.json"), "w"))
        out, csv = os.path.join(d, "t.json"), os.path.join(d, "t.csv")
        env = dict(os.environ, PYTHONPATH=REPO)
        r = subprocess.run([sys.executable, "-m", "physharm.analysis.crlb_table",
                            "--manifest-dir", md, "--out", out, "--csv", csv],
                           capture_output=True, text=True, cwd=REPO, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        a = json.load(open(out))
        assert a["n_cases"] == 60
        assert min(a["worst_spearman"].values()) >= 0.85
        assert a["prior_sd"] == 10.0
        assert a["icc"]["A1"]["T2"] is not None
        assert a["a5_counterfactual"] is True, "A5 must be flagged, not silently reported"
        assert len(open(csv).read().splitlines()) == 61
        assert "COUNTERFACTUAL" in r.stdout


if __name__ == "__main__":
    import traceback
    npass = nfail = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); npass += 1; print(f"  PASS {n}")
            except Exception:
                nfail += 1; print(f"  FAIL {n}"); traceback.print_exc(limit=2)
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
