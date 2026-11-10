"""Tests for the exclusion list (issue #3).

Run with pytest, or standalone:  python tests/test_exclusions.py
Numpy only.

These check the gates AND the reporting, because the reporting is the point: a gate that
fires unevenly across scanners is a finding, and this cohort's b_hi gate is 89.8% Siemens.
"""
from __future__ import annotations

import json
import os
import tempfile

from physharm.data import exclusions as E


def _man(case="c", model="SIGNA Premier", tr=6000.0, te=150.0, b_hi=1000.0,
         t2w=True, dwi=True, adc=True):
    con = {}
    if t2w:
        con["T2W"] = {"TR_ms": tr, "TE_ms": te}
    if dwi:
        con["DWI"] = {"TR_ms": 4000.0, "TE_ms": 70.0, "b_values": [b_hi]}
    if adc:
        con["ADC"] = {"render": "anchor"}
    return {"case_id": case, "scanner_model": model, "b_hi": b_hi, "contrasts": con}


def test_good_case_passes():
    assert E.gate_case(_man()) is None


def test_wrong_sequence_t2w_is_gated():
    """onboarding 4: T2W-labelled series below ~2000 ms are a different sequence, and
    gating them moved ICC(CRLB T2) 0.204 -> 0.224 on the real cohort."""
    assert "different sequence" in E.gate_case(_man(tr=900.0))
    assert "different sequence" in E.gate_case(_man(tr=15000.0))


def test_non_diffusion_weighted_band_is_gated():
    """A b=50 band is not diffusion-weighted; keeping it makes GE and Siemens incomparable."""
    assert "not diffusion-weighted" in E.gate_case(_man(b_hi=50.0))
    assert E.gate_case(_man(b_hi=1500.0)) is None


def test_missing_contrast_is_gated():
    for kw in ({"t2w": False}, {"dwi": False}, {"adc": False}):
        assert "missing" in E.gate_case(_man(**kw))


def test_mislabelled_bands_are_gated_by_id():
    """Case IDs come from audit/physics_sanity.py, which measures the ordering on images."""
    assert E.gate_case(_man(case="bad1"), bad_bands=("bad1", "bad2")) is not None
    assert E.gate_case(_man(case="ok"), bad_bands=("bad1",)) is None


def test_no_case_is_dropped_silently():
    mans = [_man("a"), _man("b", tr=900.0), _man("c", b_hi=50.0), _man("d", t2w=False)]
    kept, rep = E.apply(mans)
    assert len(kept) == 1
    assert rep["n_in"] == 4 and rep["n_kept"] == 1
    assert len(rep["excluded"]) == 3
    assert all(r["reason"] and r["case_id"] for r in rep["excluded"])


def test_uneven_gate_is_visible_per_scanner():
    """THE reason this module exists. A gate concentrated on one vendor must not hide in a
    total -- on the real cohort the TR gate is 7.5% on SIGNA Premier and ~0 elsewhere.
    """
    mans = ([_man(f"ge{i}", "SIGNA Premier", b_hi=1000.0) for i in range(90)] +
            [_man(f"si{i}", "MAGNETOM Vida", b_hi=50.0) for i in range(60)])
    kept, rep = E.apply(mans)
    assert len(kept) == 90
    ge = rep["by_scanner"]["SIGNA Premier"]
    si = rep["by_scanner"]["MAGNETOM Vida"]
    assert ge.get("KEPT") == 90 and "b_hi < 500 (not diffusion-weighted)" not in ge
    assert si["b_hi < 500 (not diffusion-weighted)"] == 60 and si.get("KEPT", 0) == 0
    text = E.format_report(rep)
    assert "MAGNETOM Vida" in text and "100%" in text, "the 100% rate must be legible"
    assert "UNEVEN GATES" in text, "the disproportion must be called out, not just tabulated"


def test_uneven_detector_ignores_a_handful_of_cases():
    """Leave-one-out makes ANY exclusive hit infinite, so a couple of cases would flag.

    The real dry run reported "DISCOVERY MR750w 0.3% vs 0.0% elsewhere (inf)" off exactly two
    mislabelled-band cases -- arithmetically true, evidentially nothing. A disproportion needs
    a few events behind it before it is called a finding.
    """
    mans = ([_man(f"a{i}", "MODEL_A", tr=6000.0) for i in range(800)] +
            [_man(f"b{i}", "MODEL_B", tr=6000.0) for i in range(800)])
    mans[0] = _man("a0", "MODEL_A", tr=900.0)          # a single offender, exclusive to A
    mans[1] = _man("a1", "MODEL_A", tr=900.0)          # ...and a second
    _, rep = E.apply(mans)
    text = E.format_report(rep)
    assert "UNEVEN GATES" not in text, "2 cases must not be reported as a finding"
    # ...but a real disproportion still is
    for i in range(2, 40):
        mans[i] = _man(f"a{i}", "MODEL_A", tr=900.0)
    _, rep2 = E.apply(mans)
    assert "UNEVEN GATES" in E.format_report(rep2)


def test_uneven_detector_is_leave_one_out():
    """A model's own failures inflate the cohort rate it would be judged against.

    With one group at 40% of the cohort failing 100%, the cohort rate is already 40% so the
    ratio is only 2.5x and a 3x threshold misses it entirely -- which is exactly what
    happened on first write. Comparing against the OTHER models gives 100% vs 0%.
    """
    mans = ([_man(f"ok{i}", "GOOD", b_hi=1000.0) for i in range(90)] +
            [_man(f"bad{i}", "BAD", b_hi=50.0) for i in range(60)])   # 40% of cohort, 100% hit

    _, rep = E.apply(mans)
    text = E.format_report(rep)
    assert "UNEVEN GATES" in text and "BAD" in text
    assert "elsewhere" in text, "the contrast must be against other models, not the cohort"


def test_commentary_is_derived_not_hardcoded():
    """An earlier version ended with a fixed line asserting the b_hi gate was 89.8% Siemens.
    On the real manifest set that gate is a NO-OP -- build_manifest_site already rejects
    low-b cases -- so the prose contradicted the table. It must follow the data."""
    # only the TR gate fires, evenly
    mans = ([_man(f"a{i}", "MODEL_A", tr=6000.0) for i in range(100)] +
            [_man(f"b{i}", "MODEL_B", tr=900.0) for i in range(2)] +
            [_man(f"c{i}", "MODEL_A", tr=900.0) for i in range(2)])
    _, rep = E.apply(mans)
    text = E.format_report(rep)
    assert "89.8" not in text and "99.5% GE" not in text, "no hardcoded cohort claims"
    assert "NOTE:" in text, "an absent gate should be flagged as possibly applied upstream"


def test_bad_bands_coverage_warns_on_a_sampled_run():
    """The gate is silently under-powered when physics_sanity ran with --limit."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ps.json")
        json.dump({"bad_bands": ["x1", "x2"], "n_cases": 149}, open(p, "w"))
        msg = E.bad_bands_coverage(p, n_cohort=4849)
        assert msg and "149/4849" in msg and "without --limit" in msg
        assert E.bad_bands_coverage(p, n_cohort=150) is None      # full coverage, no warning
    assert "INACTIVE" in E.bad_bands_coverage(None, 4849)


def test_report_is_stable_when_nothing_is_excluded():
    _, rep = E.apply([_man("a"), _man("b")])
    assert "no exclusions" in E.format_report(rep)


def test_missing_physics_sanity_file_raises():
    """Same class as the traveling-file bug: a missing file silently disabled the gate, and
    'no offenders' is indistinguishable from 'never looked'."""
    try:
        E.load_bad_bands("/nonexistent/physics_sanity.json")
    except FileNotFoundError as e:
        assert "silently disables" in str(e)
        return
    raise AssertionError("a missing physics-sanity file must raise")


def test_load_bad_bands_reads_physics_sanity_output():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "physics_sanity.json")
        json.dump({"bad_bands": ["x1", "x2"], "adc_ratio_median": 1.037e6}, open(p, "w"))
        assert E.load_bad_bands(p) == ("x1", "x2")
    assert E.load_bad_bands(None) == ()      # omitted entirely is an explicit choice


def test_gates_are_deterministic_and_reusable():
    """Two arms must run on the same cohort or they are not comparable."""
    mans = [_man(f"c{i}", tr=6000.0 + i, b_hi=1000.0) for i in range(50)]
    a, _ = E.apply(mans)
    b, _ = E.apply(mans)
    assert [m["case_id"] for m in a] == [m["case_id"] for m in b]


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
