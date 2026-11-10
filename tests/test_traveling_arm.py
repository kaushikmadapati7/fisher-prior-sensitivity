"""Tests for the traveling-subject arm (issue #11).

Run with pytest, or standalone:  python tests/test_traveling_arm.py
Numpy only -- builds synthetic .npz cache files, so no images and no torch.

The arm's correctness is mostly about what it REFUSES to do: count a missing model tag as a
scanner change, report a cross-scanner number without its noise floor, or let a planted
scanner effect pass unnoticed.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from physharm.analysis import traveling_arm as TA
from physharm.analysis.synthetic_arm import PARAMS, TISSUE

ACQ = dict(TR_T2W=8136.0, TE_T2W=153.0, TR_DWI=4000.0, TE_DWI=70.0, b_hi=1000.0, b_lo=0.0)


def _render_case(theta, acq, shape=(4, 16, 16), noise=0.0, seed=0):
    """A synthetic cached case: T2W, DWI and ADC rendered from known theta."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    PD, T1, T2, D = (np.full(n, theta[k], dtype=np.float64) for k in PARAMS)
    E1 = 1.0 - np.exp(-acq["TR_T2W"] / T1)
    t2w = PD * E1 * np.exp(-acq["TE_T2W"] / T2)
    E1d = 1.0 - np.exp(-acq["TR_DWI"] / T1)
    dwi = PD * E1d * np.exp(-acq["TE_DWI"] / T2) * np.exp(-acq["b_hi"] * D)
    if noise:
        t2w = t2w * np.exp(rng.normal(0, noise, n))
        dwi = dwi * np.exp(rng.normal(0, noise, n))
    mask = np.ones(n, dtype=np.float32)
    return {"T2W": t2w.reshape(shape).astype(np.float32),
            "DWI": dwi.reshape(shape).astype(np.float32),
            "__adc": D.reshape(shape).astype(np.float32),
            "__mask": mask.reshape(shape)}


def _write_cache(d, cases):
    for cid, arrs in cases.items():
        np.savez_compressed(os.path.join(d, f"{cid}.npz"), **arrs)


def test_estimate_case_recovers_planted_theta():
    truth = {"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}
    est = TA.estimate_case(_render_case(truth, ACQ), ACQ, n_vox=500)
    assert est is not None
    for k in ("T2", "D"):                       # the identifiable ones
        assert abs(np.exp(est[k]) / truth[k] - 1.0) < 0.1, f"{k}: {np.exp(est[k]):.4g}"


def test_missing_contrast_returns_none_rather_than_guessing():
    c = _render_case({"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}, ACQ)
    del c["DWI"]
    assert TA.estimate_case(c, ACQ, n_vox=200) is None


def test_empty_mask_returns_none():
    c = _render_case({"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}, ACQ)
    c["__mask"] = np.zeros_like(c["__mask"])
    assert TA.estimate_case(c, ACQ, n_vox=200) is None


def test_missing_model_tag_is_not_a_scanner_change():
    """The bug that inflated the traveling count 11x in the first audit run. An exam whose
    ManufacturerModelName is absent has an UNKNOWN scanner, not a distinct one."""
    with tempfile.TemporaryDirectory() as d:
        anat = os.path.join(d, "a.json")
        json.dump({"tol": 0.09, "false_match_rate": 0.101,
                   "high_conf_pairs": [{"patient": "p1", "a": "c1", "b": "c2"},
                                       {"patient": "p2", "a": "c3", "b": "c4"}]},
                  open(anat, "w"))
        by_case = {"c1": {"model": "SIGNA Premier"}, "c2": {"model": ""},      # unknown
                   "c3": {"model": "SIGNA Premier"}, "c4": {"model": "MAGNETOM Vida"}}
        cls = TA.classify_pairs(None, anat, by_case)
        assert cls["cross"] == [("c3", "c4")], cls["cross"]
        assert cls["same"] == []
        assert cls["false_match_rate"] == 0.101, "the caveat must travel with the number"


def test_same_unit_pairs_become_the_noise_floor():
    with tempfile.TemporaryDirectory() as d:
        anat = os.path.join(d, "a.json")
        json.dump({"tol": 0.09, "false_match_rate": 0.101,
                   "high_conf_pairs": [{"patient": "p", "a": "s1", "b": "s2"}]},
                  open(anat, "w"))
        by_case = {"s1": {"model": "SIGNA Premier"}, "s2": {"model": "SIGNA Premier"}}
        cls = TA.classify_pairs(None, anat, by_case)
        assert cls["same"] == [("s1", "s2")] and cls["cross"] == []


def test_excess_is_measured_against_the_floor_not_in_absolute_terms():
    """Both groups drift by the same amount -> zero excess, even though |dtheta| is large.

    This is how the arm survives the destroyed study dates: the same-unit controls carry the
    same unknown interval distribution, so shared biological drift cancels.
    """
    rng = np.random.default_rng(0)
    drift = lambda: {k: abs(rng.normal(0.30, 0.02)) for k in PARAMS}
    cross = [{"delta": drift()} for _ in range(40)]
    same = [{"delta": drift()} for _ in range(40)]
    res = TA.summarize(cross, same)
    for k in PARAMS:
        assert abs(res[k]["excess"]) < 0.05, f"{k}: {res[k]['excess']}"
        assert res[k]["ci95"][0] < 0 < res[k]["ci95"][1], f"{k}: CI excludes zero"


def test_a_planted_scanner_effect_is_detected():
    rng = np.random.default_rng(1)
    same = [{"delta": {k: abs(rng.normal(0.10, 0.02)) for k in PARAMS}} for _ in range(40)]
    cross = [{"delta": {**{k: abs(rng.normal(0.10, 0.02)) for k in PARAMS},
                        "T2": abs(rng.normal(0.30, 0.02))}} for _ in range(40)]
    res = TA.summarize(cross, same)
    assert res["T2"]["excess"] > 0.1 and res["T2"]["ci95"][0] > 0
    assert abs(res["D"]["excess"]) < 0.05, "unaffected parameters must stay at the floor"


def test_too_few_pairs_reports_none_rather_than_a_number():
    res = TA.summarize([{"delta": {k: 0.1 for k in PARAMS}}],
                       [{"delta": {k: 0.1 for k in PARAMS}}])
    for k in PARAMS:
        assert res[k]["excess"] is None


def test_null_pairs_are_different_patients_and_different_scanners():
    """The identity control must not accidentally pair a patient with himself."""
    by_case = {f"c{i}": {"patient_id": f"p{i//2}", "model": "A" if i % 2 else "B"}
               for i in range(40)}
    rng = np.random.default_rng(0)
    pairs = TA.null_pairs(sorted(by_case), by_case, 50, rng)
    assert pairs, "should produce pairs"
    for a, b in pairs:
        assert by_case[a]["patient_id"] != by_case[b]["patient_id"], "same patient in null"
        assert by_case[a]["model"] != by_case[b]["model"], "must match the cross-scanner design"


def test_pair_deltas_reports_why_pairs_dropped():
    """Half the cross-scanner pairs failed on the first real run with no reason given."""
    with tempfile.TemporaryDirectory() as d:
        truth = {"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}
        _write_cache(d, {"c1": _render_case(truth, ACQ, seed=1)})
        out, why = TA.pair_deltas([("c1", "missing")], d, {"c1": ACQ}, n_vox=200)
        assert out == []
        assert sum(why.values()) == 1 and "cache" in "".join(why)


def test_pair_deltas_skips_uncached_exams():
    with tempfile.TemporaryDirectory() as d:
        truth = {"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}
        _write_cache(d, {"c1": _render_case(truth, ACQ, seed=1),
                         "c2": _render_case(truth, ACQ, seed=2)})
        acqs = {"c1": ACQ, "c2": ACQ, "c3": ACQ}
        out, _why = TA.pair_deltas([("c1", "c2"), ("c1", "c3")], d, acqs, n_vox=300)
        assert len(out) == 1 and out[0]["a"] == "c1" and out[0]["b"] == "c2"
        # identical tissue and protocol -> near-zero delta
        assert out[0]["delta"]["T2"] < 0.05


def test_exclude_scanners_drops_pairs_on_either_side():
    """#56: a per-scanner scale fault forces an offset onto every cross pair it touches."""
    import json, os, tempfile
    from physharm.analysis.traveling_arm import classify_pairs
    by = {"a1": {"model": "SIGNA PREMIER", "patient_id": "p1"},
          "a2": {"model": "MAGNETOM VIDA", "patient_id": "p1"},
          "b1": {"model": "SIGNA PREMIER", "patient_id": "p2"},
          "b2": {"model": "DISCOVERY MR750", "patient_id": "p2"},
          "c1": {"model": "SIGNA PREMIER", "patient_id": "p3"},
          "c2": {"model": "SIGNA PREMIER", "patient_id": "p3"}}
    d = tempfile.mkdtemp()
    f = os.path.join(d, "anat.json")
    json.dump({"high_conf_pairs": [{"a": "a1", "b": "a2"}, {"a": "b1", "b": "b2"},
                                   {"a": "c1", "b": "c2"}]}, open(f, "w"))
    keep = classify_pairs(None, f, by)
    assert len(keep["cross"]) == 2 and len(keep["same"]) == 1
    drop = classify_pairs(None, f, by, exclude=["magnetom vida"])
    assert len(drop["cross"]) == 1 and len(drop["same"]) == 1
    assert drop["n_pairs_dropped"] == 1 and drop["excluded_scanners"] == ["MAGNETOM VIDA"]


def test_exclude_is_a_no_op_when_empty():
    import json, os, tempfile
    from physharm.analysis.traveling_arm import classify_pairs
    by = {"a1": {"model": "SIGNA PREMIER"}, "a2": {"model": "DISCOVERY MR750"}}
    d = tempfile.mkdtemp(); f = os.path.join(d, "anat.json")
    json.dump({"high_conf_pairs": [{"a": "a1", "b": "a2"}]}, open(f, "w"))
    assert len(classify_pairs(None, f, by, exclude=[])["cross"]) == 1


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


def test_classify_pairs_canonicalizes_scanner_names(tmp_path):
    """Spelling variants of one model must not be filed as a cross-scanner pair.

    The raw ManufacturerModelName carries case and punctuation variants of the same
    physical scanner ('SIGNA Premier' / 'Signa  Premier' / 'GE MEDICAL SYSTEMS SIGNA
    Premier'). Comparing raw strings counts a same-machine repeat as cross-scanner and
    simultaneously drops it from the same-unit floor, so the excess is biased toward zero
    twice over. This test fails against the pre-fix module.
    """
    from physharm.analysis.traveling_arm import classify_pairs

    by_case = {
        "a": {"model": "SIGNA Premier"},
        "b": {"model": "GE MEDICAL SYSTEMS  signa premier"},
        "c": {"model": "Discovery MR750"},
    }
    anat = tmp_path / "anatomy_repeats.json"
    anat.write_text(json.dumps({"high_conf_pairs": [{"a": "a", "b": "b"},
                                                    {"a": "a", "b": "c"}]}))
    got = classify_pairs(None, anat, by_case)
    assert got["same"] == [("a", "b")], got["same"]
    assert got["cross"] == [("a", "c")], got["cross"]

    # --exclude-scanners is documented as canonical + case-insensitive; make it so.
    got = classify_pairs(None, anat, by_case, exclude=("signa premier",))
    assert got["n_pairs_dropped"] == 2, got["n_pairs_dropped"]
    assert got["cross"] == [] and got["same"] == []
