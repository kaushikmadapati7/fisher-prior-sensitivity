"""Tests for patient-level splits (issue #2).

Run with pytest, or standalone:  python tests/test_splits.py

The cohort has 1,031 repeat exams across 876 patients and a traveling-subject control set,
so these tests are adversarial about leakage rather than about the split ratios: a leak is
silent and invalidates every downstream number.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter

from physharm.data.splits import (assert_no_patient_overlap, format_report,
                                  load_traveling_patients, split_patients)

MODELS = ["SIGNA Premier"] * 5 + ["DISCOVERY MR750"] * 3 + ["DISCOVERY MR750w"] * 2


def _cohort(n_patients=300, repeat_every=3):
    """Patients, some with two exams -- the structure a case-level split would leak."""
    mans = []
    for i in range(n_patients):
        model = MODELS[i % len(MODELS)]
        n_ex = 2 if i % repeat_every == 0 else 1
        for j in range(n_ex):
            mans.append({"case_id": f"c{i:04d}_{j}", "patient_id": f"p{i:04d}",
                         "scanner_model": model})
    return mans


def _pids(ms):
    return {m["patient_id"] for m in ms}


def test_no_patient_appears_in_two_splits():
    """THE test. Everything downstream is void if this fails."""
    splits, _ = split_patients(_cohort())
    assert_no_patient_overlap(splits)
    seen = Counter()
    for ms in splits.values():
        seen.update(_pids(ms))
    assert seen and max(seen.values()) == 1


def test_repeat_exams_stay_together():
    """A patient's two exams must land in the SAME split, not be divided."""
    mans = _cohort()
    splits, _ = split_patients(mans)
    where = {}
    for name, ms in splits.items():
        for m in ms:
            where.setdefault(m["patient_id"], set()).add(name)
    assert all(len(v) == 1 for v in where.values())
    # and the repeats really exist, or the test proves nothing
    assert len(mans) > len({m["patient_id"] for m in mans})


def test_every_exam_is_assigned_exactly_once():
    mans = _cohort()
    splits, _ = split_patients(mans)
    ids = [m["case_id"] for ms in splits.values() for m in ms]
    assert len(ids) == len(mans) and len(set(ids)) == len(mans)


def test_traveling_subjects_are_held_out_and_never_trained_on():
    """They are the gold-standard control; training on them destroys the measurement."""
    mans = _cohort()
    trav = {f"p{i:04d}" for i in range(0, 60)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "traveling_usable.json")
        json.dump({"usable": {"cross-model": sorted(trav)[:40],
                              "same-unit": sorted(trav)[40:]}}, open(p, "w"))
        splits, rep = split_patients(mans, traveling_json=p)
    assert _pids(splits["traveling"]) == trav
    for name in ("train", "val", "test"):
        assert not (_pids(splits[name]) & trav), f"{name} contains traveling subjects"
    assert rep["n_traveling_patients"] == len(trav)


def test_missing_traveling_file_raises_instead_of_silently_emptying():
    """THE bug from the first real cache build, and the most dangerous one so far.

    A relative --traveling path resolved to a file that did not exist. load_traveling_patients
    returned an empty set, the "no --traveling given" warning did not fire because the argument
    WAS given, and the run printed "0 traveling-subject patients held out" -- one line in a
    long report. Every traveling subject moved into TRAIN. The full build would have destroyed
    the gold-standard control with no error anywhere.
    """
    try:
        load_traveling_patients("/nonexistent/traveling_usable.json")
    except FileNotFoundError as e:
        assert "gold-standard control" in str(e) or "TRAIN" in str(e)
    else:
        raise AssertionError("a missing traveling file must raise, not return an empty set")
    # omitting it entirely is still allowed -- that is an explicit choice, not a typo
    assert load_traveling_patients(None) == set()


def test_traveling_file_with_no_patients_raises():
    """An empty hold-out is indistinguishable from not having one."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.json")
        json.dump({"usable": {"cross-model": [], "same-unit": []}}, open(p, "w"))
        try:
            load_traveling_patients(p)
        except ValueError as e:
            assert "anatomy_repeats" in str(e), "the message should name the likely mix-up"
            return
        raise AssertionError("an empty hold-out must raise")


def test_same_unit_controls_are_held_out_too():
    """The noise floor is part of the control -- without it a cross-scanner difference
    cannot be interpreted, so it must not be trained on either."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.json")
        json.dump({"usable": {"same-unit": ["p0001"], "cross-model": ["p0002"]}}, open(p, "w"))
        assert load_traveling_patients(p) == {"p0001", "p0002"}


def test_validation_is_carved_from_train_not_test():
    """Selection on the test split invalidates it. Ratios follow from carving val out of
    the post-test remainder, which is also the definition being asserted here."""
    mans = _cohort(n_patients=1000, repeat_every=97)
    splits, rep = split_patients(mans, val_frac=0.2, test_frac=0.2)
    n = rep["n_patients"]
    assert abs(rep["patients"]["test"] / n - 0.2) < 0.03
    # val is 20% of the REMAINING 80%, i.e. ~16% of the whole -- not 20%
    assert abs(rep["patients"]["val"] / n - 0.16) < 0.03
    assert rep["patients"]["train"] > rep["patients"]["val"]


def test_deterministic_in_the_seed():
    a, _ = split_patients(_cohort(), seed=7)
    b, _ = split_patients(_cohort(), seed=7)
    c, _ = split_patients(_cohort(), seed=8)
    assert _pids(a["test"]) == _pids(b["test"])
    assert _pids(a["test"]) != _pids(c["test"])


def test_rare_scanner_is_not_stranded_in_one_split():
    """INGENIA n=1 and VERIO n=3 are real. Hard stratification would put a whole scanner
    into a single split; pooling rare models prevents that from silently happening."""
    mans = _cohort(n_patients=200)
    mans += [{"case_id": "rare0", "patient_id": "prare0", "scanner_model": "INGENIA"}]
    mans += [{"case_id": f"v{i}", "patient_id": f"pv{i}", "scanner_model": "VERIO"}
             for i in range(3)]
    splits, rep = split_patients(mans, min_group=10)
    assert "INGENIA" in rep["rare_models_pooled"] and "VERIO" in rep["rare_models_pooled"]
    assert_no_patient_overlap(splits)


def test_missing_patient_id_is_counted_not_hidden():
    """If manifests were never patched, the split is only as safe as a case-level one --
    so say so loudly rather than silently degrading."""
    mans = [{"case_id": f"c{i}", "scanner_model": "M"} for i in range(20)]
    splits, rep = split_patients(mans)
    assert rep["n_missing_patient_id"] == 20
    assert "WARNING" in format_report(rep)
    assert_no_patient_overlap(splits)


def test_overlap_guard_actually_fires():
    """A guard that cannot fail is not a guard."""
    m = {"case_id": "c1", "patient_id": "p1", "scanner_model": "M"}
    try:
        assert_no_patient_overlap({"train": [m], "test": [m]})
    except AssertionError as e:
        assert "p1" in str(e)
        return
    raise AssertionError("guard failed to detect an obvious leak")


def test_report_shows_the_scanner_cross_tab():
    splits, rep = split_patients(_cohort())
    text = format_report(rep)
    for mdl in set(MODELS):
        assert mdl[:26] in text
    assert "repeat exams" in text


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
