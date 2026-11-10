"""Guards for the central-difference cohort sweep.

The sweep is a long cluster job against a cache that must be rebuilt first, so the failure
mode to protect against is discovering a bug six hours in. These run on a synthetic cache.
"""
import json
import os

import numpy as np

from physharm.analysis import prior_shift as PS
from physharm.analysis.synthetic_arm import PARAMS
from tests.test_traveling_arm import ACQ, _render_case, _write_cache

import experiments.cohort_shift_limit as CSL

TRUE = {"PD": 1.4, "T1": 900.0, "T2": 120.0, "D": 1.1e-3}


def _fixture(tmp_path, n=3):
    cache, man = tmp_path / "cache", tmp_path / "man"
    cache.mkdir(); man.mkdir()
    _write_cache(str(cache), {f"c{i}": _render_case(TRUE, ACQ, shape=(2, 8, 8),
                                                    noise=0.01, seed=i) for i in range(n)})
    for i in range(n):
        json.dump({"case_id": f"c{i}", "patient_id": f"p{i}", "scanner_model": "TEST",
                   "contrasts": {"T2W": {"TR_ms": ACQ["TR_T2W"], "TE_ms": ACQ["TE_T2W"]},
                                 "DWI": {"TR_ms": ACQ["TR_DWI"], "TE_ms": ACQ["TE_DWI"],
                                         "b_values": [ACQ["b_hi"]]}}},
                  open(man / f"c{i}.json", "w"))
    return str(cache), str(man)


def test_every_delta_is_returned_in_both_signs(tmp_path):
    """The whole sweep is useless if +d and -d do not both come back under the keys we look up.

    The lookup is by formatted string, so this also pins the formatting: `case_shifts` keys
    positive deltas without a sign, and `f"{+h:.4f}"` must reproduce that exactly.
    """
    cache, _ = _fixture(tmp_path, n=1)
    deltas = tuple(sorted({d for h in CSL.HALVINGS for d in (+h, -h)}, reverse=True))
    sh = PS.case_shifts(PS.load_case(cache, "c0"), ACQ, deltas, 10.0, 0.05, 64,
                        n_iter=25, arm="A1", n_perm=0)
    assert sh is not None
    for h in CSL.HALVINGS:
        assert f"{+h:.4f}" in sh, f"missing +{h}"
        assert f"{-h:.4f}" in sh, f"missing -{h}"


def test_opposite_shifts_move_the_estimate_in_opposite_directions(tmp_path):
    """If they did not, the half-difference would not be a central derivative at all."""
    cache, _ = _fixture(tmp_path, n=1)
    h = CSL.HALVINGS[0]
    sh = PS.case_shifts(PS.load_case(cache, "c0"), ACQ, (+h, -h), 10.0, 0.05, 64,
                        n_iter=25, arm="A1", n_perm=0)
    for p in ("PD", "T1", "T2"):
        op = sh[f"{+h:.4f}"][p]["observed"][p]
        om = sh[f"{-h:.4f}"][p]["observed"][p]
        assert op * om < 0, (p, op, om)


def test_central_difference_is_flatter_in_shift_size_than_one_sided(tmp_path, capsys):
    """The point of the instrument. One-sided carries an O(d) error, central an O(d^2) one,
    so the central ratio must vary less across the sweep. If this inverts, the driver is
    combining the two shifts wrongly and the cohort answer would be meaningless."""
    cache, man = _fixture(tmp_path, n=3)
    out = tmp_path / "o.json"
    import sys
    argv = sys.argv
    sys.argv = ["cohort_shift_limit", "--cache-dir", cache, "--manifest-dir", man,
                "--n-vox", "64", "--out", str(out)]
    try:
        CSL.main()
    finally:
        sys.argv = argv
    rec = json.load(open(out))["per_delta"]
    assert rec, "no records produced"
    for p in ("PD", "T1", "T2"):
        one = [r["one_sided_over_hessian"] for r in rec if r["param"] == p]
        cen = [r["central_over_hessian"] for r in rec if r["param"] == p]
        assert len(one) == len(CSL.HALVINGS)
        assert np.ptp(cen) <= np.ptp(one) + 1e-9, (p, np.ptp(cen), np.ptp(one))
