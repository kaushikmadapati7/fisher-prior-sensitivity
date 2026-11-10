"""End-to-end smoke of the prior-shift CLI, so a cohort run cannot die in its own report.

WHY THIS FILE EXISTS. Everything above `main` is unit-tested, but the reporting block is not
reachable from any of those tests: it needs a list of collected rows, which needs a cache and
a manifest directory. It has twice been the case on this project that a change was deployed to
the cluster, ran for hours, and failed -- or silently reported the wrong statistic -- in code
no test executed. A run over 4,608 exams is too expensive to use as the first execution of a
print statement, so this builds a three-case cohort on disk and runs the CLI end to end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from tests.test_traveling_arm import ACQ, _render_case

TRUE = {"PD": 1.4, "T1": 900.0, "T2": 120.0, "D": 1.1e-3}
SCANNERS = ["Siemens Skyra", "Siemens Vida", "Siemens Verio"]


def _cohort(tmp_path, n_cases=110, n_patients=25):
    """110 cases because `main` refuses to report below 100 -- the guard is the point of the
    threshold, and a smoke test that dodged it would not exercise the code that matters."""
    cache, mans = tmp_path / "cache", tmp_path / "manifests"
    cache.mkdir(), mans.mkdir()
    for i in range(n_cases):
        case = _render_case(TRUE, ACQ, shape=(1, 10, 10), noise=0.01, seed=i)
        np.savez(cache / f"case{i}.npz", **case)
        json.dump({"case_id": f"case{i}", "patient_id": f"pt{i % n_patients}",
                   "scanner_model": SCANNERS[i % len(SCANNERS)],
                   "contrasts": {"T2W": {"TR_ms": ACQ["TR_T2W"], "TE_ms": ACQ["TE_T2W"]},
                                 "DWI": {"TR_ms": ACQ["TR_DWI"], "TE_ms": ACQ["TE_DWI"],
                                         "b_values": [0.0, ACQ["b_hi"]]}}},
                  open(mans / f"case{i}.json", "w"))
    return cache, mans


def test_the_cli_runs_to_completion_and_reports_the_scalar_and_the_sign_test(tmp_path):
    cache, mans = _cohort(tmp_path)
    out = tmp_path / "runs" / "prior_shift.json"
    r = subprocess.run(
        [sys.executable, "-m", "physharm.analysis.prior_shift",
         "--cache-dir", str(cache), "--manifest-dir", str(mans), "--n-vox", "60",
         "--n-iter", "40", "--n-perm", "20", "--out", str(out)],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]

    # Both new sections must actually print, not merely not crash.
    assert "THE ONE-SCALAR CORRECTION" in r.stdout, r.stdout[-3000:]
    assert "THE SIGN CONDITION" in r.stdout, r.stdout[-3000:]
    assert "obs/scal" in r.stdout
    assert "nan" not in r.stdout.split("THE ONE-SCALAR CORRECTION")[1].split("PER SCANNER")[0]

    rep = json.load(open(out))["report"]
    for k in ("median", "p05", "p95", "vCv_over_lam_median", "identity_rel_err_median",
              "hessian_ms", "scalar_ms", "n"):
        assert k in rep["scalar"], k
    for k in ("observed_median", "frac_exams_positive", "z_perm_median", "z_flip_median",
              "frac_voxels_positive_median", "n"):
        assert k in rep["sign"], k
    assert rep["scalar"]["n"] == rep["sign"]["n"] == 110

    # The identity is the load-bearing check; a cohort run whose two routes disagree is not
    # measuring what section 3 says it is.
    assert rep["scalar"]["identity_rel_err_median"] < 1e-4


def test_disabling_the_permutations_still_produces_a_report(tmp_path):
    """--n-perm 0 is the switch for a run that only wants the correction. It must not divide
    by an empty null or emit a KeyError three hours in."""
    cache, mans = _cohort(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "physharm.analysis.prior_shift",
         "--cache-dir", str(cache), "--manifest-dir", str(mans), "--n-vox", "60",
         "--n-iter", "30", "--n-perm", "0"],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
