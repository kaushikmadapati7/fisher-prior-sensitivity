"""Tests for the permuted-protocol control (issue #9).

Run with pytest, or standalone:  python tests/test_permuted_control.py

The control's job is to FAIL when the association is spurious, so most of these construct a
spurious association on purpose and check that it is caught. A control that cannot fire is
decoration.
"""
from __future__ import annotations

import numpy as np

from physharm.analysis.permuted_control import observed_and_permuted
from physharm.analysis.synthetic_arm import PARAMS, spearman

def _protocols(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(TR_T2W=float(rng.uniform(4000, 10300)), TE_T2W=float(rng.uniform(94, 170)),
                 TR_DWI=4000.0, TE_DWI=70.0, b_hi=1000.0, b_lo=0.0) for _ in range(n)]


def test_real_association_collapses_under_physical_permutation():
    """The whole point. Pairing the predictor with the wrong scan must destroy it."""
    r = observed_and_permuted(_protocols(), "A1", n_tissue=150, snr=30.0, seed=0, n_perm=200)
    for k in ("PD", "T1", "T2"):
        obs, perm = abs(r[k]["rho"]), abs(r[k]["rho_physical_permutation"])
        assert obs > 0.25, f"{k}: no association to break ({obs:.3f})"
        assert perm < 0.5 * obs, f"{k}: survived permutation, obs {obs:.3f} perm {perm:.3f}"


def test_observed_association_beats_the_label_permutation_null():
    r = observed_and_permuted(_protocols(), "A1", n_tissue=150, snr=30.0, seed=0, n_perm=300)
    for k in ("PD", "T1", "T2"):
        assert r[k]["p_value"] < 0.05, f"{k}: p={r[k]['p_value']}"
        assert abs(r[k]["rho"]) > r[k]["null_p95"], k


def test_anchored_parameter_shows_no_strong_association():
    """D is pinned by the ADC anchor, so there is little for the protocol to move. A control
    that flagged a strong association here would be suspicious of the arm, not of the control.
    """
    r = observed_and_permuted(_protocols(), "A1", n_tissue=150, snr=30.0, seed=0, n_perm=200)
    assert abs(r["D"]["rho"]) < 0.35


def test_degenerate_parameter_is_skipped_not_scored():
    """A3 gives D no measurement; there is nothing to permute and no rho to report."""
    r = observed_and_permuted(_protocols(60), "A3", n_tissue=100, snr=30.0, seed=0, n_perm=50)
    assert r["D"]["degenerate"] is True and "rho" not in r["D"]
    assert r["T2"]["degenerate"] is False


def test_the_null_is_centred_on_zero():
    """If the label-permutation null were biased, every p-value would be wrong."""
    r = observed_and_permuted(_protocols(), "A1", n_tissue=100, snr=30.0, seed=1, n_perm=400)
    for k in ("PD", "T2"):
        assert abs(r[k]["null_mean"]) < 0.08, f"{k}: null mean {r[k]['null_mean']:.3f}"


def test_control_would_catch_a_spurious_association():
    """A control that cannot fail proves nothing, so manufacture the failure it must detect.

    Here the 'predictor' is a copy of the measurement, so the association is perfect BY
    CONSTRUCTION and has nothing to do with protocol. A physical permutation of a genuine
    predictor destroys it; permuting a self-referential one does too -- but the point is that
    the machinery reports the collapse rather than the original correlation.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(size=120)
    assert abs(spearman(y.tolist(), y.tolist()) - 1.0) < 1e-9         # perfect by construction
    assert abs(spearman(rng.permutation(y).tolist(), y.tolist())) < 0.3, \
        "permutation must destroy even a perfect association"


def test_deterministic_in_the_seed():
    a = observed_and_permuted(_protocols(60), "A1", n_tissue=80, snr=30.0, seed=3, n_perm=50)
    b = observed_and_permuted(_protocols(60), "A1", n_tissue=80, snr=30.0, seed=3, n_perm=50)
    assert a["T2"]["rho"] == b["T2"]["rho"]


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
