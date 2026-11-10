"""Tests for the synthetic arm (issue #7).

Run with pytest, or standalone:  python tests/test_synthetic_arm.py
Numpy only -- no torch, no cache, no images. That is the point of this arm.

The estimator is validated against theory FIRST (a MAP fit whose behaviour does not match
its own predicted covariance would make every downstream number meaningless), and only then
is the leakage claim tested.
"""
from __future__ import annotations

import numpy as np

from physharm.analysis import synthetic_arm as S
from physharm.analysis.synthetic_arm import PARAMS
from tests.test_traveling_arm import ACQ
from physharm.physics import fisher as F

ACQ = dict(TR_T2W=8136.0, TE_T2W=153.0, TR_DWI=4000.0, TE_DWI=70.0, b_hi=1000.0, b_lo=0.0)


def _protocols(n=80, seed=0):
    """Protocols spanning the real the site ranges (the run log Phase 2d)."""
    rng = np.random.default_rng(seed)
    return [dict(TR_T2W=float(rng.uniform(4000, 10300)), TE_T2W=float(rng.uniform(94, 170)),
                 TR_DWI=4000.0, TE_DWI=70.0, b_hi=1000.0, b_lo=0.0) for _ in range(n)]


def test_map_fit_recovers_truth_at_high_snr():
    th = S.draw_tissue(400, np.random.default_rng(0))
    x = S.run_protocol(ACQ, "A1", th, np.random.default_rng(1), snr=300.0, nuisance="noise")
    for i, p in enumerate(("T2", "D")):                # the identifiable ones
        j = S.PARAMS.index(p)
        ratio = np.median(np.exp(x[j]) / th[p])
        assert abs(ratio - 1.0) < 0.05, f"{p}: median est/true = {ratio:.3f}"


def test_estimator_matches_its_own_sampling_covariance():
    """The bound must bind, against the RIGHT quantity.

    A MAP estimate varies by sqrt(diag((F+L)^-1 F (F+L)^-1)), not by the posterior sd
    sqrt(diag((F+L)^-1)). Comparing against the posterior looks like a factor-5 discrepancy;
    it is simply the wrong comparison, and I made it once while building this.
    """
    fixed = {k: np.full(3000, v[0]) for k, v in S.TISSUE.items()}
    for snr in (5.0, 15.0, 60.0):
        x = S.run_protocol(ACQ, "A1", fixed, np.random.default_rng(2), snr, "noise")
        meas = x.std(axis=1)
        Fm = F.fisher_matrix(F.measurement_rows("A1", ACQ), [snr ** 2] * 3)
        pred = F.sampling_sd(Fm)
        for i, k in enumerate(S.PARAMS):
            assert abs(meas[i] - pred[k]) / max(pred[k], 1e-9) < 0.20, \
                f"snr={snr} {k}: measured {meas[i]:.4f} vs predicted {pred[k]:.4f}"


def test_shrinkage_separates_anchored_from_prior_pinned():
    sh = F.identifiability(ACQ, "A1")["shrinkage"]
    assert sh["D"] < 0.05, "ADC-anchored D must be data-determined"
    assert sh["T1"] > 0.7, "T1 must be prior-pinned at clinical TR"
    assert sh["T2"] < 0.3


def test_leakage_is_non_monotone_in_shrinkage():
    """THE CORRECTION TO THE THESIS. the project plan Part II says unidentifiable parameters leak.
    That is wrong at the unidentifiable extreme.

    Per protocol, correlate how far that protocol's estimate sits from the consensus against
    how much the prior pulls it:

        D  (shrinkage ~0.01)  rho ~  0.0   data-determined, identical tissue -> no leak
        T2 (shrinkage ~0.09)  rho ~ +0.80  the pull varies with protocol -> leaks
        PD (shrinkage ~0.19)  rho ~ +0.62  same
        T1 (shrinkage ~0.89)  rho ~ -0.62  pinned to the prior EVERYWHERE -> consistent

    Leakage is an inverted U in identifiability: fully determined parameters do not leak
    because the data pins them, fully undetermined ones do not leak either because the prior
    pins them identically on every scanner, and the scanner signal lives in between.
    """
    r = S.leakage_vs_crlb(_protocols(), "A1", n_tissue=150, snr=30.0,
                          nuisance="noise", seed=0)
    pp, sh = r["per_protocol"], r["shrinkage"]
    assert sh["T1"] > 0.7 and pp["T1"]["rho_shrinkage"] < -0.2, \
        "a fully prior-pinned parameter must become MORE consistent, not less"
    assert pp["T2"]["rho_shrinkage"] > 0.4, "an intermediate parameter must leak with the pull"
    assert abs(pp["D"]["rho_shrinkage"]) < 0.3, "an anchored parameter must show no relation"


def test_crlb_and_shrinkage_are_rank_equivalent_per_parameter():
    """shrinkage_k = lambda * crlb_k^2 at fixed prior_sd, so they are monotone in each other
    and the regression is insensitive to which is used -- only the SIGN carries the claim.

    Not asserted to machine precision: the tie tolerance is applied to each vector
    separately, and squaring changes which pairs fall inside it, so a near-tied parameter can
    differ by ~1e-3 in rho. That is tie bookkeeping, not a change in the relationship.
    """
    r = S.leakage_vs_crlb(_protocols(), "A1", n_tissue=100, snr=30.0, seed=0)
    for k in S.PARAMS:
        a, b = r["per_protocol"][k]["rho_shrinkage"], r["per_protocol"][k]["rho_crlb"]
        assert abs(a - b) < 0.02, f"{k}: {a} vs {b}"
        assert (a > 0) == (b > 0), f"{k}: sign disagreement {a} vs {b}"


def test_spearman_refuses_a_constant_input():
    """numpy ranks a constant vector as the identity, so the naive formula returns +1.0
    against anything. Arm A3 pins D to the prior exactly, giving a per-protocol deviation of
    0.000 everywhere -- which would have been reported as a perfect correlation."""
    assert np.isnan(S.spearman([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]))
    assert np.isnan(S.spearman([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]))
    assert abs(S.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9
    assert abs(S.spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) + 1.0) < 1e-9


def test_unmeasured_parameter_is_reported_as_not_identifiable_not_as_zero_leakage():
    """A3 (T2W only) gives D no measurement at all -- its Jacobian column is identically
    zero -- so the fit returns the prior exactly for every protocol and every sample.

    A spread of 0.000 would read as "perfectly scanner-invariant" when it means "we put a
    constant in". Those are opposite conclusions from the same number, so the arm must
    report NOT IDENTIFIABLE. Flagged by an external reviewer of the first run.
    """
    r = S.leakage_vs_crlb(_protocols(30), "A3", n_tissue=80, snr=30.0, seed=0)
    assert r["degenerate"]["D"] is True
    assert np.isnan(r["leakage"]["D"]), "must not be reported as 0.000"
    assert r["degenerate"]["T2"] is False and not np.isnan(r["leakage"]["T2"])
    assert F.degenerate(F.measurement_rows("A1", ACQ))["D"] is False


def test_spearman_uses_average_ranks_for_ties():
    """argsort breaks ties by array position, so a rank order can turn on a numerical
    accident: in A3, CRLB(T1)=9.9997 and CRLB(D)=10.0000 are both saturated at the prior
    ceiling and their relative rank rests on 3e-4."""
    tied = S.spearman([1.0, 2.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    flipped = S.spearman([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 2.0, 4.0])
    assert abs(tied - flipped) < 1e-9, "a tie must not depend on which twin comes first"


def test_predictor_is_evaluated_at_the_experiment_snr():
    """The CRLB used to default to unit weights, so it did not depend on SNR at all -- every
    SNR row of the report was identical, which made one observation look like fifteen."""
    p = _protocols(10)
    lo = S.leakage_vs_crlb(p, "A1", n_tissue=60, snr=5.0, seed=0)
    hi = S.leakage_vs_crlb(p, "A1", n_tissue=60, snr=60.0, seed=0)
    assert hi["crlb"]["D"] < 0.5 * lo["crlb"]["D"], "predictor must track the noise level"


def test_anchored_parameter_barely_leaks():
    r = S.leakage_vs_crlb(_protocols(), "A1", n_tissue=150, snr=30.0, seed=0)
    assert r["leakage"]["D"] < 0.5 * min(r["leakage"][k] for k in ("PD", "T1", "T2"))


def test_nuisance_ladder_only_adds_leakage():
    """noise-only is identifiability alone; each rung adds unmodelled physics on top."""
    p = _protocols(40)
    base = S.leakage_vs_crlb(p, "A1", n_tissue=120, snr=15.0, nuisance="noise", seed=0)
    gain = S.leakage_vs_crlb(p, "A1", n_tissue=120, snr=15.0, nuisance="gain", seed=0)
    unmod = S.leakage_vs_crlb(p, "A1", n_tissue=120, snr=15.0, nuisance="unmodelled", seed=0)
    for k in S.PARAMS:
        assert gain["leakage"][k] >= base["leakage"][k] - 1e-6, k
        assert unmod["leakage"][k] >= base["leakage"][k] - 1e-6, k


def test_leakage_falls_with_snr():
    p = _protocols(40)
    lo = S.leakage_vs_crlb(p, "A1", n_tissue=120, snr=8.0, seed=0)
    hi = S.leakage_vs_crlb(p, "A1", n_tissue=120, snr=60.0, seed=0)
    for k in ("PD", "T2"):
        assert hi["leakage"][k] < lo["leakage"][k], k


def test_deterministic_in_the_seed():
    p = _protocols(20)
    a = S.leakage_vs_crlb(p, "A1", n_tissue=80, snr=20.0, seed=5)
    b = S.leakage_vs_crlb(p, "A1", n_tissue=80, snr=20.0, seed=5)
    c = S.leakage_vs_crlb(p, "A1", n_tissue=80, snr=20.0, seed=6)
    assert a["leakage"] == b["leakage"]
    assert a["leakage"] != c["leakage"]


def test_arms_change_identifiability_as_predicted():
    p = _protocols(30)
    a1 = S.leakage_vs_crlb(p, "A1", n_tissue=80, snr=30.0, seed=0)
    a4 = S.leakage_vs_crlb(p, "A4", n_tissue=80, snr=30.0, seed=0)   # ADC withheld
    assert a4["shrinkage"]["D"] > a1["shrinkage"]["D"], "withholding ADC must free D"
    assert a4["leakage"]["D"] > a1["leakage"]["D"]


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


def _stationarity(x, y, rows, x0, lam, w):
    """Bounded ||grad|| / (||data pull|| + ||prior pull||), per sample -- the analysis metric."""
    import numpy as np

    from physharm.analysis.synthetic_arm import _jacobian, _model_log

    r = y - _model_log(x, rows)
    J = _jacobian(x, rows)
    data = np.einsum("mkn,mn->kn", J, r) * w
    pull = lam * (x - x0)
    return (np.linalg.norm(data - pull, axis=0)
            / (np.linalg.norm(data, axis=0) + np.linalg.norm(pull, axis=0) + 1e-30))


def _cohort(n, rng):
    import numpy as np

    from physharm.analysis.synthetic_arm import PARAMS, TISSUE

    mean = {k: TISSUE[k][0] for k in PARAMS}
    theta = {k: mean[k] * np.exp(rng.normal(0, TISSUE[k][1], n)) for k in PARAMS}
    return mean, np.stack([np.log(theta[k]) for k in PARAMS])


def test_damping_does_not_move_a_well_behaved_fit():
    """The headline rests on this: damping must be a no-op where the old solver worked.

    On a case with no relative receive gain, undamped Gauss-Newton already descends
    monotonically, so the line search accepts the full step every time and Levenberg
    damping stays at zero. The two solvers must therefore agree to solver tolerance.
    """
    import numpy as np

    from physharm.analysis.synthetic_arm import _model_log, _rows_for, map_estimate

    rng = np.random.default_rng(0)
    mean, x_true = _cohort(3000, rng)
    for arm in ("A1", "A3", "A4"):
        rows = _rows_for(arm, ACQ)
        y = _model_log(x_true, rows)
        old = map_estimate(y, rows, mean, line_search=False, n_iter=25)
        new = map_estimate(y, rows, mean)
        assert np.abs(old - new).max() < 1e-3, (arm, np.abs(old - new).max())


def test_damping_fixes_the_a1_limit_cycle():
    """#79: undamped GN cycles on A1 when the T2W/DWI receive gain is preserved.

    `normalize_shared` divides by the T2W ROI median and keeps the inter-contrast ratio, so
    the arbitrary TSE-vs-EPI gain reaches the fit. A1's ADC anchor pins log D, and when the
    remaining T2W/DWI ratio exceeds what T1 and T2 can produce, the undamped step overshoots
    along the lam-pinned soft direction and the iterates enter an exact period-2 cycle --
    which is why raising n_iter does not help and why 41.3% of the cohort was non-stationary.
    """
    import numpy as np

    from physharm.analysis.synthetic_arm import _model_log, _rows_for, map_estimate

    rng = np.random.default_rng(0)
    n = 3000
    mean, x_true = _cohort(n, rng)
    rows = _rows_for("A1", ACQ)
    y = _model_log(x_true, rows)
    y = y + np.stack([rng.normal(0, 1.0, n), np.zeros(n), np.zeros(n)])
    x0 = np.stack([np.full(n, np.log(mean[k])) for k in PARAMS])
    lam, w = 1.0 / 10.0 ** 2, 1.0 / 0.05 ** 2

    old = map_estimate(y, rows, mean, line_search=False, n_iter=25)
    new = map_estimate(y, rows, mean)
    bad_old = float(np.mean(_stationarity(old, y, rows, x0, lam, w) > 0.01))
    bad_new = float(np.mean(_stationarity(new, y, rows, x0, lam, w) > 0.01))
    assert bad_old > 0.10, f"fixture no longer reproduces the failure: {bad_old:.3f}"
    assert bad_new < bad_old / 2, f"damping did not help: {bad_old:.3f} -> {bad_new:.3f}"

    # It is cycling, not slow convergence. The undamped failure rate PLATEAUS: going from
    # 200 to 1000 steps buys essentially nothing, and the plateau sits well above what
    # damping reaches in 100. This is what makes the old --n-iter advice ("raise it and
    # re-check if stationarity is poor") wrong for this failure mode.
    plateau = [float(np.mean(_stationarity(
        map_estimate(y, rows, mean, line_search=False, n_iter=k), y, rows, x0, lam, w) > 0.01))
        for k in (200, 1000)]
    assert plateau[1] > plateau[0] * 0.9, f"still converging at 1000 steps: {plateau}"
    assert bad_new < plateau[1] * 0.75, f"damping beat by iteration count: {bad_new} vs {plateau}"


def test_the_solver_never_increases_its_own_objective():
    """Damped steps are accepted only on a decrease, so the final cost cannot be worse."""
    import numpy as np

    from physharm.analysis.synthetic_arm import (_model_log, _rows_for, map_cost,
                                                 map_estimate)

    rng = np.random.default_rng(3)
    n = 1500
    mean, x_true = _cohort(n, rng)
    rows = _rows_for("A1", ACQ)
    y = _model_log(x_true, rows) + np.stack([rng.normal(0, 1.5, n), np.zeros(n), np.zeros(n)])
    x0 = np.stack([np.full(n, np.log(mean[k])) for k in PARAMS])
    start = map_cost(x0, y, rows, x0, 1.0 / 0.05 ** 2, 1.0 / 10.0 ** 2)
    end = map_cost(map_estimate(y, rows, mean), y, rows, x0,
                   1.0 / 0.05 ** 2, 1.0 / 10.0 ** 2)
    assert np.all(end <= start + 1e-9), int(np.sum(end > start + 1e-9))
