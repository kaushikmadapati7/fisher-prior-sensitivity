"""Tests for the prior-shift experiment (#10). Synthetic renders, no images, no torch."""
from __future__ import annotations

import numpy as np

from physharm.analysis import prior_shift as PS
from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _jacobian, map_estimate
from physharm.analysis.traveling_arm import _rows_and_signal
from physharm.physics import fisher as F
from physharm.analysis.traveling_arm import _rows_and_signal
from tests.test_traveling_arm import ACQ, _render_case

#: truth deliberately away from the prior mean, so a leak has somewhere to pull the estimate
TRUE = {"PD": 1.4, "T1": 900.0, "T2": 120.0, "D": 1.1e-3}


def _case(seed=0, noise=0.0):
    return _render_case(TRUE, ACQ, shape=(4, 20, 20), noise=noise, seed=seed)


def _M(acq=ACQ, prior_sd=10.0):
    return F.shrinkage_matrix(F.fisher_matrix(F.measurement_rows("A1", acq)), prior_sd)


def test_shrinkage_matrix_diagonal_is_the_shrinkage_vector():
    Fm = F.fisher_matrix(F.measurement_rows("A1", ACQ))
    d = F.shrinkage(Fm)
    assert np.allclose(np.diag(F.shrinkage_matrix(Fm)), [d[p] for p in PARAMS])


def test_shrinkage_matrix_is_identity_when_the_data_says_nothing():
    """F = 0 means the estimate IS the prior, so every unit of Delta must arrive intact."""
    assert np.allclose(F.shrinkage_matrix(np.zeros((4, 4))), np.eye(4))


def test_shrinkage_matrix_vanishes_when_the_data_says_everything():
    assert np.allclose(F.shrinkage_matrix(np.eye(4) * 1e12), np.zeros((4, 4)), atol=1e-9)


def test_off_diagonal_terms_are_not_all_zero():
    """If M were diagonal there would be no cross-parameter leak to report."""
    M = _M()
    assert np.abs(M - np.diag(np.diag(M))).max() > 1e-6


def test_the_true_hessian_predicts_the_shift_and_fisher_does_not():
    """The finding. Right model, no noise, converged fit -- and Fisher is still 25-30% high.

    Both matrices agree to four digits on the diagonal; they disagree about the inverse
    because F + Lambda is near-singular. Nothing here is a fitting artifact: the residual at
    the MAP is 3e-5 and the numbers below are unchanged from 25 to 2000 Gauss-Newton steps.
    """
    per = PS.case_shifts(_case(seed=0), ACQ, deltas=(0.0100,), n_vox=800)["0.0100"]
    for p in ("PD", "T1", "T2"):
        ob, pf, ph = (per[p]["observed"][p], per[p]["predicted"][p],
                      per[p]["predicted_hessian"][p])
        assert abs(ob / ph - 1.0) < 0.05, ("hessian", p, ph, ob)
        assert pf > ph * 1.15, ("fisher should over-predict", p, pf, ph)


def test_the_dropped_term_is_parameter_effects_not_extrinsic_curvature():
    """What kind of curvature this is, which decides how strong the claim may be.

    Arm A1 has three measurements and four parameters, so the model manifold maps R^4 into
    R^3 and FILLS it: rank(J) = 3 = the data-space dimension. There is no direction normal to
    the tangent space, so the manifold's extrinsic curvature is identically zero and the term
    the Fisher matrix drops can only be PARAMETER-EFFECTS curvature -- which Transtrum, Machta
    & Sethna (arXiv:1010.1449) classify as removable by a change of coordinates.

    That weakens the finding and sharpens it at once. It is not a statement about the geometry
    of the model manifold. It is a statement about the coordinates the prior is written in,
    and those cannot be changed without changing the prior you meant to impose.
    """
    from physharm.analysis.synthetic_arm import _model_log, map_estimate
    from physharm.analysis.traveling_arm import _rows_and_signal
    rows, y = _rows_and_signal(_case(seed=0), ACQ, n_vox=800)
    xs = np.median(map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()},
                                n_iter=600), axis=1)
    J = _jacobian(xs[:, None], rows)[:, :, 0]
    assert np.linalg.matrix_rank(J) == J.shape[0] == 3


def test_the_dropped_term_is_five_orders_below_F_and_still_moves_the_leak_30pc():
    """The amplification, stated as a ratio so it cannot be read as a large correction."""
    from physharm.analysis.synthetic_arm import _model_log, map_estimate
    from physharm.analysis.traveling_arm import _rows_and_signal
    rows, y = _rows_and_signal(_case(seed=0), ACQ, n_vox=800)
    base = map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()}, n_iter=600)
    xs = np.median(base, axis=1)
    w, lam, h = 400.0, 0.01, 1e-5
    fn = lambda v: _model_log(v[:, None], rows)[:, 0]
    J = _jacobian(xs[:, None], rows)[:, :, 0]
    r = y[:, 0] - fn(xs)
    A = np.zeros((3, 4, 4))
    for i in range(4):
        for j in range(4):
            ei = np.zeros(4); ei[i] = h
            ej = np.zeros(4); ej[j] = h
            A[:, i, j] = (fn(xs+ei+ej) - fn(xs+ei-ej) - fn(xs-ei+ej) + fn(xs-ei-ej)) / (4*h*h)
    C = -np.einsum("m,mij->ij", w * r, A)
    Fm = J.T @ (w * np.eye(3)) @ J
    assert np.abs(C).max() / np.abs(Fm).max() < 1e-4          # five orders below F
    leak_F = np.diag(np.linalg.inv(Fm + lam*np.eye(4)) * lam)
    leak_H = np.diag(np.linalg.inv(Fm + C + lam*np.eye(4)) * lam)
    assert leak_F[1] / leak_H[1] > 1.25                       # T1: 0.927 vs 0.701


def _hetero_case(seed=0, spread=0.35):
    """Voxels carrying DIFFERENT tissue -- which every real ROI does and no earlier fixture did."""
    rng = np.random.default_rng(seed)
    n = 4 * 20 * 20
    theta = {k: TRUE[k] * np.exp(rng.normal(0, spread, n)) for k in PARAMS}
    PD, T1, T2, D = (theta[k] for k in PARAMS)
    e1 = 1.0 - np.exp(-ACQ["TR_T2W"] / T1)
    e1d = 1.0 - np.exp(-ACQ["TR_DWI"] / T1)
    sh = (4, 20, 20)
    return {"T2W": (PD * e1 * np.exp(-ACQ["TE_T2W"] / T2)).reshape(sh).astype(np.float32),
            "DWI": (PD * e1d * np.exp(-ACQ["TE_DWI"] / T2)
                    * np.exp(-ACQ["b_hi"] * D)).reshape(sh).astype(np.float32),
            "__adc": D.reshape(sh).astype(np.float32),
            "__mask": np.ones(sh, dtype=np.float32)}


def test_hessian_prediction_survives_a_heterogeneous_roi():
    """Regression for the defect the cohort run exposed (#43).

    The first posterior_curvature evaluated the gradient at the ROI-median fit while feeding
    it VOXEL ZERO's signal. Every voxel of the homogeneous fixture carries identical tissue,
    so the two agreed and all 13 tests passed. On a real prostate the mismatch between one
    voxel and the ROI median is the tissue heterogeneity of the organ, which inflated the
    curvature term by two to three orders of magnitude and drove every Hessian prediction to
    zero -- observed +0.00377 against a predicted +0.00001.

    Any fixture whose voxels are identical is blind to this. This one is not.
    """
    per = PS.case_shifts(_hetero_case(), ACQ, deltas=(0.0100,), n_vox=800)["0.0100"]
    for p in ("PD", "T1", "T2"):
        ob, ph = per[p]["observed"][p], per[p]["predicted_hessian"][p]
        assert abs(ob / ph - 1.0) < 0.35, (p, ph, ob)


def test_the_model_second_derivative_is_what_finite_differences_say_it_is():
    """The curvature term is analytic; this pins the finite difference against it.

    d log S / d log theta depends on one parameter per term, so the second-derivative array is
    diagonal: zero for PD (which enters linearly in log space), -TE/T2, -bD, and for T1
    -t da/dt with t = TR/T1 and a = -t/(e^t - 1). Checked because the cohort's Hessian column
    was wild and the finite difference was the first suspect. It is not the cause: the two
    agree to 1e-7 even when the residual is large.
    """
    from physharm.analysis.synthetic_arm import _model_log
    rows, y = _rows_and_signal(_hetero_case(), ACQ, n_vox=600)
    x = np.median(map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()},
                               n_iter=400), axis=1)[:, None]
    PD, T1, T2, D = (float(np.exp(x[i, 0])) for i in range(4))
    for m, r in enumerate(rows):
        if r == "anchor":
            continue
        TR, TE, b = r
        t = TR / T1
        dadt = (t*np.exp(t) - np.exp(t) + 1.0) / np.expm1(t)**2
        want = np.array([0.0, float(-t*dadt), float(-TE/T2), float(-b*D)])
        got = np.empty(4)
        h = 1e-5
        for k in range(4):
            e = np.zeros((4, 1)); e[k] = h
            got[k] = (_model_log(x+e, rows)[m, 0] - 2*_model_log(x, rows)[m, 0]
                      + _model_log(x-e, rows)[m, 0]) / h**2
        # atol tracks the second-difference floor, eps/h^2 ~ 2e-6, not the algebra
        assert np.allclose(got, want, rtol=1e-3, atol=1e-5), (m, want, got)


def test_dispersion_is_reported_not_just_the_centre():
    """T1's obs/Hessian is 1.008 at the median and 10,090 at p95 on the cohort."""
    d = PS._fit_slope([1.0]*100 + [0.001]*100, [1.0]*100 + [1.0]*100)
    assert d["frac_wild"] > 0.4 and d["ratio_p95"] > 100


def test_the_output_identifies_the_code_that_produced_it():
    """A run on the cluster cannot otherwise be traced to a version of this file.

    The deployed tree is an scp'd copy, not a checkout. On 2026-08-27 a full 4,608-case run
    reproduced an earlier one byte for byte because a fix was committed and PR'd but never
    deployed, and nothing in the output revealed it. Two runs sharing a fingerprint ran the same
    code; two that differ did not.
    """
    fp = PS.code_fingerprint()
    assert len(fp) == 12 and fp != "unknown" and all(c in "0123456789abcdef" for c in fp)
    assert PS.code_fingerprint() == fp


def test_stationarity_cannot_exceed_one():
    """The first version divided by the prior pull alone and reported 1e4 on the cohort.

    That denominator vanishes for any fit sitting near the prior mean, so the ratio exploded
    where the gradient was small too -- a bimodal split with 30% at machine zero and the rest at
    1e3-1e4. Normalising by the sum of both terms is bounded in [0,1] by the triangle inequality
    and still exactly 0 at a stationary point.
    """
    rows, y = _rows_and_signal(_hetero_case(), ACQ, n_vox=600)
    pm = {k: v[0] for k, v in TISSUE.items()}
    for n_iter in (1, 3, 25, 400):
        v = PS.stationarity(map_estimate(y, rows, pm, n_iter=n_iter), y, rows, pm)
        assert 0.0 <= v <= 1.0, (n_iter, v)
    # a deliberately unconverged fit must score worse than a converged one
    assert (PS.stationarity(map_estimate(y, rows, pm, n_iter=1), y, rows, pm)
            > PS.stationarity(map_estimate(y, rows, pm, n_iter=400), y, rows, pm))


def test_stationarity_is_zero_for_a_converged_fit():
    rows, y = _rows_and_signal(_hetero_case(), ACQ, n_vox=600)
    pm = {k: v[0] for k, v in TISSUE.items()}
    x = map_estimate(y, rows, pm, n_iter=400)
    assert PS.stationarity(x, y, rows, pm) < 1e-3


def test_every_arm_reads_the_rows_its_fisher_matrix_describes():
    """The invariant #48 depends on: estimator rows and Fisher rows must be the same experiment.

    `_rows_and_signal` and `fisher.measurement_rows` are two hand-maintained lists of the same
    measurement set. If they drift, the predicted leak describes an acquisition nobody ran and
    every ratio in the arm sweep is meaningless. Checked per arm rather than assumed.
    """
    for arm, n in (("A1", 3), ("A3", 1), ("A4", 2)):
        got = _rows_and_signal(_hetero_case(), ACQ, arm=arm, n_vox=400)
        assert got is not None, arm
        rows, y = got
        want = F.measurement_rows(arm, ACQ)
        assert len(rows) == len(want) == n, (arm, len(rows), len(want))
        assert y.shape[0] == n, (arm, y.shape)
        # the anchor row is the one that is not a (TR, TE, b) triple; it must appear in A1 only
        assert ("anchor" in rows) == (arm == "A1"), arm


def test_arms_the_cache_cannot_support_return_none_rather_than_silently_giving_a1():
    """A2 needs the low-b band and A5 is counterfactual. Before #48 every arm returned A1."""
    for arm in ("A2", "A5"):
        assert _rows_and_signal(_hetero_case(), ACQ, arm=arm, n_vox=400) is None, arm


def test_dropping_a_measurement_actually_changes_the_fit():
    """Regression: the arm argument existed and was ignored, so every caller silently got A1."""
    a1 = PS.case_shifts(_hetero_case(), ACQ, deltas=(0.2231,), n_vox=400, arm="A1")["0.2231"]
    a4 = PS.case_shifts(_hetero_case(), ACQ, deltas=(0.2231,), n_vox=400, arm="A4")["0.2231"]
    # removing the ADC anchor must move D from pinned to prior-dominated
    assert a1["D"]["predicted"]["D"] < 0.01 * a4["D"]["predicted"]["D"]


def test_the_weights_are_the_ones_the_fit_actually_runs_at():
    """Leaving fisher_matrix at its unit default inflates every shrinkage by 1/sigma^2.

    A global noise scale cancels out of the CRLB ORDERING, which is why this survived
    review -- but shrinkage is an absolute fraction, and at sigma = 0.05 that is 400x.
    """
    per = PS.case_shifts(_case(seed=0), ACQ, deltas=(0.0100,), sigma=0.05, n_vox=800)["0.0100"]
    unweighted = F.shrinkage(F.fisher_matrix(F.measurement_rows("A1", ACQ)))["PD"]
    assert per["PD"]["predicted"]["PD"] / 0.01 < unweighted / 2


def test_prediction_scales_with_delta():
    """M*Delta is linear in Delta; a doubled shift must move the estimate twice as far."""
    cached = _case(seed=1)
    got = PS.case_shifts(cached, ACQ, deltas=(0.2231, 0.4462), n_vox=800)
    a = got["0.2231"]["T1"]["observed"]["T1"]
    b = got["0.4462"]["T1"]["observed"]["T1"]
    assert abs(b / a - 2.0) < 0.25


def test_shifting_one_prior_moves_another_estimate():
    """T1 and PD share one product in the T2W signal, so the leak crosses between them."""
    cached = _case(seed=2)
    per = PS.case_shifts(cached, ACQ, deltas=(0.6931,), n_vox=800)["0.6931"]
    assert abs(per["T1"]["observed"]["PD"]) > 1e-4


def test_fit_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.01, 0.5, 500)
    assert abs(PS._fit_slope(pred, 0.7 * pred + rng.normal(0, 0.001, 500))["slope"] - 0.7) < 0.02


def test_fit_slope_flags_a_regressor_with_no_spread():
    """D's predicted shift is the same number every case -- a level test, not a slope."""
    rng = np.random.default_rng(0)
    flat = np.full(500, 0.0062) + rng.normal(0, 1e-7, 500)
    assert PS._fit_slope(flat, flat * 0.9)["x_spread"] < 0.05
    assert PS._fit_slope(rng.uniform(0.01, 0.5, 500), rng.uniform(0.01, 0.5, 500))["x_spread"] > 0.05


def test_fit_slope_refuses_too_few_points():
    assert PS._fit_slope([0.1] * 5, [0.1] * 5)["slope"] != PS._fit_slope([0.1] * 5, [0.1] * 5)["slope"]


def test_the_protocol_only_predictor_uses_no_image_information():
    """The paper says the prediction is computable before the images exist. Only one column is.

    `header` linearizes the Fisher rows at this case's fitted ROI median, so it reaches the
    images through theta-hat. `protocol` uses the nominal point and the acquisition alone.
    They are not close: on the exact render the ratios are PD 0.852 vs 0.486, T2 0.708 vs 0.885.
    """
    a = PS.case_shifts(_case(seed=0), ACQ, deltas=(0.2231,), n_vox=600)["0.2231"]
    # noise, or the render is identical whatever the seed and the test proves nothing
    b = PS.case_shifts(_case(seed=3, noise=0.05), ACQ, deltas=(0.2231,), n_vox=600)["0.2231"]
    # protocol depends only on the acquisition, so two different images give the same prediction
    for p in PARAMS:
        assert abs(a[p]["predicted_protocol"][p] - b[p]["predicted_protocol"][p]) < 1e-12, p
    # header does not
    assert any(abs(a[p]["predicted"][p] - b[p]["predicted"][p]) > 1e-9 for p in PARAMS)


def test_the_curvature_gap_collapses_as_the_prior_tightens():
    """A sharper prediction than any single ratio, and the same for every parameter.

    obs/Fisher is not a constant of the method: it is set by how far the curvature term moves a
    near-singular F+Lambda, so it must go to 1 as the prior dominates F and to 0 as it vanishes.
    On the exact render the Hessian/Fisher leak ratio runs 1.000, 0.973, 0.758, 0.257, 0.030 at
    prior_sd 1, 3, 10, 30, 100 -- and PD, T1 and T2 agree to three decimals at every point,
    because the gap is a property of the matrix rather than of the parameter.

    Our operating point is prior_sd = 10, where this predicts 0.758. The cohort measures 0.764,
    0.688 and 0.733.
    """
    from physharm.analysis.synthetic_arm import _model_log
    rows, y = _rows_and_signal(_case(seed=0), ACQ, n_vox=800)
    x = map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()}, n_iter=400)
    xs = np.median(x, axis=1)
    w, h = 400.0, 1e-5
    fn = lambda v: _model_log(v[:, None], rows)[:, 0]
    J = _jacobian(xs[:, None], rows)[:, :, 0]
    r = y[:, 0] - fn(xs)
    A = np.zeros((3, 4, 4))
    for i in range(4):
        for j in range(4):
            ei = np.zeros(4); ei[i] = h
            ej = np.zeros(4); ej[j] = h
            A[:, i, j] = (fn(xs+ei+ej) - fn(xs+ei-ej) - fn(xs-ei+ej) + fn(xs-ei-ej)) / (4*h*h)
    C = -np.einsum("m,mij->ij", w * r, A)
    Fm = J.T @ (w * np.eye(3)) @ J

    def ratio(ps):
        lam = 1.0 / ps ** 2
        mf = np.diag(np.linalg.inv(Fm + lam*np.eye(4)) * lam)
        mh = np.diag(np.linalg.inv(Fm + C + lam*np.eye(4)) * lam)
        return [mh[i] / mf[i] for i in range(3)]        # PD, T1, T2

    tight, mid, loose = ratio(1.0), ratio(10.0), ratio(100.0)
    assert all(v > 0.99 for v in tight), tight          # strong prior: no gap
    assert all(0.70 < v < 0.80 for v in mid), mid       # operating point: ~0.758
    assert all(v < 0.10 for v in loose), loose          # weak prior: prediction collapses
    for band in (tight, mid, loose):                    # same for every parameter
        assert max(band) - min(band) < 0.01, band


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


# ---------------------------------------------------------------- the one-scalar correction

def _scalar_case(noise=0.0, seed=0):
    return PS.case_shifts(_case(seed=seed, noise=noise), ACQ, deltas=(0.0100,), n_vox=800,
                          n_perm=20)


def test_the_scalar_correction_reproduces_the_full_hessian():
    """The theorem used as a method: correcting one eigendirection does the Hessian's job.

    `posterior_curvature` builds the whole 4x4 curvature from 2p gradient evaluations. The
    derivation says the leak's error lives entirely in the near-null eigendirection, so shifting
    that one eigenvalue by v'Cv should reproduce it. If this test fails the derivation is
    describing something other than what the correction actually does.
    """
    per = _scalar_case()["0.0100"]
    for p in PARAMS:
        ps, ph = per[p]["predicted_scalar"][p], per[p]["predicted_hessian"][p]
        assert abs(ps / ph - 1.0) < 0.01, ("scalar vs hessian", p, ps, ph)


def test_the_scalar_correction_leaves_the_directly_measured_parameter_alone():
    """Consequence (ii), and the bug that a uniform rescale introduces.

    D is measured directly by the ADC map, has no loading on v, and its Fisher leak is already
    right at a ratio of 1.000. The first version of this correction multiplied the WHOLE leak
    matrix by 1 - v'Cv/lam, which is the obvious reading of the theorem and is wrong: it moved
    D to 1.474. The correction has to be applied in the eigenbasis, to the v-v term alone.
    """
    per = _scalar_case()["0.0100"]
    ob, pf, ps = (per["D"]["observed"]["D"], per["D"]["predicted_voxel"]["D"],
                  per["D"]["predicted_scalar"]["D"])
    assert abs(ob / pf - 1.0) < 0.01, ("fisher was already right for D", ob / pf)
    assert abs(ob / ps - 1.0) < 0.01, ("scalar must not touch D", ob / ps)


def test_equation_8_holds_by_two_independent_routes():
    """v'Cv from differentiating the gradient equals -<Wr, g> from the model's own curvature.

    The two share nothing but the fitted point: one differentiates J'Wr along v and subtracts
    v'Fv, the other evaluates v'd2f_m v by a second difference of the model and contracts it
    with the weighted residual. Agreement is evidence the derivation is right rather than
    merely self-consistent, which is the whole claim of section 3.
    """
    from physharm.analysis.synthetic_arm import map_estimate
    rows, y = _rows_and_signal(_case(seed=0), ACQ, n_vox=400)
    base = map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()}, n_iter=600)
    J = _jacobian(base, rows)
    w, lam = 400.0, 0.01
    Fv = np.einsum("mkn,mln->kln", J, J) * w + lam * np.eye(4)[:, :, None]
    v, _ = PS.null_direction(Fv - lam * np.eye(4)[:, :, None])
    ca = PS.curvature_along(base, y, rows, v, w, J=J)
    rel = np.abs(ca["vCv"] + ca["rg"]) / np.maximum(np.abs(ca["vCv"]), 1e-300)
    assert np.median(rel) < 1e-4, ("equation 8", float(np.median(rel)))


def test_the_correction_costs_two_gradient_evaluations_and_the_hessian_costs_2p():
    """The scaling claim, counted rather than asserted.

    A gradient evaluation is one model call and one Jacobian call. The full-Hessian route needs
    2p of them and doubles with the parameter count; the scalar route needs 2 whatever p is,
    because J at the fitted point is already built for the Fisher matrix and v'Fv reuses it.
    Wall clock at p = 4 does NOT favour the scalar route -- the per-voxel eigendecomposition
    dominates -- so the count is the honest form of the claim and this test pins it.
    """
    from physharm.analysis import synthetic_arm as SA
    from physharm.analysis.synthetic_arm import map_estimate
    rows, y = _rows_and_signal(_case(seed=0), ACQ, n_vox=200)
    base = map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()}, n_iter=600)
    J = _jacobian(base, rows)
    w, lam = 400.0, 0.01
    Fv = np.einsum("mkn,mln->kln", J, J) * w + lam * np.eye(4)[:, :, None]
    v, _ = PS.null_direction(Fv - lam * np.eye(4)[:, :, None])

    counts = {"model": 0, "jac": 0}
    real_model, real_jac = PS._model_log, PS._jacobian

    def cm(*a, **k):
        counts["model"] += 1
        return real_model(*a, **k)

    def cj(*a, **k):
        counts["jac"] += 1
        return real_jac(*a, **k)

    PS._model_log, PS._jacobian = cm, cj
    try:
        PS.curvature_along(base, y, rows, v, w, J=J, with_sign=False)
        scalar = dict(counts)
        counts.update(model=0, jac=0)
        PS.posterior_curvature(base, y, rows)
        hess = dict(counts)
    finally:
        PS._model_log, PS._jacobian = real_model, real_jac

    assert scalar == {"model": 2, "jac": 2}, scalar
    assert hess == {"model": 8, "jac": 8}, hess          # 2p at p = 4


# ------------------------------------------------------------------- the sign condition

def _sign_inputs(seed=0, m=3, n=2000):
    rng = np.random.default_rng(seed)
    return rng, rng.normal(1.0, 0.4, (m, n))


def test_both_nulls_are_quiet_when_the_residual_is_noise_independent_of_the_curvature():
    """The case the standard justification assumes. Neither null may fire on it."""
    rng, g = _sign_inputs()
    d = PS.sign_diagnostics(rng.normal(0, 0.05, g.shape), g, 400.0, n_perm=200)
    assert abs(d["z_perm"]) < 3.0, d
    assert abs(d["z_flip"]) < 3.0, d


def test_the_flip_null_catches_a_biased_residual_and_the_permutation_null_does_not():
    """Misspecification as a systematic offset, with no voxel-by-voxel covariance.

    Permuting voxels preserves both marginals, so it CANNOT see an offset -- the permuted mean
    is the observed mean. The sign flip can, because "r is zero-mean noise" makes the statistic
    symmetric about zero. Reporting only the permutation null would call this case null.
    """
    rng, g = _sign_inputs()
    d = PS.sign_diagnostics(rng.normal(0, 0.05, g.shape) + 0.03, g, 400.0, n_perm=200)
    assert abs(d["z_perm"]) < 3.0, ("permutation cannot see an offset", d)
    assert d["z_flip"] > 10.0, ("flip must see it", d)


def test_the_permutation_null_catches_covariance_that_the_marginals_do_not_explain():
    """The other mechanism: r and g covarying voxel by voxel, with r zero-mean overall.

    This is the case the paper's argument turns on -- the fit absorbs misspecification by
    moving, which changes the curvature along v, so residual and curvature covary whatever the
    sign of the misspecification. The permutation null is what isolates it.
    """
    rng, g = _sign_inputs()
    r = 0.02 * (g - g.mean()) + rng.normal(0, 0.05, g.shape)
    d = PS.sign_diagnostics(r - r.mean(), g, 400.0, n_perm=200)
    assert d["z_perm"] > 10.0, ("permutation must see pairing", d)


def test_null_direction_is_the_smallest_eigenvector_and_is_unit_norm():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 4, 4))
    Fd = np.moveaxis(np.einsum("nij,nkj->nik", A, A), 0, 2)      # (4, 4, n), PSD
    v, f1 = PS.null_direction(Fd)
    for n in range(Fd.shape[2]):
        ev = np.linalg.eigvalsh(Fd[:, :, n])
        assert abs(f1[n] - ev.min()) < 1e-10
        assert abs(np.linalg.norm(v[:, n]) - 1.0) < 1e-12
        assert np.allclose(Fd[:, :, n] @ v[:, n], f1[n] * v[:, n], atol=1e-9)
