r"""Does the protocol predict, in absolute terms, how far the prior moves the estimate? (#10)

THE PROBLEM THIS FIXES. The within-scanner rank test in `real_arm.py` asks whether cases with
more shrinkage sit closer to the prior. It works only where shrinkage VARIES across cases, and
on this cohort it barely does: (1-s) swings 1.21x for PD, 1.09x for T2, and 1.0009x for D. So
three of four parameters were untestable, and the one that passed (T1) is the one the
acquisition barely measures.

THE SHARPER TEST. Move the prior mean and watch the estimate follow. For a MAP fit,

    theta_hat(x0 + Delta) - theta_hat(x0)  =  M Delta,     M = (F + L)^-1 L

with M computed from the DICOM acquisition parameters alone -- TR, TE, b -- at a fixed
linearization point, never from the images. Two things this buys:

  * A MAGNITUDE, not an ordering. Every case makes a numeric prediction that the images
    either meet or miss, so the residual is interpretable instead of a rank statistic.
  * LEVERAGE EVERYWHERE, because Delta is ours to choose. D's shrinkage is 0.009 and does not
    move across the cohort, so no correlation exists to measure -- but M_DD * Delta is a
    definite displacement, and a 2x shift of the prior mean predicts a 0.6% move in ADC that
    is either there or not.

The claim under test is the one that would be useful if it held: **the protocol alone tells you
how much of the prior you are about to read back out of the image.** A slope of 1 means the
header predicts the leak; 0.6 means it over-predicts it by 40%; 0 means the Fisher calculation
describes an estimator nobody is running.

OFF-DIAGONAL TERMS ARE INCLUDED. T1 and PD enter the T2W signal through one product, so
shifting the T1 prior moves the PD estimate. Shifting one parameter at a time recovers a
column of M, which is why the run does four shifted fits per case and not one.

WHAT WOULD FALSIFY IT: observed displacements uncorrelated with M Delta, or a slope far from 1
that varies by scanner -- either would mean the identifiability calculation does not describe
what the fit does to real images.

    python -m physharm.analysis.prior_shift --cache-dir cache_shared \
        --manifest-dir manifests_site --out runs/prior_shift.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

from physharm.analysis.crlb_table import acquisition, canonical_scanner
from physharm.analysis.synthetic_arm import (PARAMS, TISSUE, _jacobian, _model_log,
                                             map_estimate, spearman)
from physharm.analysis.traveling_arm import _rows_and_signal, load_case
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

#: prior-mean displacements to test, in log units. log 2 is a 2x change -- large enough to
#: clear fit noise on D, small enough that the linearization is not obviously abused. The
#: smaller value is carried so the two can be compared: a mechanism linear in Delta must give
#: the same slope at both, and a discrepancy is nonlinearity rather than a failed prediction.
DELTAS = (0.2231, 0.6931)          # log 1.25, log 2


def code_fingerprint():
    """First 12 hex of a sha256 over this module's own source.

    The cluster copy is an scp'd tree, not a checkout, so there is no commit to read and no way
    to tell from an output whether it came from current code. On 2026-08-27 a full 4,608-case run
    was spent reproducing an earlier one byte for byte because a fix had been committed and PR'd
    but never deployed -- and nothing in the output said so. Printing this makes the question
    answerable from the log alone: two runs sharing a fingerprint ran the same code, whatever
    anyone believes about what was deployed.
    """
    try:
        return hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]
    except OSError:
        return "unknown"


def posterior_curvature(x, y, rows, prior_sd=10.0, sigma=0.05, h=1e-5):
    """True curvature of the negative log posterior AT EACH VOXEL, shape (4, 4, n).

    The Fisher matrix is the GAUSS-NEWTON approximation to this: it keeps J^T W J and drops
    sum_m r_m d2f_m/dx2, the term carrying the residual. Textbook advice is that the dropped
    term is negligible when the residual is small. On a noiseless render of the exact model
    the residual is 3e-5 and the two matrices agree to four digits on the diagonal -- and the
    leak coefficients they predict still differ by 25-30%, because F + Lambda is near-singular
    and a 1.4e-5 relative perturbation of a near-singular matrix is not a small perturbation
    of its inverse.

    PER VOXEL, AND AGAINST THAT VOXEL'S OWN DATA. The first version of this evaluated the
    gradient at the ROI-median fit while feeding it voxel 0's signal. Every voxel of the
    synthetic render carries identical tissue, so the two agreed and the tests passed; on real
    images the residual between one voxel's data and the ROI median is the tissue heterogeneity
    of the whole prostate, which inflated the curvature term by two to three orders of
    magnitude and drove every Hessian prediction to zero.
    """
    w, lam = 1.0 / sigma ** 2, 1.0 / prior_sd ** 2

    def grad(X):                                   # (4, n); the prior term cancels in the
        r = y - _model_log(X, rows)                # central difference, so it is omitted
        return np.einsum("mkn,mn->kn", _jacobian(X, rows), w * r)

    H = np.empty((4, 4, x.shape[1]))
    for k in range(4):
        e = np.zeros((4, 1))
        e[k] = h
        H[:, k, :] = (grad(x + e) - grad(x - e)) / (2.0 * h)
    return -H + lam * np.eye(4)[:, :, None]


def _leak_per_voxel(A, lam):
    """median over voxels of diag-and-off-diag of A^-1 * lam, as a (4, 4) matrix."""
    M = np.linalg.inv(np.moveaxis(A, 2, 0)) * lam   # (n, 4, 4)
    return np.median(M, axis=0)


def baseline_leaks(J, r, w, lam, H=None):
    """The three curvature approximations the ML literature actually uses, on the same fits.

    Each one is a named method's matrix, not a strawman:

      empirical Fisher   sum_m g_m g_m' over per-measurement gradients g_m = w r_m J_m. This is
                         the matrix Kunstner et al. (2019) argue is a poor stand-in for the
                         Hessian. We cite that argument; this measures it here.
      diagonal Fisher    diag(J'WJ). What EWC (Kirkpatrick et al., 2017) penalises with, and
                         the cheapest thing anyone ships.
      diagonal Hessian   diag of the true posterior curvature. Separates the cost of dropping
                         the residual term from the cost of dropping the OFF-DIAGONALS.

    THE OFF-DIAGONALS ARE THE POINT. v is a near-null direction of F, and in these protocols it
    is not axis-aligned -- it mixes the parameters. Any diagonal approximation discards exactly
    the structure that makes F near-singular, so it cannot see the amplification the paper is
    about. Expect the diagonal baselines to fail differently from the Fisher one: not too large
    by a third, but insensitive to the near-null direction altogether.

    Returns {name: (4, 4) leak matrix}, each a median over voxels like _leak_per_voxel.
    """
    out = {}
    Femp = np.einsum("mn,mkn,mln->kln", (w * r) ** 2, J, J)
    out["empirical_fisher"] = _leak_per_voxel(Femp + lam * np.eye(4)[:, :, None], lam)

    Fv = np.einsum("mkn,mln->kln", J, J) * w
    d = np.einsum("kkn->kn", Fv)                             # diag only, off-diagonals zeroed
    out["diagonal_fisher"] = _leak_per_voxel(
        d[:, None, :] * np.eye(4)[:, :, None] + lam * np.eye(4)[:, :, None], lam)

    if H is not None:                                        # H already carries +lam*I
        dh = np.einsum("kkn->kn", H)
        out["diagonal_hessian"] = _leak_per_voxel(
            dh[:, None, :] * np.eye(4)[:, :, None], lam)
    return out


def leak_scalar_corrected(A, lam, v, f1, vCv):
    """Fisher leak with the near-null eigenvalue shifted by v'Cv. A is F + lam*I, (4, 4, n).

    NOT A UNIFORM RESCALE OF M, which is the first thing one writes and is wrong. Multiplying
    the whole leak matrix by 1 - v'Cv/lam corrects every parameter that loads on v and corrupts
    the one that does not: D is measured directly by the ADC map, its Fisher leak is already
    right (ratio 1.000), and the uniform rescale drove it to 1.474. Consequence (ii) of the
    derivation is explicit that a parameter orthogonal to v is untouched, so the correction
    belongs in the eigenbasis, on the v-v term alone:

        M = M_F + [ lam/(f1 + v'Cv + lam) - lam/(f1 + lam) ] v v'

    For a parameter whose leak is dominated by v this is (f1 + lam)/(f1 + v'Cv + lam), which is
    the paper's 1 - v'Cv/lam once f1 << lam; for a parameter orthogonal to v it is nothing. It
    is also the RESUMMED form rather than the first-order one, so it cannot go negative when
    v'Cv is a sizeable fraction of lam -- which on this cohort it is.
    """
    M = np.linalg.inv(np.moveaxis(A, 2, 0)) * lam                  # (n, 4, 4)
    den = f1 + vCv + lam
    dl = np.where(den > 0, lam / np.where(den > 0, den, 1.0) - lam / (f1 + lam), 0.0)
    Vt = v.T                                                       # (n, 4)
    return np.median(M + dl[:, None, None] * (Vt[:, :, None] * Vt[:, None, :]), axis=0)


def null_direction(F_data):
    """Unit eigenvector of F's smallest eigenvalue, per voxel. F_data is (4, 4, n), no lam.

    This is `v`: the direction the measurements do not constrain. It costs nothing extra --
    establishing that there IS a near-null direction to worry about already requires this
    eigendecomposition -- and it is the only thing the scalar correction below needs beyond
    the Fisher analysis a practitioner is running anyway.
    """
    A = np.moveaxis(F_data, 2, 0)
    ev, V = np.linalg.eigh(0.5 * (A + np.swapaxes(A, 1, 2)))    # eigenvalues ascending
    return (np.ascontiguousarray(V[:, :, 0].T),                 # (4, n); sign is arbitrary
            np.ascontiguousarray(ev[:, 0]))                     # and cancels in v'Cv


def curvature_along(x, y, rows, v, w, J=None, h=1e-5, h2=1e-4, with_sign=True):
    """The one scalar the correction needs, from TWO gradient evaluations whatever p is.

    THE THEOREM USED AS A METHOD. `posterior_curvature` builds the whole 4x4 Hessian: 2p
    gradient evaluations and a p x p inverse per voxel. The derivation says the correction to
    EVERY parameter's leak is the single scalar 1 - v'Cv/lam, so only the second derivative
    ALONG v is needed -- one central difference in the direction v, which is two gradient
    evaluations and does not grow with p.

    TWO INDEPENDENT ROUTES TO v'Cv, and the tests check they agree. The first differentiates
    the gradient along v and subtracts v'Fv. The second evaluates the model's own curvature
    g_m = v' d2f_m v and forms -<Wr, g>, which is equation 8. They share no intermediate
    quantity except the fitted point, so agreement is evidence the derivation is right and not
    merely self-consistent.

    A LARGER STEP FOR THE SECOND ROUTE. `g` is a pure second difference, so its denominator is
    h^2; at h = 1e-5 that is 1e-10 against O(1) model values and the cancellation eats six
    digits. h2 = 1e-4 keeps the truncation error negligible and the cancellation at 1e-8.

    WHAT THE CORRECTION ACTUALLY COSTS. Only the first route is needed to correct a leak, and
    it is two gradient evaluations: `J` at the fitted point is passed in because the caller has
    already built it for the Fisher matrix, so v'Fv is free. The second route and the residual
    it needs are for the sign test and the identity check, and `with_sign=False` switches them
    off -- which is what the cost comparison against `posterior_curvature` must measure, since
    that function computes no diagnostics either.

    Returns a dict with the scalar per voxel and, unless disabled, the pieces the sign test
    needs.
    """
    def grad(X):                                   # J'W r, which is -grad of the data term;
        r = y - _model_log(X, rows)                # the prior term cancels in the difference
        return np.einsum("mkn,mn->kn", _jacobian(X, rows), w * r)

    vHv = -np.einsum("kn,kn->n", v, (grad(x + h * v) - grad(x - h * v)) / (2.0 * h))
    Jv = np.einsum("mkn,kn->mn", _jacobian(x, rows) if J is None else J, v)
    vFv = w * np.einsum("mn,mn->n", Jv, Jv)
    out = {"vCv": vHv - vFv, "vFv": vFv}
    if not with_sign:
        return out

    f0 = _model_log(x, rows)
    g = (_model_log(x + h2 * v, rows) - 2.0 * f0 + _model_log(x - h2 * v, rows)) / (h2 * h2)
    r0 = y - f0
    out.update({"r": r0, "g": g, "rg": w * np.einsum("mn,mn->n", r0, g)})
    return out


def sign_diagnostics(r, g, w, n_perm=200, seed=0):
    """Is the residual correlated with the model's curvature along v, or is it noise?

    Equation 8 says the correction's sign is <Wr, g>, and that a residual which is zero-mean
    noise INDEPENDENT of g gives zero in expectation. That independence is the assumption to
    test, and on a fixed clinical cohort it can be tested directly rather than simulated: the
    marginals of r and of g are whatever the anatomy and the acquisition make them, and only
    their PAIRING carries the claim.

      voxel permutation -- pair each voxel's residual with a different voxel's curvature.
          Both marginals are preserved exactly and only the correspondence is destroyed, so
          the gap to the observed value is the part of the inner product that comes from r
          and g covarying voxel by voxel.
      sign flip -- negate a random half of the per-voxel inner products. Under "r is zero-mean
          noise" the statistic is symmetric about zero, so this null is centred at zero by
          construction and the gap measures the systematic part of the residual.

    The two answer different objections -- "you are seeing the marginals" and "you are seeing
    noise" -- and neither is a simulation. Reported as z against each null; a coin flip is 0.
    """
    rng = np.random.default_rng(seed)
    n = r.shape[1]
    t = w * np.einsum("mn,mn->n", r, g)                          # per-voxel <Wr, g>
    obs = float(np.mean(t))
    if n_perm <= 0:                                              # correction-only runs still
        return {"observed": obs, "median": float(np.median(t)),  # want the inner product, and
                "frac_positive": float(np.mean(t > 0)),          # np.mean of an empty null is
                "perm_mean": float("nan"), "perm_sd": float("nan"),   # a warning and a NaN
                "z_perm": float("nan"), "z_flip": float("nan")}

    perm = np.array([w * float(np.mean(np.einsum("mn,mn->n", r[:, rng.permutation(n)], g)))
                     for _ in range(n_perm)])
    flip = np.array([float(np.mean(t * rng.choice((-1.0, 1.0), size=n)))
                     for _ in range(n_perm)])

    def _z(null):
        sd = float(np.std(null))
        return float((obs - float(np.mean(null))) / sd) if sd > 0 else float("nan")

    return {"observed": obs, "median": float(np.median(t)),
            "frac_positive": float(np.mean(t > 0)),
            "perm_mean": float(np.mean(perm)), "perm_sd": float(np.std(perm)),
            "z_perm": _z(perm), "z_flip": _z(flip)}


def stationarity(x, y, rows, prior_mean, prior_sd=10.0, sigma=0.05):
    """How far the fit is from a stationary point, on a scale that cannot blow up.

    At the MAP, J^T W r - Lambda (x - x0) = 0, and the displacement identity
    theta_hat(x0 + Delta) - theta_hat(x0) = M Delta follows from perturbing that condition. A
    fit that has not reached it does not obey the identity.

    NORMALISED BY THE SUM OF BOTH TERMS, NOT BY THE PRIOR PULL ALONE. The first version divided
    by ||Lambda (x - x0)||, which vanishes for any case whose fit sits near the prior mean -- so
    the ratio exploded on cases where the gradient was tiny too. On the cohort it flagged 41.3%
    of A1 with a clean bimodal split, 30% at machine zero and the rest at 1e3-1e4, which is the
    signature of a vanishing denominator rather than of a solver walking away. That number should
    not be quoted.

    ||g|| / (||J^T W r|| + ||Lambda (x - x0)||) is bounded in [0, 1] by the triangle inequality,
    is 0 exactly at a stationary point, and has no denominator that can vanish while the
    numerator does not.
    """
    w, lam = 1.0 / sigma ** 2, 1.0 / prior_sd ** 2
    x0 = np.array([np.log(prior_mean[k]) for k in PARAMS])[:, None]
    r = y - _model_log(x, rows)
    data = np.einsum("mkn,mn->kn", _jacobian(x, rows), w * r)
    pull = lam * (x - x0)
    g = data - pull
    denom = np.linalg.norm(data, axis=0) + np.linalg.norm(pull, axis=0)
    ok = denom > 0
    if not ok.any():
        return 0.0
    return float(np.median(np.linalg.norm(g, axis=0)[ok] / denom[ok]))


def case_shifts(cached, acq, deltas=DELTAS, prior_sd=10.0, sigma=0.05, n_vox=4000, seed=0,
                n_iter=25, arm="A1", n_perm=200):
    """Observed vs predicted displacement of every parameter when one prior mean moves.

    Returns {delta: {shifted_param: {observed: {p: dx}, predicted: {p: dx}}}}. The observed
    number is the MEDIAN OVER VOXELS OF THE PER-VOXEL DIFFERENCE, not the difference of two
    medians -- the fits are paired voxel by voxel, and pairing is free accuracy.
    """
    got = _rows_and_signal(cached, acq, arm=arm, n_vox=n_vox, rng=np.random.default_rng(seed))
    if got is None:
        return None
    rows, y = got
    base_mean = {k: v[0] for k, v in TISSUE.items()}
    base = map_estimate(y, rows, base_mean, prior_sd=prior_sd, sigma=sigma, n_iter=n_iter)
    theta = {k: float(np.exp(np.median(base[i]))) for i, k in enumerate(PARAMS)}

    # THE WEIGHTS MATTER. map_estimate runs at W = 1/sigma^2; leaving fisher_matrix at its
    # unit default made F smaller by that factor and every shrinkage correspondingly larger.
    # A global noise scale cancels out of the CRLB ORDERING, which is why it went unnoticed --
    # but shrinkage is an absolute fraction, and at sigma = 0.05 the difference is 400x.
    lam = 1.0 / prior_sd ** 2
    w = 1.0 / sigma ** 2

    # THREE PREDICTORS, all of the same quantity.
    #   header   -- what a practitioner can compute before touching the images: Fisher from
    #               TR/TE/b at a single linearization point.
    #   fisher   -- Gauss-Newton per voxel, at that voxel's own fit. Isolates the effect of
    #               the linearization point from the effect of the dropped curvature term.
    #   hessian  -- true posterior curvature per voxel. What the estimator actually follows.
    # THE PAPER CLAIMS THIS IS COMPUTABLE BEFORE THE IMAGES EXIST. Only `protocol` is.
    # `header` evaluates the Fisher rows at the FITTED ROI median, so it reaches the images
    # through theta-hat; `protocol` uses the nominal linearization point and nothing else.
    # Measured on the exact render, the two differ by a lot for PD (0.852 vs 0.486) and little
    # for T1 (0.719 vs 0.744), so which one is reported is not a detail.
    M_header = F.shrinkage_matrix(
        F.fisher_matrix(F.measurement_rows(arm, acq, theta), [w] * len(rows)), prior_sd)
    M_protocol = F.shrinkage_matrix(
        F.fisher_matrix(F.measurement_rows(arm, acq, None), [w] * len(rows)), prior_sd)
    J = _jacobian(base, rows)
    Fv = np.einsum("mkn,mln->kln", J, J) * w + lam * np.eye(4)[:, :, None]

    # THE SCALAR CORRECTION, timed against the full-Hessian one it is meant to replace. Both
    # clocks cover only the curvature work, not the fit, because the fit is shared and would
    # dilute the ratio the paper quotes.
    t0 = time.perf_counter()
    try:
        M_hess = _leak_per_voxel(
            posterior_curvature(base, y, rows, prior_sd, sigma), lam)
    except np.linalg.LinAlgError:
        return None
    t_hess = time.perf_counter() - t0

    t0 = time.perf_counter()
    v, f1 = null_direction(Fv - lam * np.eye(4)[:, :, None])
    ca = curvature_along(base, y, rows, v, w, J=J, with_sign=False)
    try:
        M_scalar = leak_scalar_corrected(Fv, lam, v, f1, ca["vCv"])
    except np.linalg.LinAlgError:
        return None
    t_scal = time.perf_counter() - t0
    # The sign test and the equation-8 identity are diagnostics, not part of the correction,
    # so they sit outside the clock that is compared against the Hessian route.
    ca.update(curvature_along(base, y, rows, v, w, J=J))

    try:
        M_fisher = _leak_per_voxel(Fv, lam)            # the baseline, needed either way, so
    except np.linalg.LinAlgError:                      # it is outside both clocks
        return None
    scal = np.where(f1 + ca["vCv"] + lam > 0,
                    (f1 + lam) / np.where(f1 + ca["vCv"] + lam > 0,
                                          f1 + ca["vCv"] + lam, 1.0), np.nan)

    if not all(np.all(np.isfinite(m)) for m in (M_fisher, M_hess, M_scalar)):
        return None

    out = {"__stationarity": stationarity(base, y, rows, base_mean, prior_sd, sigma),
           "__cost": {"hessian_s": t_hess, "scalar_s": t_scal, "n_vox": int(y.shape[1])},
           "__scalar": {"median": float(np.nanmedian(scal)),
                        "p05": float(np.nanpercentile(scal, 5)),
                        "p95": float(np.nanpercentile(scal, 95)),
                        "vCv_over_lam": float(np.median(ca["vCv"]) / lam),
                        "f1_over_lam": float(np.median(f1) / lam),
                        # v'Cv from the gradient route against -<Wr, g> from the model route.
                        # Two derivations of equation 8 that share only the fitted point; a
                        # cohort-wide disagreement would mean one of them is wrong.
                        "identity_rel_err": float(np.median(
                            np.abs(ca["vCv"] + ca["rg"])
                            / np.maximum(np.abs(ca["vCv"]), 1e-300)))},
           "__sign": sign_diagnostics(ca["r"], ca["g"], w, n_perm=n_perm, seed=seed)}
    for d in deltas:
        per_shift = {}
        for j, pj in enumerate(PARAMS):
            mean = dict(base_mean)
            mean[pj] = float(base_mean[pj] * np.exp(d))
            x = map_estimate(y, rows, mean, prior_sd=prior_sd, sigma=sigma, n_iter=n_iter)
            obs = np.median(x - base, axis=1)
            if not np.all(np.isfinite(obs)):
                return None
            per_shift[pj] = {
                "observed": {p: float(obs[i]) for i, p in enumerate(PARAMS)},
                "predicted": {p: float(M_header[i, j] * d) for i, p in enumerate(PARAMS)},
                "predicted_protocol": {p: float(M_protocol[i, j] * d)
                                       for i, p in enumerate(PARAMS)},
                "predicted_voxel": {p: float(M_fisher[i, j] * d) for i, p in enumerate(PARAMS)},
                "predicted_hessian": {p: float(M_hess[i, j] * d) for i, p in enumerate(PARAMS)},
                "predicted_scalar": {p: float(M_scalar[i, j] * d) for i, p in enumerate(PARAMS)},
            }
        out[f"{d:.4f}"] = per_shift
    return out


def collect(cache_dir, manifests, prior_sd=10.0, sigma=0.05, n_vox=4000, limit=0,
            deltas=DELTAS, log=print, n_iter=25, arm="A1", n_perm=200):
    rows, why = [], Counter()
    for i, m in enumerate(manifests):
        if limit and len(rows) >= limit:
            break
        acq, bad = acquisition(m)
        if acq is None:
            why[bad or "no acquisition params"] += 1
            continue
        cached = load_case(cache_dir, m.get("case_id"))
        if cached is None:
            why["not in cache"] += 1
            continue
        sh = case_shifts(cached, acq, deltas, prior_sd, sigma, n_vox, n_iter=n_iter, arm=arm,
                         n_perm=n_perm)
        if sh is None:
            why["fit failed"] += 1
            continue
        rows.append({"case_id": m.get("case_id"), "patient_id": m.get("patient_id"),
                     "scanner": canonical_scanner(m.get("scanner_model")),
                     "stationarity": sh.pop("__stationarity"),
                     "cost": sh.pop("__cost"), "scalar": sh.pop("__scalar"),
                     "sign": sh.pop("__sign"), "shifts": sh})
        if log and (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(manifests)}  kept {len(rows)}")
    return rows, why


def _bootstrap_ratio(pred, obs, groups, n=400, seed=0):
    """95% CI on the median ratio, resampling PATIENTS rather than cases."""
    rng = np.random.default_rng(seed)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    keys = list(idx)
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    out = []
    for _ in range(n):
        take = [i for k in rng.choice(keys, len(keys)) for i in idx[k]]
        pr, ob = pred[take], obs[take]
        ok = np.isfinite(pr) & np.isfinite(ob) & (pr != 0)
        if ok.sum() > 20:
            out.append(float(np.median(ob[ok] / pr[ok])))
    if not out:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5))}


def _fit_slope(pred, obs):
    """Least squares through the origin, plus the spread of the regressor that supports it.

    REPORT THE MEDIAN RATIO, NOT THE SLOPE. A slope through the origin is tail-weighted, and
    on the cohort the two disagree badly: PD's observed/predicted ratio falls from 1.05 in the
    smallest quintile of predicted shift to 0.51 in the largest, so the relation is not a line
    through zero and the slope reports mostly the tail. Cohort medians 0.764 / 0.688 / 0.733
    against slopes 0.745 / 0.569 / 1.091 -- the slope makes T2 look like an under-prediction
    when every quintile of it is an over-prediction.

    The slope is only a slope if `pred` varies. Where it does not -- D's predicted shift is
    the same number for every case -- it degenerates into a comparison of levels, which is
    still a real test of the magnitude but is not evidence about the trend. `x_spread` is
    reported so the two cannot be confused, the same distinction the leverage gate draws in
    real_arm.py.
    """
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    if pred.size < 20 or not np.any(pred):
        # Every key main() reads, so a structurally zero predictor column (A3's D-involving
        # off-diagonals) degrades to NaN instead of raising KeyError deep in the report.
        return {"n": int(pred.size), "slope": float("nan"), "median_ratio": float("nan"),
                "rho": float("nan"), "ratio_p05": float("nan"), "ratio_p95": float("nan"),
                "frac_wild": float("nan"), "x_spread": float("nan"),
                "median_predicted": float("nan"),
                "median_observed": float(np.median(obs)) if obs.size else float("nan")}
    denom = float(np.percentile(np.abs(pred), 50)) or 1.0
    nz = pred != 0
    rat = obs[nz] / pred[nz]
    return {
        # A MEDIAN IS NOT ENOUGH FOR THIS COLUMN. The curvature-corrected prediction has a
        # median of 1.008 for T1 on the cohort and a 95th percentile of 10,090: right on
        # average, unusable per case. Reporting the centre alone would have hidden that.
        "ratio_p05": float(np.percentile(rat, 5)) if rat.size else float("nan"),
        "ratio_p95": float(np.percentile(rat, 95)) if rat.size else float("nan"),
        "frac_wild": float(np.mean((rat < 0) | (rat > 3))) if rat.size else float("nan"),
        "slope": float(pred @ obs / (pred @ pred)),
        # nanmedian, not median: the np.where above deliberately inserts NaN at zero
        # predictors, and plain np.median propagates it -- one structurally zero entry
        # (A3's D-involving off-diagonals, or any case whose b_hi resolves to 0) turns the
        # headline statistic into NaN. Every sibling reduction here filters zeros already.
        "median_ratio": float(np.nanmedian(obs / np.where(pred == 0, np.nan, pred)))
        if np.any(pred != 0) else float("nan"),
        "rho": spearman(pred.tolist(), obs.tolist()),
        "x_spread": float((np.percentile(pred, 95) - np.percentile(pred, 5)) / denom),
        "median_predicted": float(np.median(pred)),
        "median_observed": float(np.median(obs)),
        "n": int(pred.size),
    }


def diagonal(rows, delta_key):
    """Per parameter: does shifting its own prior move its own estimate by the predicted amount?"""
    out = {}
    groups = [r["patient_id"] or r["case_id"] for r in rows]
    for p in PARAMS:
        pr = [r["shifts"][delta_key][p]["predicted"][p] for r in rows]
        ob = [r["shifts"][delta_key][p]["observed"][p] for r in rows]
        d = _fit_slope(pr, ob)
        d["ci"] = _bootstrap_ratio(pr, ob, groups)
        d["protocol"] = _fit_slope([r["shifts"][delta_key][p]["predicted_protocol"][p]
                                    for r in rows], ob)
        d["voxel"] = _fit_slope([r["shifts"][delta_key][p]["predicted_voxel"][p]
                                 for r in rows], ob)
        d["hessian"] = _fit_slope([r["shifts"][delta_key][p]["predicted_hessian"][p]
                                   for r in rows], ob)
        d["scalar"] = _fit_slope([r["shifts"][delta_key][p]["predicted_scalar"][p]
                                  for r in rows], ob)
        d["scalar_ci"] = _bootstrap_ratio([r["shifts"][delta_key][p]["predicted_scalar"][p]
                                           for r in rows], ob, groups)
        d["hessian_ci"] = _bootstrap_ratio([r["shifts"][delta_key][p]["predicted_hessian"][p]
                                            for r in rows], ob, groups)
        by = defaultdict(lambda: ([], []))
        for r in rows:
            a, b = by[r["scanner"]]
            a.append(r["shifts"][delta_key][p]["predicted"][p])
            b.append(r["shifts"][delta_key][p]["observed"][p])
        # Store the MEDIAN RATIO, which is what the stdout table prints and what the paper
        # quotes. This field used to hold the slope under the same name, so anyone reading
        # per_scanner out of the JSON got a different statistic from the one in the table --
        # and for T2 the two disagree on the direction (median 0.733, slope 1.091).
        d["per_scanner"] = {s: _fit_slope(*v)["median_ratio"]
                            for s, v in by.items() if len(v[0]) >= 50}
        d["per_scanner_slope"] = {s: _fit_slope(*v)["slope"]
                                  for s, v in by.items() if len(v[0]) >= 50}
        out[p] = d
    return out


def off_diagonal(rows, delta_key):
    """Cross-parameter leak: shift j's prior, watch i's estimate. i != j."""
    out = {}
    for j in PARAMS:
        for i in PARAMS:
            if i == j:
                continue
            pr = [r["shifts"][delta_key][j]["predicted"][i] for r in rows]
            ob = [r["shifts"][delta_key][j]["observed"][i] for r in rows]
            out[f"{j}->{i}"] = _fit_slope(pr, ob)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-vox", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default="A1", choices=["A1", "A3", "A4"],
                    help="measurement set. A3 is T2W alone, A4 drops the ADC anchor; both are "
                         "subsets of the cache. A2 needs the low-b band and A5 is counterfactual.")
    ap.add_argument("--n-iter", type=int, default=100,
                    help="Damped Gauss-Newton steps. The displacement prediction assumes a "
                         "STATIONARY fit. Raising this no longer fixes a poor stationarity "
                         "rate on its own: the old failure was cycling, not slow "
                         "convergence, and the solver is Levenberg-damped now (#79). The "
                         "fit stops early once every voxel is stationary, so the higher "
                         "default costs nothing on well-behaved cases.")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations for the sign-condition nulls, per exam (0 disables)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mans = read_manifests(args.manifest_dir)
    print(f"code {code_fingerprint()}  |  {len(mans)} manifests; arm {args.arm}; "
          f"1 baseline + 4 shifted MAP fits per case, per delta")
    rows, why = collect(args.cache_dir, mans, args.prior_sd, args.sigma, args.n_vox,
                        args.limit, n_iter=args.n_iter, arm=args.arm,
                        n_perm=args.n_perm)
    print(f"\n{len(rows)} cases fitted")
    for k, v in why.most_common():
        print(f"  skipped {v:5d}  {k}")
    if len(rows) < 100:
        raise SystemExit("too few cases fitted to say anything")

    st = np.array([r["stationarity"] for r in rows if r["stationarity"] == r["stationarity"]])
    print(f"\n=== IS THE FIT STATIONARY? (the displacement identity assumes it is) ===")
    print(f"  ||grad|| / (||data|| + ||prior pull||), bounded in [0,1], at {args.n_iter} steps")
    print(f"  median {np.median(st):.4f}   p95 {np.percentile(st, 95):.4f}   "
          f"max {st.max():.4f}   cases above 0.01: {100*np.mean(st > 0.01):.1f}%")
    if np.median(st) > 0.01:
        print("  NOT STATIONARY. theta_hat(x0+Delta) - theta_hat(x0) = M Delta does not hold,")
        print("  and because map_estimate starts at x = x0 an unconverged fit inherits part of")
        print("  the shift from its moved starting point. Re-run with a larger --n-iter before")
        print("  reading anything below.")

    report = {}
    for d in DELTAS:
        key = f"{d:.4f}"
        dg = diagonal(rows, key)
        od = off_diagonal(rows, key)
        report[key] = {"diagonal": dg, "off_diagonal": od}
        print(f"\n=== PRIMARY: prior mean moved by {np.exp(d):.2f}x (Delta = {d:.4f} log units) ===")
        print("  Predicted displacement M*Delta comes from TR/TE/b in the header, never from")
        print("  the images. A slope of 1.00 means the protocol predicts the leak exactly.\n")
        print(f"  {'param':>5s} {'observed':>10s} {'PROTOCOL':>10s} {'obs/prot':>9s} "
              f"{'header':>10s} {'obs/hdr':>8s} {'95% CI':>16s} {'obs/scal':>9s} "
              f"{'obs/Hess':>9s}   reading")
        for p in PARAMS:
            v, vv, vh = dg[p], dg[p]["voxel"], dg[p]["hessian"]
            r = v["median_ratio"]
            if abs(v["median_observed"]) < 1e-4:
                note = "leak is ~0 — agreement is trivial, not evidence"
            elif abs(r - 1.0) < 0.10:
                note = "Fisher predicts the leak"
            elif r < 1.0:
                note = f"Fisher OVER-predicts by {100*(1-r):.0f}%"
            else:
                note = f"Fisher UNDER-predicts by {100*(r-1):.0f}%"
            ci = f"[{v['ci']['lo']:.3f}, {v['ci']['hi']:.3f}]"
            vp = dg[p]["protocol"]
            print(f"  {p:>5s} {v['median_observed']:>+10.5f} {vp['median_predicted']:>+10.5f} "
                  f"{vp['median_ratio']:>9.3f} {v['median_predicted']:>+10.5f} {r:>8.3f} "
                  f"{ci:>16s} {dg[p]['scalar']['median_ratio']:>9.3f} "
                  f"{vh['median_ratio']:>9.3f}   {note}")
        print(f"\n  {'':>5s} dispersion of obs/Hessian, which the median hides:")
        for p in PARAMS:
            vh = dg[p]["hessian"]
            flag = "  <-- median is not a usable per-case prediction" if vh["frac_wild"] > 0.1 else ""
            print(f"  {p:>5s}   p05 {vh['ratio_p05']:>9.3f}   p95 {vh['ratio_p95']:>12.1f}   "
                  f"{100*vh['frac_wild']:>5.1f}% outside [0, 3]{flag}")
        print("\n  PROTOCOL = Fisher from TR/TE/b at the NOMINAL linearization point. This is the")
        print("             only column computable before the images exist, and the one the")
        print("             paper's claim is about.")
        print("  header   = same, but linearized at this case's fitted ROI median, so it reaches")
        print("             the images through theta-hat. Reported for continuity with earlier runs.")
        print("  Fisher  = Gauss-Newton per voxel; separates linearization from curvature.")
        print("  Hessian = true posterior curvature per voxel. sl.H near 1.00 means the")
        print("            dropped curvature term is what the header prediction is missing.")
        print("\n  cross-parameter leak (shift j's prior, read i's estimate):")
        worst = sorted(od.items(), key=lambda kv: -abs(kv[1].get("median_observed", 0)))[:4]
        for k, v in worst:
            print(f"    {k:>10s}  predicted {v['median_predicted']:>+8.4f}  "
                  f"observed {v['median_observed']:>+8.4f}  slope {v['slope']:>7.3f}")

    # ------------------------------------------------------------------ scalar correction
    key = f"{DELTAS[0]:.4f}"
    dg0 = report[key]["diagonal"]
    print("\n=== THE ONE-SCALAR CORRECTION: does shifting one eigenvalue do the Hessian's job? ===")
    print("  The derivation says the leak's error lives entirely in the near-null direction, so")
    print("  only the second derivative ALONG v is needed: two gradient evaluations, not 2p, and")
    print("  no p x p curvature. If this column matches the Hessian column the theorem is a")
    print("  method and not only an explanation.\n")
    print(f"  {'param':>5s} {'obs/Fisher':>11s} {'obs/scalar':>11s} {'95% CI':>16s} "
          f"{'obs/Hessian':>12s} {'95% CI':>16s}")
    for p_ in PARAMS:
        d = dg0[p_]
        print(f"  {p_:>5s} {d['voxel']['median_ratio']:>11.3f} "
              f"{d['scalar']['median_ratio']:>11.3f} "
              f"[{d['scalar_ci']['lo']:>6.3f}, {d['scalar_ci']['hi']:>6.3f}] "
              f"{d['hessian']['median_ratio']:>12.3f} "
              f"[{d['hessian_ci']['lo']:>6.3f}, {d['hessian_ci']['hi']:>6.3f}]")

    sc = np.array([r["scalar"]["median"] for r in rows], float)
    vc = np.array([r["scalar"]["vCv_over_lam"] for r in rows], float)
    idr = np.array([r["scalar"]["identity_rel_err"] for r in rows], float)
    th = np.array([r["cost"]["hessian_s"] for r in rows], float)
    ts = np.array([r["cost"]["scalar_s"] for r in rows], float)
    print(f"\n  v'Cv/lam over the cohort: median {np.median(vc):.3f}")
    print(f"  correction factor (f1+lam)/(f1+v'Cv+lam): median {np.median(sc):.3f}, "
          f"[p05 {np.percentile(sc, 5):.3f}, p95 {np.percentile(sc, 95):.3f}]")
    print(f"      the paper's first-order form 1 - v'Cv/lam would give "
          f"{1.0 - np.median(vc):.3f}; the gap is the resummation, not a discrepancy")
    print(f"  equation-8 identity v'Cv = -<Wr,g>, two independent routes: median relative "
          f"disagreement {np.median(idr):.2e}")
    print(f"\n  cost of the curvature work per exam (the shared Fisher leak is outside both):")
    print(f"      Hessian {np.median(th)*1e3:7.1f} ms   2p = 8 gradient evaluations, dense 4x4")
    print(f"      scalar  {np.median(ts)*1e3:7.1f} ms   2 gradient evaluations, one eigenvector")
    print("  THE CLAIM IS THE COUNT, NOT THIS WALL CLOCK. At p = 4 with a three-exponential")
    print("  forward model both routes are sub-millisecond and the per-voxel eigendecomposition")
    print("  is most of the scalar route's time, so the measured ratio here says little. What")
    print("  does not depend on the model is that the scalar route needs 2 gradient evaluations")
    print("  whatever p is, and the Hessian route 2p.")
    report["scalar"] = {
        "median": float(np.median(sc)), "p05": float(np.percentile(sc, 5)),
        "p95": float(np.percentile(sc, 95)), "vCv_over_lam_median": float(np.median(vc)),
        "first_order_equivalent": float(1.0 - np.median(vc)),
        "identity_rel_err_median": float(np.median(idr)),
        "hessian_ms": float(np.median(th) * 1e3), "scalar_ms": float(np.median(ts) * 1e3),
        "n": int(sc.size)}

    # ------------------------------------------------------------------ the sign condition
    print("\n=== THE SIGN CONDITION: is the residual correlated with the curvature along v? ===")
    print("  Equation 8 makes the sign <Wr, g>. Noise independent of g gives zero in")
    print("  expectation; that independence is what the standard justification assumes. Both")
    print("  nulls below are built from this cohort's own residuals, not from a simulation.\n")
    zp = np.array([r["sign"]["z_perm"] for r in rows], float)
    zf = np.array([r["sign"]["z_flip"] for r in rows], float)
    fp = np.array([r["sign"]["frac_positive"] for r in rows], float)
    ob = np.array([r["sign"]["observed"] for r in rows], float)
    zp, zf = zp[np.isfinite(zp)], zf[np.isfinite(zf)]
    print(f"  <Wr, g> per exam: median {np.median(ob):+.4g}, "
          f"positive in {100*np.mean(ob > 0):.1f}% of exams")
    if zp.size == 0:
        print("  (nulls not computed: --n-perm 0)")
    else:
        print(f"  voxel permutation (breaks r-g pairing, keeps both marginals):")
        print(f"      median z {np.median(zp):+.2f}, |z| > 2 in "
              f"{100*np.mean(np.abs(zp) > 2):.1f}% of exams")
        print(f"  sign flip (null centred at zero by construction):")
        print(f"      median z {np.median(zf):+.2f}, |z| > 2 in "
              f"{100*np.mean(np.abs(zf) > 2):.1f}% of exams")
    print(f"  per-voxel <Wr,g> positive in a median {100*np.median(fp):.1f}% of voxels "
          f"(a coin flip is 50%)")
    report["sign"] = {
        "observed_median": float(np.median(ob)),
        "frac_exams_positive": float(np.mean(ob > 0)),
        "z_perm_median": float(np.median(zp)) if zp.size else float("nan"),
        "frac_perm_sig": float(np.mean(np.abs(zp) > 2)) if zp.size else float("nan"),
        "z_flip_median": float(np.median(zf)) if zf.size else float("nan"),
        "frac_flip_sig": float(np.mean(np.abs(zf) > 2)) if zf.size else float("nan"),
        "frac_voxels_positive_median": float(np.median(fp)), "n": int(ob.size)}

    print("\n=== PER SCANNER: is the mis-prediction a property of the method or the machine? ===")
    by = defaultdict(list)
    for r in rows:
        by[r["scanner"]].append(r)
    big = {k: v for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(v) >= 50}
    key = f"{DELTAS[0]:.4f}"
    print(f"  {'param':>5s} " + "  ".join(f"{s.split()[-1]:>10s}" for s in big))
    for p in PARAMS:
        print(f"  {p:>5s} " + "  ".join(
            f"{diagonal(rs, key)[p]['median_ratio']:>10.3f}" for rs in big.values()))

    ka, kb = f"{DELTAS[0]:.4f}", f"{DELTAS[1]:.4f}"
    print(f"\n=== LINEARITY: the ratio must not depend on Delta ===")
    for p in PARAMS:
        a = report[ka]["diagonal"][p]["median_ratio"]
        b = report[kb]["diagonal"][p]["median_ratio"]
        print(f"  {p:>5s}  {np.exp(DELTAS[0]):.2f}x: {a:>7.3f}   {np.exp(DELTAS[1]):.2f}x: "
              f"{b:>7.3f}   {'consistent' if abs(a-b) < 0.10 else 'NONLINEAR — Delta too large'}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"n_cases": len(rows), "prior_sd": args.prior_sd, "deltas": list(DELTAS),
                   "n_iter": args.n_iter, "arm": args.arm,
                   "code_fingerprint": code_fingerprint(),
                   "stationarity_median": float(np.median(st)),
                   "report": report, "skipped": dict(why), "cases": rows},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
