r"""Fisher information and identifiability for clinical bpMRI protocols.

This replaces the archived ``physharm/eval/fisher_identifiability.py``, which was
defective in a way that would have inverted this project's central result. See
``crlb()`` and ``the triage notes`` §2a.

The predictor for the whole study is :func:`identifiability`: given a case's *measured*
acquisition parameters it returns, with no training, how well each tissue parameter is
determined. Everything downstream regresses scanner leakage against it.

Deliberately numpy, not torch. The sensitivities are analytic and verified against finite
differences (``tests/test_fisher.py``), so autograd buys nothing, and the predictor must be
computable for 4,673 cases on a login node with no GPU.
"""
from __future__ import annotations

import math

import numpy as np

PARAMS = ("PD", "T1", "T2", "D")

# Linearization point. The Fisher matrix is LOCAL, so these set where the sensitivities are
# evaluated; they are not fitted. Every downstream claim is about the parameter ORDERING,
# which `tests/test_fisher.py` checks is stable under them.
#   T1, T2: prostate at 3T. NEEDS A CITATION before publication -- de Bazelaire 2004
#           (Radiology) is the candidate. Tracked in issue #14.
#   D:      MEASURED in our own cohort (GE 1.30 / Siemens 1.31 x1e-3 mm^2/s, onboarding 3.2).
#   PD:     arbitrary scale; absolute PD is unidentifiable anyway (confounded with the gain).
THETA0 = {"PD": 0.6, "T1": 1400.0, "T2": 80.0, "D": 1.3e-3}


def log_sensitivities(TR_ms, TE_ms, b, theta=None):
    r"""Row of dimensionless log-sensitivities d log S / d log theta for one measurement.

    Signal (spin echo with Stejskal-Tanner attenuation; see ``decoders.render_se_epi``):

        S = g * PD * (1 - exp(-TR/T1)) * exp(-TE/T2) * exp(-b*D)

    With E1 = exp(-TR/T1):

        dlogS/dlogPD =  1
        dlogS/dlogT1 = -(TR/T1) * E1 / (1 - E1)
        dlogS/dlogT2 =  TE / T2
        dlogS/dlogD  = -b * D

    The T1 row carries the factor E1, which is 0.114 / 0.033 / 0.0023 at TR = 2602 / 4084 /
    7300 for T1 = 1200 ms. That is the whole reason T1 is unrecoverable from routine
    T2-weighted acquisitions -- and, because TR differs ~3x across vendors, the reason the
    residual T1 error is vendor-correlated.

    Units: TR/TE/T1/T2 in ms, b in s/mm^2, D in mm^2/s (so b*D is dimensionless).
    Pass b=0 for a non-diffusion-weighted contrast.
    """
    t = dict(THETA0 if theta is None else theta)
    T1, T2, D = t["T1"], t["T2"], t["D"]
    E1 = math.exp(-TR_ms / T1)
    dT1 = -(TR_ms / T1) * E1 / (1.0 - E1) if (1.0 - E1) > 1e-12 else 0.0
    return np.array([1.0, dT1, TE_ms / T2, -b * D], dtype=float)


def anchor_row(param="D"):
    """Row for a directly measured parameter (the scanner's ADC map measures D).

    In log space d log ADC / d log D = 1 and the other entries are zero, so the anchor
    clamps exactly one axis and adds no information about the rest.
    """
    r = np.zeros(len(PARAMS))
    r[PARAMS.index(param)] = 1.0
    return r


def fisher_matrix(rows, weights=None):
    r"""F = J^T W J, with W = diag(1/sigma_c^2) the per-contrast inverse noise variance.

    A GLOBAL noise scale cancels out of the parameter ordering (F scales by 1/sigma^2 and
    the CRLB by sigma), so only the RELATIVE per-contrast weighting matters. That is
    fortunate, because DWI SNR could not be measured reliably on registered volumes
    (the run log Phase 2c) -- and the ordering is invariant over a 625x sweep of relative
    weighting, which ``tests/test_fisher.py`` pins.
    """
    J = np.asarray(rows, dtype=float)
    w = np.ones(len(J)) if weights is None else np.asarray(weights, dtype=float)
    return J.T @ (w[:, None] * J)


def crlb(F, prior_sd=10.0):
    r"""Per-parameter uncertainty from the Fisher matrix. **Never use ``np.linalg.pinv``.**

    For clinical bpMRI there are fewer independent measurements than parameters, so F is
    singular or near-singular with a T1-dominated sloppy direction. The true Cramer-Rao
    bound along a null direction is +infinity: the parameter is unrecoverable.

    ``pinv`` reports the OPPOSITE. It zeroes the null-space contribution -- asserting the
    unidentifiable direction is known exactly -- so a parameter's reported variance FALLS
    as its overlap with that direction RISES. Measured on the archived vendor protocols:

        vendor    |T1 projection onto null|    pinv CRLB(T1)    this function
        GE                  0.834                   4.23             8.97
        Siemens             0.992                   0.61             9.94
        Philips             0.172                  40.21             9.69

    Perfectly monotone, in the wrong direction. At Siemens -- longest TR, therefore the
    LEAST T1 information of the three -- pinv calls T1 the best-determined parameter in the
    model, beating the directly measured D. This project's central regression is leakage on
    predicted CRLB; run with pinv it would have had **the sign backwards for T1**, the one
    parameter the paper is about, and would have looked like a clean refutation.

    Instead, regularize explicitly as a Bayesian bound:

        Cov >= (F + I / prior_sd^2)^-1

    ``prior_sd`` is in log units and is an ASSUMPTION, so it is a named argument. A sloppy
    direction returns ~prior_sd and is thereby visibly prior-dominated rather than silently
    well-determined.

    **prior_sd is NOT a free knob.** The regression predictor is only meaningful in the
    Fisher-dominated regime. Measured across 300 protocols spanning the real the site TR/TE
    ranges, the ACROSS-CASE ranking of CRLB -- which is what leakage is regressed against --
    behaves like this against a prior_sd = 10 reference (Spearman):

        prior_sd      PD       T1       T2        D
             1.0  -0.941    0.674    0.970   -0.314     <- INVERTED
             3.0   0.953    0.909    0.975    1.000
            10.0   1.000    1.000    1.000    1.000
            30.0   0.873    0.994    0.869    1.000
           100.0   0.820    0.992    0.748    1.000

    At small prior_sd the bound saturates -- every case returns ~prior_sd -- so the residual
    ordering is numerical noise on a flat function and can invert the sign of the whole
    regression. Use :func:`prior_dominance` to check the regime rather than assuming it, and
    keep prior_sd well above the sloppiest scale (>= 10 for clinical bpMRI).

    By contrast the relative per-contrast NOISE weighting barely matters: worst-case
    across-case Spearman 0.908-1.000 over a 625x sweep. That is what makes the unmeasurable
    DWI SNR (the run log Phase 2c) a non-issue for the regression.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    sd = np.sqrt(np.clip(np.diag(np.linalg.inv(F + np.eye(n) / prior_sd ** 2)), 0.0, None))
    return dict(zip(PARAMS, sd.tolist()))


def prior_dominance(F, prior_sd=10.0):
    """Fraction of the parameters whose bound is set by the PRIOR rather than the data.

    A parameter is prior-dominated when the prior's precision 1/prior_sd^2 exceeds the
    Fisher information along its own axis. If most parameters are prior-dominated the CRLB
    has saturated and its across-case ordering is meaningless -- see :func:`crlb`.
    """
    F = np.asarray(F, dtype=float)
    lam = 1.0 / prior_sd ** 2
    diag = np.diag(F)
    return {"lambda": lam,
            "fraction_prior_dominated": float(np.mean(diag < lam)),
            "per_param": dict(zip(PARAMS, (diag < lam).tolist()))}


def degenerate(rows):
    """Which parameters have NO measurement at all under this measurement set.

    A parameter whose Jacobian column is identically zero is not estimated -- the fit returns
    the prior, exactly, for every input. Its across-protocol spread is therefore 0 BY
    CONSTRUCTION, which reads as "perfectly scanner-invariant" when it means "we put a
    constant in". Arm A3 (T2W only) does this to D. Report it as not-identifiable, never as
    a leakage of 0.000.
    """
    J = np.asarray(rows, dtype=float)
    return dict(zip(PARAMS, (np.abs(J).sum(axis=0) == 0).tolist()))


def shrinkage(F, prior_sd=10.0):
    """Fraction of each parameter's estimate that is set by the PRIOR rather than the data.

        S = diag( (F + L)^-1 L ),      L = I / prior_sd^2

    This is the leak mechanism stated directly. A MAP fit has

        x_hat - x_true  ~  (noise)  -  (F+L)^-1 L (x_true - x_prior)

    so a parameter with shrinkage ~1 is pinned to the prior and carries no information from
    that scan; one with shrinkage ~0 is determined by the data. Since the pull depends on how
    much information the PROTOCOL supplies, and protocol varies with scanner, the estimate of
    a shrunk parameter inherits the scanner. That is the whole thesis in one number.

    PASS `weights`. This is an ABSOLUTE fraction, so unlike the CRLB ordering it does not
    survive a global rescaling of F: at sigma = 0.05 the unit default inflates every value by
    400x. On the SIGNA Premier median protocol at prior_sd=10 the difference is
    T1 0.896 -> 0.087, PD 0.084 -> 0.0003, T2 0.021 -> 0.0001, D 0.000 -> 0.00002.

    Note also that shrinkage is the GAUSS-NEWTON leak coefficient. The displacement an
    estimator actually shows follows the true posterior curvature, which on this sloppy model
    is 25-30% smaller; `analysis/prior_shift.py` measures both.

    Related to but distinct from :func:`crlb`, which returns the POSTERIOR sd. Note the two
    disagree about what "uncertain" means: a heavily shrunk parameter has a LARGE posterior
    sd and a SMALL sampling variance, because shrinkage trades variance for bias. Leakage
    comes from the bias, so `shrinkage` is the more direct predictor and `crlb` a monotone
    proxy for it.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    L = np.eye(n) / prior_sd ** 2
    return dict(zip(PARAMS, np.diag(np.linalg.inv(F + L) @ L).tolist()))


def shrinkage_matrix(F, prior_sd=10.0):
    """The full leak operator M = (F + L)^-1 L, not just its diagonal.

    :func:`shrinkage` returns diag(M), which answers "how much of parameter i comes from its
    own prior". M answers the sharper question: move the prior mean of parameter j by Delta
    and the estimate of parameter i moves by M_ij * Delta. The off-diagonal terms are real --
    T1 and PD enter the T2W signal through one product, so a prior on one leaks into the
    estimate of the other -- and they are what makes a single-parameter reading of shrinkage
    incomplete.

    This is the prediction `analysis/prior_shift.py` tests against real images, and it is
    sharper than a rank correlation in two ways: it predicts a MAGNITUDE per case rather than
    an ordering, and it has leverage even where shrinkage barely varies across the cohort
    (D moves 0.0002 case to case, so no correlation is measurable, but M_DD * Delta is a
    definite number that either shows up in the estimates or does not).

    Rows and columns are in PARAMS order.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    L = np.eye(n) / prior_sd ** 2
    return np.linalg.inv(F + L) @ L


def sampling_sd(F, prior_sd=10.0):
    """SD the MAP estimate actually varies by under noise: sqrt(diag((F+L)^-1 F (F+L)^-1)).

    NOT the posterior sd. Provided because comparing a measured estimator spread against
    :func:`crlb` looks like a factor-5 discrepancy when it is simply the wrong comparison --
    a mistake made once already while validating the synthetic arm.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    P = np.linalg.inv(F + np.eye(n) / prior_sd ** 2)
    return dict(zip(PARAMS, np.sqrt(np.diag(P @ F @ P)).tolist()))


def spectrum(F):
    """Eigenvalues, condition number and per-parameter null-space overlap.

    ``null_overlap`` is prior-free but **discontinuous** -- it jumps 0.87 -> 0.00 when an
    added measurement changes the rank -- and b-band availability is vendor-linked, so
    regressing on it would manufacture the scanner confound this study exists to avoid.
    Diagnostic only; the regression uses :func:`crlb`.
    """
    evals, evecs = np.linalg.eigh(np.asarray(F, dtype=float))
    evals = np.clip(evals, 0.0, None)
    tol = max(evals.max(), 1e-30) * 1e-8
    null = evecs[:, evals <= tol]
    overlap = (null ** 2).sum(axis=1) if null.size else np.zeros(len(PARAMS))
    nz = evals[evals > tol]
    return dict(eigenvalues=evals.tolist(),
                condition_number=float(nz.max() / nz.min()) if nz.size >= 2 else float("inf"),
                rank=int((evals > tol).sum()),
                null_overlap=dict(zip(PARAMS, overlap.tolist())))


# ---------------------------------------------------------------------------
# The measurement sets of the project plan §3.3 -- the identifiability lever.
# Each arm changes exactly one thing about what the encoder is given.
# ---------------------------------------------------------------------------
def measurement_rows(arm, acq, theta=None):
    """Rows for one arm. ``acq`` needs TR/TE for T2W and DWI plus the b-values.

    Expected keys: ``TR_T2W, TE_T2W, TR_DWI, TE_DWI, b_hi``; ``b_lo`` for A2.
    ``theta`` overrides the linearization point (:data:`THETA0`).
    """
    t2w = log_sensitivities(acq["TR_T2W"], acq["TE_T2W"], 0.0, theta)
    dwi = log_sensitivities(acq["TR_DWI"], acq["TE_DWI"], acq["b_hi"], theta)
    if arm == "A1":                       # clinical bpMRI baseline
        return [t2w, dwi, anchor_row("D")]
    if arm == "A2":                       # + low-b band (S0)
        lo = log_sensitivities(acq["TR_DWI"], acq["TE_DWI"], acq["b_lo"], theta)
        return [t2w, dwi, lo, anchor_row("D")]
    if arm == "A3":                       # T2W only -- removes the TE ratio
        return [t2w]
    if arm == "A4":                       # ADC withheld -- D becomes gain-degenerate
        return [t2w, dwi]
    if arm == "A5":                       # + a short-TR acquisition (the T1 lever)
        short = log_sensitivities(acq.get("TR_SHORT", 600.0), acq["TE_T2W"], 0.0, theta)
        return [t2w, dwi, short, anchor_row("D")]
    raise ValueError(f"unknown arm {arm!r}; choose from A1..A5")


def identifiability(acq, arm="A1", weights=None, prior_sd=10.0, theta=None):
    """The study's predictor: CRLB + spectrum for one case under one measurement set."""
    F = fisher_matrix(measurement_rows(arm, acq, theta), weights)
    rows = measurement_rows(arm, acq, theta)
    out = {"arm": arm, "crlb": crlb(F, prior_sd),
           "shrinkage": shrinkage(F, prior_sd),
           "shrinkage_matrix": shrinkage_matrix(F, prior_sd).tolist(),
           "sampling_sd": sampling_sd(F, prior_sd),
           "degenerate": degenerate(rows)}
    out.update(spectrum(F))
    return out
