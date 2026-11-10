r"""Synthetic arm: does predicted CRLB track measured leakage when biology is fixed? (Issue #7)

THE MECHANISM UNDER TEST. Physics-based harmonization estimates tissue parameters and calls
them scanner-invariant. When a parameter is poorly determined by the acquisition, its estimate
is not set by the data -- it falls back on whatever prior or regularizer the fit carries. How
far it falls back depends on how much information that protocol supplies. Protocol varies with
scanner. So the estimate inherits the scanner, and it does so *in proportion to
unidentifiability*, which is computable in advance.

Writing that as an estimator makes the prediction exact. For a MAP fit in log-parameter space,

    x_hat = argmin ||W^(1/2) (y - f(x))||^2 + ||(x - x0) / prior_sd||^2

linearizing about the truth gives

    x_hat - x_true  ~  (F + L)^-1 J^T W eps        <- noise, covariance ~ CRLB^2
                     - (F + L)^-1 L (x_true - x0)  <- SHRINKAGE toward the prior

with L = I/prior_sd^2. The second term is the leak. It vanishes when F >> L (well determined)
and approaches the full distance to the prior when F << L (sloppy). Since the shrinkage factor
(F+L)^-1 L and the bound sqrt(diag((F+L)^-1)) are built from the same matrices, **the CRLB this
project already computes is a prediction about the bias, not only about the variance.**

WHY A MAP FIT AND NOT A NEURAL ENCODER. Issue #7 says the reference theta should come from an
encoder. That belongs to the real arm (#10): a learned encoder adds a learned prior, which
confounds identifiability with whatever the network absorbed from the training distribution.
The estimator the Cramer-Rao bound actually describes is maximum-likelihood/MAP, so that is
what the synthetic arm uses -- and it needs no GPU, no cache and no torch, which is why this
is the minimum publishable result and runs anywhere.

WHAT IS SWEPT. SNR, because it could not be measured on resampled volumes (the run log Phase
2c) -- so the claim is that the association holds across the plausible range rather than at
one assumed value. And the nuisance ladder, which separates identifiability from the live
alternative explanation:

    noise            leakage here is identifiability ALONE -- the forward model is exact
    + gain           an unmodelled per-contrast receive-gain mismatch
    + unmodelled     a protocol-dependent term the model does not contain: the MT / B1+ /
                     imperfect-spoiling family that the relaxometry literature blames for
                     vendor-dependent quantitative values (the project plan Part I 3)

    LIMITATION, measured on the real protocol set: the noise -> gain step is informative
    (A1 PD leakage 0.073 -> 0.232 at SNR 60, a 3x increase, so unmodelled nuisance dominates
    identifiability once SNR is good). The gain -> unmodelled step is NOT: the two differ by
    <0.003 everywhere, because `unmodelled` applies the gain perturbation as well and its own
    TE-scaled term is both small and largely absorbed by T2 -- physically correct, since MT
    does look like a T2 change, but it means this ladder separates "modelled" from
    "unmodelled" and does NOT separate the kinds of unmodelled physics. Do not claim a
    decomposition it does not deliver.

    python -m physharm.analysis.synthetic_arm --manifest-dir manifests_site \
        --out runs/synthetic_arm.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from physharm.analysis.crlb_table import acquisition
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

PARAMS = F.PARAMS
#: Prostate tissue priors at 3T. The MEAN is the estimator's prior; the SPREAD is the
#: biological variation we draw truth from. Log-normal keeps every draw positive.
#: T2/D spreads are wide enough to cover benign vs tumour; T1 and PD are nuisances here.
TISSUE = {"PD": (0.6, 0.25), "T1": (1400.0, 0.20), "T2": (80.0, 0.35), "D": (1.3e-3, 0.30)}


def draw_tissue(n, rng, tissue=TISSUE):
    """Sample n tissue vectors (log-normal about the prior mean)."""
    return {k: float(m) * np.exp(rng.normal(0.0, s, n)) for k, (m, s) in tissue.items()}


def _model_log(x, rows):
    """log-signal for log-parameters x (shape (4,) or (4,n)) under stacked sensitivity rows.

    The forward model is evaluated exactly, not linearly: rows only supply which measurement
    each entry is. `rows` here is a list of (TR, TE, b) or the string "anchor".
    """
    PD, T1, T2, D = (np.exp(x[i]) for i in range(4))
    out = []
    for r in rows:
        if r == "anchor":
            out.append(np.log(D))
            continue
        TR, TE, b = r
        E1 = 1.0 - np.exp(-TR / T1)
        out.append(np.log(PD) + np.log(np.clip(E1, 1e-12, None)) - TE / T2 - b * D)
    return np.array(out)


def _rows_for(arm, acq):
    """Measurement descriptors for an arm, parallel to fisher.measurement_rows."""
    t2w = (acq["TR_T2W"], acq["TE_T2W"], 0.0)
    dwi = (acq["TR_DWI"], acq["TE_DWI"], acq["b_hi"])
    if arm == "A1":
        return [t2w, dwi, "anchor"]
    if arm == "A2":
        return [t2w, dwi, (acq["TR_DWI"], acq["TE_DWI"], acq["b_lo"]), "anchor"]
    if arm == "A3":
        return [t2w]
    if arm == "A4":
        return [t2w, dwi]
    if arm == "A5":
        return [t2w, dwi, (acq.get("TR_SHORT", 600.0), acq["TE_T2W"], 0.0), "anchor"]
    raise ValueError(arm)


def _jacobian(x, rows, theta_names=PARAMS):
    """d log S / d log theta, evaluated at x. Same algebra as fisher.log_sensitivities,
    vectorized over samples and reusing it for the single-sample case."""
    PD, T1, T2, D = (np.exp(x[i]) for i in range(4))
    J = []
    for r in rows:
        if r == "anchor":
            row = [np.zeros_like(D), np.zeros_like(D), np.zeros_like(D), np.ones_like(D)]
        else:
            TR, TE, b = r
            E1 = np.exp(-TR / T1)
            dT1 = -(TR / T1) * E1 / np.clip(1.0 - E1, 1e-12, None)
            row = [np.ones_like(PD), dT1, TE / T2, -b * D]
        J.append(np.stack(row))
    return np.array(J)                       # (n_meas, 4, n_samples)


X_LO, X_HI = np.log(1e-8), np.log(1e6)
MU_MAX = 1e12          # past this the damped step is numerically zero; going further
                       # only overflows diag(A)*mu and makes the solve singular


def _lm_trial(x, A, g, mu, scale, y, rows, x0, w, lam):
    """One damped step and its cost. mu == 0 reproduces the plain Gauss-Newton step exactly."""
    Ad = A + (mu * scale)[:, None, :] * np.eye(4)[:, :, None]
    dx = np.linalg.solve(np.moveaxis(Ad, 2, 0), np.moveaxis(g, 1, 0)[..., None])
    trial = np.clip(x + np.moveaxis(dx[..., 0], 0, 1), X_LO, X_HI)
    return trial, map_cost(trial, y, rows, x0, w, lam)


def map_cost(x, y, rows, x0, w, lam):
    """Per-sample MAP objective: w*||y - f(x)||^2 + lam*||x - x0||^2. Shape (n,).

    This is the function `map_estimate` descends. It is module-level so the line search and
    the tests can score the same quantity the solver is minimizing, rather than a
    re-derivation that might drift from it.
    """
    r = y - _model_log(x, rows)
    return w * np.sum(r * r, axis=0) + lam * np.sum((x - x0) ** 2, axis=0)


def map_estimate(y, rows, prior_mean, prior_sd=10.0, sigma=0.05, n_iter=100,
                 line_search=True, n_backtrack=20, tol=1e-4):
    """Gauss-Newton MAP fit in log space. Returns log-parameter estimates, shape (4, n).

    This is deliberately the estimator the Cramer-Rao bound describes. `prior_sd` and the
    noise scale enter exactly as they do in `fisher.crlb`, so the predicted covariance and
    the predicted shrinkage are the same objects the prediction is made from.

    WHY THERE IS A LINE SEARCH (#79). Undamped Gauss-Newton does not converge on arm A1 for
    a large minority of real exams -- 41.3% of the cohort was non-stationary. It is not slow
    convergence: the iterates fall into an exact period-2 limit cycle, so 25, 200 and 2000
    steps give bit-identical answers and raising `n_iter` cannot help. The mechanism is that
    A1's ADC anchor pins log D, leaving the T2W/DWI log-ratio to be explained by T1 and T2
    alone. `normalize_shared` divides by the T2W ROI median and deliberately preserves the
    inter-contrast ratio, so the arbitrary TSE-vs-EPI receive gain survives into `y`. When
    that ratio demands more than the model's supremum of max(0, log(TR_T2W/TR_DWI)), the MAP
    sits at an extreme where A's smallest eigenvalue is pinned at lam and the softest
    eigenvector is T1 (or the PD-T1 ridge, correlation -0.994). The GN step along it is g/lam
    = 100g, which overshoots into a strongly curved region; the cost RISES and the iterates
    cycle. The old synthetic fixture never caught this because it rendered both contrasts
    from one PD with no relative gain.

    A backtracking line search fixes it: the step is halved until the objective actually
    decreases, and is rejected outright if it never does. On a well-behaved case the full
    step always decreases the cost, the search accepts it on the first try, and the iterates
    are bit-identical to the undamped ones -- which is why this does not move the headline
    numbers. Pass `line_search=False` to reproduce pre-#79 artifacts exactly.
    """
    n = y.shape[1]
    x = np.stack([np.full(n, np.log(prior_mean[k])) for k in PARAMS])
    x0 = x.copy()
    lam = 1.0 / prior_sd ** 2
    w = 1.0 / sigma ** 2
    if not line_search:
        for _ in range(n_iter):
            r = y - _model_log(x, rows)
            J = _jacobian(x, rows)
            A = np.einsum("mkn,mln->kln", J, J) * w + lam * np.eye(4)[:, :, None]
            g = np.einsum("mkn,mn->kn", J, r) * w - lam * (x - x0)
            dx = np.linalg.solve(np.moveaxis(A, 2, 0), np.moveaxis(g, 1, 0)[..., None])
            x = np.clip(x + np.moveaxis(dx[..., 0], 0, 1), X_LO, X_HI)
        return x

    cost = map_cost(x, y, rows, x0, w, lam)
    mu = np.zeros(n)                                      # LM damping, per sample; 0 == GN
    act = np.arange(n)                                    # samples still worth iterating
    for _ in range(n_iter):
        if act.size == 0:
            break
        # Everything below is computed on the ACTIVE subset only. Retiring samples as they
        # finish is what makes this affordable: the stopping rule is per-sample, so a handful
        # of voxels that cannot be improved must not drag every converged voxel through the
        # remaining iterations. Doing that cost ~25x the whole analysis.
        xa, x0a, ya, mua = x[:, act], x0[:, act], y[:, act], mu[act]
        r = ya - _model_log(xa, rows)
        J = _jacobian(xa, rows)
        A = np.einsum("mkn,mln->kln", J, J) * w + lam * np.eye(4)[:, :, None]
        data = np.einsum("mkn,mn->kn", J, r) * w
        pull = lam * (xa - x0a)
        g = data - pull

        # Stop a sample once it is stationary, by the same bounded ratio the analysis reports,
        # so "converged" here and "converged" in the results table mean the same thing.
        resid = (np.linalg.norm(g, axis=0)
                 / (np.linalg.norm(data, axis=0) + np.linalg.norm(pull, axis=0) + 1e-30))
        keep = resid >= tol
        if not keep.all():
            act = act[keep]
            if act.size == 0:
                break
            xa, x0a, ya, mua = xa[:, keep], x0a[:, keep], ya[:, keep], mua[keep]
            A, g = A[:, :, keep], g[:, keep]

        scale = np.einsum("kkn->kn", A)                   # Marquardt scaling, diag(A)
        floor = 1e-9 * np.maximum(scale.mean(axis=0), 1e-30)
        ca = cost[act]
        trial, c_trial = _lm_trial(xa, A, g, mua, scale, ya, rows, x0a, w, lam)
        for _ in range(n_backtrack):
            idx = np.flatnonzero(c_trial >= ca)
            if idx.size == 0:
                break
            mua[idx] = np.clip(np.maximum(mua[idx] * 10.0, floor[idx]), 0.0, MU_MAX)
            sub, c_sub = _lm_trial(xa[:, idx], A[:, :, idx], g[:, idx], mua[idx],
                                   scale[:, idx], ya[:, idx], rows, x0a[:, idx], w, lam)
            trial[:, idx], c_trial[idx] = sub, c_sub

        # Reject rather than accept an increase. A sample whose cost never falls is already
        # somewhere GN cannot improve on, and stepping anyway is what produced the cycling.
        better = c_trial < ca
        x[:, act] = np.where(better, trial, xa)
        cost[act] = np.where(better, c_trial, ca)
        mu[act] = np.where(better, mua / 3.0, mua)        # relax back toward pure GN
        # A sample that could not improve even at maximum damping is done; carrying it
        # further only burns iterations on every other sample in the batch.
        act = act[better | (mu[act] < MU_MAX)]
    return x


def run_protocol(acq, arm, theta_true, rng, snr, nuisance="noise", prior_sd=10.0,
                 gain_sd=0.05, unmodelled_sd=0.05):
    """Render one protocol, corrupt it per the nuisance level, and fit. Returns log theta_hat.

    `snr` is per-measurement SNR on the linear signal; sigma in log space is ~1/snr.
    """
    rows = _rows_for(arm, acq)
    x_true = np.stack([np.log(theta_true[k]) for k in PARAMS])
    y = _model_log(x_true, rows)
    sigma = 1.0 / snr
    y = y + rng.normal(0.0, sigma, y.shape)               # always: measurement noise
    if nuisance in ("gain", "unmodelled"):
        # an unknown per-contrast receive gain the estimator does not model
        y = y + rng.normal(0.0, gain_sd, (len(rows), 1))
    if nuisance == "unmodelled":
        # a PROTOCOL-DEPENDENT term absent from the forward model -- the MT / B1+ /
        # imperfect-spoiling family. Scaled by TE so it differs across protocols the way
        # the relaxometry literature reports.
        extra = np.array([[0.0] if r == "anchor" else [-unmodelled_sd * r[1] / 100.0]
                          for r in rows])
        y = y + extra
    return map_estimate(y, rows, {k: v[0] for k, v in TISSUE.items()},
                        prior_sd=prior_sd, sigma=sigma)


def leakage_vs_crlb(protocols, arm="A1", n_tissue=400, snr=15.0, nuisance="noise",
                    seed=0, prior_sd=10.0):
    """The experiment. Identical tissue through every protocol; spread of the estimate IS leak.

    Returns per-parameter measured leakage (across-protocol sd of the estimate, in log units,
    averaged over tissue samples) and the mean predicted CRLB.
    """
    rng = np.random.default_rng(seed)
    theta = draw_tissue(n_tissue, rng)
    est, crlbs = [], []
    # Evaluate the predictor at the SAME noise level as the experiment. Previously the
    # weights defaulted to 1, so the CRLB did not depend on SNR at all -- which made every
    # SNR row of the report identical and gave the false impression that a result had been
    # replicated across conditions when it was one observation printed many times.
    w = lambda a: [snr ** 2] * len(F.measurement_rows(arm, a))
    for acq in protocols:
        est.append(run_protocol(acq, arm, theta, rng, snr, nuisance, prior_sd))
        crlbs.append(F.identifiability(acq, arm, weights=w(acq), prior_sd=prior_sd)["crlb"])
    est = np.array(est)                                   # (n_proto, 4, n_tissue)
    leak = est.std(axis=0).mean(axis=1)                   # across protocols, then over tissue
    bias = (est.mean(axis=0) - np.stack([np.log(theta[k]) for k in PARAMS])).mean(axis=1)
    pred = [F.identifiability(a, arm, weights=w(a), prior_sd=prior_sd) for a in protocols]
    degen = pred[0]["degenerate"]
    # PER-PROTOCOL statistic -- the one with power. `leakage` above is a single number per
    # parameter, so correlating it against a predictor is an n=4 rank test that can only take
    # five values. Here each protocol contributes a point: how far ITS estimate sits from the
    # consensus, against how much the prior pulls it. n = number of protocols, and it is the
    # same regression the real arm (#10) runs per case.
    consensus = est.mean(axis=0)                          # (4, n_tissue)
    dev = np.abs(est - consensus).mean(axis=2)            # (n_proto, 4)
    per_param_rho = {}
    for i, k in enumerate(PARAMS):
        sh = [pr["shrinkage"][k] for pr in pred]
        cr = [pr["crlb"][k] for pr in pred]
        per_param_rho[k] = {"rho_shrinkage": spearman(sh, dev[:, i].tolist()),
                            "rho_crlb": spearman(cr, dev[:, i].tolist()),
                            "dev_median": float(np.median(dev[:, i]))}
    return {"arm": arm, "snr": snr, "nuisance": nuisance, "seed": seed,
            "n_protocols": len(protocols), "n_tissue": n_tissue,
            "degenerate": degen,
            # a parameter with no measurement is NOT "zero leakage" -- it is not estimated
            "leakage": {k: (float("nan") if degen[k] else float(v))
                        for k, v in zip(PARAMS, leak.tolist())},
            "bias": dict(zip(PARAMS, bias.tolist())),
            "crlb": {k: float(np.mean([c[k] for c in crlbs])) for k in PARAMS},
            # the DIRECT predictor: how much of the estimate the prior sets, and how much
            # that pull VARIES across protocols -- which is what makes the leak scanner-shaped
            "shrinkage": {k: float(np.mean([p["shrinkage"][k] for p in pred])) for k in PARAMS},
            "shrinkage_spread": {k: float(np.std([p["shrinkage"][k] for p in pred]))
                                 for k in PARAMS},
            "per_protocol": per_param_rho}


def spearman(x, y):
    """Spearman rank correlation, returning NaN when either input is constant.

    numpy's argsort ranks a constant vector as 0,1,2,... -- the identity -- so the naive
    version returns a spurious +1.0 against ANY x. That is not hypothetical here: in arm A3
    the diffusion coefficient has no information at all, so its estimate is pinned to the
    prior and its per-protocol deviation is exactly 0.000 for every protocol. The naive
    formula reports a perfect correlation for a quantity that never varies.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")

    def rank(v):
        # AVERAGE ranks for ties. argsort assigns tied values distinct ranks by array
        # position, so a rank order can be decided by a numerical accident: in arm A3,
        # CRLB(T1)=9.9997 and CRLB(D)=10.0000 are both saturated at the prior ceiling and
        # their relative rank turns on 3e-4. That is noise, not an ordering.
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        i = 0
        vs = v[order]
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and np.isclose(vs[j + 1], vs[i], rtol=1e-6, atol=0):
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", required=True,
                    help="protocols are drawn from the REAL per-case acquisition parameters, "
                         "not from a handful of per-vendor constants")
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--snr", nargs="+", type=float, default=[5.0, 10.0, 15.0, 30.0, 60.0])
    ap.add_argument("--nuisance", nargs="+", default=["noise", "gain", "unmodelled"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-protocols", type=int, default=150)
    ap.add_argument("--n-tissue", type=int, default=400)
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mans = read_manifests(args.manifest_dir)
    acqs = [a for a in (acquisition(m)[0] for m in mans) if a is not None]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(acqs), size=min(args.n_protocols, len(acqs)), replace=False)
    protocols = [acqs[i] for i in idx]
    have_lo = [a for a in protocols if "b_lo" in a]
    print(f"{len(acqs)} protocols available; using {len(protocols)} "
          f"({len(have_lo)} with a low-b band)")

    rows, results = [], []
    for arm in args.arms:
        pr = have_lo if arm == "A2" else protocols
        if not pr:
            continue
        for nu in args.nuisance:
            for snr in args.snr:
                per_seed = [leakage_vs_crlb(pr, arm, args.n_tissue, snr, nu, s,
                                            args.prior_sd) for s in args.seeds]
                results += per_seed
                leak = {k: float(np.mean([r["leakage"][k] for r in per_seed])) for k in PARAMS}
                sd = {k: float(np.std([r["leakage"][k] for r in per_seed])) for k in PARAMS}
                crlb = per_seed[0]["crlb"]
                shr = per_seed[0]["shrinkage"]
                shr_sd = per_seed[0]["shrinkage_spread"]
                pp = {k: {m: float(np.mean([r["per_protocol"][k][m] for r in per_seed]))
                          for m in ("rho_shrinkage", "rho_crlb", "dev_median")}
                      for k in PARAMS}
                rows.append({"per_protocol": pp,
                             "arm": arm, "nuisance": nu, "snr": snr,
                             "rho_crlb": spearman([crlb[k] for k in PARAMS],
                                                  [leak[k] for k in PARAMS]),
                             "rho_shrinkage": spearman([shr_sd[k] for k in PARAMS],
                                                       [leak[k] for k in PARAMS]),
                             "leakage": leak, "leakage_sd": sd, "crlb": crlb,
                             "shrinkage": shr, "shrinkage_spread": shr_sd})

    print("\n=== LEAKAGE vs PREDICTED CRLB (identical tissue through every protocol) ===")
    print(f"{'arm':>4s} {'nuisance':>11s} {'snr':>6s}  " +
          "  ".join(f"{p:>16s}" for p in PARAMS) + f"  {'rho_crlb':>9s} {'rho_shrink':>10s}")
    for r in rows:
        cells = "  ".join(f"{r['leakage'][p]:.3f}+-{r['leakage_sd'][p]:.3f}" for p in PARAMS)
        print(f"{r['arm']:>4s} {r['nuisance']:>11s} {r['snr']:>6.0f}  {cells}  "
              f"{r['rho_crlb']:>+9.2f} {r['rho_shrinkage']:>+10.2f}")

    print("\n=== PER-PROTOCOL REGRESSION (n = protocols, the statistic with power) ===")
    print("  Per protocol: how far ITS estimate sits from the consensus, against how hard")
    print("  the prior pulls it. This is the regression the real arm runs per case.")
    print("  The rho columns above are an n=4 rank test over parameters and can only take")
    print("  five values -- read these instead.\n")
    print(f"  {'arm':>4s} {'nuisance':>11s} {'snr':>5s}  " +
          "  ".join(f"{p:>7s}" for p in PARAMS) + "   sign pattern")
    for r in rows:
        vals = [r["per_protocol"][p]["rho_shrinkage"] for p in PARAMS]
        pat = "".join("." if v != v else "+" if v > 0.2 else "-" if v < -0.2 else "0"
                      for v in vals)
        print(f"  {r['arm']:>4s} {r['nuisance']:>11s} {r['snr']:>5.0f}  " +
              "  ".join("      ." if v != v else f"{v:>+7.2f}" for v in vals) + f"   {pat}")
    print("\n  Expected under the inverted-U correction: '0' for an anchored parameter,")
    print("  '+' for intermediate ones, '-' for a fully prior-pinned one.")
    print("  '.' means the deviation never varied across protocols, so no correlation is")
    print("  defined -- which is itself the signature of a parameter pinned entirely to the")
    print("  prior (arm A3's D is exactly 0.000 everywhere).")

    print("\n=== READING ===")
    for nu in args.nuisance:
        rr = [r for r in rows if r["nuisance"] == nu]
        if rr:
            print(f"  {nu:>11s}: rho_crlb {min(r['rho_crlb'] for r in rr):+.2f}..."
                  f"{max(r['rho_crlb'] for r in rr):+.2f}   "
                  f"rho_shrinkage {min(r['rho_shrinkage'] for r in rr):+.2f}..."
                  f"{max(r['rho_shrinkage'] for r in rr):+.2f}")
    print("  'noise' is identifiability ALONE (the forward model is exact there).")
    print("  The increments to 'gain' and 'unmodelled' are the contribution of physics the")
    print("  model does not contain -- the MT / B1+ / spoiling alternative explanation.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"summary": rows, "runs": results,
                   "n_protocols": len(protocols), "prior_sd": args.prior_sd},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
