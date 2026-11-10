r"""Self-supervised arm: does an amortised estimator inherit the curvature failure? (#82)

    python -m physharm.analysis.selfsup_arm --cache-dir CACHE --manifest-dir MANIFESTS

WHAT THIS TESTS. The theory (run log, phases 25-27) locates the failure in the per-exam
optimisation: the correction is signed because the FIT moves along the sloppy direction in
response to misspecification, which changes the curvature it is evaluated against. A network that
learns an amortised map y -> x never solves that optimisation. So the theory predicts an amortised
estimator does NOT inherit the signed shortfall. This is that prediction's test, and it is
falsifiable: if the network tracks the MAP fit, the mechanism as stated is wrong.

WHY IT CAN RUN ON REAL IMAGES, WHERE THE SUPERVISED VERSION CANNOT. A supervised network needs
(y, theta) pairs and this cohort has no ground-truth parameter maps. A self-supervised one needs
none: it is trained to minimise the same objective the MAP fit minimises,

    L(x) = w * ||y - f(x)||^2  +  lambda * ||x - x0||^2

which is computable from the images alone. This is the construction van Lune et al. use, where the
prior is an explicit soft bound on proton density. Because the prior is written down rather than
implied by a training population, it can be SHIFTED, which is the intervention used everywhere
else in this project. Move x0 by Delta, retrain, and measure how far the network's output follows.

The network therefore differs from the MAP fit in exactly one respect: it is amortised. Same data,
same objective, same prior, same shift. That isolates the estimator.

Gradients are analytic rather than autodiff, so physharm keeps numpy as its only dependency:

    dL/dx = -2w J^T (y - f(x)) + 2 lambda (x - x0)

and that vector is backpropagated through the network in the usual way.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from physharm.analysis.crlb_table import acquisition, canonical_scanner
from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _jacobian, _model_log, map_estimate
from physharm.analysis.traveling_arm import _rows_and_signal, load_case
from physharm.data.manifests import read_manifests
from physharm.physics import fisher as F

SIGMA, PRIOR_SD, ARM = 0.05, 10.0, "A1"


def physics_grad(x, y, rows, w, lam, x0):
    """dL/dx for the MAP objective, shape (4, n). The only physics the network sees."""
    r = y - _model_log(x, rows)                      # (m, n)
    J = _jacobian(x, rows)                           # (m, 4, n)
    return -2.0 * w * np.einsum("mkn,mn->kn", J, r) + 2.0 * lam * (x - x0)


def physics_loss(x, y, rows, w, lam, x0):
    r = y - _model_log(x, rows)
    return float(np.mean(w * np.sum(r * r, axis=0) + lam * np.sum((x - x0) ** 2, axis=0)))


class MLP:
    """Two hidden layers, tanh, Adam. Trained by an externally supplied output gradient."""

    def __init__(self, d_in, d_out, h=96, seed=0, lr=2e-3):
        rng = np.random.default_rng(seed)
        self.P = [rng.normal(0, np.sqrt(1 / d_in), (h, d_in)), np.zeros((h, 1)),
                  rng.normal(0, np.sqrt(1 / h), (h, h)), np.zeros((h, 1)),
                  rng.normal(0, np.sqrt(1 / h), (d_out, h)), np.zeros((d_out, 1))]
        self.M = [np.zeros_like(p) for p in self.P]
        self.V = [np.zeros_like(p) for p in self.P]
        self.t, self.lr = 0, lr

    def forward(self, a0, offset=0.0):
        """Returns `offset + net(a0)`.

        The offset is the prior mean, so an untrained network predicts the prior rather than
        exp(0) = 1 for every parameter. Without it the initial T1 is 1 ms against a true 1400,
        the physics gradient at that point is enormous, and training diverges to a loss worse
        than doing nothing -- measured at 2264 against the prior-mean baseline's 526.
        """
        z1 = self.P[0] @ a0 + self.P[1]; a1 = np.tanh(z1)
        z2 = self.P[2] @ a1 + self.P[3]; a2 = np.tanh(z2)
        return offset + self.P[4] @ a2 + self.P[5], (a0, a1, a2)

    def step(self, cache, g_out):
        a0, a1, a2 = cache
        n = a0.shape[1]
        g5 = g_out / n
        G = [None] * 6
        G[4], G[5] = g5 @ a2.T, g5.sum(1, keepdims=True)
        g2 = (self.P[4].T @ g5) * (1 - a2 ** 2)
        G[2], G[3] = g2 @ a1.T, g2.sum(1, keepdims=True)
        g1 = (self.P[2].T @ g2) * (1 - a1 ** 2)
        G[0], G[1] = g1 @ a0.T, g1.sum(1, keepdims=True)
        self.t += 1
        for i in range(6):
            self.M[i] = 0.9 * self.M[i] + 0.1 * G[i]
            self.V[i] = 0.999 * self.V[i] + 0.001 * G[i] ** 2
            self.P[i] -= self.lr * (self.M[i] / (1 - 0.9 ** self.t)) / \
                (np.sqrt(self.V[i] / (1 - 0.999 ** self.t)) + 1e-8)


def acq_features(acq):
    """Log acquisition settings, as network inputs.

    WITHOUT THESE THE PROBLEM IS UNSOLVABLE ON REAL DATA. Protocols differ between exams, so the
    same measurement vector implies different tissue parameters depending on TR, TE and b. A
    network given only the measurements is being asked to invert a model whose settings it cannot
    see. This was invisible on single-protocol synthetic data and would have produced a confident
    wrong answer on the cohort.
    """
    return np.array([[np.log(acq["TR_T2W"])], [np.log(acq["TE_T2W"])],
                     [np.log(acq["TR_DWI"])], [np.log(acq["TE_DWI"])],
                     [np.log(max(acq["b_hi"], 1.0))]])


def train(per_exam, x0, w, lam, seed=0, steps=4000, batch=512, scale=None, offset=None):
    """Fit the network by descending the physics loss.

    `per_exam` is a list of (rows, y, acq_features) — one entry per exam, NOT a pooled array.
    Each batch is drawn from a SINGLE exam, so the row set used to compute the physics gradient
    is the one that actually produced those voxels. Pooling and then sampling across exams
    applies one exam's acquisition to another's measurements, which is wrong physics for most of
    the batch and was the second defect this module had.

    `x0` is the prior mean IN THE LOSS; shifting it and retraining is the whole experiment.

    `offset` is the output parameterisation and must be held FIXED across base and shifted runs.
    If it moves with the prior, the two networks learn different functions rather than the same
    function under different objectives, and the measured displacement contains the
    parameterisation change itself.
    """
    if offset is None:
        offset = x0
    if scale is None:
        allX = np.hstack([np.vstack([y, np.repeat(af, y.shape[1], 1)]) for _, y, af in per_exam])
        scale = (allX.mean(1, keepdims=True), allX.std(1, keepdims=True) + 1e-9)
    xmu, xsd = scale
    net = MLP(xmu.shape[0], 4, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        rows, y, af = per_exam[rng.integers(0, len(per_exam))]
        idx = rng.integers(0, y.shape[1], min(batch, y.shape[1]))
        yb = y[:, idx]
        inp = (np.vstack([yb, np.repeat(af, idx.size, 1)]) - xmu) / xsd
        out, cache = net.forward(inp, offset=offset)
        net.step(cache, physics_grad(out, yb, rows, w, lam, x0))
    return net, scale


def predict(net, y, af, scale, x0):
    xmu, xsd = scale
    inp = (np.vstack([y, np.repeat(af, y.shape[1], 1)]) - xmu) / xsd
    return net.forward(inp, offset=x0)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--delta", type=float, default=0.2231)
    ap.add_argument("--limit", type=int, default=400,
                    help="exams to pool voxels from; 0 means all, as everywhere else here")
    ap.add_argument("--n-vox", type=int, default=200, help="voxels per exam")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w, lam = 1.0 / SIGMA ** 2, 1.0 / PRIOR_SD ** 2
    base_mean = {k: v[0] for k, v in TISSUE.items()}
    x0 = np.array([[np.log(base_mean[k])] for k in PARAMS])

    mans = read_manifests(args.manifest_dir)
    if args.limit:
        mans = mans[:args.limit]
    per_exam, acqs = [], []
    for m in mans:
        acq, why = acquisition(m)
        if acq is None:
            continue
        cached = load_case(args.cache_dir, m.get("case_id"))
        if cached is None:
            continue
        got = _rows_and_signal(cached, acq, arm=ARM, n_vox=args.n_vox,
                               rng=np.random.default_rng(0))
        if got is None:
            continue
        rows, y = got
        per_exam.append((rows, y, acq_features(acq)))
        acqs.append(acq)
    if not per_exam:
        raise SystemExit("no usable exams; check --cache-dir and --manifest-dir")
    n_vox = sum(y.shape[1] for _, y, _ in per_exam)
    print(f"{len(per_exam)} exams, {n_vox} voxels")

    def loss_over_cohort(pred_fn):
        tot = 0.0
        for rows, y, af in per_exam:
            tot += physics_loss(pred_fn(rows, y, af), y, rows, w, lam, x0) * y.shape[1]
        return tot / n_vox

    net, scale = train(per_exam, x0, w, lam, steps=args.steps)
    l_net = loss_over_cohort(lambda rows, y, af: predict(net, y, af, scale, x0))
    l_map = loss_over_cohort(lambda rows, y, af: map_estimate(y, rows, base_mean,
                                                              prior_sd=PRIOR_SD, sigma=SIGMA))
    print(f"  physics loss   network {l_net:.5g}   MAP {l_map:.5g}   ratio {l_net/l_map:.1f}x")
    if l_net > 3 * l_map:
        print("  ** NOT AT LOSS PARITY. Displacements measured from this network are confounded")
        print("     with how well it was trained; see the run log, phase 29. Raise --steps.")

    base_net = {i: predict(net, y, af, scale, x0) for i, (_, y, af) in enumerate(per_exam)}
    base_map = {i: map_estimate(y, r, base_mean, prior_sd=PRIOR_SD, sigma=SIGMA)
                for i, (r, y, _) in enumerate(per_exam)}
    M = F.shrinkage_matrix(
        F.fisher_matrix(F.measurement_rows(ARM, acqs[0], base_mean),
                        [w] * len(per_exam[0][0])), PRIOR_SD)

    report = {"n_exams": len(per_exam), "n_vox": int(n_vox), "delta": args.delta,
              "loss_net": l_net, "loss_map": l_map, "by_param": {}}
    print(f"\n{'param':>6} {'predicted':>11} {'net ratio':>11} {'MAP ratio':>11}")
    for j, pj in enumerate(PARAMS):
        xs = x0.copy(); xs[j] += args.delta
        sm = dict(base_mean); sm[pj] = float(base_mean[pj] * np.exp(args.delta))
        net_s, _ = train(per_exam, xs, w, lam, steps=args.steps, scale=scale, offset=x0)
        dn = np.concatenate([predict(net_s, y, af, scale, x0)[j] - base_net[i][j]
                             for i, (_, y, af) in enumerate(per_exam)])
        dm = np.concatenate([map_estimate(y, r, sm, prior_sd=PRIOR_SD, sigma=SIGMA)[j]
                             - base_map[i][j] for i, (r, y, _) in enumerate(per_exam)])
        pred = float(M[j, j] * args.delta)
        report["by_param"][pj] = {"predicted": pred,
                                  "net_ratio": float(np.median(dn)) / pred if pred else None,
                                  "map_ratio": float(np.median(dm)) / pred if pred else None}
        print(f"{pj:>6} {pred:>11.5f} {float(np.median(dn))/pred:>11.3f} "
              f"{float(np.median(dm))/pred:>11.3f}")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
