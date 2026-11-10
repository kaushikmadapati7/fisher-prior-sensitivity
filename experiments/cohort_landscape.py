"""Objective landscape of one clinical exam, for the paper's Figure 1.

    PYTHONPATH=. python -u experiments/cohort_landscape.py \
        --cache-dir "$CACHE" --manifest-dir "$MANIFESTS" --out runs/cohort_landscape.json

Evaluates the data misfit of a fitted voxel on a grid in the plane spanned by F's two
least-constrained eigendirections, and records where the prior sits. That is the figure: the
valley the data cannot resolve, and the prior that decides where in it the answer lands.

WHY THE COHORT RATHER THAN A PUBLIC ARM. Figure 1 currently illustrates with a photovoltaic
curve, which is the only public arm whose landscape renders as a readable valley: the network
arms are near-isotropic in that plane (anisotropy 0.2 to 1.0) because their f2/f1 is 1.4 to 3.2.
The clinical cohort should be cleaner still, since three measurements against four parameters
makes F exactly rank-deficient and the valley exactly flat.

Writes only the grid, the eigenvalues and two marker positions. No images, no voxel data, and
nothing that identifies a case: the output is safe to bring back for plotting.
"""
import argparse, json

import numpy as np

from physharm.analysis.crlb_table import acquisition
from physharm.analysis.prior_shift import PARAMS, TISSUE, _jacobian, _model_log
from physharm.analysis.synthetic_arm import map_estimate
from physharm.analysis.traveling_arm import _rows_and_signal, load_case
from physharm.data.manifests import read_manifests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--arm", default="A1", choices=["A1", "A3", "A4"])
    ap.add_argument("--prior-sd", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--span", type=float, default=0.55, help="grid half-width, log units")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--out", default="runs/cohort_landscape.json")
    a = ap.parse_args()

    w, lam = 1.0 / a.sigma ** 2, 1.0 / a.prior_sd ** 2
    mans = read_manifests(a.manifest_dir)
    for m in mans:
        acq, _ = acquisition(m)
        if acq is None:
            continue
        cached = load_case(a.cache_dir, m.get("case_id"))
        if cached is None:
            continue
        got = _rows_and_signal(cached, acq, arm=a.arm, n_vox=400,
                               rng=np.random.default_rng(0))
        if got is None:
            continue
        rows, y = got
        break
    else:
        raise SystemExit("no usable case found")

    x0 = np.array([np.log(TISSUE[k][0]) for k in PARAMS])[:, None] * np.ones((1, y.shape[1]))
    xh = map_estimate(y, rows, {k: TISSUE[k][0] for k in PARAMS},
                      prior_sd=a.prior_sd, sigma=a.sigma, n_iter=a.n_iter)

    j = int(np.argmin(np.abs(xh[1] - np.median(xh[1]))))      # the median-T1 voxel
    xv, yv, x0v = xh[:, j], y[:, j], x0[:, j]
    rr = [r for r in rows]

    J = _jacobian(xv[:, None], rr)[:, :, 0]                   # (m, 4)
    F = w * (J.T @ J)
    ev, U = np.linalg.eigh(0.5 * (F + F.T))
    v, u = U[:, 0], U[:, 1]

    def misfit(dv, du):
        r = yv - _model_log((xv + dv * v + du * u)[:, None], rr)[:, 0]
        return 0.5 * w * float(r @ r)

    base = misfit(0.0, 0.0)
    ax = np.linspace(-a.span, a.span, a.grid)
    grid = [[float(misfit(p, q) - base) for p in ax] for q in ax]

    out = dict(arm=a.arm, span=a.span, n=a.grid, axis=ax.tolist(), grid=grid,
               eigenvalues=[float(e) for e in ev], lam=lam, sigma=a.sigma,
               prior_at=[float((x0v - xv) @ v), float((x0v - xv) @ u)],
               n_meas=len(rr), n_par=len(PARAMS))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  {len(rr)} measurements, {len(PARAMS)} parameters")
    print(f"  eigenvalues/lambda: {[round(float(e)/lam, 4) for e in ev]}")
    print(f"  misfit range along v: {max(r[a.grid//2] for r in grid):.4g}   "
          f"along u: {max(grid[a.grid//2]):.4g}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
