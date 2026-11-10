#!/usr/bin/env python3
"""Draw the paper's headline figure from a prior_shift artifact.

    python paper/make_figure.py runs/prior_shift_lm_A1.json

Writes paper/fig_leak.pdf (vector, for LaTeX) and paper/fig_leak.png (for quick viewing).

WHAT IT SHOWS. One point per exam: how far the estimate actually moved when the prior was
shifted, against how far it was predicted to move. A correct predictor puts the cloud on the
identity line. The Fisher panel sits visibly below it; the curvature panel sits on it. This is
the whole result, and it also shows what the medians in Table 1 hide -- the Fisher cloud is
tight and consistently low rather than scattered, which is why a 29-38% shortfall is a
systematic bias and not noise.

matplotlib is imported here and nowhere in physharm, so the analysis package stays numpy-only
and the test suite does not gain a dependency.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

DELTA = "0.2231"
PARAM = "T1"
LABEL = r"$T_1$"


def series(rows, param, key):
    ob = np.array([r["shifts"][DELTA][param]["observed"][param] for r in rows], float)
    pr = np.array([r["shifts"][DELTA][param][key][param] for r in rows], float)
    ok = np.isfinite(ob) & np.isfinite(pr) & (pr != 0)
    return pr[ok], ob[ok]


PANELS = (("PD", "PD"), ("T1", r"$T_1$"), ("T2", r"$T_2$"))


def main(path):
    rows = json.load(open(path))["cases"]
    # TWO ROWS. The rank-one repair was tried as a third cloud and as a binned median line in
    # the same axes; neither worked. Three translucent clouds cannot be separated by hue, and a
    # binned line in a sparse tail collapses each bin to a point, so the long straight segments
    # between them caricature a curved departure. Its own row keeps the cloud intact and makes
    # the columns directly comparable.
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.7), sharex="col", sharey="col")

    for col, (param, label) in enumerate(PANELS):
        top, bot = axes[0, col], axes[1, col]
        # Per-column limits: the three parameters leak by very different amounts (PD and T2 by
        # about a twentieth of T1), so a shared axis would collapse two of the three to a dot.
        hi = 0.0
        clouds = {}
        for key, colour, name in (("predicted", "#b2182b", "Fisher"),
                                  ("predicted_hessian", "#2166ac", "curvature restored"),
                                  ("predicted_scalar", "#e69f00", "rank-one")):
            pr, ob = series(rows, param, key)
            clouds[name] = (pr, ob, colour)
            hi = max(hi, float(np.percentile(pr, 99.5)), float(np.percentile(ob, 99.5)))
        hi *= 1.08

        def draw(ax, names):
            ax.plot([0, hi], [0, hi], ls="--", lw=0.9, color="0.3", zorder=5)
            for nm in names:
                pr, ob, colour = clouds[nm]
                ax.scatter(pr, ob, s=1.6, alpha=0.08, color=colour, linewidths=0, zorder=2,
                           rasterized=True)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

        draw(top, ("Fisher", "curvature restored"))
        draw(bot, ("Fisher", "rank-one"))

        # SPEARMAN, not Pearson. The surrounding argument is about rank -- a median can be
        # rescaled, an ordering cannot -- and Pearson answers a different question. It also
        # punishes the rank-one repair for a magnitude tail on T2, where the leak is carried by
        # directions other than v so the v-v correction cannot reach it: rank-one's Pearson is
        # 0.508 against a Spearman of 0.933. Ties matter, because the Fisher prediction
        # saturates at the shift value for much of the cohort, so average ranks are required
        # and a naive argsort would be wrong. Series are looked up by NAME, never by position.
        rho = lambda nm: float(spearmanr(clouds[nm][0], clouds[nm][1]).statistic)
        top.set_title(f"{label}   $r$: {rho('Fisher'):.2f} "
                      f"$\\rightarrow$ {rho('curvature restored'):.2f}", fontsize=8.5, pad=4)
        bot.set_title(f"rank-one   $r$: {rho('Fisher'):.2f} "
                      f"$\\rightarrow$ {rho('rank-one'):.2f}", fontsize=8.5, pad=4)
        bot.set_xlabel("predicted (log units)", fontsize=7.5)

    for r in (0, 1):
        axes[r, 0].set_ylabel("measured (log units)", fontsize=7.5)
    # No legend: the caption names the colours, and a legend strip costs vertical space the
    # paper does not have.

    below = int((np.asarray([r["shifts"][DELTA]["T1"]["observed"]["T1"]
                             for r in rows], float) < 0).sum())
    print(f"note: {below} of {len(rows)} exams have a negative measured T1 displacement and fall "
          f"below the axis floor ({100*below/len(rows):.2f}%)")
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = f"paper/fig_leak.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
