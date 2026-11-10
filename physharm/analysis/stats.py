r"""Small statistics helpers shared by the analysis scripts."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def icc(values, groups):
    r"""One-way ICC: the fraction of variance that is BETWEEN groups.

    ``ICC = var_between / (var_between + var_within)``, with the unequal-group-size
    correction ``n0``. For this project the groups are scanner models and the value is a
    predicted CRLB, so the ICC answers the question the whole study rests on:

        1.0  the predictor is a function of the scanner label -- "CRLB predicts leakage" and
             "scanner predicts leakage" are the same sentence, and nothing is testable.
        0.0  the predictor varies entirely WITHIN scanners -- cases can be matched across
             scanners at equal identifiability, the strongest available design.

    Measured at 0.266 for CRLB(T2) on the site, and exactly 1.000 on a vendor-lookup-table
    cohort (the run log Phase 2a). Returns ``None`` when fewer than two groups have >=2
    finite values.
    """
    by = defaultdict(list)
    for v, g in zip(values, groups):
        if v is not None and np.isfinite(v):
            by[g].append(float(v))
    by = {g: np.asarray(v) for g, v in by.items() if len(v) >= 2}
    if len(by) < 2:
        return None
    n_tot = sum(len(v) for v in by.values())
    k = len(by)
    grand = np.concatenate(list(by.values())).mean()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for v in by.values())
    ss_w = sum(((v - v.mean()) ** 2).sum() for v in by.values())
    df_b, df_w = k - 1, n_tot - k
    if df_b <= 0 or df_w <= 0:
        return None
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    n0 = (n_tot - sum(len(v) ** 2 for v in by.values()) / n_tot) / df_b
    var_b = max(0.0, (ms_b - ms_w) / n0)
    denom = var_b + ms_w
    return float(var_b / denom) if denom > 0 else None


def pcts(v, ps=(10, 50, 90)):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {int(p): float(np.percentile(v, p)) for p in ps} if v else {}
