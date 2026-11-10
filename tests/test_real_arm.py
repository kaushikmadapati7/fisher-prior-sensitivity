"""Tests for the real arm (issue #10). Synthetic caches, no images, no torch."""
from __future__ import annotations

import numpy as np

from physharm.analysis import real_arm as RA
from physharm.analysis.synthetic_arm import PARAMS, TISSUE
from physharm.analysis.traveling_arm import estimate_case
from tests.test_traveling_arm import ACQ, _render_case


def _rows(n=400, seed=0, slope=-0.5):
    """Cases with a planted relation between shrinkage and distance-from-prior."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = float(rng.uniform(0.05, 0.9))
        base = float(rng.uniform(0.1, 0.6))
        d = max(0.01, base + slope * s + rng.normal(0, 0.05))
        out.append({"case_id": f"c{i}", "patient_id": f"p{i//2}",
                    "scanner": ["A", "B", "C"][i % 3],
                    "dist": {k: d for k in PARAMS},
                    "theta": {k: d for k in PARAMS},
                    "shrinkage": {k: s for k in PARAMS},
                    "crlb": {k: s * 10 for k in PARAMS}})
    return out


def test_regression_recovers_a_planted_negative_slope():
    """The prediction: higher shrinkage -> estimate sits closer to the prior."""
    reg = RA.within_scanner_regression(_rows(slope=-0.5))
    for p in PARAMS:
        assert reg[p]["pooled"] < -0.5, f"{p}: {reg[p]['pooled']}"
        assert reg[p]["median_within"] < -0.5
        assert reg[p]["n_scanners"] == 3


def test_regression_reports_flat_when_there_is_no_relation():
    """A flat slope must read as 'no mechanism', not be rounded toward the hypothesis."""
    reg = RA.within_scanner_regression(_rows(slope=0.0))
    for p in PARAMS:
        assert abs(reg[p]["median_within"]) < 0.25, f"{p}: {reg[p]['median_within']}"


def test_regression_detects_a_contradicting_slope():
    reg = RA.within_scanner_regression(_rows(slope=+0.5))
    for p in PARAMS:
        assert reg[p]["median_within"] > 0.5


def test_within_scanner_holds_the_instrument_fixed():
    """A pure BETWEEN-scanner effect with no within-scanner relation must not register.

    This is the confound the within-scanner column exists to exclude: a scanner that happens
    to have both high shrinkage and distant estimates would produce a pooled correlation that
    is really a scanner effect wearing a protocol costume.
    """
    rows = []
    for i in range(300):
        s_id = ["A", "B", "C"][i % 3]
        s = {"A": 0.2, "B": 0.5, "C": 0.8}[s_id]          # constant within a scanner
        d = {"A": 0.6, "B": 0.4, "C": 0.2}[s_id]
        rows.append({"case_id": f"c{i}", "patient_id": f"p{i}", "scanner": s_id,
                     "dist": {k: d for k in PARAMS}, "theta": {k: d for k in PARAMS},
                     "shrinkage": {k: s for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}})
    reg = RA.within_scanner_regression(rows)
    for p in PARAMS:
        assert reg[p]["pooled"] < -0.9, "pooled sees the between-scanner effect"
        assert np.isnan(reg[p]["median_within"]), \
            "within-scanner must be undefined when protocol is constant inside a scanner"


def test_chance_counts_only_the_classes_actually_scored():
    """The pilot reported chance = 0.20 while the classifier used 2-3 classes (true chance
    0.33-0.50), which made an at-chance result look 2x above it. Chance must come from the
    classes that were scored, not from every scanner present."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):                     # two big scanners, three tiny ones
        s = ["A", "B"][i % 2] if i < 285 else ["X", "Y", "Z"][i % 3]
        rows.append({"case_id": f"c{i}", "patient_id": f"p{i}", "scanner": s,
                     "dist": {k: 0.0 for k in PARAMS},
                     "theta": {k: float(rng.normal()) for k in PARAMS},
                     "shrinkage": {k: 0.1 for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}})
    sep = RA.separability(rows, n_perm=100)
    for p in PARAMS:
        assert sep[p]["n_effective_classes"] <= 3, sep[p]["n_effective_classes"]
        assert sep[p]["chance"] > 0.3, f"chance must reflect the scored classes: {sep[p]}"
        assert sep[p]["above_chance"] is False, \
            f"random features must not read as separable: {sep[p]}"


def test_within_scanner_median_is_undefined_below_three_scanners():
    """A median of two numbers is not a summary. The pilot printed one as if it were."""
    rows = _rows(n=200, slope=-0.5)
    for r in rows:
        r["scanner"] = ["A", "B"][hash(r["case_id"]) % 2]
    reg = RA.within_scanner_regression(rows)
    for p in PARAMS:
        assert reg[p]["n_scanners"] == 2
        assert np.isnan(reg[p]["median_within"]), "2-scanner median must be n/a"
        assert not np.isnan(reg[p]["median_within_raw"]), "but still recorded"


def test_separability_is_at_chance_when_the_parameter_carries_nothing():
    rng = np.random.default_rng(0)
    rows = [{"case_id": f"c{i}", "patient_id": f"p{i}", "scanner": ["A", "B"][i % 2],
             "dist": {k: 0.0 for k in PARAMS},
             "theta": {k: float(rng.normal()) for k in PARAMS},
             "shrinkage": {k: 0.1 for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}}
            for i in range(400)]
    sep = RA.separability(rows, n_perm=100)
    for p in PARAMS:
        assert abs(sep[p]["balanced_acc"] - 0.5) < 0.12, f"{p}: {sep[p]['balanced_acc']}"
        assert sep[p]["above_chance"] is False


def test_above_chance_uses_a_permutation_null():
    """Two earlier rules were wrong. `acc > 1/n_scanners_present` understated chance because
    most scanners never enter the classifier. A 2-standard-error margin over 5 folds is itself
    too noisy -- uninformative features cleared it in 2 of 12 trials. This checks the
    permutation null across several seeds, which is the point: it must hold every time."""
    rng = np.random.default_rng(3)
    rows = [{"case_id": f"c{i}", "patient_id": f"p{i}", "scanner": ["A", "B"][i % 2],
             "dist": {k: 0.0 for k in PARAMS},
             "theta": {k: float(rng.normal()) for k in PARAMS},
             "shrinkage": {k: 0.1 for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}}
            for i in range(600)]
    for seed in (3, 4, 5):
        sep = RA.separability(rows, seed=seed, n_perm=120)
        for p in PARAMS:
            assert not sep[p]["above_chance"], f"seed {seed} {p}: {sep[p]}"
            assert sep[p]["n_perm"] > 50


def test_separability_detects_a_planted_scanner_offset():
    rng = np.random.default_rng(1)
    rows = [{"case_id": f"c{i}", "patient_id": f"p{i}", "scanner": ["A", "B"][i % 2],
             "dist": {k: 0.0 for k in PARAMS},
             "theta": {k: float(rng.normal(0 if i % 2 else 4, 0.3)) for k in PARAMS},
             "shrinkage": {k: 0.1 for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}}
            for i in range(400)]
    sep = RA.separability(rows, n_perm=100)
    for p in PARAMS:
        assert sep[p]["balanced_acc"] > 0.9
        assert sep[p]["above_chance"] is True, "a planted offset must clear the null"


def test_distance_from_prior_is_zero_at_the_prior_mean():
    """Sanity on the quantity being regressed."""
    truth = {k: v[0] for k, v in TISSUE.items()}
    th = estimate_case(_render_case(truth, ACQ), ACQ, n_vox=400)
    d = {k: abs(th[k] - float(np.log(TISSUE[k][0]))) for k in PARAMS}
    assert d["T2"] < 0.15 and d["D"] < 0.15


def _flat_rows(n=400, seed=1, spread=0.0004):
    """The D case: shrinkage is pinned, so there is no regressor to correlate against."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = 0.009 + float(rng.uniform(-spread / 2, spread / 2))
        d = float(rng.uniform(0.1, 0.6))
        out.append({"case_id": f"c{i}", "patient_id": f"p{i//2}",
                    "scanner": ["A", "B", "C"][i % 3],
                    "dist": {k: d for k in PARAMS}, "theta": {k: d for k in PARAMS},
                    "shrinkage": {k: s for k in PARAMS}, "crlb": {k: 1.0 for k in PARAMS}})
    return out


def test_leverage_separates_a_pinned_regressor_from_a_moving_one():
    """D's shrinkage moves 0.0002 IQR over 4608 real cases -- that is not a null result."""
    assert RA.leverage(_flat_rows())["D"]["ratio"] < RA.MIN_LEVERAGE
    assert RA.leverage(_rows())["T1"]["ratio"] > RA.MIN_LEVERAGE


def test_leverage_matches_the_hand_computed_ratio():
    lev = RA.leverage(_rows(n=2000))["PD"]
    assert abs(lev["ratio"] - (1 - lev["p05"]) / (1 - lev["p95"])) < 1e-9


def test_sign_test_needs_agreement_not_just_a_median():
    """PD: median +0.113 but 2 of 5 scanners negative. The median alone would call it."""
    assert RA.sign_test([-0.41, 0.13, 0.11, -0.14, 0.28])["n_neg"] == 2
    assert RA.sign_test([-0.46, -0.12, -0.34, -0.13, -0.63])["p"] < 0.05
    assert RA.sign_test([-0.41, 0.13, 0.11, -0.14, 0.28])["p"] > 0.5


def test_sign_test_is_undefined_below_three_scanners():
    assert RA.sign_test([-0.4, -0.3])["p"] != RA.sign_test([-0.4, -0.3])["p"]


def test_regression_carries_the_sign_count():
    d = RA.within_scanner_regression(_rows())["T1"]
    assert d["sign"]["n"] == d["n_scanners"] and d["sign"]["n_neg"] == 3


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
