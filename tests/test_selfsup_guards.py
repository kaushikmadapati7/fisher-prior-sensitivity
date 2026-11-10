"""Guards on the self-supervised arm. These exist because each one failed in a cohort run.

The arm is outside `physharm` and needs torch, so these skip when it is absent rather than
forcing the dependency on the whole suite.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_clip_default_matches_the_trainer():
    """A CLI default left behind silently overrode a tightened clip on a cohort run.

    The launch script now passes --clip explicitly, but the defaults must agree anyway: the next
    person to run this by hand will not pass it.
    """
    import inspect
    sst, sd = _load("selfsup_torch"), _load("seed_disagreement")
    trainer_default = inspect.signature(sst.train).parameters["clip"].default
    src = (ROOT / "experiments" / "seed_disagreement.py").read_text()
    import re
    m = re.search(r'add_argument\("--clip",\s*type=float,\s*default=([0-9.eE+-]+)', src)
    assert m, "could not find the --clip default"
    assert float(m.group(1)) == trainer_default, (
        f"CLI default {m.group(1)} disagrees with trainer default {trainer_default}")


def test_the_divergence_guard_catches_a_finite_blowup():
    """`isfinite` is not enough: a cohort run reported losses of 1e117, which are finite.

    THE GUARD IS NOW HARD TO REACH, WHICH IS THE POINT. Bounding the network's output (#84)
    capped the loss at ~5e5 against an initialisation around 2e3, so the settings that used to
    blow past the 1000x threshold -- lr 0.5 with clipping effectively off -- no longer can. The
    guard is still the last line of defence and its logic still has to be exercised, so this
    drives it through `blowup` instead of through a real divergence. `test_the_output_bound_
    makes_the_old_divergence_settings_survivable` covers the other half: that those settings
    now train instead of dying.
    """
    sst = _load("selfsup_torch")
    from physharm.analysis.selfsup_arm import acq_features
    from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _model_log, _rows_for

    acq = {"TR_T2W": 6000.0, "TE_T2W": 120.0, "TR_DWI": 4000.0, "TE_DWI": 70.0,
           "b_hi": 1000.0, "b_lo": 0.0}
    rows = _rows_for("A1", acq)
    mean = {k: v[0] for k, v in TISSUE.items()}
    x0 = np.array([[np.log(mean[k])] for k in PARAMS])
    rng = np.random.default_rng(0)
    x = x0 + 0.2 * rng.normal(size=(4, 80))
    y = _model_log(x, rows) + 0.05 * rng.normal(size=(len(rows), 80))
    exams = [(rows, y, acq_features(acq).T)]

    with pytest.raises(SystemExit) as e:
        sst.train(exams, x0, 1 / 0.05 ** 2, 1 / 100.0, steps=400, seed=0, lr=0.5, clip=1e9,
                  blowup=1.05, warmup=0.01)
    assert "diverged" in str(e.value)

    net, scale = sst.train(exams, x0, 1 / 0.05 ** 2, 1 / 100.0, steps=300, seed=0)
    assert np.isfinite(sst.predict(net, y, acq_features(acq).T, scale, x0)).all()


def test_the_output_bound_removes_the_exponential_divergence_path():
    """The regression test for the actual fix, at the settings the sweeps actually used.

    WHAT THE BOUND DOES AND DOES NOT DO. The forward model is exponential and unbounded in two
    directions -- -TE/exp(T2) as T2 is driven down and -b*exp(D) as D is driven up -- and that is
    where the sweeps' losses of 1e20, 1e72 and 1e117 came from, at lr 1e-3 and clip 10. Bounding
    those two directions removes that path entirely.

    It does NOT make any learning rate safe. PD and T1 are deliberately unbounded, since neither
    can run away and bounding them is what voided run A, so an absurd step size can still
    overflow them into NaN. That is what the divergence guard is for, and the test above covers
    it. Do not "fix" a NaN here by bounding PD or T1.
    """
    sst = _load("selfsup_torch")
    from physharm.analysis.selfsup_arm import acq_features
    from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _model_log, _rows_for

    acq = {"TR_T2W": 6000.0, "TE_T2W": 120.0, "TR_DWI": 4000.0, "TE_DWI": 70.0,
           "b_hi": 1000.0, "b_lo": 0.0}
    rows = _rows_for("A1", acq)
    mean = {k: v[0] for k, v in TISSUE.items()}
    x0 = np.array([[np.log(mean[k])] for k in PARAMS])
    rng = np.random.default_rng(0)
    x = x0 + 0.2 * rng.normal(size=(4, 80))
    y = _model_log(x, rows) + 0.05 * rng.normal(size=(len(rows), 80))
    af = acq_features(acq).T
    exams = [(rows, y, af)]

    net, scale = sst.train(exams, x0, 1 / 0.05 ** 2, 1 / 100.0, steps=400, seed=0,
                           lr=1e-3, clip=10.0)
    pred = sst.predict(net, y, af, scale, x0)
    assert np.isfinite(pred).all()
    # the two divergent directions are closed, whatever the optimiser did on the way: T2 cannot
    # fall more than `bound` below the parameterisation point, and D cannot rise more than it
    dev = pred - x0
    for k, sign in sst.BOUNDED.items():
        assert (sign * dev[k]).max() <= sst.DEVIATION_BOUND + 1e-9, (k, dev[k].min(), dev[k].max())

    # and the loss such a network can reach is finite and small, which is the property that
    # makes the guard unreachable by this path rather than merely unlikely
    worst = sst.Net(len(rows) + 5, width=16, bound=sst.DEVIATION_BOUND).double()
    import torch as _t
    with _t.no_grad():
        worst.f[-1].bias.copy_(_t.tensor([0.0, 0.0, -1e3, 1e3], dtype=_t.float64))
    x0t = _t.tensor(x0.T, dtype=_t.float64)
    L = sst.physics_loss_torch(worst(_t.zeros(16, len(rows) + 5, dtype=_t.float64), x0t),
                               _t.zeros(16, len(rows), dtype=_t.float64), rows,
                               1 / 0.05 ** 2, 1 / 100.0, x0t)
    assert float(L.detach()) < 2e6, float(L.detach())
