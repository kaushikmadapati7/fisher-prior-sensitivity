"""The ensemble/architecture arm, end to end on a fixture. No images, no cluster.

WHY END TO END. Three ML sweeps on this project were launched, ran for hours and were thrown
away -- once because a fix never reached the node, once because a guard watched the wrong loss,
once because the optimisation was unstable in a way no local run had exercised. The reporting and
aggregation code in particular had never been executed before a cohort run executed it. This
builds a small cohort on disk and runs the CLI.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physharm.analysis.synthetic_arm import PARAMS, TISSUE  # noqa: E402
from tests.test_traveling_arm import ACQ, _render_case      # noqa: E402

TRUE = {"PD": 1.4, "T1": 900.0, "T2": 120.0, "D": 1.1e-3}


def _sst():
    spec = importlib.util.spec_from_file_location("sst", ROOT / "experiments" / "selfsup_torch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cohort(tmp_path, n_cases=24):
    """288 voxels a case: `_rows_and_signal` returns None below 50 surviving voxels, so a
    fixture sized to the subsample rather than to that floor yields zero usable exams."""
    cache, mans = tmp_path / "cache", tmp_path / "manifests"
    cache.mkdir(), mans.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_cases):
        acq = dict(ACQ, TR_T2W=ACQ["TR_T2W"] * float(np.exp(rng.normal(0, 0.15))),
                   TE_T2W=ACQ["TE_T2W"] * float(np.exp(rng.normal(0, 0.15))))
        np.savez(cache / f"case{i}.npz",
                 **_render_case(TRUE, acq, shape=(2, 12, 12), noise=0.02, seed=i))
        json.dump({"case_id": f"case{i}", "patient_id": f"pt{i}",
                   "scanner_model": "Siemens Skyra",
                   "contrasts": {"T2W": {"TR_ms": acq["TR_T2W"], "TE_ms": acq["TE_T2W"]},
                                 "DWI": {"TR_ms": acq["TR_DWI"], "TE_ms": acq["TE_DWI"],
                                         "b_values": [0.0, acq["b_hi"]]}}},
                  open(mans / f"case{i}.json", "w"))
    return cache, mans


# ------------------------------------------------------------------ the output bound

def test_the_bound_makes_the_loss_finite_where_the_forward_model_is_not():
    """The mechanism behind the 1e20, 1e72 and 1e117 losses three sweeps produced.

    The T2W row carries -TE/exp(T2) and the DWI row -b*exp(D); both are unbounded, so a network
    free to emit any log-parameter can drive the loss arbitrarily high. Gradient clipping bounds
    the STEP, not the region, which is why clipping never fixed this.
    """
    sst = _sst()
    from physharm.analysis.selfsup_arm import acq_features
    from physharm.analysis.synthetic_arm import _rows_for
    rows = _rows_for("A1", ACQ)
    x0 = np.array([[np.log(TISSUE[k][0])] for k in PARAMS])
    d_in = len(rows) + acq_features(ACQ).size
    losses = {}
    for tag, bound in (("unbounded", 1e6), ("bounded", sst.DEVIATION_BOUND)):
        net = sst.Net(d_in, width=16, bound=bound).double()
        with torch.no_grad():                       # drive T2 down and D up, the two bad ways
            net.f[-1].bias.copy_(torch.tensor([0.0, 0.0, -40.0, 40.0], dtype=torch.float64))
        x0t = torch.tensor(x0.T, dtype=torch.float64)
        x = net(torch.zeros(32, d_in, dtype=torch.float64), x0t)
        losses[tag] = float(sst.physics_loss_torch(
            x, torch.zeros(32, len(rows), dtype=torch.float64), rows, 400.0, 0.01, x0t).detach())
    assert losses["unbounded"] > 1e30, losses
    # and below the divergence guard's threshold, which is ~1000x an initialisation near 2e3
    assert losses["bounded"] < 2e6, losses


def test_an_untrained_network_still_predicts_the_prior_exactly():
    """tanh(0) = 0, so the bound must not disturb the zero-init that keeps T1 off exp(0) = 1 ms."""
    sst = _sst()
    net = sst.Net(7, width=16).double()
    x0 = torch.tensor([[0.3, 7.2, 4.4, -6.6]], dtype=torch.float64)
    out = net(torch.randn(8, 7, dtype=torch.float64), x0)
    assert torch.allclose(out, x0.repeat(8, 1)), out


def test_the_bound_is_a_fixed_constant_and_not_a_per_run_knob():
    """It must not differentiate the configurations being compared, or it becomes a confound."""
    sst = _sst()
    assert sst.Net(7, width=16).bound == sst.Net(7, width=512).bound == sst.DEVIATION_BOUND


def test_only_the_two_divergent_directions_are_bounded():
    """The bug that voided run A: bounding PD and T1, which cannot diverge.

    The cache is normalised per case, so the scale is an additive constant in log space that PD
    absorbs -- a quantity with no physiological range. Bounding it pinned 52% of predictions and
    the run had to be thrown away. PD enters the log-signal linearly and T1 through a term that
    is linear at one extreme and constant at the other, so the loss is a bowl in both and neither
    can run away. Only T2 driven DOWN and D driven UP are exponential.
    """
    sst = _sst()
    assert sst.BOUNDED == {2: -1.0, 3: +1.0}, sst.BOUNDED
    net = sst.Net(7, width=16).double()
    with torch.no_grad():                       # ask for the moon on every parameter, both ways
        net.f[-1].bias.copy_(torch.tensor([60.0, -60.0, -60.0, 60.0], dtype=torch.float64))
    d = net(torch.zeros(2, 7, dtype=torch.float64),
            torch.zeros(2, 4, dtype=torch.float64)).detach().numpy()[0]
    assert d[0] == 60.0 and d[1] == -60.0, ("PD and T1 must be free", d)
    assert abs(d[2]) <= sst.DEVIATION_BOUND and abs(d[3]) <= sst.DEVIATION_BOUND, d


def test_the_bounded_parameters_are_free_on_their_safe_side():
    """One-sided. T2 blows up only downward and D only upward; the other direction is identity,
    so the bound constrains the divergence and nothing else."""
    sst = _sst()
    net = sst.Net(7, width=16).double()
    with torch.no_grad():
        net.f[-1].bias.copy_(torch.tensor([0.0, 0.0, 40.0, -40.0], dtype=torch.float64))
    d = net(torch.zeros(2, 7, dtype=torch.float64),
            torch.zeros(2, 4, dtype=torch.float64)).detach().numpy()[0]
    assert d[2] == 40.0 and d[3] == -40.0, ("safe side must be untouched", d)


def test_saturation_is_reported_per_parameter():
    """A scalar max said only that something was pinned; finding out what cost a cohort run."""
    sst = _sst()
    net = sst.Net(7, width=16).double()
    with torch.no_grad():
        net.f[-1].bias.copy_(torch.tensor([0.0, 0.0, -60.0, 0.0], dtype=torch.float64))
    sat = net.saturation(torch.zeros(5, 7, dtype=torch.float64))
    assert sat.shape == (4,)
    assert sat[2] == 1.0 and sat[0] == sat[1] == sat[3] == 0.0, sat


# ------------------------------------------------------------------ the experiment

def test_the_cli_runs_end_to_end_and_reports_both_comparisons(tmp_path):
    cache, mans = _cohort(tmp_path)
    out = tmp_path / "ens.json"
    r = subprocess.run(
        [sys.executable, "experiments/ensemble_prior.py",
         "--cache-dir", str(cache), "--manifest-dir", str(mans),
         "--seeds", "4", "--steps", "150", "--n-vox", "120", "--n-eval", "6",
         "--widths", "16", "32", "--size-sweep", "6", "--size-width", "16",
         "--out", str(out)],
        capture_output=True, text=True, timeout=1800, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "DOES ENSEMBLING FIX IT?" in r.stdout, r.stdout[-2000:]
    assert "DOES MORE DATA FIX IT?" in r.stdout, r.stdout[-2000:]
    assert "IS THE ARCHITECTURE THE PRIOR?" in r.stdout, r.stdout[-2000:]

    rep = json.load(open(out))
    assert set(rep["per_width"]) == {"width 16", "width 32", "n=6"}
    for wd in rep["per_width"].values():
        for k in ("single", "ens"):
            assert {"along", "orth", "ratio", "share", "n_exams"} <= set(wd[k])
            assert np.isfinite(wd[k]["ratio"])
    # cross-comparison is between ARCHITECTURES only; a size point differs in data, not in model,
    # so pairing it against a width would confound the two axes
    assert set(rep["cross_width"]) == {"width 16 vs width 32"}
    assert rep["meta"]["n=6"]["width"] == 16 and rep["meta"]["n=6"]["n_train"] == 6
    assert rep["bar"] == rep["prior_baseline"] / 10.0
    assert rep["bound"] == _sst().DEVIATION_BOUND
    for m in rep["meta"].values():
        assert m["n_seeds"] % 2 == 0 and m["n_seeds"] >= 4
        assert set(m["saturation"]) == {"PD", "T1", "T2", "D"}
        assert all(np.isfinite(v) for v in m["saturation"].values())
        assert m["saturation"]["PD"] == m["saturation"]["T1"] == 0.0


def test_a_network_no_better_than_the_prior_is_dropped(tmp_path):
    """The defect that put a held-out loss of 867.6 in run B's table.

    The bar used to be an absolute 1e5. A network that just returns the prior scores in the low
    thousands on this cohort and converged ones score tens, so 1e5 admitted networks that had
    learned nothing. The bar is now computed from the prior baseline on the same held-out exams.
    """
    cache, mans = _cohort(tmp_path, n_cases=12)
    r = subprocess.run(
        [sys.executable, "experiments/ensemble_prior.py",
         "--cache-dir", str(cache), "--manifest-dir", str(mans), "--seeds", "4",
         "--steps", "1", "--n-vox", "120", "--n-eval", "4", "--widths", "16"],
        capture_output=True, text=True, timeout=900, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    # one step of training cannot beat the prior by 10x, so every seed must be dropped and the
    # run must end without a table rather than reporting numbers from untrained networks
    assert "does not beat the prior baseline" in r.stdout, r.stdout[-3000:]
    assert r.returncode != 0
    assert "no width produced a usable ensemble" in (r.stdout + r.stderr)


def test_checkpoints_give_the_same_network_training_would_have_stopped_at():
    """The premise of the loss curve: a schedule that is a function of t alone makes a
    mid-training snapshot identical to a run truncated there. Cosine is not such a schedule, and
    the second half of this test pins that, so nobody can switch the curve back to it."""
    sst = _sst()
    from physharm.analysis.selfsup_arm import acq_features
    from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _model_log, _rows_for
    rows = _rows_for("A1", ACQ)
    x0 = np.array([[np.log(TISSUE[k][0])] for k in PARAMS])
    rng = np.random.default_rng(0)
    x = x0 + 0.2 * rng.normal(size=(4, 60))
    y = _model_log(x, rows) + 0.05 * rng.normal(size=(len(rows), 60))
    af = acq_features(ACQ).T
    exams = [(rows, y, af)]

    kw = dict(schedule="inv", warmup=10)
    grabbed = {}
    sst.train(exams, x0, 400.0, 0.01, steps=120, seed=0, width=16, checkpoints=[60],
              on_checkpoint=lambda n, sc, st: grabbed.setdefault(
                  st, sst.predict(n, y, af, sc, x0)), **kw)
    net, scale = sst.train(exams, x0, 400.0, 0.01, steps=60, seed=0, width=16, **kw)
    assert np.allclose(grabbed[60], sst.predict(net, y, af, scale, x0)), "checkpoint != truncated"

    # and it is NOT true under the cosine schedule, which is why the curve does not use it
    g2 = {}
    sst.train(exams, x0, 400.0, 0.01, steps=120, seed=0, width=16, checkpoints=[60],
              on_checkpoint=lambda n, sc, st: g2.setdefault(st, sst.predict(n, y, af, sc, x0)))
    n2, s2 = sst.train(exams, x0, 400.0, 0.01, steps=60, seed=0, width=16)
    assert not np.allclose(g2[60], sst.predict(n2, y, af, s2, x0))


def test_the_loss_curve_runs_and_keeps_its_seed_set_fixed(tmp_path):
    cache, mans = _cohort(tmp_path, n_cases=20)
    out = tmp_path / "curve.json"
    r = subprocess.run(
        [sys.executable, "experiments/ensemble_prior.py",
         "--cache-dir", str(cache), "--manifest-dir", str(mans), "--seeds", "4",
         "--steps", "120", "--n-vox", "120", "--n-eval", "6", "--widths", "16",
         "--size-width", "16", "--loss-curve", "6", "0", "--checkpoints", "60", "120",
         "--out", str(out)],
        capture_output=True, text=True, timeout=1800, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert "RATIO AGAINST CONVERGENCE" in r.stdout, r.stdout[-2000:]
    cur = json.load(open(out))["loss_curves"]
    assert set(cur) == {"6", "14"}, set(cur)
    for c in cur.values():
        if not c["points"]:
            continue
        assert [p["step"] for p in c["points"]] == [60, 120]
        # the population must not change along the curve, or it describes two things at once
        assert len({p["n_seeds"] for p in c["points"]}) == 1, c["points"]
        for p in c["points"]:
            assert np.isfinite(p["ratio"]) and np.isfinite(p["loss"])


def test_a_seed_that_is_good_early_and_bad_late_is_kept():
    """The rule run E got wrong: it admitted on the FINAL checkpoint.

    A curve exists because checkpoints differ. n=500 at 160k steps is 320 passes over each exam,
    which overfits, so the final loss can be far worse than the best -- that run dropped a seed at
    a final loss of 1225 and threw away whatever its early points were worth. Admission is on the
    BEST checkpoint; the seed is then used at every checkpoint, so the population stays fixed
    along the curve.

    Training and the loss are faked so the curve sees a planted shape -- good at 20, ruinous at
    40 -- which is the one case that cannot be arranged reliably by actually training something.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ep", ROOT / "experiments" / "ensemble_prior.py")
    ep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ep)
    from physharm.analysis.synthetic_arm import PARAMS, TISSUE, _model_log, _rows_for
    from physharm.analysis.selfsup_arm import acq_features

    rows = _rows_for("A1", ACQ)
    rng = np.random.default_rng(0)
    x0 = np.array([[np.log(TISSUE[k][0])] for k in PARAMS])
    xt = x0 + 0.2 * rng.normal(size=(4, 40))
    Y = _model_log(xt, rows) + 0.02 * rng.normal(size=(len(rows), 40))
    ev = [(rows, Y, acq_features(ACQ).T)]
    LOSS = {20: 5.0, 40: 900.0}            # best early, hopeless late

    def fake_train(exams, x0_, w, lam, **kw):
        for st in sorted(kw["checkpoints"]):
            kw["on_checkpoint"](("net", kw["seed"], st), "scale", st)
        return ("net", kw["seed"], None), "scale"

    def fake_predict(net, Y_, af, scale, x0_):
        _, seed, st = net
        r = np.random.default_rng(1000 * seed + (st or 0))
        out = x0 + 0.05 * r.normal(size=(4, Y_.shape[1]))
        out[0, 0] = st if st is not None else -1     # carries the step to the fake loss
        return out

    real_train, real_predict, real_loss = ep._sst.train, ep._sst.predict, ep.physics_loss
    ep._sst.train, ep._sst.predict = fake_train, fake_predict
    ep.physics_loss = lambda x, *a, **k: LOSS[int(x[0, 0])]
    try:
        out, dropped = ep.loss_curve(ev, ev, x0, 400.0, 0.01,
                                     {k: v[0] for k, v in TISSUE.items()},
                                     3, 16, 1e-3, 1.0, [20, 40], 100.0)
    finally:
        ep._sst.train, ep._sst.predict, ep.physics_loss = real_train, real_predict, real_loss

    assert dropped == 0, (dropped, out)
    assert [c["step"] for c in out] == [20, 40]
    assert all(c["n_seeds"] == 3 for c in out), out
    # the overfit point is reported rather than hidden: its loss is the bad one
    assert out[0]["loss"] == 5.0 and out[1]["loss"] == 900.0, out


def test_an_odd_seed_count_is_refused_rather_than_silently_halved(tmp_path):
    """Unequal halves change the shrink factor the experiment exists to measure."""
    cache, mans = _cohort(tmp_path, n_cases=8)
    r = subprocess.run(
        [sys.executable, "experiments/ensemble_prior.py",
         "--cache-dir", str(cache), "--manifest-dir", str(mans), "--seeds", "5",
         "--steps", "10", "--n-vox", "120", "--n-eval", "2", "--widths", "16"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode != 0
    assert "must be even" in (r.stdout + r.stderr)


def test_the_timing_probe_runs_and_reports_a_cost_per_network(tmp_path):
    """Sizing a cluster run needs a measured steps/sec, which nothing in the repo had. This runs
    the probe end to end so it cannot fail on the node it exists to measure."""
    cache, mans = _cohort(tmp_path, n_cases=12)
    out = tmp_path / "t.json"
    r = subprocess.run(
        [sys.executable, "experiments/step_timing.py",
         "--cache-dir", str(cache), "--manifest-dir", str(mans),
         "--limits", "12", "--steps", "60", "--n-vox", "60", "120",
         "--widths", "16", "--threads", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=900, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "voxels/s" in r.stdout and "h per 160k" in r.stdout, r.stdout[-2000:]
    rec = json.load(open(out))["timings"]
    assert len(rec) == 2
    for t in rec:
        for k in ("exams", "threads", "n_vox", "width", "batch", "steps_per_s", "voxels_per_s",
                  "hours_per_160k", "load_s"):
            assert k in t and np.isfinite(t[k]), (k, t)
        assert t["steps_per_s"] > 0 and t["hours_per_160k"] > 0
    # the batch reported is the MEDIAN exam's, not the smallest -- the smallest understates
    # what a typical step does, which is how the first probe's voxels/sec came out low
    assert {t["batch"] for t in rec} == {60.0, 120.0}
