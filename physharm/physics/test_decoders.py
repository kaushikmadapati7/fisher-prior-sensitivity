"""Unit tests for the physics forward models.

Run with the Project 3 venv:
    .../venv/bin/python -m pytest physharm/physics/test_decoders.py -q
or standalone:
    .../venv/bin/python physharm/physics/test_decoders.py
"""
import math

import torch

from physharm.physics.decoders import (
    ContrastGains,
    assert_units,
    render_contrast,
    render_se_epi,
    render_spgr,
    render_tse_t2,
)


def _scalar(v):
    return torch.tensor(float(v), dtype=torch.float64)


def test_tse_t2_closed_form():
    PD, T1, T2, g = _scalar(0.5), _scalar(1200.0), _scalar(80.0), _scalar(1.0)
    TR, TE = 7300.0, 135.0
    got = render_tse_t2(PD, T1, T2, g, TR, TE).item()
    exp = 0.5 * (1 - math.exp(-TR / 1200.0)) * math.exp(-TE / 80.0)
    assert abs(got - exp) < 1e-6, (got, exp)  # _EPS in denominators ~1e-9 abs


def test_se_epi_closed_form():
    PD, T1, T2, D, g = (_scalar(0.5), _scalar(1200.0), _scalar(80.0),
                        _scalar(1.0e-3), _scalar(1.0))
    TR, TE, b = 6540.0, 63.0, 1500.0
    got = render_se_epi(PD, T1, T2, D, g, TR, TE, b).item()
    s0 = 0.5 * (1 - math.exp(-TR / 1200.0)) * math.exp(-TE / 80.0)
    exp = s0 * math.exp(-b * 1.0e-3)
    assert abs(got - exp) < 1e-6, (got, exp)


def test_spgr_closed_form():
    PD, T1, T2, g = _scalar(0.5), _scalar(1200.0), _scalar(80.0), _scalar(1.0)
    TR, TE, FA = 3.22, 1.18, 9.0
    got = render_spgr(PD, T1, g, TR, TE, T2, FA).item()
    a = FA * math.pi / 180.0
    E1 = math.exp(-TR / 1200.0)
    exp = (0.5 * math.sin(a) * (1 - E1) / (1 - math.cos(a) * E1)
           * math.exp(-TE / 80.0))
    assert abs(got - exp) < 1e-6, (got, exp)


def test_te_monotonicity():
    PD, T1, T2, g = _scalar(0.5), _scalar(1200.0), _scalar(80.0), _scalar(1.0)
    s_short = render_tse_t2(PD, T1, T2, g, 7300.0, 60.0).item()
    s_long = render_tse_t2(PD, T1, T2, g, 7300.0, 135.0).item()
    assert s_long < s_short, "longer TE must reduce T2W signal"


def test_te_zero_is_finite():
    PD, T1, T2, g = _scalar(0.5), _scalar(1200.0), _scalar(80.0), _scalar(1.0)
    s = render_spgr(PD, T1, g, 4.26, 0.0, T2, 10.0).item()  # Philips TE=0
    assert math.isfinite(s) and s > 0, s


def test_gradcheck():
    torch.manual_seed(0)
    PD = torch.rand(2, 1, 2, 2, dtype=torch.float64, requires_grad=True) * 0.5 + 0.25
    T1 = torch.rand(2, 1, 2, 2, dtype=torch.float64, requires_grad=True) * 1000 + 500
    T2 = torch.rand(2, 1, 2, 2, dtype=torch.float64, requires_grad=True) * 100 + 40
    D = torch.rand(2, 1, 2, 2, dtype=torch.float64, requires_grad=True) * 1e-3 + 5e-4
    g = _scalar(1.0)

    def f(PD, T1, T2, D):
        a = render_tse_t2(PD, T1, T2, g, 7300.0, 135.0)
        b = render_se_epi(PD, T1, T2, D, g, 6540.0, 63.0, 1500.0)
        c = render_spgr(PD, T1, g, 3.22, 1.18, T2, 9.0)
        return a + b + c

    assert torch.autograd.gradcheck(f, (PD, T1, T2, D), eps=1e-6, atol=1e-4)


def test_dispatch_and_gains():
    gains = ContrastGains(["T2W", "DWI", "DCE_pre"])
    theta = {k: _scalar(v) for k, v in
             dict(PD=0.5, T1=1200.0, T2=80.0, D=1.0e-3).items()}
    p_t2 = {"TR_ms": 7300.0, "TE_ms": 135.0}
    p_dwi = {"TR_ms": 6540.0, "TE_ms": 63.0, "b_values": [100.0, 1500.0]}
    p_dce = {"TR_ms": 3.22, "TE_ms": 1.18, "FA_deg": 9.0}
    assert_units(p_t2, "T2W"); assert_units(p_dwi, "DWI"); assert_units(p_dce, "DCE_pre")
    s_t2 = render_contrast("tse_t2", theta, p_t2, gains["T2W"]).item()
    s_dwi = render_contrast("se_epi", theta, p_dwi, gains["DWI"]).item()  # uses b=1500
    s_dce = render_contrast("spgr", theta, p_dce, gains["DCE_pre"]).item()
    for s in (s_t2, s_dwi, s_dce):
        assert math.isfinite(s) and s > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} physics tests passed.")
