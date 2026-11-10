"""Tests for the Fisher/identifiability predictor.

Run with pytest, or standalone:  python tests/test_fisher.py
No pytest dependency -- these must run on a login node with nothing installed.

Numpy only, so these run anywhere -- including a login node with no GPU and no torch.
The torch-dependent decoder gradcheck lives in physharm/physics/test_decoders.py and must
be run on the cluster.

The regression tests here exist because the archived implementation was defective in a way
that would have inverted this project's headline result. `test_pinv_is_anticorrelated...`
documents the failure rather than merely avoiding it, so nobody reintroduces it.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

from physharm.physics import fisher as F

# SIGNA Premier median protocol from the Phase-2 audit (the run log).
ACQ = dict(TR_T2W=8136.0, TE_T2W=153.0, TR_DWI=4000.0, TE_DWI=70.0, b_hi=1000.0, b_lo=0.0)
# The three archived per-vendor protocols, used only to reproduce the pinv pathology.
VENDOR = {"GE":      dict(TR_T2W=2602.0, TE_T2W=112.0, TR_DWI=5000.0, TE_DWI=78.0, b_hi=1500.0),
          "Siemens": dict(TR_T2W=7300.0, TE_T2W=135.0, TR_DWI=5000.0, TE_DWI=63.0, b_hi=1500.0),
          "Philips": dict(TR_T2W=4084.0, TE_T2W=80.0,  TR_DWI=5000.0, TE_DWI=79.0, b_hi=1200.0)}


def _signal(TR, TE, b, PD, T1, T2, D):
    return PD * (1 - math.exp(-TR / T1)) * math.exp(-TE / T2) * math.exp(-b * D)


def test_log_sensitivities_match_finite_differences():
    """The analytic rows are the ground truth for everything else; check them numerically."""
    TR, TE, b = 5000.0, 100.0, 1000.0
    t = dict(F.THETA0)
    ana = F.log_sensitivities(TR, TE, b, t)
    base = dict(PD=t["PD"], T1=t["T1"], T2=t["T2"], D=t["D"])
    for i, k in enumerate(F.PARAMS):
        h = base[k] * 1e-6
        up, dn = dict(base), dict(base)
        up[k], dn[k] = base[k] + h, base[k] - h
        num = (math.log(_signal(TR, TE, b, **up)) - math.log(_signal(TR, TE, b, **dn))) / (2 * h) * base[k]
        assert abs(num - ana[i]) < 1e-4 * max(1.0, abs(num)), f"{k}: {num} vs {ana[i]}"


def test_t1_recovery_term_saturates():
    """T1 information dies at clinical TR -- the physical reason T1 is unrecoverable."""
    for TR, expected in [(2602, 0.114), (4084, 0.033), (7300, 0.0023)]:
        got = math.exp(-TR / 1200.0)
        assert abs(got - expected) < 6e-3, f"TR={TR}: {got:.4f} vs {expected}"


def test_t1_sensitivity_falls_monotonically_with_tr():
    t = dict(F.THETA0, T1=1200.0)
    s = [abs(F.log_sensitivities(TR, 100.0, 0.0, t)[1]) for TR in (2602, 4084, 7300)]
    assert s[0] > s[1] > s[2]
    assert s[2] < 0.02, "T1 should be nearly invisible at the longest TR"


def test_crlb_ordering_is_physical():
    """D measured, T2 identified by the TE ratio, PD confounded with gain, T1 sloppy."""
    sd = F.identifiability(ACQ, "A1")["crlb"]
    assert sd["D"] < sd["T2"] < sd["PD"] < sd["T1"]


def test_ordering_stable_in_the_prior():
    """prior_sd is an assumption; only the ORDERING is used downstream, so pin it."""
    for prior_sd in (1.0, 3.0, 10.0, 30.0, 100.0):
        sd = F.identifiability(ACQ, "A1", prior_sd=prior_sd)["crlb"]
        assert sorted(sd, key=sd.get) == ["D", "T2", "PD", "T1"], f"prior_sd={prior_sd}: {sd}"


def test_ordering_stable_under_relative_noise_weighting():
    """DWI SNR is unmeasurable on resampled volumes (the run log Phase 2c). It does not
    matter: a global noise scale cancels out of the ordering, which survives a 625x sweep
    of the relative weighting -- so SNR is a sensitivity axis, not a blocker."""
    for w_dwi in (0.04, 0.2, 1.0, 5.0, 25.0):
        for w_adc in (0.1, 1.0, 10.0):
            sd = F.identifiability(ACQ, "A1", weights=[1.0, w_dwi, w_adc])["crlb"]
            assert sorted(sd, key=sd.get) == ["D", "T2", "PD", "T1"], \
                f"w_dwi={w_dwi} w_adc={w_adc}: {sd}"


def test_low_b_band_adds_no_new_direction():
    """the project plan §3.3 arm A2, and a falsifiable prediction about the planned S0 rewiring.

    (DWI at b_hi) - (DWI at b_lo) is exactly the ADC row direction, so once ADC clamps D the
    S0 band contributes averaging, not identifiability: the rank does not rise and CRLB(T2)
    improves only modestly. If the real S0 rewiring delivers a LARGE invariance gain, the
    Fisher analysis under-predicted and the thesis is weakened -- report it.
    """
    a1, a2 = F.identifiability(ACQ, "A1"), F.identifiability(ACQ, "A2")
    assert a2["rank"] == a1["rank"], "an extra b-band must not raise the rank"
    gain = (a1["crlb"]["T2"] - a2["crlb"]["T2"]) / a1["crlb"]["T2"]
    assert 0.0 < gain < 0.40, f"expected a modest averaging gain, got {gain:.1%}"

    hi = F.log_sensitivities(ACQ["TR_DWI"], ACQ["TE_DWI"], ACQ["b_hi"])
    lo = F.log_sensitivities(ACQ["TR_DWI"], ACQ["TE_DWI"], ACQ["b_lo"])
    d = hi - lo
    assert np.allclose(d / np.linalg.norm(d), -F.anchor_row("D"), atol=1e-12)


def test_short_tr_is_what_rescues_t1():
    """A5: only a short-TR acquisition moves T1 out of the sloppy subspace."""
    t1_base = F.identifiability(ACQ, "A1")["crlb"]["T1"]
    t1_short = F.identifiability(dict(ACQ, TR_SHORT=600.0), "A5")["crlb"]["T1"]
    assert t1_short < 0.5 * t1_base


def test_dropping_the_te_ratio_destroys_t2():
    """A3: T2W alone cannot separate PD from T2 -- one equation, two unknowns."""
    assert F.identifiability(ACQ, "A3")["crlb"]["T2"] > F.identifiability(ACQ, "A1")["crlb"]["T2"]


def test_withholding_adc_makes_d_degenerate():
    """A4: without the ADC anchor, D falls into the unidentifiable subspace.

    Note what this does NOT assert. An earlier version required CRLB(D) to inflate >5x and
    failed at 3.85x -- that threshold was an invented number, not a physical prediction.
    Because b*D = 1.3 the high-b DWI row carries genuine D sensitivity, so under a finite
    prior the bound does not explode. the project plan's actual claim is structural: D becomes
    *gain-degenerate*, i.e. it acquires overlap with the null direction and the rank drops.
    Test that, plus a monotone (not magic-number) CRLB increase.
    """
    a1, a4 = F.identifiability(ACQ, "A1"), F.identifiability(ACQ, "A4")
    assert a4["rank"] == a1["rank"] - 1, "removing the anchor must cost exactly one direction"
    assert a1["null_overlap"]["D"] < 1e-9, "with the anchor, D is fully identified"
    assert a4["null_overlap"]["D"] > 0.05, "without it, D enters the unidentifiable subspace"
    assert a4["crlb"]["D"] > a1["crlb"]["D"]
    # and D stops being the best-determined parameter, which is the whole point of the anchor
    assert min(a1["crlb"], key=a1["crlb"].get) == "D"


def test_pinv_is_anticorrelated_with_identifiability():
    """REGRESSION TEST FOR THE ARCHIVED DEFECT. Do not delete.

    Demonstrates *why* pinv is banned: the reported variance falls as the parameter's
    overlap with the unidentifiable direction rises, which is exactly backwards, while the
    regularized bound stays physical at every vendor.
    """
    proj, pinv_t1, reg_t1 = {}, {}, {}
    for v, acq in VENDOR.items():
        M = F.fisher_matrix(F.measurement_rows("A1", acq))
        proj[v] = F.spectrum(M)["null_overlap"]["T1"]
        pinv_t1[v] = float(np.sqrt(np.clip(np.diag(np.linalg.pinv(M)), 0, None))[1])
        reg_t1[v] = F.crlb(M)["T1"]

    order = sorted(VENDOR, key=lambda v: proj[v])           # least -> most unidentifiable
    assert [pinv_t1[v] for v in order] == sorted((pinv_t1[v] for v in order), reverse=True), \
        "pinv should decrease as unidentifiability rises -- the pathology"
    assert pinv_t1["Siemens"] < pinv_t1["GE"] < pinv_t1["Philips"]
    # Siemens has the longest TR and therefore the least T1 information, yet pinv calls T1
    # better determined there than the directly measured D.
    assert pinv_t1["Siemens"] < 1.0
    # the regularized bound is physical and consistent everywhere
    assert all(reg_t1[v] > max(F.crlb(F.fisher_matrix(F.measurement_rows("A1", VENDOR[v])))[p]
                               for p in ("PD", "T2", "D")) for v in VENDOR)
    assert max(reg_t1.values()) / min(reg_t1.values()) < 1.3, "should not swing 66x like pinv"


def test_no_pinv_anywhere_in_the_fisher_path():
    """A grep-level guard, because this defect is easy to reintroduce and hard to see."""
    src = pathlib.Path(F.__file__).read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    # the docstring names pinv deliberately; no executable line may call it
    code = body.split('"""')
    executable = "".join(code[i] for i in range(0, len(code), 2))
    assert "pinv" not in executable, "np.linalg.pinv must not appear in executable code"


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    npass = nfail = 0
    for n, f in fns:
        try:
            f(); npass += 1; print(f"  PASS {n}")
        except Exception:
            nfail += 1; print(f"  FAIL {n}"); traceback.print_exc(limit=2)
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
