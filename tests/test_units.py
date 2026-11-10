"""Tests for the diffusion-units boundary (issue #1).

Run with pytest, or standalone:  python tests/test_units.py
Numpy only -- no torch, so these run on a login node.

The failure this guards is silent: a factor-10^6 error in D produces no error, no NaN and no
visible artefact, only a meaningless b*D. So the tests check that the guard FIRES, not merely
that the happy path works.
"""
from __future__ import annotations

import math

import numpy as np

from physharm.data.units import (ADC_STORED_TO_MM2S, D_MAX_MM2S, D_MIN_MM2S,
                                 assert_diffusion_units, to_mm2s, to_physical_adc)


class Skipped(Exception):
    """Raised by a test that needs a dependency absent on this machine.

    `decoders.py` imports torch, which is not installed on the laptop (onboarding 1). Such a
    test is not passing -- it is UNRUN, and the runner reports it separately so it can never
    be mistaken for a pass. It must be run on the cluster before the code is trusted.
    """


def test_constant_matches_the_measurement():
    """audit/physics_sanity.py measured stored/D_est = 1.037e6 over 149 cases."""
    assert abs(1.0 / ADC_STORED_TO_MM2S - 1.037e6) / 1.037e6 < 0.05


def test_conversion_lands_in_physiological_range():
    """Typical prostate ADC ~1300 stored -> ~1.3e-3 mm^2/s."""
    d = to_mm2s(1300.0)
    assert D_MIN_MM2S < d < D_MAX_MM2S
    assert abs(d - 1.3e-3) < 1e-9


def test_guard_fires_on_unconverted_adc():
    """The whole point: stored units must be rejected, with an actionable message."""
    for stored in (1300.0, 850.0, 2000.0):
        try:
            assert_diffusion_units(stored)
        except ValueError as e:
            assert "10^-6" in str(e), "the error must say how to fix it"
        else:
            raise AssertionError(f"guard failed to fire on unconverted ADC {stored}")


def test_guard_fires_on_over_conversion():
    """Applying the factor twice is just as wrong, and must also be caught."""
    try:
        assert_diffusion_units(to_mm2s(to_mm2s(1300.0)))
    except ValueError:
        pass
    else:
        raise AssertionError("guard failed to fire on double conversion")


def test_guard_passes_physiological_values():
    for d in (0.6e-3, 1.3e-3, 2.5e-3):
        assert assert_diffusion_units(d) == d


def test_assert_units_checks_D_when_given():
    """decoders.assert_units validated TR/TE/FA/b and NOTHING about D. It does now."""
    try:
        from physharm.physics.decoders import assert_units
    except ModuleNotFoundError as e:
        raise Skipped(f"needs {e.name}; run on the cluster") from None
    p = {"TR_ms": 5000.0, "TE_ms": 100.0, "b_values": [0.0, 1000.0]}
    assert_units(p, "DWI", D_typical=1.3e-3)          # fine
    assert_units(p, "DWI")                            # D optional, still fine
    try:
        assert_units(p, "DWI", D_typical=1300.0)      # unconverted
    except ValueError:
        pass
    else:
        raise AssertionError("assert_units accepted D in the wrong units")


def test_round_trip_render_then_recover_D():
    """Render at a known D through the signal equation, recover it, agree to <1%.

    Pure numpy so it runs without torch; it exercises the same algebra the decoder uses:
        S(b) = S0 exp(-b D)  =>  D = ln(S_lo/S_hi) / (b_hi - b_lo)
    """
    for stored in (700.0, 1300.0, 2200.0):
        D = to_mm2s(stored)
        assert_diffusion_units(D)
        S0, b_lo, b_hi = 500.0, 0.0, 1000.0
        s_lo, s_hi = S0 * math.exp(-b_lo * D), S0 * math.exp(-b_hi * D)
        rec = math.log(s_lo / s_hi) / (b_hi - b_lo)
        assert abs(rec - D) / D < 0.01, f"stored={stored}: {rec:.4e} vs {D:.4e}"


def test_to_physical_adc_converts_clips_crops_and_guards():
    """The conversion path as the dataset actually calls it, WITHOUT needing nibabel.

    This exists because the guard was originally written inline in
    `dataset_physics._preprocess` and referenced the ROI mask one statement before the mask
    was assigned. It raised UnboundLocalError on the very first real case, and no test could
    see it: nothing in the suite could import that module at all (torch + SimpleITK).
    """
    raw = np.full((4, 8, 8), 1300.0, dtype=np.float32)
    mask = np.zeros((4, 8, 8), dtype=np.float32)
    mask[1:3, 2:6, 2:6] = 1.0
    out = to_physical_adc(raw, mask=mask)
    assert out.shape == raw.shape
    assert abs(float(np.median(out[mask > 0.5])) - 1.3e-3) < 1e-9

    # clipping applies before scaling
    hot = np.full((2, 4, 4), 9e4, dtype=np.float32)
    # float32 arithmetic lands 2.6e-11 above the exact bound; compare with a tolerance
    assert float(np.max(to_physical_adc(hot, clip=3000.0))) <= 3000.0 * ADC_STORED_TO_MM2S * (1 + 1e-6)

    # an all-zero mask must not crash -- it falls back to the whole volume
    to_physical_adc(raw, mask=np.zeros_like(mask))
    # no mask at all is also fine
    to_physical_adc(raw, mask=None)

    # a crop callable is applied
    cropped = to_physical_adc(raw, mask=None, crop=lambda a: a[:, :4, :4])
    assert cropped.shape == (4, 4, 4)


def test_to_physical_adc_guard_fires_on_wrong_units():
    """Passing an already-converted map means the factor is applied twice."""
    already = np.full((4, 8, 8), 1.3e-3, dtype=np.float32)
    try:
        to_physical_adc(already)
    except ValueError:
        return
    raise AssertionError("double conversion was not caught")


def test_wrong_units_would_have_been_silent():
    """Documents WHY the guard is needed: nothing else notices.

    With D in stored units, b*D = 1000*1300 = 1.3e6 and exp(-b*D) underflows to exactly 0 --
    no exception, no NaN, just an all-zero diffusion term that trains happily.
    """
    b, D_wrong = 1000.0, 1300.0
    sig = np.exp(-b * D_wrong)
    assert sig == 0.0 and np.isfinite(sig), "underflow is silent, which is the problem"
    try:
        assert_diffusion_units(D_wrong)
    except ValueError:
        return
    raise AssertionError("the guard is the only thing that catches this")


def test_the_units_guard_cannot_see_a_scale_fault():
    """Why Vida went out silently: the only ADC guard was a units check two orders wide."""
    from physharm.data.units import assert_diffusion_units
    assert_diffusion_units(0.00056)          # Vida as cached -- passes, and should not have


def test_the_roi_scale_check_catches_what_the_units_check_misses():
    from physharm.data.units import roi_scale_problem
    assert roi_scale_problem(0.00146) is None            # GE cohort
    assert roi_scale_problem(0.001585) is None           # Verio as cached
    assert "below" in roi_scale_problem(0.00056)         # Vida as cached, 2.4x low
    assert "above" in roi_scale_problem(0.0119)          # Verio at source, 8x high
    assert roi_scale_problem(float("nan")) is not None
    assert roi_scale_problem(0.0) is not None


def test_scanner_scale_report_flags_only_the_offender():
    from physharm.data.units import scanner_scale_report
    rows, bad = scanner_scale_report({
        "SIGNA PREMIER": [0.00146] * 2075, "DISCOVERY MR750": [0.00149] * 1514,
        "MAGNETOM VIDA": [0.00056] * 20, "VERIO": [0.001585] * 3})
    assert bad == ["MAGNETOM VIDA"]
    assert {r["scanner"]: round(r["ratio"], 2) for r in rows}["MAGNETOM VIDA"] < 0.45


def test_a_scanner_below_min_n_is_reported_but_not_fatal():
    """Three Verio cases should not stop a 4,600-case build on their own."""
    from physharm.data.units import scanner_scale_report
    _, bad = scanner_scale_report({"SIGNA PREMIER": [0.00146] * 500,
                                   "ODDBALL": [0.0004] * 3}, min_n=5)
    assert bad == []


def test_the_audit_reads_the_cache_and_fails_the_build():
    """End to end on a synthetic cache: one scanner off by 2.4x must stop the build."""
    import os, tempfile
    import numpy as np
    from physharm.data.build_cache import audit_adc_scale
    d = tempfile.mkdtemp()
    mans = []
    for i, (sc, adc) in enumerate([("SIGNA PREMIER", 0.00146)] * 30
                                  + [("MAGNETOM VIDA", 0.00056)] * 10):
        cid = f"c{i}"
        np.savez_compressed(os.path.join(d, f"{cid}.npz"),
                            __adc=np.full((4, 8, 8), adc, dtype=np.float32),
                            __mask=np.ones((4, 8, 8), dtype=np.float32))
        mans.append({"case_id": cid, "scanner_model": sc})
    out = []
    try:
        audit_adc_scale(d, mans, strict=True, log=out.append)
        raise AssertionError("strict audit should have stopped the build")
    except SystemExit as e:
        assert "MAGNETOM VIDA" in str(e)
    rows = audit_adc_scale(d, mans, strict=False, log=out.append)
    assert [r["flagged"] for r in rows if r["scanner"] == "MAGNETOM VIDA"] == [True]


def test_in_over_out_of_mask_separates_a_scaling_fault_from_a_mask_fault():
    """The diagnostic that redirected #56 (2026-08-23).

    adc_rescale came back [1.0] on every scanner including MAGNETOM Vida, and Vida's
    whole-volume source median is normal at 1355 x10^-6 while its in-mask median is 561 --
    0.41x its own volume. A global scale factor (adc_rescale, the 3000 clip, a units
    convention) multiplies every voxel equally, so it moves the in-mask and out-of-mask
    medians together and leaves their RATIO untouched. Only voxel selection moves the ratio.

    The shifted-mask scenario below reproduces 0.39 against Vida's observed 0.41.
    """
    import numpy as np
    sh = (4, 16, 16)

    def case(scale=1.0, mask_shift=0):
        adc = np.full(sh, 1.45e-3, dtype=np.float32)
        adc[:, :6, :] = 0.56e-3                     # a low-ADC structure nearby
        mask = np.zeros(sh, dtype=np.float32)
        mask[:, 8 + mask_shift:12 + mask_shift, 6:12] = 1.0
        return adc * scale, mask

    def in_over_out(adc, mask):
        return float(np.median(adc[mask > 0.5])) / float(np.median(adc[mask <= 0.5]))

    assert abs(in_over_out(*case()) - 1.0) < 1e-6                      # healthy
    assert abs(in_over_out(*case(scale=0.41)) - 1.0) < 1e-6            # global scale: unmoved
    assert in_over_out(*case(mask_shift=-6)) < 0.5                     # displaced mask: moved


if __name__ == "__main__":
    import traceback
    npass = nfail = nskip = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f) and n != "Skipped":
            try:
                f(); npass += 1; print(f"  PASS {n}")
            except Skipped as e:
                nskip += 1; print(f"  SKIP {n}  ({e})")
            except Exception:
                nfail += 1; print(f"  FAIL {n}"); traceback.print_exc(limit=2)
    print(f"\n{npass} passed, {nfail} failed, {nskip} SKIPPED (unrun -- not a pass)")
    raise SystemExit(1 if nfail else 0)
