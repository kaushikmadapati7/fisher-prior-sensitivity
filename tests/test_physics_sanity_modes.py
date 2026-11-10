"""Tests for physics_sanity's reporting modes (issue #29 follow-up).

Run with pytest, or standalone:  python tests/test_physics_sanity_modes.py

Uses a stub loader, so no nibabel and no images. These exist because the first real
--bands-only run printed two things that were not true: it labelled a band-check count as an
"ADC-unit measurement", and it printed EMPTY ADC and SNR tables -- which reads as "measured,
found nothing" rather than "skipped". That is the same defect class as the hardcoded b_hi
footer fixed in #22, and it is worth a test rather than a resolution to be careful.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUB = textwrap.dedent('''
    import numpy as np, sys
    import audit.physics_sanity as PS
    def fake_load(path):
        base = path.rsplit("/", 1)[-1]; case = path.split("/")[-2]
        n = (4, 8, 8)
        if "mask" in base: return np.ones(n, np.float32)
        if base.startswith("ADC"): return np.full(n, 1300., np.float32)
        if "b0000" in base: return np.full(n, 500., np.float32)
        if "b1000" in base:
            return np.full(n, 900. if case == "c3" else 200., np.float32)
        return np.full(n, 400., np.float32)
    PS._load = fake_load
    sys.argv = sys.argv[1:]
    PS.main()
''')


def _run(mode, tmp):
    md, img = os.path.join(tmp, "m"), os.path.join(tmp, "img")
    os.makedirs(md, exist_ok=True)
    for i in range(6):
        cid = f"c{i}"
        os.makedirs(os.path.join(img, cid), exist_ok=True)
        json.dump({"case_id": cid, "scanner_model": "SIGNA Premier",
                   "contrasts": {"T2W": {"TR_ms": 8000., "TE_ms": 150., "file_stem": "T2W"},
                                 "DWI": {"TR_ms": 4000., "TE_ms": 70., "b_values": [1000.],
                                         "file_stem": "DWI_b1000_to_T2W"},
                                 "ADC": {"render": "anchor", "file_stem": "ADC_to_T2W"},
                                 "S0": {"TR_ms": 4000., "TE_ms": 70., "b_values": [0.],
                                        "file_stem": "DWI_b0000_to_T2W"}}},
                  open(os.path.join(md, f"{cid}.json"), "w"))
    sp = os.path.join(tmp, "run.py")
    open(sp, "w").write(STUB)
    out = os.path.join(tmp, f"o{len(mode)}.json")
    r = subprocess.run([sys.executable, sp, "x", "--manifest-dir", md, "--image-root", img,
                        "--limit", "0", *mode, "--out", out],
                       capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ, PYTHONPATH=REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, json.load(open(out))


def test_bands_only_does_not_print_empty_adc_or_snr_tables():
    """An empty table means 'measured, found nothing'. Skipping must SAY it skipped."""
    with tempfile.TemporaryDirectory() as d:
        out, _ = _run(["--bands-only"], d)
    assert "SKIPPED (--bands-only)" in out
    assert "median stored/D_est" not in out, "empty ADC table must not be printed"
    assert "SNR bg" not in out, "empty SNR table must not be printed"


def test_bands_only_labels_the_count_for_the_mode_it_counted():
    with tempfile.TemporaryDirectory() as d:
        out, _ = _run(["--bands-only"], d)
    assert "cases band-checked:" in out
    assert "ADC-unit measurement" not in out, "no ADC measurement is made in this mode"


def test_bands_only_finds_the_planted_inverted_case_and_reports_a_rate():
    with tempfile.TemporaryDirectory() as d:
        out, js = _run(["--bands-only"], d)
    assert js["bad_bands"] == ["c3"], js["bad_bands"]
    assert "rate this shard: 1/6" in out


def test_full_mode_still_reports_adc_and_snr():
    """The guard must not have disabled the normal path."""
    with tempfile.TemporaryDirectory() as d:
        out, js = _run([], d)
    assert "=== 1. ADC UNITS" in out and "median stored/D_est" in out
    assert "=== 2. PER-SCANNER SNR" in out
    assert js["adc_ratio_median"] is not None
    assert js["bad_bands"] == ["c3"]


def test_both_modes_agree_on_the_bad_band_list():
    """The exclusion list is the point; the modes must not disagree about it."""
    with tempfile.TemporaryDirectory() as d:
        _, a = _run(["--bands-only"], d)
        _, b = _run([], d)
    assert a["bad_bands"] == b["bad_bands"] == ["c3"]


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
