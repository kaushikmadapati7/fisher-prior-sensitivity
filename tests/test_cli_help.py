"""Every command-line entry point must be able to print its own help. (#57)

WHY THIS EXISTS. `traveling_arm` shipped an argparse help string containing a literal "23%".
argparse %-interpolates help text, so an unescaped % raises `ValueError: badly formed help
string` -- at `add_argument` time, which means the module could not run AT ALL, with or without
the new flag. Every unit test passed, because they call the analysis functions directly and
nothing in the suite had ever invoked a `main()`.

This runs each entry point with --help in a subprocess. It is the cheapest possible check that
the thing a person actually types still works.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _entry_points():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "physharm", "**", "*.py"), recursive=True)
                    + glob.glob(os.path.join(ROOT, "audit", "*.py"))):
        src = open(f).read()
        if "add_argument" not in src or "__main__" not in src:
            continue
        rel = os.path.relpath(f, ROOT)[:-3].replace(os.sep, ".")
        out.append(rel)
    return out


def test_every_cli_prints_help():
    """A ModuleNotFoundError is the environment; anything else is the code.

    `physharm.manifest.build_manifest` needs torch, and two audit scripts import a sibling by
    bare name so they only resolve when run from inside `audit/`. Neither is what this test is
    for, and failing on them would make it noise that gets muted. Everything else -- and in
    particular an argparse construction error, which fires before any argument is parsed -- is
    a real break and fails here.
    """
    mods = _entry_points()
    assert len(mods) >= 5, f"expected several entry points, found {mods}"
    env = dict(os.environ, PYTHONPATH=ROOT)
    broken, skipped = [], []
    for m in mods:
        r = subprocess.run([sys.executable, "-m", m, "--help"],
                           capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        if r.returncode == 0 and "usage" in (r.stdout + r.stderr).lower():
            continue
        err = (r.stderr or r.stdout).strip()
        last = err.splitlines()[-1] if err.splitlines() else "?"
        if "ModuleNotFoundError" in err:
            skipped.append((m, last))
        else:
            broken.append((m, last))
    if skipped:
        print("    (skipped, missing dependency: "
              + ", ".join(m for m, _ in skipped) + ")")
    assert not broken, "entry points that cannot print --help:\n" + "\n".join(
        f"  {m}: {e}" for m, e in broken)


def test_the_guard_catches_a_bad_help_string():
    """The exact defect this file exists for: an unescaped %% in argparse help."""
    import argparse
    ap = argparse.ArgumentParser()
    try:
        # Older argparse raises here; 3.12+ defers validation to formatting time, which is
        # why the guard has to render the help rather than just build the parser. The real
        # #62 defect surfaced at --help, not at add_argument.
        ap.add_argument("--x", help="which is 23% of them")
        ap.format_help()
        raise AssertionError("argparse accepted an unescaped percent")
    except (ValueError, TypeError) as e:
        # 3.11 and earlier: ValueError("badly formed help string").
        # 3.12+: the deferred %-substitution surfaces as TypeError instead.
        assert "badly formed help string" in str(e) or "format" in type(e).__name__.lower() \
            or isinstance(e, TypeError), e


def test_image_root_that_cannot_work_is_refused_not_silently_ignored():
    """Two rounds of the ADC diagnostic reported nan for every scanner and never said why."""
    import tempfile
    from audit.adc_scale_provenance import _check_image_root
    try:
        _check_image_root("/nope/does/not/exist")
        raise AssertionError("a missing image root was accepted")
    except SystemExit as e:
        assert "not a directory" in str(e) and "registered volumes" in str(e)
    empty = tempfile.mkdtemp()
    try:
        _check_image_root(empty)
        raise AssertionError("an image root with no NIfTI was accepted")
    except SystemExit as e:
        assert "no .nii.gz" in str(e)


def test_adc_scale_provenance_actually_runs_on_a_synthetic_cache():
    """--help is not enough: it never enters the loop where the work happens.

    `whole_c` was spliced one indent too deep, landing inside an `except` block after its
    `continue`, so it was dead code and the line that used it raised UnboundLocalError on the
    first real case. Every test passed -- test_every_cli_prints_help included, because --help
    returns before the loop runs.

    This builds a two-scanner cache and a manifest dir and runs main() end to end. It is the
    same defect class as the mask_c UnboundLocalError from #1, which also survived a green
    suite, and the fifth time on this project that tests passed while an entry point was broken.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import numpy as np

    d = tempfile.mkdtemp()
    cd_, md = os.path.join(d, "cache"), os.path.join(d, "man")
    os.makedirs(cd_); os.makedirs(md)
    for i in range(24):
        sc = "SIGNA PREMIER" if i % 2 else "MAGNETOM VIDA"
        adc = np.full((4, 8, 8), 0.00145 if i % 2 else 0.00056, dtype=np.float32)
        mask = np.zeros((4, 8, 8), dtype=np.float32); mask[:, 2:6, 2:6] = 1.0
        np.savez_compressed(os.path.join(cd_, f"c{i}.npz"), __adc=adc, __mask=mask)
        json.dump({"case_id": f"c{i}", "scanner_model": sc,
                   "contrasts": {"ADC": {"file_stem": f"ADC_to_T2W_{i}"}}},
                  open(os.path.join(md, f"c{i}.json"), "w"))
    r = subprocess.run([sys.executable, "-m", "audit.adc_scale_provenance",
                        "--cache-dir", cd_, "--manifest-dir", md],
                       capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=ROOT), timeout=180)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert "MAGNETOM VIDA" in r.stdout and "SIGNA PREMIER" in r.stdout
    assert "Traceback" not in r.stderr


def _tiny_cohort(dirpath, n=120):
    """A cache + manifest pair small enough to run main() on, real enough to exercise it."""
    import json
    import os
    import numpy as np
    from physharm.analysis.synthetic_arm import PARAMS
    cd_, md = os.path.join(dirpath, "cache"), os.path.join(dirpath, "man")
    os.makedirs(cd_, exist_ok=True); os.makedirs(md, exist_ok=True)
    rng = np.random.default_rng(0)
    true = {"PD": 1.4, "T1": 900.0, "T2": 120.0, "D": 1.1e-3}
    for i in range(n):
        a = dict(TR_T2W=float(rng.uniform(4000, 9000)), TE_T2W=float(rng.uniform(90, 180)),
                 TR_DWI=float(rng.uniform(3000, 6000)), TE_DWI=float(rng.uniform(55, 95)))
        sh = (2, 10, 10); nv = int(np.prod(sh))
        th = {k: true[k] * np.exp(rng.normal(0, 0.3, nv)) for k in PARAMS}
        PD, T1, T2, D = (th[k] for k in PARAMS)
        t2w = PD * (1 - np.exp(-a["TR_T2W"] / T1)) * np.exp(-a["TE_T2W"] / T2)
        dwi = (PD * (1 - np.exp(-a["TR_DWI"] / T1)) * np.exp(-a["TE_DWI"] / T2)
               * np.exp(-1000.0 * D))
        np.savez_compressed(os.path.join(cd_, f"c{i}.npz"),
                            T2W=t2w.reshape(sh).astype("f4"), DWI=dwi.reshape(sh).astype("f4"),
                            __adc=D.reshape(sh).astype("f4"),
                            __mask=np.ones(sh, dtype="f4"))
        json.dump({"case_id": f"c{i}", "patient_id": f"p{i//2}",
                   "scanner_model": ["SIGNA PREMIER", "DISCOVERY MR750"][i % 2],
                   "contrasts": {"T2W": {"TR_ms": a["TR_T2W"], "TE_ms": a["TE_T2W"]},
                                 "DWI": {"TR_ms": a["TR_DWI"], "TE_ms": a["TE_DWI"],
                                         "b_values": [1000.0]}}},
                  open(os.path.join(md, f"c{i}.json"), "w"))
    return cd_, md


def test_prior_shift_and_real_arm_main_actually_run():
    """The two modules every headline number came from had never had main() executed.

    Found in the 2026-08-23 audit. Both have thorough unit tests -- 20 and 15 -- and all of them
    call analysis functions directly. Nothing had ever run the path a person types. That is the
    same gap that produced the missing-modules crash, the mask_c UnboundLocalError, the silent
    empty hold-out, the `23%` argparse crash and the whole_c UnboundLocalError: five defects, all
    in entry points, all under a green suite.

    Deliberately small and synthetic. It asserts the run completes and emits its primary table,
    not that any number is right -- correctness of the numbers is what the unit tests are for.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    d = tempfile.mkdtemp()
    cd_, md = _tiny_cohort(d)
    env = dict(os.environ, PYTHONPATH=ROOT)
    for mod, marker, extra in (
            ("physharm.analysis.prior_shift", "PRIMARY", ["--n-vox", "200"]),
            ("physharm.analysis.real_arm", "PRIMARY", ["--n-vox", "200"])):
        out = os.path.join(d, f"{mod.rsplit('.', 1)[-1]}.json")
        r = subprocess.run([sys.executable, "-m", mod, "--cache-dir", cd_,
                            "--manifest-dir", md, "--out", out] + extra,
                           capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
        assert r.returncode == 0, f"{mod}\n" + (r.stdout + r.stderr)[-1200:]
        assert marker in r.stdout, f"{mod} produced no primary table"
        assert os.path.exists(out), f"{mod} wrote no output file"
        json.load(open(out))


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
